"""Unified benchmark: one model, one tier per process.

    python bench/bench.py --family cls --model resnet50 --tier cuda-kernels --out results/runs.jsonl
    python bench/bench.py --family seg --model segformer-b0 --tier flags
    python bench/bench.py --family bb --model dinov2-vitl14 --tier portable

Tiers: upstream (fp32, PyTorch defaults) | flags (unmodified upstream + inference_mode, cudnn.benchmark, TF32, model.half() or
fp16 autocast, channels_last; classification and ViT backbones also allow_fp16_accumulation) | portable | max.
portable and max run as shipped: they set their own fp16-accumulate flags, no cudnn.benchmark, no global switches.
Records (JSON line): startup_s (process start -> model ready, incl. import torch and CUDA init), first_call_s (first call minus
median, size 1), median ms and peak allocated / reserved MB per size (weights included), new_size_s (first call at size 2
minus its median, incl. cudnn.benchmark autotuning) and a quality check against upstream fp32, computed after timing:
cls top-1 agreement on ImageNet val images ($FVM_IMAGENET class folders; else logits rel L2), seg pixel agreement on
ADE20K val images ($FVM_ADE20K), bb rel L2 of the output on the timed input.
Sizes: cls b1 -> b64 (halved on OOM); seg 512x512 -> 1024x2048; bb DINOv2 896x1344 -> 448x672, DINOv3 512 -> 1024,
timm 1024 b1 -> 224 b32.
"""

import time

T0 = time.perf_counter()

import argparse  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

os.environ.setdefault("XFORMERS_DISABLED", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

CL = torch.channels_last
# models whose unmodified upstream gives wrong / non-finite output after model.half(): flags tier uses fp16 autocast
AUTOCAST = {"efficientnet_b4", "segnext-t", "eomt-l-dinov3", "dinov3-vitl16"}
SEG_FIXED = {"unet-s5", "ddrnet-23", "pidnet-l"}  # Cityscapes nets: quality images resized to 512x1024


def set_flags(fp16acc):
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_accumulation = bool(fp16acc)


def reset_flags():
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True  # PyTorch default
    torch.backends.cuda.matmul.allow_fp16_accumulation = False


def free():
    gc.collect()
    torch.cuda.empty_cache()


def wrap_prec(fn, mode, cl=False):
    """flags tier: run fn on half / channels_last inputs or under fp16 autocast."""
    if mode == "half":
        return lambda x: fn(x.half().contiguous(memory_format=CL) if cl else x.half())

    def run(x):
        with torch.autocast("cuda", dtype=torch.float16):
            return fn(x.contiguous(memory_format=CL) if cl else x)

    return run


def to_flags(obj, half, seen=None):
    """in place: modules -> (half) channels_last, float tensors -> half, recursing into function closures (seg wrappers)."""
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return
    seen.add(id(obj))
    if isinstance(obj, torch.nn.Module):
        if half:
            obj.half()
        obj.to(memory_format=CL)
    for cell in getattr(obj, "__closure__", None) or ():
        v = cell.cell_contents
        if isinstance(v, torch.Tensor) and half and v.is_floating_point():
            cell.cell_contents = v.half()
        elif isinstance(v, torch.nn.Module) or callable(v):
            to_flags(v, half, seen)


# ------------------------------------------------------------------------------------------------ families
def cls_build(name, tier):
    from fast_vision_models.classification import zoo

    if tier == "flags":
        set_flags(True)
    m = zoo.load(name, "base" if tier in ("upstream", "flags") else tier)
    C, H, W = m.input_size
    fn, prec = m.model, "fp32" if tier == "upstream" else "own fp16 path"
    if tier == "flags":
        prec = "autocast-fp16" if name in AUTOCAST else "half"
        if prec == "half":
            m.model.half()
        m.model.to(memory_format=CL)
        fn = wrap_prec(m.model, prec, cl=True)
    g = torch.Generator(device="cuda").manual_seed(0)
    sizes = [("b1", lambda: torch.randn(1, C, H, W, device="cuda", generator=g)),
             ("b64", lambda: torch.randn(64, C, H, W, device="cuda", generator=g))]
    return fn, sizes, prec, m


def cls_quality(name, fn, x_timed, out_timed):
    from fast_vision_models.classification import zoo

    root = os.environ.get("FVM_IMAGENET")
    xs = None
    if root:
        from PIL import Image

        m = zoo.load(name, "base", device="cpu")
        files = [os.path.join(root, c, f) for c in sorted(os.listdir(root)) for f in sorted(os.listdir(os.path.join(root, c)))]
        xs = torch.stack([m.transform(Image.open(p).convert("RGB")) for p in files[::195][:256]])
        del m
    with torch.inference_mode():
        outs = None if xs is None else torch.cat([fn(xs[i : i + 32].cuda()).float().cpu() for i in range(0, len(xs), 32)])
    return xs, outs


def seg_build(name, tier):
    from fast_vision_models.segmentation import common as C
    from fast_vision_models.segmentation import zoo

    if tier == "flags":
        set_flags(False)
    w = zoo.build(name, "base" if tier in ("upstream", "flags") else tier)
    prec = "fp32" if tier == "upstream" else "own fp16 path"
    fn = lambda x: C.classmap(w, x)  # noqa: E731
    if tier == "flags":
        prec = "autocast-fp16" if name in AUTOCAST else "half"
        to_flags(w.logits, prec == "half")
        if w.module is not None:
            to_flags(w.module, prec == "half")
        fn = wrap_prec(fn, prec, cl=True)
    g = torch.Generator(device="cuda").manual_seed(0)
    sizes = [("512x512", lambda: torch.randn(1, 3, 512, 512, device="cuda", generator=g)),
             ("1024x2048", lambda: torch.randn(1, 3, 1024, 2048, device="cuda", generator=g))]
    return fn, sizes, prec, w


def seg_inputs(name, w):
    from fast_vision_models.segmentation import common as C

    if not os.path.isdir(C.ADE):
        return None
    xs = []
    for ip, ap in C.ade_val("all")[::41][:12]:
        x = C.preprocess(C.load_pair(ip, ap)[0], w.protocol["resize"], w.norm)
        xs.append(F.interpolate(x, (512, 1024), mode="bilinear") if name in SEG_FIXED else x)
    return xs


def bb_sizes(name):
    if name.startswith("dinov2"):
        return [("896x1344 b1", (1, 896, 1344)), ("448x672 b1", (1, 448, 672))]
    if name.startswith("dinov3"):
        return [("512px b1", (1, 512, 512)), ("1024px b1", (1, 1024, 1024))]
    return [("1024px b1", (1, 1024, 1024)), ("224px b32", (32, 224, 224))]


def bb_input(b, shape):
    """ImageNet-normalized image: $FVM_IMAGE (any RGB image) resized, else a smooth random field (seed 0)."""
    B, H, W = shape
    p = os.environ.get("FVM_IMAGE")
    if p:
        from PIL import Image

        im = torch.from_numpy(__import__("numpy").asarray(Image.open(p).convert("RGB"))).cuda().permute(2, 0, 1)[None].float() / 255
        im = F.interpolate(im, size=(H, W), mode="bilinear", antialias=True)
    else:
        g = torch.Generator(device="cuda").manual_seed(0)
        im = F.interpolate(torch.rand(1, 3, H // 16, W // 16, device="cuda", generator=g), size=(H, W), mode="bicubic").clamp(0, 1)
    m = torch.tensor(b.mean, device="cuda").view(1, 3, 1, 1)
    s = torch.tensor(b.std, device="cuda").view(1, 3, 1, 1)
    return ((im - m) / s).expand(B, -1, -1, -1).contiguous()


def bb_build(name, tier):
    from fast_vision_models import backbones

    vit = name.startswith("dinov2") or name.startswith("dinov3-vit")
    if tier == "flags":
        set_flags(vit)
    b = backbones.build(name, "upstream" if tier == "flags" else tier)
    fn, prec = b.fn, "fp32" if tier == "upstream" else "own fp16 path"
    if tier == "flags":
        prec = "autocast-fp16" if name in AUTOCAST else "half"
        if prec == "half":
            b.module.half()
        if not vit:
            b.module.to(memory_format=CL)
        fn = wrap_prec(b.fn, prec, cl=not vit)
    sizes = [(lab, lambda s=s: bb_input(b, s)) for lab, s in bb_sizes(name)]
    return fn, sizes, prec, b


# ------------------------------------------------------------------------------------------------ timing
def timed(fn, x, n):
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn(x)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return ts


def measure(fn, x, n, warm):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    first = timed(fn, x, 1)[0]
    timed(fn, x, warm)
    probe = statistics.median(timed(fn, x, 2))
    n = max(5, n // 4) if probe > 1.0 else (max(5, n // 2) if probe > 0.2 else n)
    med = statistics.median(timed(fn, x, n))
    return {"first_ms": round(first * 1e3, 2), "ms": round(med * 1e3, 3), "n": n,
            "alloc_mb": torch.cuda.max_memory_allocated() >> 20, "res_mb": torch.cuda.max_memory_reserved() >> 20}


def rel_l2(o, r):
    o, r = o.float().cpu(), r.float().cpu()
    return float("nan") if not torch.isfinite(o).all() else round(float((o - r).norm() / r.norm()), 5)


def git_rev():
    try:
        return subprocess.check_output(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "--short", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["cls", "seg", "bb"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--tier", required=True, choices=["upstream", "flags", "portable", "cuda-kernels"])
    ap.add_argument("--n", type=int, default=0, help="timed runs (default cls 30, seg 20, bb 10)")
    ap.add_argument("--no-quality", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    torch.zeros(1, device="cuda")
    t_cuda = time.perf_counter()
    build = {"cls": cls_build, "seg": seg_build, "bb": bb_build}[a.family]
    with torch.inference_mode():
        fn, sizes, prec, obj = build(a.model, a.tier)
    torch.cuda.synchronize()
    t_ready = time.perf_counter()
    n = a.n or {"cls": 30, "seg": 20, "bb": 10}[a.family]
    warm = {"cls": 12, "seg": 4, "bb": 3}[a.family]
    rec = {"family": a.family, "model": a.model, "tier": a.tier, "precision": prec,
           "startup_s": round(t_ready - T0, 3), "cuda_init_s": round(t_cuda - T0, 3), "build_s": round(t_ready - t_cuda, 3)}
    xq = oq = None
    with torch.inference_mode():
        for i, (lab, mk) in enumerate(sizes):
            free()
            for _ in range(3):
                try:
                    x = mk()
                    r = measure(fn, x, n if i == 0 else max(5, n // 3), warm if i == 0 else 3)
                    break
                except torch.OutOfMemoryError:
                    if a.family != "cls":
                        raise
                    free()
                    B = x.shape[0] // 2 if "x" in dir() else 32
                    lab, mk = f"b{B}", (lambda B=B, s=x.shape[1:]: torch.randn(B, *s, device="cuda"))
            r["first_call_s" if i == 0 else "new_size_s"] = round((r["first_ms"] - r["ms"]) / 1e3, 3)
            rec[lab] = r
            if i == 0:
                xq, oq = x.clone(), fn(x)
                oq = oq.float().cpu() if a.family != "seg" else oq.cpu()
            del x
    if not a.no_quality and a.tier != "upstream":
        rec["quality"] = quality(a, fn, obj, xq, oq)
    rec.update(gpu=torch.cuda.get_device_name(), torch=torch.__version__, commit=git_rev(), date=time.strftime("%Y-%m-%d %H:%M"),
               command=" ".join(["python", *sys.argv]))
    line = json.dumps(rec)
    print(line, flush=True)
    if a.out:
        with open(a.out, "a") as f:
            f.write(line + "\n")


def quality(a, fn, obj, xq, oq):
    """tier output vs upstream fp32 (PyTorch default flags), computed after all timing."""
    with torch.inference_mode():
        if a.family == "cls":
            xs, outs = cls_quality(a.model, fn, xq, oq)
            del fn, obj
            free()
            reset_flags()
            from fast_vision_models.classification import zoo

            ref = zoo.load(a.model, "base").model
            if xs is not None:
                r = torch.cat([ref(xs[i : i + 32].cuda()).float().cpu() for i in range(0, len(xs), 32)])
                ok = torch.isfinite(outs).all()
                return {"top1_agree_pct": round(100 * float((outs.argmax(1) == r.argmax(1)).float().mean()), 2) if ok else 0.0,
                        "images": len(xs)}
            return {"logits_rel_l2": rel_l2(oq, ref(xq))}
        if a.family == "seg":
            from fast_vision_models.segmentation import common as C
            from fast_vision_models.segmentation import zoo

            xs = seg_inputs(a.model, obj)
            if xs is None:
                return None
            outs = []
            for x in xs:
                torch.manual_seed(0)
                outs.append(fn(x).reshape(-1).cpu())
            del fn, obj
            free()
            reset_flags()
            ref = zoo.build(a.model, "base")
            ok = n = 0
            for x, o in zip(xs, outs):
                torch.manual_seed(0)
                r = C.classmap(ref, x).reshape(-1).cpu()
                ok += int((o == r).sum())
                n += r.numel()
            return {"pixel_agree_pct": round(100 * ok / n, 2), "images": len(xs)}
        from fast_vision_models import backbones

        del fn, obj
        free()
        reset_flags()
        ref = backbones.build(a.model, "upstream")
        return {"rel_l2": rel_l2(oq, ref.fn(xq))}


if __name__ == "__main__":
    main()
    os._exit(0)
