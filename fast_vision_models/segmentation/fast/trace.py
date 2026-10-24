"""Graph-free launch-overhead removal for the max path: a TorchScript trace of a module, run by the C++ executor with all
fusers off (plain aten dispatch, nothing compiled or specialized per input size). Pattern from exp 008 clf_fast.Fast.

    net = Traced(module)          # module(x) -> tensor / tuple / list of tensors; x one tensor
    y = net(x)                    # 1st call: eager warm-up, trace, validate; later calls: the trace (or eager on mismatch)

Rules (exp 008 lessons):
- every custom kernel must be a torch op (torch.ops.segx.*): a raw ctypes launch is invisible to the tracer;
- shape-dependent decisions belong inside the C++ ops (output sizes, scales, tile choices) or must be derived from traced
  tensor sizes; Python-side caches are built by the eager warm-up before tracing;
- validation on FRESH random inputs, traced first (stale allocator memory cannot fake a match), at the call's shape AND a
  second shape with different H and W (catches sizes frozen into the trace); CPU / CUDA RNG forked and re-seeded so random
  draws inside the net (SegNeXt NMF bases) match. Any mismatch -> eager for good (printed once).
- the caller keeps global flags that the trace cannot record (fp16-accumulate GEMM flag, F.interpolate patch) around the call.
"""
import os

import torch

ENABLED = os.environ.get("SEG_TRACE", "1") == "1"
TOL = float(os.environ.get("SEG_TRACE_TOL", "1e-2"))


_INVISIBLE = []


def mark_invisible(what):
    """called by code paths a trace cannot record (raw ctypes launches, cuDNN-frontend executions) while tracing: the trace
    is then rejected (a replay would read uninitialized memory) instead of relying on the validation to notice."""
    if torch.jit.is_tracing():
        _INVISIBLE.append(what)


def jit_plain():
    """TorchScript executor without fusers / codegen: plain aten dispatch from C++."""
    torch._C._jit_set_texpr_fuser_enabled(False)
    torch._C._jit_override_can_fuse_on_gpu(False)
    torch._C._jit_override_can_fuse_on_cpu(False)
    if hasattr(torch._C, "_jit_set_nvfuser_enabled"):
        try:
            torch._C._jit_set_nvfuser_enabled(False)
        except Exception:  # noqa: BLE001
            pass
    if os.environ.get("SEG_TRACE_OPT", "0") != "1":   # no profiling runs / shape-guarded re-optimization: run the graph as traced
        torch._C._set_graph_executor_optimize(False)


def _flat(y):
    if isinstance(y, torch.Tensor):
        return [y]
    if isinstance(y, (list, tuple)):
        return [t for v in y for t in _flat(v)]
    if isinstance(y, dict):
        return [t for v in y.values() for t in _flat(v)]
    return []


def alt_shape(shape):
    """a second validation shape: different H and W (not multiples of each other's ratio), >= 32."""
    n, c, h, w = shape
    h2 = max(32, (h // 2) // 32 * 32)
    w2 = max(32, (w * 3 // 4) // 32 * 32)
    if h2 == h:
        h2 = h + 32
    if w2 == w:
        w2 = w + 64
    return (n, c, h2, w2)


def _same(r, t, tol):
    fr, ft = _flat(r), _flat(t)
    if len(fr) != len(ft):
        return False, "n_out"
    for a, b in zip(fr, ft):
        if a.shape != b.shape or a.dtype != b.dtype:
            return False, f"shape {tuple(a.shape)} vs {tuple(b.shape)}"
        if a.is_floating_point():
            a, b = a.float(), b.float()
            d = float((a - b).abs().max()) if a.numel() else 0.0
            if not d <= tol * (float(a.abs().max()) + 1e-3):
                return False, f"max diff {d:.3g}"
        elif not bool((a == b).all()):
            return False, "int mismatch"
    return True, ""


class Traced:
    def __init__(self, module, name="", shapes=alt_shape, tol=TOL, enabled=ENABLED):
        self.module, self.name, self.shapes, self.tol, self.enabled = module, name, shapes, tol, enabled
        self.tr = None           # None: not built yet; False: rejected (eager); else the ScriptModule
        self.info = {"trace": "pending" if enabled else "off"}

    def _run_seeded(self, f, x, seed):
        with torch.random.fork_rng(devices=[x.device] if x.is_cuda else []):
            torch.manual_seed(seed)
            return f(x)

    def build(self, x):
        # the whole build runs on forked RNG states: warm-up / tracing / validation consume no draws, so the caller's
        # random stream (SegNeXt's per-forward NMF bases) is the same as without tracing
        with torch.random.fork_rng(devices=[x.device] if x.is_cuda else []):
            self._build(x)

    def _build(self, x):
        jit_plain()
        with torch.no_grad():
            self.module(x)                                       # eager warm-up: Python-side caches, cuDNN plans
            del _INVISIBLE[:]
            tr = None
            try:
                mod = self.module
                if isinstance(mod, torch.nn.Module):           # trace a function, not the module: no recursive
                    mod.requires_grad_(False)                  # ScriptModule wrapping of ~1000 submodules (HRNet trace
                                                               # 0.66 s -> see results); weights become graph constants

                    def fn(t):
                        return mod(t)
                else:
                    fn = mod
                tr = torch.jit.trace(fn, x, check_trace=False, strict=False)
                why, ok = "", True
                if _INVISIBLE:
                    ok, why = False, "invisible ops: " + ",".join(sorted(set(_INVISIBLE)))
                for k, shp in enumerate([tuple(x.shape)] + ([self.shapes(tuple(x.shape))] if self.shapes else [])):
                    if not ok:
                        break
                    xi = torch.randn(shp, device=x.device, dtype=torch.float32).to(x.dtype)
                    if x.dim() == 4 and x.is_contiguous(memory_format=torch.channels_last) and not x.is_contiguous():
                        xi = xi.contiguous(memory_format=torch.channels_last)
                    t = self._run_seeded(tr, xi, 1234 + k)
                    r = self._run_seeded(self.module, xi, 1234 + k)
                    ok, why = _same(r, t, self.tol)
                    if not ok:
                        why = f"shape {shp}: {why}"
                        break
            except Exception as e:  # noqa: BLE001 - untraceable Python (e.g. size checks on traced ints) -> eager
                ok, why = False, f"trace failed: {type(e).__name__}: {str(e).splitlines()[0][:120]}"
        self.tr = tr if ok else False
        self.info = {"trace": "ok" if ok else f"rejected ({why})"}
        if not ok:
            print(f"[seg_fast.trace] {self.name}: trace mismatch -> eager ({why})")

    def __call__(self, x):
        if not self.enabled:
            return self.module(x)
        if self.tr is None:
            self.build(x)
        return (self.tr or self.module)(x)
