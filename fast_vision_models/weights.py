"""Checkpoint lookup: weights are never shipped with this package.

``fetch(url)`` returns a local path: the file of the same name in one of the ``FVM_WEIGHTS`` directories
(``os.pathsep``-separated, searched recursively one level deep), else a download from the official URL into
``~/.cache/fast-vision-models/weights``. Hugging Face / timm / torchvision weights use their own caches.
"""

import os

import torch

from ._build import cache_dir


def fetch(url):
    name = url.rsplit("/", 1)[-1]
    for root in filter(None, os.environ.get("FVM_WEIGHTS", "").split(os.pathsep)):
        for d in [root] + [os.path.join(root, s) for s in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, s))]:
            p = os.path.join(d, name)
            if os.path.exists(p):
                return p
    p = os.path.join(cache_dir("weights"), name)
    if not os.path.exists(p):
        torch.hub.download_url_to_file(url, p + ".part")
        os.replace(p + ".part", p)
    return p
