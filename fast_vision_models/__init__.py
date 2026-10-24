"""Fast inference paths for vision models: classification, semantic segmentation and feature backbones.

Each family offers the tiers "upstream" (official code, fp32), "portable" (plain PyTorch only, ONNX / TensorRT
exportable) and "max" (custom CUDA kernels). See README.md for the measured speed, memory, startup and quality.
"""

__version__ = "0.1.0"
