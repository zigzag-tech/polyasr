#!/usr/bin/env python3
"""Cheap CUDA runtime validation performed before model weights are loaded."""

import importlib


def _cuda_version(module):
    return getattr(getattr(module, "version", None), "cuda", None)


def validate_pytorch_stack(import_module=importlib.import_module):
    try:
        torch = import_module("torch")
        torchaudio = import_module("torchaudio")
        torchvision = import_module("torchvision")
        import_module("silero_vad")
        import_module("resemblyzer")
    except Exception as exc:
        raise RuntimeError(
            "PyTorch audio stack failed to import. Reinstall the pinned CUDA "
            "stack with cuda/install.sh before starting PolyASR."
        ) from exc

    torch_cuda = _cuda_version(torch)
    torchaudio_cuda = _cuda_version(torchaudio)
    # Some TorchAudio releases do not expose torchaudio.version.cuda. Its
    # extension import above still performs the compiled CUDA ABI check.
    if torchaudio_cuda is not None and torch_cuda != torchaudio_cuda:
        raise RuntimeError(
            "PyTorch/TorchAudio CUDA mismatch: "
            f"torch={torch.__version__} (CUDA {torch_cuda}), "
            f"torchaudio={torchaudio.__version__} (CUDA {torchaudio_cuda}). "
            "Reinstall the pinned CUDA stack with cuda/install.sh."
        )

    torch_release = torch.__version__.split("+", 1)[0].rsplit(".", 1)[0]
    torchaudio_release = torchaudio.__version__.split("+", 1)[0].rsplit(".", 1)[0]
    if torch_release != torchaudio_release:
        raise RuntimeError(
            "PyTorch/TorchAudio release mismatch: "
            f"torch={torch.__version__}, torchaudio={torchaudio.__version__}. "
            "Reinstall the pinned CUDA stack with cuda/install.sh."
        )

    return {
        "torch": torch.__version__,
        "torchaudio": torchaudio.__version__,
        "torchvision": torchvision.__version__,
        "cuda": torch_cuda,
    }


if __name__ == "__main__":
    versions = validate_pytorch_stack()
    print(
        "PolyASR runtime preflight passed: "
        + ", ".join(f"{name}={value}" for name, value in versions.items())
    )
