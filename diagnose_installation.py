from __future__ import annotations

import sys


def main() -> int:
    import cv2
    import numpy
    import PIL
    import PySide6
    import torch

    print("Python:", sys.version)
    print("PyTorch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
    print("PySide6:", PySide6.__version__)
    print("NumPy:", numpy.__version__, "OpenCV:", cv2.__version__, "Pillow:", PIL.__version__)
    if not torch.cuda.is_available():
        print("ERROR: PyTorch cannot access an NVIDIA GPU.")
        return 2
    print("GPU:", torch.cuda.get_device_name(0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
