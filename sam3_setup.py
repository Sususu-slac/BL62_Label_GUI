from __future__ import annotations

import json
import os
from pathlib import Path


def valid_repository(path: str | Path) -> bool:
    root = Path(path).expanduser()
    return (root / "pyproject.toml").is_file() and (root / "sam3" / "__init__.py").is_file()


def valid_checkpoint(path: str | Path) -> bool:
    file = Path(path).expanduser()
    return file.is_file() and file.suffix.lower() in {".pt", ".pth", ".ckpt"} and file.stat().st_size > 100 * 1024 * 1024


def detect_repository(app_dir: Path, configured: str = "") -> str:
    candidates = [
        configured,
        os.environ.get("SAM3_REPOSITORY", ""),
        app_dir / "sam3",
        app_dir.parent / "sam3",
        Path(r"D:\SAM3p1\sam3"),
        Path(r"C:\SAM3p1\sam3"),
        Path.home() / "sam3",
    ]
    for candidate in candidates:
        if candidate and valid_repository(candidate):
            return str(Path(candidate).resolve())
    return ""


def detect_checkpoint(app_dir: Path, repository: str = "", configured: str = "") -> str:
    candidates = [configured, os.environ.get("SAM3_CHECKPOINT", "")]
    roots = [app_dir / "checkpoints", app_dir.parent / "checkpoints", Path(r"D:\SAM3p1"), Path(r"C:\SAM3p1")]
    if repository:
        roots.extend([Path(repository), Path(repository).parent, Path(repository) / "checkpoints"])
    for root in roots:
        if root.is_dir():
            candidates.extend(sorted(root.glob("*.pt")))
            candidates.extend(sorted(root.glob("*.pth")))
            candidates.extend(sorted(root.glob("*.ckpt")))
    for candidate in candidates:
        if candidate and valid_checkpoint(candidate):
            return str(Path(candidate).resolve())
    return ""


def save_settings(path: Path, repository: str, checkpoint: str) -> None:
    path.write_text(json.dumps({"repo": repository, "checkpoint": checkpoint}, indent=2), encoding="utf-8")


def diagnostic(repository: str, checkpoint: str) -> list[tuple[str, bool, str]]:
    results = []
    results.append(("SAM3 repository", valid_repository(repository), repository or "Not found"))
    results.append(("Checkpoint", valid_checkpoint(checkpoint), checkpoint or "Not found"))
    try:
        import torch
        available = bool(torch.cuda.is_available())
        detail = torch.cuda.get_device_name(0) if available else "CUDA is not available"
        results.append(("NVIDIA GPU / CUDA", available, detail))
        results.append(("PyTorch", True, f"{torch.__version__} (CUDA {torch.version.cuda})"))
    except Exception as exc:
        results.append(("PyTorch", False, f"{type(exc).__name__}: {exc}"))
    return results
