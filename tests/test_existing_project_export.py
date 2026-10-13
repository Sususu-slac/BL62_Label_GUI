from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from app_en import MainWindow


def wait_for_export(app, window, timeout=180):
    ticks = 0
    timer = QTimer()
    timer.setInterval(20)
    timer.timeout.connect(lambda: None)
    timer.start()
    deadline = time.monotonic() + timeout
    while window.export_thread is not None and time.monotonic() < deadline:
        app.processEvents()
        ticks += 1
        time.sleep(0.01)
    timer.stop()
    app.processEvents()
    if window.export_thread is not None:
        raise TimeoutError("Export did not finish within the timeout")
    return ticks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = args.project.resolve()
    output = args.output or (Path.home() / "Desktop" / f"test_export_result_{datetime.now():%Y%m%d_%H%M%S}")
    app = QApplication.instance() or QApplication([])

    with tempfile.TemporaryDirectory(prefix="bl62_existing_project_") as temporary:
        copied = Path(temporary) / "project"
        copied.mkdir()
        shutil.copy2(source / "annotations.json", copied / "annotations.json")
        if (source / "project_state.json").is_file():
            shutil.copy2(source / "project_state.json", copied / "project_state.json")
        data = json.loads((source / "annotations.json").read_text(encoding="utf-8"))
        for record in data["images"]:
            name = Path(record["file_name"]).name
            candidate = source / record["file_name"]
            if not candidate.is_file(): candidate = source / "images" / record["file_name"]
            shutil.copy2(candidate, copied / name)

        window = MainWindow()
        window.project.open_json(copied / "annotations.json")
        window.populate()
        with patch("app.QFileDialog.getExistingDirectory", return_value=str(output)), \
             patch("app.QMessageBox.information"), patch("app.QMessageBox.warning"), patch("app.QMessageBox.critical"):
            started = time.perf_counter()
            window.export_all()
            event_ticks = wait_for_export(app, window)
            elapsed = time.perf_counter() - started

        exported = json.loads((output / "training_dataset" / "annotations.json").read_text(encoding="utf-8"))
        training_images = list((output / "training_dataset" / "images").glob("*"))
        labels = list((output / "auxiliary_outputs" / "instance_labels").glob("*.png"))
        previews = list((output / "auxiliary_outputs" / "previews").glob("*.png"))
        result = {
            "source": str(source),
            "output": str(output),
            "source_images": len(data["images"]),
            "source_annotations": len(data["annotations"]),
            "exported_images": len(exported["images"]),
            "exported_annotations": len(exported["annotations"]),
            "copied_training_images": len(training_images),
            "instance_label_files": len(labels),
            "preview_files": len(previews),
            "incomplete_marker": (output / "EXPORT_INCOMPLETE.txt").exists(),
            "event_loop_ticks": event_ticks,
            "elapsed_seconds": elapsed,
        }
        ok = (
            result["source_images"] == result["exported_images"] == result["copied_training_images"]
            == result["instance_label_files"] == result["preview_files"]
            and result["source_annotations"] == result["exported_annotations"]
            and not result["incomplete_marker"] and event_ticks > 1
        )
        (output / "REAL_PROJECT_EXPORT_TEST.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
        window.project.dirty = False
        window.close()
        return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
