from __future__ import annotations

import json
import tempfile
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
from PySide6.QtWidgets import QApplication

from app import MainWindow


def main() -> int:
    app = QApplication.instance() or QApplication([])
    desktop = Path.home() / "Desktop"
    output = desktop / f"BL62_Label_Export_Test_{datetime.now():%Y%m%d_%H%M%S}"

    with tempfile.TemporaryDirectory(prefix="bl62_label_export_test_") as temp:
        source = Path(temp) / "source_images"
        source.mkdir()
        for index in range(2):
            yy, xx = np.mgrid[:64, :64]
            image = np.zeros((64, 64, 3), np.uint8)
            image[..., 0] = np.clip(xx * 4, 0, 255)
            image[..., 1] = np.clip(yy * 4, 0, 255)
            image[..., 2] = 50 + index * 100
            Image.fromarray(image).save(source / f"test_image_{index + 1:02d}.png")

        window = MainWindow()
        window.project.new_from_folder(source)
        window.populate()

        first = np.zeros((64, 64), bool)
        first[12:36, 10:34] = True
        window.masks = [{"id": 0, "category_id": 1, "noun_phrase": "particle", "annotation_source": "manual", "mask": first}]
        window.view.masks = window.masks
        window.commit_current()
        window.navigate(1)

        second = np.zeros((64, 64), bool)
        yy, xx = np.mgrid[:64, :64]
        second[(xx - 38) ** 2 + (yy - 34) ** 2 <= 14 ** 2] = True
        window.masks = [{"id": 0, "category_id": 1, "noun_phrase": "particle", "annotation_source": "manual", "mask": second}]
        window.view.masks = window.masks

        with patch("app.QFileDialog.getExistingDirectory", return_value=str(output)), \
             patch("app.QMessageBox.information"), patch("app.QMessageBox.warning"), patch("app.QMessageBox.critical"):
            window.export_all()
            deadline = time.monotonic() + 15
            while window.export_thread is not None and time.monotonic() < deadline:
                QApplication.processEvents()
                time.sleep(0.01)
            QApplication.processEvents()
        window.close()

    annotations = output / "training_dataset" / "annotations.json"
    data = json.loads(annotations.read_text(encoding="utf-8"))
    expected = [
        output / "training_dataset" / "images" / "test_image_01.png",
        output / "training_dataset" / "images" / "test_image_02.png",
        output / "auxiliary_outputs" / "instance_labels" / "test_image_01_instances.png",
        output / "auxiliary_outputs" / "instance_labels" / "test_image_02_instances.png",
        output / "auxiliary_outputs" / "previews" / "test_image_01_preview.png",
        output / "auxiliary_outputs" / "previews" / "test_image_02_preview.png",
        output / "auxiliary_outputs" / "project_state.json",
        output / "auxiliary_outputs" / "dataset_statistics.csv",
    ]
    missing = [str(path) for path in expected if not path.is_file()]
    summary = {
        "output": str(output),
        "images": len(data.get("images", [])),
        "annotations": len(data.get("annotations", [])),
        "categories": len(data.get("categories", [])),
        "missing_files": missing,
    }
    (output / "EXPORT_TEST_RESULT.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["images"] == 2 and summary["annotations"] == 2 and not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
