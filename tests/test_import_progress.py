import tempfile
import time
import unittest
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
from PySide6.QtWidgets import QApplication

from app import MainWindow, ProjectImportWorker, ProjectOpenWorker


class ImportProgressTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_import_worker_reports_per_image_progress(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = []
            for index in range(4):
                path = root / f"slice_{index}.png"
                Image.fromarray(np.full((12, 16), index, dtype=np.uint8)).save(path)
                paths.append(path)
            worker = ProjectImportWorker(paths=paths, root=root)
            progress = []
            completed = []
            worker.progress.connect(lambda done, total, message: progress.append((done, total, message)))
            worker.completed.connect(completed.append)
            worker.run()
            self.assertEqual([item[0] for item in progress], list(range(9)))
            self.assertTrue(all(item[1] == 8 for item in progress))
            self.assertEqual(len(completed), 1)
            self.assertEqual(len(completed[0].image_paths), 4)

    def test_gui_import_progress_completes_without_blocking_event_loop(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for index in range(3):
                Image.fromarray(np.full((16, 20), index * 20, dtype=np.uint8)).save(root / f"{index}.png")
            window = MainWindow()
            window._start_project_import(folder=root)
            self.assertIsNotNone(window.import_dialog)
            deadline = time.monotonic() + 10
            with patch("app.QMessageBox.information"):
                while window.import_thread is not None and time.monotonic() < deadline:
                    self.app.processEvents()
                    time.sleep(0.005)
            self.app.processEvents()
            self.assertIsNone(window.import_thread)
            self.assertEqual(len(window.project.image_paths), 3)
            self.assertIsNotNone(window.view.array)
            window.project.dirty = False
            window.close()

    def test_cancel_stops_import_worker_and_does_not_reopen_progress_dialog(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for index in range(8):
                Image.fromarray(np.full((256, 256), index, dtype=np.uint8)).save(root / f"{index}.png")
            window = MainWindow()
            original_project = window.project
            original_prepare = __import__("coco_project").prepare_image

            def slow_prepare(path, **kwargs):
                result = original_prepare(path, **kwargs)
                time.sleep(0.05)
                return result

            try:
                with patch("coco_project.prepare_image", side_effect=slow_prepare):
                    window._start_project_import(folder=root)
                    deadline = time.monotonic() + 15
                    while window.import_dialog is not None and "Preparing SAM3" not in window.import_dialog.labelText() and time.monotonic() < deadline:
                        self.app.processEvents()
                        time.sleep(0.005)
                    self.assertIsNotNone(window.import_dialog)
                    window.import_dialog.canceled.emit()
                    self.assertTrue(window._import_cancel_requested)
                    deadline = time.monotonic() + 15
                    while window.import_thread is not None and time.monotonic() < deadline:
                        self.app.processEvents()
                        time.sleep(0.005)
                    self.app.processEvents()
                    self.assertIsNone(window.import_thread)
                    self.assertIsNone(window.import_dialog)
                    self.assertIs(window.project, original_project)
                    self.assertFalse(window._import_cancel_requested)
            finally:
                if window.import_thread is not None:
                    window._request_import_cancel()
                    deadline = time.monotonic() + 15
                    while window.import_thread is not None and time.monotonic() < deadline:
                        self.app.processEvents()
                        time.sleep(0.005)
                window.close()

    def test_coco_project_open_reports_image_indexing_progress(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for index in range(3):
                Image.fromarray(np.full((10, 14), index, dtype=np.uint8)).save(root / f"{index}.png")
            json_path = root / "annotations.json"
            json_path.write_text(json.dumps({"images": [
                {"id": index + 1, "file_name": f"{index}.png", "width": 14, "height": 10}
                for index in range(3)], "annotations": [], "categories": []}), encoding="utf-8")
            worker = ProjectOpenWorker(json_path)
            progress = []
            completed = []
            worker.progress.connect(lambda done, total, message: progress.append((done, total)))
            worker.completed.connect(completed.append)
            worker.run()
            self.assertEqual(progress, [(1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 6)])
            self.assertEqual(len(completed), 1)
            self.assertEqual(len(completed[0].image_paths), 3)

    def test_drag_added_images_run_in_background(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "project"
            external = Path(folder) / "incoming"
            root.mkdir(); external.mkdir()
            Image.fromarray(np.zeros((16, 20), dtype=np.uint8)).save(root / "existing.png")
            incoming = external / "new.png"
            Image.fromarray(np.full((16, 20), 255, dtype=np.uint8)).save(incoming)
            window = MainWindow()
            window.project.new_from_folder(root); window.populate()
            window._start_add_images([incoming])
            deadline = time.monotonic() + 10
            with patch("app.QMessageBox.information"):
                while window.add_thread is not None and time.monotonic() < deadline:
                    self.app.processEvents(); time.sleep(0.005)
            self.app.processEvents()
            self.assertIsNone(window.add_thread)
            self.assertEqual(len(window.project.image_paths), 2)
            self.assertTrue((root / "new.png").is_file())
            window.project.dirty = False
            window.close()


if __name__ == "__main__":
    unittest.main()
