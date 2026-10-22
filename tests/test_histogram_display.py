import unittest
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from app import AnnotationView, HistogramPanel, MainWindow, apply_tone_curve


class HistogramDisplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_tone_curve_changes_display_array_without_modifying_source(self):
        image = np.array([[[0, 50, 100], [150, 200, 255]]], dtype=np.uint8)
        original = image.copy()
        curve = np.arange(256, dtype=np.uint8)
        curve[64:192] = np.linspace(0, 255, 128).astype(np.uint8)
        adjusted = apply_tone_curve(image, curve)
        self.assertFalse(np.array_equal(adjusted, image))
        self.assertTrue(np.array_equal(image, original))

    def test_histogram_curve_generates_lut_and_reset_restores_identity(self):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        image[:, 4:] = 200
        panel = HistogramPanel()
        panel.set_image(image)
        self.assertEqual(int(panel.canvas.histogram[0]), 32)
        self.assertEqual(int(panel.canvas.histogram[255]), 32)
        self.assertEqual(panel.canvas.intensity_min, 0.0)
        self.assertEqual(panel.canvas.intensity_max, 200.0)
        panel.canvas.points = [(0, 0), (128, 220), (255, 255)]
        lut = panel.canvas._curve_lut()
        self.assertEqual(int(lut[128]), 220)
        self.assertGreater(int(lut[64]), 64)
        panel.reset_curve()
        np.testing.assert_array_equal(panel.canvas._curve_lut(), np.arange(256, dtype=np.uint8))
        panel.close()

    def test_curve_control_point_can_be_added_dragged_and_removed(self):
        panel = HistogramPanel()
        panel.resize(520, 170)
        panel.show()
        self.app.processEvents()
        canvas = panel.canvas
        left, top, width, height = canvas._plot_bounds()
        x = round(left + width / 2)
        y = round(top + height / 2)
        QTest.mouseClick(canvas, Qt.LeftButton, pos=QPoint(x, y))
        self.assertEqual(len(canvas.points), 3)
        QTest.mousePress(canvas, Qt.LeftButton, pos=QPoint(x, y))
        QTest.mouseMove(canvas, QPoint(x, round(top + 2)))
        QTest.mouseRelease(canvas, Qt.LeftButton, pos=QPoint(x, round(top + 2)))
        self.assertGreater(int(canvas._curve_lut()[128]), 220)
        QTest.mouseClick(canvas, Qt.RightButton, pos=QPoint(x, round(top + 2)))
        self.assertEqual(len(canvas.points), 2)
        panel.close()

    def test_view_updates_only_display_using_curve(self):
        image = np.array([[[0, 0, 0], [100, 100, 100], [200, 200, 200]]], dtype=np.uint8)
        original = image.copy()
        view = AnnotationView()
        view.set_image(image, [])
        curve = np.arange(256, dtype=np.uint8)
        curve[100] = 37
        view.set_display_curve(curve)
        rendered = view.image_item.pixmap().toImage()
        self.assertEqual(rendered.pixelColor(1, 0).red(), 37)
        self.assertTrue(np.array_equal(image, original))
        view.close()

    def test_high_bit_image_cache_survives_page_switch_and_curve_does_not_change_sam(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first_source = np.linspace(-1.0, 2.0, 64 * 80, dtype=np.float32).reshape(64, 80)
            second_source = np.arange(64 * 80, dtype=np.uint16).reshape(64, 80)
            Image.fromarray(first_source).save(root / "01_float.tiff")
            Image.fromarray(second_source).save(root / "02_uint16.tiff")
            window = MainWindow()
            window.project.new_from_folder(root)
            window.populate()
            first_data = window.image_cache.get(root / "01_float.tiff")
            self.assertEqual(first_data.source_dtype, "float32")
            first_sam = np.load(first_data.sam_path, allow_pickle=False)
            self.assertEqual(window.histogram.canvas.intensity_min, -1.0)
            self.assertEqual(window.histogram.canvas.intensity_max, 2.0)
            window.images.setCurrentRow(1)
            second_data = window.image_cache.get(root / "02_uint16.tiff")
            self.assertEqual(second_data.source_dtype, "uint16")
            window.images.setCurrentRow(0)
            curve = np.arange(256, dtype=np.uint8)
            curve[64:192] = 255
            window.histogram.canvas.points = [(0, 0), (64, 255), (192, 255), (255, 255)]
            window.view.set_display_curve(window.histogram.canvas._curve_lut())
            np.testing.assert_array_equal(np.load(first_data.sam_path, allow_pickle=False), first_sam)
            np.testing.assert_array_equal(np.asarray(first_data.display_rgb8)[..., 0], np.linspace(0, 255, 64 * 80, dtype=np.uint8).reshape(64, 80))
            self.assertEqual(window.histogram.canvas.intensity_min, -1.0)
            self.assertEqual(window.histogram.canvas.intensity_max, 2.0)
            window.project.dirty = False
            window.close()

    def test_histogram_hides_range_without_image_and_auto_adjusts_from_original_range(self):
        panel = HistogramPanel()
        self.assertFalse(panel.canvas.has_image)
        source = np.array([[1000, 2000, 3000, 4000]], dtype=np.uint16)
        panel.set_image(source)
        self.assertEqual(panel.canvas.intensity_min, 1000.0)
        self.assertEqual(panel.canvas.intensity_max, 4000.0)
        panel.canvas.auto_curve()
        self.assertNotEqual(panel.canvas._curve_lut()[128], 128)
        panel.close()


if __name__ == "__main__":
    unittest.main()
