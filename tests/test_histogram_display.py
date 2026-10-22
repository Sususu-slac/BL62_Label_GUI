import unittest

import numpy as np
from PySide6.QtWidgets import QApplication

from app import AnnotationView, HistogramPanel, apply_display_range


class HistogramDisplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_full_range_preserves_original_image_values(self):
        image = np.array([[[0, 20, 255], [80, 120, 160]]], dtype=np.uint8)
        adjusted = apply_display_range(image, 0, 255)
        self.assertTrue(np.array_equal(adjusted, image))
        self.assertTrue(np.array_equal(image, np.array([[[0, 20, 255], [80, 120, 160]]], dtype=np.uint8)))

    def test_adjusted_range_changes_only_display_pixels(self):
        image = np.array([[[0, 50, 100], [150, 200, 255]]], dtype=np.uint8)
        before = image.copy()
        adjusted = apply_display_range(image, 50, 200)
        self.assertTrue(np.array_equal(adjusted, np.array([[[0, 0, 85], [170, 255, 255]]], dtype=np.uint8)))
        self.assertTrue(np.array_equal(image, before))

    def test_histogram_panel_tracks_image_and_range(self):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        image[:, 4:] = 200
        panel = HistogramPanel()
        panel.set_image(image)
        self.assertEqual(int(panel.canvas.histogram[0]), 32)
        self.assertEqual(int(panel.canvas.histogram[200]), 32)
        panel.set_range(30, 210)
        self.assertEqual((panel.black_spin.value(), panel.white_spin.value()), (30, 210))
        panel.close()

    def test_view_updates_only_the_display_pixmap(self):
        image = np.array([[[0, 0, 0], [100, 100, 100], [200, 200, 200]]], dtype=np.uint8)
        original = image.copy()
        view = AnnotationView()
        view.set_image(image, [])
        view.set_display_range(50, 150)
        rendered = view.image_item.pixmap().toImage()
        self.assertEqual(rendered.pixelColor(1, 0).red(), 127)
        self.assertEqual(rendered.pixelColor(2, 0).red(), 255)
        self.assertTrue(np.array_equal(image, original))
        view.close()


if __name__ == "__main__":
    unittest.main()
