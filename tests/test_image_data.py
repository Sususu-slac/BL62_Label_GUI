import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from image_data import ImageDataCache


class ImageDataTests(unittest.TestCase):
    def test_float_tiff_prepares_original_range_display_and_cached_sam_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "float.tiff"
            source = np.linspace(-2.0, 3.0, 128 * 96, dtype=np.float32).reshape(96, 128)
            Image.fromarray(source).save(path)
            data = ImageDataCache().get(path)
            self.assertEqual(data.source_dtype, "float32")
            self.assertEqual(data.display_rgb8.shape, (96, 128, 3))
            self.assertEqual(data.display_rgb8.dtype, np.uint8)
            self.assertEqual((data.intensity_min, data.intensity_max), (-2.0, 3.0))
            self.assertTrue(data.sam_path.is_file())
            self.assertEqual(data.sam_path.suffix, ".npy")
            sam = np.load(data.sam_path, allow_pickle=False)
            self.assertEqual(sam.shape, (96, 128, 3))
            self.assertGreater(int(sam.max()), 200)

    def test_uint16_tiff_is_prepared_once_and_reused_on_return(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "uint16.tiff"
            source = np.arange(128 * 96, dtype=np.uint16).reshape(96, 128) * 5
            Image.fromarray(source).save(path)
            cache = ImageDataCache()
            first = cache.get(path)
            second = cache.get(path)
            self.assertIs(first, second)
            self.assertEqual(cache.cached_images, 1)
            self.assertEqual(first.intensity_min, float(source.min()))
            self.assertEqual(first.intensity_max, float(source.max()))
            self.assertTrue(first.sam_path.is_file())
            self.assertTrue((first.cache_dir / "display.npy").is_file())
            with patch("image_data._read_source", side_effect=AssertionError("source TIFF decoded again")):
                reopened = ImageDataCache().get(path)
            self.assertEqual(reopened.source_dtype, "uint16")
            self.assertTrue(reopened.sam_path.is_file())

    def test_rgb8_input_is_displayed_unchanged_and_passed_directly_to_sam(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rgb.png"
            source = np.zeros((20, 24, 3), dtype=np.uint8)
            source[..., 0] = 17
            source[..., 1] = 128
            source[..., 2] = 249
            Image.fromarray(source).save(path)
            data = ImageDataCache().get(path)
            np.testing.assert_array_equal(data.display_rgb8, source)
            self.assertEqual(data.sam_path, path)
            self.assertEqual(data.source_dtype, "uint8")


if __name__ == "__main__":
    unittest.main()
