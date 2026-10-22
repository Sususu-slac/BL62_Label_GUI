import unittest

import numpy as np

from coco_project import CocoProject, mask_annotation_data, polygons_to_mask


class MaskHoleRoundTripTests(unittest.TestCase):
    def test_hollow_mask_round_trips_through_coco_rle(self):
        mask = np.zeros((64, 80), dtype=bool)
        mask[8:56, 12:68] = True
        mask[22:42, 29:51] = False

        segmentation, bbox, area = mask_annotation_data(mask)

        self.assertIsInstance(segmentation, dict)
        self.assertEqual(bbox, [12.0, 8.0, 56.0, 48.0])
        self.assertEqual(area, float(mask.sum()))
        self.assertTrue(np.array_equal(polygons_to_mask(segmentation, 80, 64), mask))

    def test_solid_mask_keeps_polygon_format(self):
        mask = np.zeros((64, 80), dtype=bool)
        mask[8:56, 12:68] = True

        segmentation, _, _ = mask_annotation_data(mask)

        self.assertIsInstance(segmentation, list)
        self.assertTrue(np.array_equal(polygons_to_mask(segmentation, 80, 64), mask))

    def test_replace_masks_keeps_hole_after_image_switch_and_reload(self):
        project = CocoProject()
        project.data = {
            "info": {},
            "images": [{"id": 1, "file_name": "sample.png", "width": 80, "height": 64}],
            "annotations": [],
            "categories": [{"id": 1, "name": "particle", "supercategory": "particle"}],
        }
        project._rebuild_annotation_index()
        mask = np.zeros((64, 80), dtype=bool)
        mask[8:56, 12:68] = True
        mask[22:42, 29:51] = False

        project.replace_masks(1, [{"mask": mask, "category_id": 1}])
        saved = project.data["annotations"][0]
        reloaded = polygons_to_mask(saved["segmentation"], 80, 64)

        self.assertIsInstance(saved["segmentation"], dict)
        self.assertEqual(int(reloaded.sum()), int(mask.sum()))
        self.assertTrue(np.array_equal(reloaded, mask))


if __name__ == "__main__":
    unittest.main()
