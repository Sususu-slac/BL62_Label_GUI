from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def mask_to_polygons(mask: np.ndarray) -> list[list[float]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = []
    for contour in contours:
        if len(contour) >= 3 and cv2.contourArea(contour) >= 1:
            points = contour.reshape(-1, 2).astype(float).ravel().tolist()
            if len(points) >= 6:
                result.append(points)
    return result


def polygons_to_mask(segmentation, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), np.uint8)
    if isinstance(segmentation, list):
        polys = []
        for polygon in segmentation:
            arr = np.asarray(polygon, dtype=np.float32)
            if arr.size >= 6 and arr.size % 2 == 0:
                polys.append(np.rint(arr.reshape(-1, 2)).astype(np.int32))
        if polys:
            cv2.fillPoly(mask, polys, 1)
    elif isinstance(segmentation, dict):
        try:
            from pycocotools import mask as mask_utils
            decoded = mask_utils.decode(segmentation)
            mask = (decoded[..., 0] if decoded.ndim == 3 else decoded).astype(np.uint8)
        except Exception as exc:
            raise ValueError("RLE mask requires pycocotools") from exc
    return mask.astype(bool)


def mask_geometry(mask: np.ndarray) -> tuple[list[float], float]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return [0.0, 0.0, 0.0, 0.0], 0.0
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    return [float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)], float(mask.sum())


class CocoProject:
    def __init__(self):
        self.root: Path | None = None
        self.json_path: Path | None = None
        self.data: dict = {"info": {"description": "Particle annotations"}, "images": [], "annotations": [], "categories": [{"id": 1, "name": "particle", "supercategory": "particle"}]}
        self.image_paths: list[Path] = []
        self.reviewed: set[int] = set()
        self.dirty = False
        self._backup_done = False

    def new_from_folder(self, folder: Path) -> None:
        folder = folder.resolve()
        paths = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if not paths:
            raise ValueError("No supported images found in this folder")
        self.new_from_files(paths, folder)

    def new_from_files(self, paths: list[Path], root: Path | None = None) -> None:
        paths = [Path(path).resolve() for path in paths]
        paths = sorted(dict.fromkeys(path for path in paths if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS))
        if not paths:
            raise ValueError("No supported images were selected")
        parents = {path.parent for path in paths}
        if root is None:
            if len(parents) != 1:
                raise ValueError("Selected images must be in the same folder")
            root = next(iter(parents))
        root = Path(root).resolve()
        from PIL import Image
        self.root, self.json_path, self.image_paths = root, root / "annotations.json", paths
        images = []
        for idx, path in enumerate(paths, 1):
            with Image.open(path) as im:
                width, height = im.size
            images.append({"id": idx, "file_name": path.name, "width": width, "height": height})
        self.data = {"info": {"description": "Particle Label GUI project", "created": datetime.now().isoformat(timespec="seconds")}, "images": images, "annotations": [], "categories": [{"id": 1, "name": "particle", "supercategory": "particle"}]}
        self.reviewed, self.dirty, self._backup_done = set(), True, False

    def add_images(self, paths: list[Path], copy_into_project: bool = True) -> list[int]:
        if self.root is None:
            raise RuntimeError("Open or create a project before adding images")
        from PIL import Image

        sources = [Path(path).resolve() for path in paths]
        sources = list(dict.fromkeys(path for path in sources if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS))
        if not sources:
            raise ValueError("No supported images were provided")
        existing_paths = {path.resolve() for path in self.image_paths if path.exists()}
        existing_names = {path.name.casefold() for path in self.image_paths}
        next_id = max((int(item.get("id", 0)) for item in self.data.get("images", [])), default=0) + 1
        added_indices = []
        for source in sources:
            if source in existing_paths:
                continue
            target = source
            if copy_into_project and source.parent != self.root:
                target = self.root / source.name
                counter = 1
                while target.name.casefold() in existing_names or target.exists():
                    target = self.root / f"{source.stem}_{counter}{source.suffix.lower()}"
                    counter += 1
                shutil.copy2(source, target)
            elif target.name.casefold() in existing_names:
                continue
            with Image.open(target) as image:
                width, height = image.size
                image.verify()
            self.data.setdefault("images", []).append(
                {"id": next_id, "file_name": target.name, "width": width, "height": height}
            )
            self.image_paths.append(target)
            existing_paths.add(target.resolve())
            existing_names.add(target.name.casefold())
            added_indices.append(len(self.image_paths) - 1)
            next_id += 1
        if added_indices:
            self.dirty = True
        return added_indices

    def remove_images(self, indices: list[int]) -> tuple[int, int]:
        valid = sorted({int(index) for index in indices if 0 <= int(index) < len(self.image_paths)})
        if not valid:
            return 0, 0
        removed_ids = {int(self.data["images"][index]["id"]) for index in valid}
        annotation_count = sum(int(item.get("image_id", -1)) in removed_ids for item in self.data.get("annotations", []))
        self.data["images"] = [item for index, item in enumerate(self.data["images"]) if index not in valid]
        self.image_paths = [path for index, path in enumerate(self.image_paths) if index not in valid]
        self.data["annotations"] = [item for item in self.data.get("annotations", []) if int(item.get("image_id", -1)) not in removed_ids]
        self.reviewed.difference_update(removed_ids)
        self.dirty = True
        return len(valid), annotation_count

    def open_json(self, json_path: Path) -> None:
        json_path = json_path.resolve()
        with json_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data.get("images"), list) or not isinstance(data.get("annotations"), list):
            raise ValueError("Not a COCO file: images and annotations lists are required")
        self.json_path, self.root, self.data = json_path, json_path.parent, data
        if not isinstance(self.data.get("categories"), list):
            self.data["categories"] = []
        known = {int(c.get("id", 0)) for c in self.data["categories"]}
        referenced = {int(a.get("category_id", 1)) for a in self.data["annotations"]}
        for category_id in sorted(referenced - known):
            self.data["categories"].append({"id": category_id, "name": f"category_{category_id}", "supercategory": "unknown"})
        if not self.data["categories"]:
            self.data["categories"] = [{"id": 1, "name": "particle", "supercategory": "particle"}]
        candidates = [self.root / "images", self.root]
        self.image_paths = []
        for record in self.data["images"]:
            name = Path(str(record.get("file_name", "")).replace("\\", "/"))
            found = next((base / name for base in candidates if (base / name).is_file()), None)
            if found is None:
                found = next((base / name.name for base in candidates if (base / name.name).is_file()), self.root / name)
            self.image_paths.append(found)
        state = self.root / "project_state.json"
        if state.is_file():
            try:
                self.reviewed = set(json.loads(state.read_text(encoding="utf-8")).get("reviewed_image_ids", []))
            except Exception:
                self.reviewed = set()
        else:
            self.reviewed = set()
        self.dirty, self._backup_done = False, False

    def image_record(self, index: int) -> dict:
        return self.data["images"][index]

    def masks_for(self, image_id: int, width: int, height: int) -> list[dict]:
        result = []
        for ann in self.data["annotations"]:
            if ann.get("image_id") == image_id:
                item = copy.deepcopy(ann)
                item["mask"] = polygons_to_mask(item.get("segmentation", []), width, height)
                result.append(item)
        return result

    def categories(self) -> list[dict]:
        return sorted(self.data.get("categories", []), key=lambda item: int(item.get("id", 0)))

    def category(self, category_id: int) -> dict | None:
        return next((item for item in self.categories() if int(item.get("id", 0)) == int(category_id)), None)

    def add_category(self, name: str, supercategory: str = "") -> dict:
        name = name.strip()
        if not name:
            raise ValueError("Category name cannot be empty")
        if any(str(item.get("name", "")).casefold() == name.casefold() for item in self.categories()):
            raise ValueError(f"Category already exists: {name}")
        category_id = max({int(item.get("id", 0)) for item in self.categories()} | {0}) + 1
        item = {"id": category_id, "name": name, "supercategory": supercategory.strip() or name}
        self.data.setdefault("categories", []).append(item); self.dirty = True
        return item

    def rename_category(self, category_id: int, name: str, supercategory: str) -> None:
        item = self.category(category_id)
        if item is None:
            raise KeyError(f"Unknown category ID: {category_id}")
        name = name.strip()
        if not name:
            raise ValueError("Category name cannot be empty")
        if any(int(other.get("id", 0)) != category_id and str(other.get("name", "")).casefold() == name.casefold() for other in self.categories()):
            raise ValueError(f"Category already exists: {name}")
        item["name"], item["supercategory"] = name, supercategory.strip() or name; self.dirty = True
        for annotation in self.data.get("annotations", []):
            if int(annotation.get("category_id", 0)) == int(category_id):
                annotation["noun_phrase"] = name

    def category_usage(self, category_id: int) -> int:
        return sum(int(ann.get("category_id", 0)) == int(category_id) for ann in self.data["annotations"])

    def delete_category(self, category_id: int) -> None:
        usage = self.category_usage(category_id)
        if usage:
            raise ValueError(f"Category {category_id} is used by {usage} annotations")
        if len(self.categories()) <= 1:
            raise ValueError("A project must contain at least one category")
        self.data["categories"] = [item for item in self.categories() if int(item.get("id", 0)) != int(category_id)]; self.dirty = True

    def merge_category(self, source_id: int, target_id: int) -> None:
        if source_id == target_id:
            return
        if self.category(source_id) is None or self.category(target_id) is None:
            raise KeyError("Unknown source or target category")
        target_name = str(self.category(target_id).get("name", "particle"))
        for ann in self.data["annotations"]:
            if int(ann.get("category_id", 0)) == source_id:
                ann["category_id"] = target_id; ann["noun_phrase"] = target_name
        self.data["categories"] = [item for item in self.categories() if int(item.get("id", 0)) != source_id]; self.dirty = True

    def validate_categories(self) -> None:
        ids = [int(item.get("id", 0)) for item in self.categories()]
        if any(category_id <= 0 for category_id in ids) or len(ids) != len(set(ids)):
            raise ValueError("Category IDs must be unique positive integers")
        unknown = sorted({int(ann.get("category_id", 0)) for ann in self.data["annotations"]} - set(ids))
        if unknown:
            raise ValueError(f"Annotations reference unknown category IDs: {unknown}")

    def replace_masks(self, image_id: int, masks: list[dict]) -> None:
        others = [ann for ann in self.data["annotations"] if ann.get("image_id") != image_id]
        used = {int(a.get("id", 0)) for a in others}
        next_id = max(used | {0}) + 1
        saved = []
        for item in masks:
            mask = item["mask"].astype(bool)
            polygons = mask_to_polygons(mask)
            bbox, area = mask_geometry(mask)
            if not polygons or area <= 0:
                continue
            ann_id = int(item.get("id", 0))
            if ann_id <= 0 or ann_id in used:
                ann_id, next_id = next_id, next_id + 1
            used.add(ann_id)
            saved_item = {"id": ann_id, "image_id": image_id, "category_id": int(item.get("category_id", 1)), "segmentation": polygons, "area": area, "bbox": bbox, "iscrowd": 0, "noun_phrase": item.get("noun_phrase", "particle"), "annotation_source": item.get("annotation_source", "manual")}
            if item.get("score") is not None:
                saved_item["score"] = float(item["score"])
            if item.get("mask_prompt_source") is not None:
                saved_item["mask_prompt_source"] = str(item["mask_prompt_source"])
            saved.append(saved_item)
        self.data["annotations"] = others + saved
        self.dirty = True

    def save(self) -> None:
        if self.json_path is None or self.root is None:
            raise RuntimeError("No project is open")
        self.validate_categories(); self.root.mkdir(parents=True, exist_ok=True)
        if self.json_path.exists() and not self._backup_done:
            backup_dir = self.root / "backups"
            backup_dir.mkdir(exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            shutil.copy2(self.json_path, backup_dir / f"annotations_{stamp}.json")
            self._backup_done = True
        fd, temp_name = tempfile.mkstemp(prefix=".annotations_", suffix=".tmp", dir=self.root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.data, handle, ensure_ascii=False, indent=2)
                handle.flush(); os.fsync(handle.fileno())
            os.replace(temp_name, self.json_path)
        finally:
            if os.path.exists(temp_name): os.unlink(temp_name)
        (self.root / "project_state.json").write_text(json.dumps({"reviewed_image_ids": sorted(self.reviewed), "updated": datetime.now().isoformat(timespec="seconds")}, ensure_ascii=False, indent=2), encoding="utf-8")
        self.dirty = False
