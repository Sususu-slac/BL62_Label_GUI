# BL62 Label GUI

GPU-assisted image annotation with SAM3 and COCO dataset export.

## Windows installation

1. Install a current NVIDIA driver and Miniforge/Conda.
2. Extract this repository to a local folder.
3. Open Miniforge Prompt in that folder and run `install_windows.bat`.
4. Start the application with `BL62 Label GUI.vbs`.
5. The installer checks GPU/CUDA availability. On first launch, the application auto-detects common SAM3 locations; otherwise use **Select SAM3 Repository** and **Select Checkpoint**. There is no Setup / GPU Test dialog in the GUI.

The installer creates the dedicated `BL62_Label_GUI` environment and does not modify other project environments.

## SAM3 and checkpoint

SAM3 source code and model weights are not included. Obtain them from the official provider and comply with their license. A valid repository contains both `pyproject.toml` and `sam3/__init__.py`. A checkpoint must be a `.pt`, `.pth`, or `.ckpt` file larger than 100 MB.

Configuration is saved locally in `sam3_settings.json`, which is excluded from Git.

## Project workflow

- Create a project from a folder or selected images.
- Drag images into an empty window to create a project.
- Drag images into an open project to add them.
- Use one unified **Label** for both the SAM3 text prompt and COCO category.
- Use the single **Run SAM3 (N pictures)** button (also available in the image list right-click menu). Select one image for the existing Text / Box / Instance Prompt workflow. Ctrl/Shift-select multiple images for batch Text Prompt inference. Confirm that **all existing labels, including manual labels**, will be replaced on the selected images. This uses the current Text Prompt, category, and confidence on full images, without the current SAM3 Box.
- The first run loads SAM3 in a persistent background process. Single images, batches, Text / Box / Instance Prompts, and later runs reuse the same model. **Load SAM3 on GPU** optionally preloads it; it now keeps the model loaded.
- **Release GPU** unloads the worker. The next run reloads automatically. Changing the repository/checkpoint reloads an already loaded model; closing the application releases it. Changing the prompt or confidence does not reload the model.
- Inference and loading use a modal progress window to protect the labels being processed. Editing, switching images, saving and exporting remain available while the model is idle. Cancelling an active task terminates the worker and releases GPU memory; the next run reloads.
- The batch saves each successful image automatically. Cancel or failure preserves unfinished images and completed results. A successful prediction with zero detections clears that image’s labels. Reprocessed images are marked unreviewed.
- **Auto Run after Box** is on by default. On one selected image, draw a SAM3 Box and release the left mouse button to append results with the current Text or Instance Prompt. Existing labels are kept, including overlapping or duplicate labels; no replacement confirmation is shown. Undo removes one completed addition. Turn the switch off to draw boxes without running inference.
- In Instance Prompt mode, **Use Selected Instance** freezes a copy of the example and activates the box tool. Keep drawing boxes anywhere in the same image; the example may be outside the box and survives deletion or editing of its original label. Results are clipped to the requested box. Clear/set the example to change it; switching images clears it.
- The ordinary Run button still asks to replace existing labels. Multi-image batch confirmation and replacement are unchanged.
- Save and export the training dataset from the GUI.
- Eraser edits that create interior holes are saved as COCO RLE masks, so holes stay open after switching images or reopening the project.

## Development

Run the English GUI:

```bat
conda run --no-capture-output -n BL62_Label_GUI python app_en.py
```

Run tests:

```bat
set QT_QPA_PLATFORM=offscreen
conda run --no-capture-output -n BL62_Label_GUI python -m unittest discover -s tests -v
```

Do not commit checkpoints, datasets, annotations, exported results, or `sam3_settings.json`.
