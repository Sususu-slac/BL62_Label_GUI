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
- The batch loads the model once and saves each successful image automatically. Cancel or failure preserves unfinished images and completed results. A successful prediction with zero detections clears that image’s labels. Reprocessed images are marked unreviewed.
- Save and export the training dataset from the GUI.

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
