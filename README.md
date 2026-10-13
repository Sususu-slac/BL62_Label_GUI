# BL62 Label GUI

GPU-assisted image annotation with SAM3 and COCO dataset export.

## Windows installation

1. Install a current NVIDIA driver and Miniforge/Conda.
2. Extract this repository to a local folder.
3. Open Miniforge Prompt in that folder and run `install_windows.bat`.
4. Start the application with `BL62 Label GUI.vbs`.
5. On first launch, use **SAM3 Setup & GPU Test**. The application auto-detects common locations; otherwise browse to an existing SAM3 source repository and compatible checkpoint.

The installer creates the dedicated `BL62_Label_GUI` environment and does not modify other project environments.

## SAM3 and checkpoint

SAM3 source code and model weights are not included. Obtain them from the official provider and comply with their license. A valid repository contains both `pyproject.toml` and `sam3/__init__.py`. A checkpoint must be a `.pt`, `.pth`, or `.ckpt` file larger than 100 MB.

Configuration is saved locally in `sam3_settings.json`, which is excluded from Git.

## Project workflow

- Create a project from a folder or selected images.
- Drag images into an empty window to create a project.
- Drag images into an open project to add them.
- Use one unified **Label** for both the SAM3 text prompt and COCO category.
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
