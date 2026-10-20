"""Isolated SAM3 inference, including a persistent JSON-lines request server."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def progress(message: str) -> None:
    print(json.dumps({"type": "progress", "message": message}), flush=True)


def predict_image(processor, torch, rgb, prompt, confidence, box=None, mode="text", exemplar_path=None):
    processor.set_confidence_threshold(confidence)
    full_h, full_w = rgb.shape[:2]
    if box:
        x0, y0, x1, y1 = box
        x0=max(0,min(int(x0),full_w-1)); x1=max(x0+1,min(int(x1)+1,full_w))
        y0=max(0,min(int(y0),full_h-1)); y1=max(y0+1,min(int(y1)+1,full_h))
    else:
        x0,y0,x1,y1=0,0,full_w,full_h
    cropped = rgb[y0:y1, x0:x1]
    progress(f"Extracting image features ({x1-x0}×{y1-y0})…")
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        state = processor.set_image(Image.fromarray(cropped))
        if mode == "text":
            progress("Running Text Prompt grounding…")
            state = processor.set_text_prompt(prompt=prompt, state=state)
        else:
            if not exemplar_path or not Path(exemplar_path).is_file():
                raise RuntimeError("Instance Prompt mask was not provided")
            exemplar = np.load(exemplar_path, allow_pickle=False).astype(bool)[y0:y1, x0:x1]
            if not exemplar.any():
                raise RuntimeError("The Instance Prompt is outside the current SAM3 box")
            progress("Running Instance Prompt grounding…")
            state["backbone_out"].update(processor.model.backbone.forward_text(["visual"], device="cuda"))
            state["geometric_prompt"] = processor.model._get_dummy_prompt()
            prompt_mask = torch.as_tensor(exemplar, device="cuda", dtype=torch.float32).view(1,1,1,*exemplar.shape)
            state["geometric_prompt"].append_masks(prompt_mask)
            state = processor._forward_grounding(state)
    cropped_masks = state["masks"].squeeze(1).detach().cpu().numpy().astype(bool)
    masks = np.zeros((len(cropped_masks), full_h, full_w), dtype=bool)
    masks[:, y0:y1, x0:x1] = cropped_masks
    scores = state["scores"].detach().float().cpu().numpy()
    return masks, scores


def serve(processor, torch):
    print(json.dumps({"type": "ready", "pid": __import__("os").getpid(), "gpu": torch.cuda.get_device_name(0)}), flush=True)
    for line in sys.stdin:
        request_id = None
        try:
            request = json.loads(line)
            request_id = request["id"]
            path = Path(request["image"])
            if path.suffix.lower() == ".npy":
                rgb = np.load(path, allow_pickle=False)
            else:
                with Image.open(path) as image:
                    rgb = np.asarray(image.convert("RGB"))
            masks, scores = predict_image(processor, torch, rgb, request["prompt"],
                request["confidence"], request.get("box"), request.get("mode", "text"), request.get("exemplar"))
            np.savez(request["output"], masks=masks, scores=scores)
            print(json.dumps({"type": "result", "id": request_id, "count": len(masks)}), flush=True)
            del rgb, masks, scores
        except Exception as exc:
            print(json.dumps({"type": "error", "id": request_id, "message": f"{type(exc).__name__}: {exc}"}), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prompt", default="particle")
    parser.add_argument("--confidence", type=float, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--box", default="")
    parser.add_argument("--mode", choices=("text", "instance"), default="text")
    parser.add_argument("--exemplar", type=Path)
    args = parser.parse_args()

    try:
        repo = args.repo.resolve(); checkpoint = args.checkpoint.resolve()
        if not (repo / "sam3").is_dir():
            raise FileNotFoundError(f"SAM3 package not found: {repo / 'sam3'}")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        sys.path.insert(0, str(repo))
        progress("Loading SAM3 model into GPU memory…")
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in the SAM3p1 environment")
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        model = build_sam3_image_model(
            checkpoint_path=str(checkpoint), load_from_HF=False,
            device="cuda", eval_mode=True,
        )
        processor = Sam3Processor(model=model, device="cuda", confidence_threshold=args.confidence)
        if args.serve:
            return serve(processor, torch)
        if args.check_only:
            print(json.dumps({"type": "done", "message": f"SAM3 loaded successfully on {torch.cuda.get_device_name(0)}"}), flush=True)
            return 0
        if args.batch:
            if args.mode != "text" or args.box or not args.output:
                raise ValueError("Batch inference requires full-image Text Prompt and an output path")
            paths = json.loads(args.batch.read_text(encoding="utf-8"))
            for index, path in enumerate(paths):
                progress(f"{index+1}/{len(paths)}: {Path(path).name}")
                with Image.open(path) as image:
                    rgb = image.convert("RGB")
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    state = processor.set_image(rgb)
                    state = processor.set_text_prompt(prompt=args.prompt, state=state)
                masks = state["masks"].squeeze(1).detach().cpu().numpy().astype(bool)
                scores = state["scores"].detach().float().cpu().numpy()
                np.savez(args.output, masks=masks, scores=scores)
                del state, masks, scores
                print(json.dumps({"type": "result", "index": index}), flush=True)
                # Keep at most one full-resolution result on disk; wait until
                # the GUI has safely saved it before processing the next image.
                if sys.stdin.readline().strip() != "next":
                    return 0
            return 0
        if not args.image or not args.image.is_file():
            raise FileNotFoundError("Input image array was not provided")
        if not args.output:
            raise ValueError("Output path was not provided")
        rgb = np.load(args.image, allow_pickle=False)
        masks, scores = predict_image(processor, torch, rgb, args.prompt, args.confidence,
            json.loads(args.box) if args.box else None, args.mode, args.exemplar)
        np.savez(args.output, masks=masks, scores=scores)
        print(json.dumps({"type": "done", "count": len(masks)}), flush=True)
        return 0
    except Exception as exc:
        print(json.dumps({"type": "error", "message": f"{type(exc).__name__}: {exc}"}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
