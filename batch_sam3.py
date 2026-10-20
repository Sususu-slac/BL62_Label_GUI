"""Selected-image SAM3 jobs, isolated from the GUI with one model per batch."""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from PySide6.QtCore import QProcess, Qt
from PySide6.QtWidgets import QMessageBox, QProgressDialog

from sam3_setup import valid_checkpoint, valid_repository


class BatchSam3Mixin:
    def _batch_text(self, english, chinese):
        return english if self.ui_language == "en" else chinese

    def update_batch_button(self):
        if not hasattr(self, "run_sam_btn"):
            return
        count = len(self.images.selectedIndexes())
        multiple = count > 1
        ready = (not self.sam_busy and count > 0 and
                 ((multiple and self.prompt_mode == "text") or
                  (not multiple and self.sam_loaded_flag and
                   (self.prompt_mode == "text" or self.mask_prompt is not None))))
        self.run_sam_btn.setEnabled(ready)
        self.run_sam_btn.setText(self._batch_text(
            f"Run SAM3 ({count} picture{'s' if count != 1 else ''})",
            f"运行 SAM3（{count} 张图片）"))
        self.run_sam_btn.setToolTip(self._batch_text(
            "Select one image for Text / Box / Instance Prompt; select multiple images for full-image Text Prompt.",
            "单张支持文字、Box 和实例提示；多张使用整图文字提示。"))

    def run_selected_or_current(self):
        if self.sam_busy:
            return
        rows = sorted({index.row() for index in self.images.selectedIndexes()})
        if len(rows) > 1:
            self.run_sam3_selected()
        elif rows:
            if self.index != rows[0]:
                self.load_image(rows[0])
            self.run_sam3()

    def run_sam3_selected(self):
        if self.sam_busy:
            return
        rows = sorted({index.row() for index in self.images.selectedIndexes()})
        prompt = self.prompt.currentText().strip()
        if not rows or not prompt or self.prompt_mode != "text":
            QMessageBox.warning(self, "SAM3", self._batch_text(
                "Select images and enter a Text Prompt first.", "请先选择图片并填写文字提示词。"))
            return
        if not valid_repository(self.sam_repo) or not valid_checkpoint(self.sam_checkpoint):
            QMessageBox.warning(self, "SAM3", self._batch_text(
                "Select a valid SAM3 repository and checkpoint first.", "请先选择有效的 SAM3 仓库和 checkpoint。"))
            return
        title = self._batch_text("Clear All Labels and Run SAM3", "清除全部标签并运行 SAM3")
        message = self._batch_text(
            f"Run SAM3 again on {len(rows)} selected image(s)?\n\nALL existing labels on these images, including manual labels, will be replaced.\nText Prompt: {prompt}\nConfidence: {self.confidence.value()}%\nFull images will be processed; the current SAM3 Box is not used.\n\nEach successful image is saved automatically. If cancelled or failed, unfinished images keep their original labels. Unselected images are unchanged.",
            f"对选中的 {len(rows)} 张图片重新运行 SAM3？\n\n这些图片的全部现有标签（包括人工标签）都会被替换。\n文字提示：{prompt}\n置信度：{self.confidence.value()}%\n处理整张图片，不使用当前 SAM3 Box。\n\n每张成功后自动保存；取消或失败时，未完成图片保留原标签。未选中的图片不受影响。")
        dialog = QMessageBox(QMessageBox.Warning, title, message, parent=self)
        run = dialog.addButton(self._batch_text("Clear All Labels and Run", "清除全部标签并运行"), QMessageBox.AcceptRole)
        dialog.addButton(QMessageBox.Cancel)
        dialog.setDefaultButton(QMessageBox.Cancel)
        dialog.exec()
        if dialog.clickedButton() is not run:
            return
        self._start_selected_batch(rows, prompt)

    def _start_selected_batch(self, rows, prompt):
        try:
            self.commit_current()
            self.project.save()
            self._batch_temp = tempfile.TemporaryDirectory(prefix="bl62_sam3_batch_")
            folder = Path(self._batch_temp.name)
            self._batch_rows = list(rows)
            self._batch_completed = set()
            self._batch_prompt = prompt
            self._batch_category = self.current_category_id()
            self._batch_cancelled = False
            self._batch_error = ""
            self._batch_stdout = ""
            self._batch_stderr = ""
            manifest = folder / "batch.json"
            manifest.write_text(json.dumps([str(self.project.image_paths[row]) for row in rows]), encoding="utf-8")
            self._batch_dialog = QProgressDialog(self._batch_text("Loading SAM3…", "正在加载 SAM3……"), self._batch_text("Cancel", "取消"), 0, len(rows), self)
            self._batch_dialog.setWindowTitle(self._batch_text("Run SAM3 for Selected Pictures", "对选中图片运行 SAM3"))
            self._batch_dialog.setWindowModality(Qt.ApplicationModal)
            self._batch_dialog.setMinimumDuration(0)
            self._batch_dialog.setAutoClose(False)
            self._batch_dialog.setAutoReset(False)
            self._batch_dialog.canceled.connect(self.cancel_selected_batch)
            self.sam_busy = True
            self.update_run_button()
            self.load_sam_btn.setEnabled(False)
            process = self._batch_process = QProcess(self)
            process.setProgram(str(Path(sys.executable).with_name("python.exe")) if os.name == "nt" else sys.executable)
            process.setArguments([str(Path(__file__).with_name("sam3_inference_process.py")),
                "--repo", self.sam_repo, "--checkpoint", self.sam_checkpoint,
                "--confidence", str(self.confidence.value()/100), "--prompt", prompt,
                "--batch", str(manifest), "--output", str(folder / "result.npz")])
            process.readyReadStandardOutput.connect(self._read_batch_output)
            process.readyReadStandardError.connect(self._read_batch_error)
            process.finished.connect(self._batch_finished)
            process.errorOccurred.connect(self._batch_process_error)
            process.start()
            self._batch_dialog.show()
        except Exception as exc:
            self._batch_error = str(exc)
            self._finish_selected_batch()

    def _read_batch_error(self):
        self._batch_stderr = (self._batch_stderr + bytes(self._batch_process.readAllStandardError()).decode("utf-8", errors="replace"))[-8000:]

    def _read_batch_output(self):
        self._batch_stdout += bytes(self._batch_process.readAllStandardOutput()).decode("utf-8", errors="replace")
        lines = self._batch_stdout.split("\n")
        self._batch_stdout = lines.pop()
        for line in lines:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if self._batch_cancelled or self._batch_error:
                continue
            if event.get("type") == "progress":
                self._batch_dialog.setLabelText(event["message"])
            elif event.get("type") == "error":
                self._batch_error = event["message"]
            elif event.get("type") == "result":
                try:
                    self._accept_batch_result(event["index"])
                    if self._batch_process is not None and not self._batch_cancelled:
                        self._batch_process.write(b"next\n")
                except Exception as exc:
                    self._batch_error = str(exc)
                    if self._batch_process is not None:
                        self._batch_process.kill()

    def _accept_batch_result(self, position):
        if position in self._batch_completed or not 0 <= position < len(self._batch_rows):
            raise ValueError("Unexpected SAM3 batch result index")
        row = self._batch_rows[position]
        rec = self.project.image_record(row)
        with np.load(Path(self._batch_temp.name)/"result.npz", allow_pickle=False) as result:
            masks, scores = result["masks"], result["scores"]
            if masks.ndim != 3 or masks.shape[1:] != (rec["height"], rec["width"]) or len(scores) != len(masks):
                raise ValueError("SAM3 result dimensions do not match the image")
            entries = [{"id": 0, "category_id": self._batch_category, "noun_phrase": self._batch_prompt,
                "annotation_source": "sam3_text", "score": float(score), "mask": mask}
                for mask, score in zip(masks, scores)]
        previous_annotations = self.project.data["annotations"]
        previous_reviewed = self.project.reviewed.copy()
        try:
            self.project.replace_masks(rec["id"], entries)
            self.project.reviewed.discard(rec["id"])
            self.project.save()
        except Exception:
            self.project.data["annotations"] = previous_annotations
            self.project.reviewed = previous_reviewed
            self.project._rebuild_annotation_index()
            self.project.dirty = True
            raise
        # Never let stale displayed masks overwrite a newly accepted result.
        if row == self.index:
            self.current_masks_dirty = False
            self.load_image(row)
        self._batch_completed.add(position)
        self.update_image_status(row)
        self._batch_dialog.setLabelText(self._batch_text(
            f"Saved {len(self._batch_completed)}/{len(self._batch_rows)}: {self.project.image_paths[row].name}",
            f"已保存 {len(self._batch_completed)}/{len(self._batch_rows)}：{self.project.image_paths[row].name}"))
        self._batch_dialog.setValue(len(self._batch_completed))

    def cancel_selected_batch(self):
        self._batch_cancelled = True
        process = getattr(self, "_batch_process", None)
        if process is not None and process.state() != QProcess.NotRunning:
            process.kill()

    def _batch_process_error(self, error):
        if error == QProcess.FailedToStart:
            self._batch_error = self._batch_process.errorString()
            self._finish_selected_batch()

    def _batch_finished(self, code, _status):
        self._read_batch_output()
        self._read_batch_error()
        if not self._batch_cancelled and not self._batch_error and (code or len(self._batch_completed) != len(self._batch_rows)):
            self._batch_error = self._batch_stderr or "SAM3 batch exited before all images were processed"
        self._finish_selected_batch()

    def _finish_selected_batch(self):
        self._batch_process = None
        if getattr(self, "_batch_dialog", None):
            self._batch_dialog.canceled.disconnect(self.cancel_selected_batch)
            self._batch_dialog.close()
            self._batch_dialog.deleteLater()
            self._batch_dialog = None
        if getattr(self, "_batch_temp", None):
            self._batch_temp.cleanup()
            self._batch_temp = None
        self.sam_busy = False
        self.load_sam_btn.setEnabled(True)
        self.update_run_button()
        count = len(getattr(self, "_batch_completed", set()))
        message = self._batch_text(f"SAM3 batch: {count} image(s) saved.", f"SAM3 批量处理：已保存 {count} 张图片。")
        if getattr(self, "_batch_error", ""):
            QMessageBox.critical(self, "SAM3", message + "\n" + self._batch_error)
        else:
            message += self._batch_text(" Cancelled." if self._batch_cancelled else " Complete.", " 已取消。" if self._batch_cancelled else " 已完成。")
            QMessageBox.information(self, "SAM3", message)
        self.sam_status.setText(message)
