"""English user interface for Particle Label GUI.

The annotation/SAM3/COCO implementation is shared with app.py so both language
variants always write exactly the same project format.
"""
from pathlib import Path
import json
import sys
import tempfile
import time

from PIL import Image
import numpy as np
from PySide6.QtCore import Qt, QProcess, QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QDialog, QFileDialog, QHBoxLayout, QInputDialog,
    QLabel, QLineEdit, QMessageBox, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout)

from app import MainWindow as ChineseMainWindow
from coco_project import mask_geometry
from sam3_setup import detect_checkpoint, detect_repository, valid_checkpoint, valid_repository


TRANSLATIONS = {
    "平移": "Pan", "选择": "Select", "多边形": "Polygon", "画笔": "Brush",
    "橡皮擦": "Eraser", "SAM3框": "SAM3 Box", "隐藏标签": "Hide Labels", "显示标签": "Show Labels", "笔刷 ": " Brush size ",
    "新建项目": "New Project", "导入已有项目": "Import Existing Project",
    "从项目移除": "Remove from Project",
    "保存 Ctrl+S": "Save  Ctrl+S", "上一张": "Previous", "下一张": "Next",
    "标为已审核": "Toggle Reviewed", "导出": "Export",
    "未打开项目": "No project open", "图片（✓ 已审核，● 有标注）": "Images  (✓ reviewed, ● annotated)",
    "SAM3 GPU 辅助标注": "SAM3 GPU-Assisted Annotation",
    "选择 SAM3 仓库": "Select SAM3 Repository", "选择 checkpoint": "Select Checkpoint",
    "Noun phrase / SAM3 提示词": "Noun Phrase / SAM3 Prompt", "置信度": "Confidence", "SAM3 未加载": "SAM3 not loaded",
    "加载 SAM3 到 GPU": "Load SAM3 on GPU",
    "对选中图片运行 SAM3": "Run SAM3 for Selected Pictures",
    "标签": "Label", "管理标签": "Manage Labels",
    "运行 SAM3（文本 + 可选框）": "Run SAM3 (Text + Optional Box)",
    "清除 SAM3 框": "Clear SAM3 Box", "标注实例（Ctrl 可多选）": "Annotations (Ctrl: multi-select)",
    "SAM3 提示方式": "SAM3 Prompt Mode", "Text Prompt": "Text Prompt", "Mask Prompt": "Instance Prompt",
    "搜索范围：全图": "Search area: full image", "Mask Prompt：尚未设置": "Instance Prompt: not set",
    "使用当前选中实例": "Use Selected Instance", "绘制 Mask Prompt": "Draw Instance Prompt", "清除 Mask Prompt": "Clear Instance Prompt",
    "删除选中实例 Delete": "Delete Selected Instance  Del",
    "合并所选 Masks 为一个实例": "Merge Selected Masks into One Instance",
    "训练标签 Category": "Training Category", "Category ID / 名称": "Category ID / Name",
    "新建 Category": "New Category", "管理 Categories": "Manage Categories",
    "将当前标签分配给选中实例": "Assign Current Label to Selected",
}

STATUS = {
    "默认光标": "Default cursor",
    "平移/缩放": "Pan / zoom", "点击选择实例": "Click a mask to select an instance",
    "左键加点，右键或 Enter 完成": "Left-click points; right-click or Enter to finish",
    "直接绘制新实例；选中实例时修改它；Enter 完成当前实例": "Draw to create an instance; edit a selected instance; Enter finishes it",
    "擦除选中实例": "Erase from the selected instance",
    "拖动绘制 SAM3 正框提示": "Drag to draw a positive SAM3 box", "当前实例已完成；继续绘制会创建新实例": "Current instance finished; drawing again creates a new instance",
    "请先选择一个实例": "Select an instance first",
}


class MainWindow(ChineseMainWindow):
    def __init__(self):
        self.ui_language="en"
        super().__init__()
        self.setWindowTitle("Particle Label GUI — English — 2026-08-25")
        for widget in self.findChildren(QLabel) + self.findChildren(QPushButton):
            if widget.text() in TRANSLATIONS:
                widget.setText(TRANSLATIONS[widget.text()])
        for action in self.findChildren(QAction):
            if action.text() in TRANSLATIONS:
                action.setText(TRANSLATIONS[action.text()])
        try: self.view.status.disconnect(self.statusBar().showMessage)
        except Exception: pass
        self.view.status.connect(lambda text: self.statusBar().showMessage(STATUS.get(text, text)))
        self.statusBar().showMessage("Create a project from a folder or selected images; drag images here to add them to the current project")
        self.inference_process = None
        self.load_process = None
        self.load_stdout = ""
        self.task_cancelled = False
        self.inference_temp = None
        self.inference_stdout = ""
        self.inference_started = 0.0
        self.inference_timer = QTimer(self)
        self.inference_timer.timeout.connect(self._update_inference_elapsed)
        self.cancel_sam_btn = QPushButton("Cancel SAM3 Task")
        self.cancel_sam_btn.setEnabled(False)
        self.cancel_sam_btn.clicked.connect(self.cancel_sam_inference)
        self.run_sam_btn.parentWidget().layout().insertWidget(
            self.run_sam_btn.parentWidget().layout().indexOf(self.run_sam_btn) + 1,
            self.cancel_sam_btn,
        )
        self._update_sam_labels()
        QTimer.singleShot(0,self.ensure_sam3_configuration)

    def ensure_sam3_configuration(self):
        app_dir=Path(__file__).resolve().parent
        repository=detect_repository(app_dir,self.sam_repo); checkpoint=detect_checkpoint(app_dir,repository,self.sam_checkpoint)
        if repository:self.sam_repo=repository
        if checkpoint:self.sam_checkpoint=checkpoint
        self._update_sam_labels()
        if valid_repository(self.sam_repo) and valid_checkpoint(self.sam_checkpoint):
            self.statusBar().showMessage("SAM3 repository and checkpoint detected. Click Load SAM3 on GPU when ready.",8000)
        else:self.statusBar().showMessage("Select the SAM3 repository and checkpoint using the buttons above.",8000)

    def _update_sam_labels(self):
        self.repo_label.setText(f"Repository: {self.sam_repo}")
        self.ckpt_label.setText(f"Checkpoint: {self.sam_checkpoint}")
        self.settings_path.write_text(__import__('json').dumps({"repo":self.sam_repo,"checkpoint":self.sam_checkpoint},ensure_ascii=False,indent=2),encoding="utf-8")

    def choose_sam_repo(self):
        name=QFileDialog.getExistingDirectory(self,"Select the folder containing the sam3 Python package",self.sam_repo)
        if name:self.sam_repo=name; self._update_sam_labels()

    def choose_checkpoint(self):
        name,_=QFileDialog.getOpenFileName(self,"Select SAM3 checkpoint",str(Path(self.sam_checkpoint).parent),"PyTorch checkpoint (*.pt *.pth *.ckpt);;All files (*)")
        if name:self.sam_checkpoint=name; self._update_sam_labels()

    def new_project(self):
        self._choose_new_project(True)

    def open_project(self):
        name,_=QFileDialog.getOpenFileName(self,"Import Existing Project: select annotations.json","","COCO JSON (*.json)")
        if not name:return
        self._open_project_in_background(Path(name))

    def load_image(self,index):
        if index<0 or index>=len(self.project.image_paths):return
        if self.index>=0:self.commit_current()
        self.index=index; path=self.project.image_paths[index]
        try:
            rgb=np.asarray(Image.open(path).convert("RGB")); rec=self.project.image_record(index)
            self.masks=self.project.masks_for(rec["id"],rgb.shape[1],rgb.shape[0]); self.undo_stack=[]; self.redo_stack=[]
            self.mask_prompt=None; self.mask_prompt_source=None; self.capture_mask_prompt=False; self.view.exemplar_mask=None; self.view.sam_box=None
            self.view.set_image(rgb,self.masks); self.current_masks_dirty=False; self.mask_prompt_status.setText("Instance Prompt: not set"); self.update_roi_status(); self.update_run_button(); self.refresh_masks(); self.counter.setText(f"{index+1}/{len(self.project.image_paths)}  {path.name}")
        except Exception as exc:self.error(exc)

    def apply_labels_to_selected(self):
        rows=sorted({item.row() for item in self.mask_list.selectedIndexes()})
        if not rows and 0<=self.view.selected<len(self.masks):rows=[self.view.selected]
        if not rows:self.statusBar().showMessage("Select one or more instances first");return
        self.push_undo(); phrase=self.prompt.currentText().strip(); category_id=self.current_category_id(); category=self.project.category(category_id)
        for row in rows:self.masks[row]["category_id"]=category_id; self.masks[row]["noun_phrase"]=phrase or category["name"]
        self.view.redraw(); self.refresh_masks(); self.statusBar().showMessage(f"Updated training labels for {len(rows)} instance(s)")

    def toggle_review(self):
        if self.index<0:return
        self.commit_current(); image_id=self.project.image_record(self.index)["id"]
        if image_id in self.project.reviewed:self.project.reviewed.remove(image_id)
        else:self.project.reviewed.add(image_id)
        self.project.dirty=True; self.update_image_status(self.index); self.statusBar().showMessage("Review status updated; press Ctrl+S to save")

    def save(self):
        if self.project.json_path is None:return
        try:
            self.commit_current(force=True); self.project.save(); self.update_image_status(self.index); self.statusBar().showMessage(f"Saved: {self.project.json_path}",5000)
            QMessageBox.information(self,"Save Complete",f"Saved to:\n{self.project.json_path}")
        except Exception as exc:self.error(exc)

    def load_sam3(self):
        if self.sam_busy:return
        if not valid_repository(self.sam_repo) or not valid_checkpoint(self.sam_checkpoint):
            QMessageBox.warning(self,"SAM3","Configure a valid SAM3 repository and checkpoint first.");return
        self.task_cancelled=False; self.sam_busy=True; self.sam_loaded_flag=False; self.update_run_button(); self.load_sam_btn.setEnabled(False); self.cancel_sam_btn.setEnabled(True)
        python=Path(sys.executable).with_name("python.exe"); worker=Path(__file__).with_name("sam3_inference_process.py")
        self.load_stdout=""; self.inference_started=time.monotonic(); self.load_process=QProcess(self); self.load_process.setProgram(str(python))
        self.load_process.setArguments([str(worker),"--repo",self.sam_repo,"--checkpoint",self.sam_checkpoint,"--confidence",str(self.confidence.value()/100),"--check-only"])
        self.load_process.setProcessChannelMode(QProcess.SeparateChannels); self.load_process.readyReadStandardOutput.connect(self._read_load_output)
        self.load_process.finished.connect(self._load_process_finished); self.load_process.start(); self.inference_timer.start(1000)
        self.sam_status.setText("Loading SAM3 in an isolated process… The GUI remains responsive.")

    def _read_load_output(self):
        if not self.load_process:return
        self.load_stdout += bytes(self.load_process.readAllStandardOutput()).decode("utf-8",errors="replace")
        lines=self.load_stdout.split("\n"); self.load_stdout=lines.pop()
        for line in lines:
            try:item=json.loads(line)
            except Exception:continue
            if item.get("type")=="progress":self.sam_status.setText(item.get("message","Loading SAM3…"))

    def _load_process_finished(self,code,_status):
        self.inference_timer.stop(); self.cancel_sam_btn.setEnabled(False); self._read_load_output()
        try:
            if self.task_cancelled:
                self.sam_busy=False; self.sam_loaded_flag=False; self.load_sam_btn.setEnabled(True); self.update_run_button(); self.sam_status.setText("SAM3 loading cancelled"); self.task_cancelled=False; return
            if code!=0:
                error=bytes(self.load_process.readAllStandardError()).decode("utf-8",errors="replace").strip()
                raise RuntimeError(error or "The isolated SAM3 loading process failed")
            self.sam_busy=False; self.sam_loaded_flag=True; self.load_sam_btn.setEnabled(True); self.update_run_button(); self.sam_status.setText("SAM3 load test completed successfully. Run SAM3 is enabled.")
        except Exception as exc:
            self.sam_busy=False; self.sam_loaded_flag=False; self.load_sam_btn.setEnabled(True); self.update_run_button(); self.sam_status.setText(f"SAM3 loading failed: {exc}"); QMessageBox.critical(self,"SAM3",self.sam_status.text())
        finally:self.load_process=None

    def sam_loaded(self,message):
        english=(message.replace("SAM3 已加载：","SAM3 loaded: ")
                 .replace("，耗时 ",", elapsed ").replace(" 秒"," s"))
        self.sam_busy=False; self.sam_loaded_flag=True; self.load_sam_btn.setEnabled(True); self.update_run_button(); self.sam_status.setText(english)

    def sam_progress(self,message):
        english=(message.replace("正在导入 PyTorch 和 SAM3（首次约 5–10 秒）……","Importing PyTorch and SAM3 (first load may take 5–10 seconds)…")
            .replace("正在读取 checkpoint 并加载到 GPU……","Reading checkpoint and loading it onto the GPU…")
            .replace("正在提取图像特征","Extracting image features").replace("范围 ","region ").replace("正在执行 Text Prompt grounding……","Running Text Prompt grounding…")
            .replace("正在执行 Mask Prompt grounding……","Running Mask Prompt grounding…").replace("推理完成，正在传回","Inference complete; returning").replace("个 Masks……"," masks…"))
        self.sam_status.setText(english); self.statusBar().showMessage(english)

    def sam_failed(self,message):
        english=(message.replace("SAM3 加载失败：","SAM3 loading failed: ").replace("SAM3 推理失败：","SAM3 inference failed: ")
            .replace("请先加载 SAM3 模型","Load the SAM3 model first")
            .replace("请先指定或绘制一个 Mask Prompt","Select or draw a Mask Prompt first")
            .replace("Mask Prompt 不在当前 SAM3 框范围内","The Mask Prompt is outside the current SAM3 box"))
        self._sam_replace_requested=False; self._sam_replace_box=None
        self.sam_busy=False; self.load_sam_btn.setEnabled(True); self.run_sam_btn.setEnabled(self.sam_engine.processor is not None); self.sam_status.setText(english); QMessageBox.critical(self,"SAM3",english)

    def run_sam3(self):
        if self.view.array is None:return
        prompt=self.prompt.currentText().strip()
        if self.prompt_mode=="text" and not prompt:QMessageBox.warning(self,"SAM3","The Text Prompt cannot be empty");return
        if self.prompt_mode=="mask" and self.mask_prompt is None:QMessageBox.warning(self,"SAM3","Select or draw a Mask Prompt first");return
        if not prompt:prompt=(self.project.category(self.current_category_id()) or {"name":"particle"})["name"]
        if not self._confirm_sam_replacement():return
        self._start_isolated_inference(prompt)

    def _start_isolated_inference(self, prompt):
        self.task_cancelled=False; self.sam_busy=True; self.update_run_button(); self.cancel_sam_btn.setEnabled(True)
        self.inference_temp=tempfile.TemporaryDirectory(prefix="bl62_sam3_")
        folder=Path(self.inference_temp.name); image_path=folder/"image.npy"; output_path=folder/"result.npz"
        np.save(image_path,self.view.array,allow_pickle=False)
        python=Path(sys.executable).with_name("python.exe")
        worker=Path(__file__).with_name("sam3_inference_process.py")
        args=[str(worker),"--repo",self.sam_repo,"--checkpoint",self.sam_checkpoint,
              "--image",str(image_path),"--output",str(output_path),"--prompt",prompt,
              "--confidence",str(self.confidence.value()/100),
              "--mode","instance" if self.prompt_mode=="mask" else "text"]
        if self.prompt_mode=="mask":
            exemplar_path=folder/"exemplar.npy"
            np.save(exemplar_path,self.mask_prompt,allow_pickle=False)
            args += ["--exemplar",str(exemplar_path)]
        if self.view.sam_box:args += ["--box",json.dumps(self.view.sam_box)]
        self.inference_stdout=""; self.inference_started=time.monotonic()
        self.inference_process=QProcess(self); self.inference_process.setProgram(str(python)); self.inference_process.setArguments(args)
        self.inference_process.setProcessChannelMode(QProcess.SeparateChannels)
        self.inference_process.readyReadStandardOutput.connect(self._read_inference_output)
        self.inference_process.finished.connect(lambda code,_status:self._text_inference_finished(code,output_path))
        self.inference_process.start(); self.inference_timer.start(1000)
        kind="Instance Prompt" if self.prompt_mode=="mask" else "Text Prompt"
        self.sam_status.setText(f"Starting isolated SAM3 {kind} inference…")

    def _read_inference_output(self):
        self.inference_stdout += bytes(self.inference_process.readAllStandardOutput()).decode("utf-8",errors="replace")
        lines=self.inference_stdout.split("\n"); self.inference_stdout=lines.pop()
        for line in lines:
            try:item=json.loads(line)
            except Exception:continue
            if item.get("type")=="progress":self.sam_status.setText(item["message"])

    def _update_inference_elapsed(self):
        process=self.inference_process or self.load_process
        if process and process.state()!=QProcess.NotRunning:
            base=self.sam_status.text().split("  [")[0]
            self.sam_status.setText(f"{base}  [{time.monotonic()-self.inference_started:.0f} s]")

    def _text_inference_finished(self, code, output_path):
        self.inference_timer.stop(); self.cancel_sam_btn.setEnabled(False)
        self._read_inference_output()
        try:
            if self.task_cancelled:
                self._sam_replace_requested=False; self._sam_replace_box=None
                self.sam_busy=False; self.update_run_button(); self.sam_status.setText("SAM3 inference cancelled"); self.task_cancelled=False; return
            if code!=0 or not output_path.exists():
                error=bytes(self.inference_process.readAllStandardError()).decode("utf-8",errors="replace").strip()
                raise RuntimeError(error or "The isolated SAM3 process exited without a result")
            result=np.load(output_path,allow_pickle=False)
            self.sam_predicted(result["masks"],result["scores"].tolist())
        except Exception as exc:
            self._sam_replace_requested=False; self._sam_replace_box=None
            self.sam_busy=False; self.update_run_button(); self.sam_status.setText(f"SAM3 inference failed: {exc}")
            QMessageBox.critical(self,"SAM3",self.sam_status.text())
        finally:
            self.inference_process=None
            if self.inference_temp:self.inference_temp.cleanup(); self.inference_temp=None

    def cancel_sam_inference(self):
        self.task_cancelled=True
        if self.load_process and self.load_process.state()!=QProcess.NotRunning:
            self.load_process.kill(); self.load_process.waitForFinished(3000)
        if self.inference_process and self.inference_process.state()!=QProcess.NotRunning:
            self.inference_process.kill(); self.inference_process.waitForFinished(3000)
        self.inference_timer.stop(); self._sam_replace_requested=False; self._sam_replace_box=None; self.sam_busy=False; self.update_run_button(); self.cancel_sam_btn.setEnabled(False)
        self.sam_status.setText("SAM3 task cancelled")

    def sam_predicted(self,masks,scores):
        self.push_undo(); prompt=self.prompt.currentText().strip()
        self._apply_pending_sam_replacement()
        category_id=self.current_category_id(); source="sam3_text" if self.prompt_mode=="text" else "sam3_mask_prompt"
        for mask,score in zip(masks,scores):
            item={"id":0,"category_id":category_id,"noun_phrase":prompt,"score":float(score),"annotation_source":source,"mask":mask}
            if self.prompt_mode=="mask":item["mask_prompt_source"]=self.mask_prompt_source or "drawn_or_selected_mask"
            self.masks.append(item)
        self.view.masks=self.masks; self.view.redraw(); self.refresh_masks(); self.sam_busy=False; self.update_run_button(); self.sam_status.setText(f"Inference complete: added {len(masks)} masks ({source})")

    def clear_sam_box(self):
        self.view.sam_box=None; self.view.redraw(); self.update_roi_status(); self.statusBar().showMessage("SAM3 box cleared; search restored to full image")

    def update_roi_status(self,box=None):
        box=self.view.sam_box if box is None else box
        if box:
            x0,y0,x1,y1=box; self.roi_status.setText(f"Search area: SAM3 box  x={x0}, y={y0}, w={x1-x0}, h={y1-y0}")
        else:self.roi_status.setText("Search area: full image")

    def set_prompt_mode(self,mode):
        self.prompt_mode=mode; self.text_prompt_btn.setChecked(mode=="text"); self.mask_prompt_btn.setChecked(mode=="mask"); self.mask_prompt_controls.setVisible(mode=="mask"); self.update_run_button(); self.statusBar().showMessage("Text Prompt mode" if mode=="text" else "Instance Prompt mode: use a selected instance or draw a prompt mask")

    def update_run_button(self):
        self.update_batch_button()
        ready=self.sam_loaded_flag and not self.sam_busy and (self.prompt_mode=="text" or self.mask_prompt is not None); self.run_sam_btn.setEnabled(ready); self.run_sam_btn.setText("Run SAM3 — Text Prompt" if self.prompt_mode=="text" else "Run SAM3 — Instance Prompt")

    def use_selected_as_mask_prompt(self):
        index=self.view.selected
        if index<0 or index>=len(self.masks):QMessageBox.warning(self,"Mask Prompt","Select an instance in the image or instance list first");return
        self.mask_prompt=self.masks[index]["mask"].copy(); self.mask_prompt_source=f"Instance {index+1} / Ann ID {self.masks[index].get('id',0)}"; self.view.exemplar_mask=self.mask_prompt; self.view.redraw(); self.mask_prompt_status.setText(f"Mask Prompt: {self.mask_prompt_source}, area {int(self.mask_prompt.sum())} px"); self.update_run_button()

    def draw_mask_prompt(self):
        if self.view.array is None:QMessageBox.warning(self,"Mask Prompt","Import an image first");return
        self.capture_mask_prompt=True; self.view.selected=-1; self.activate_tool("brush"); self.statusBar().showMessage("Draw the Mask Prompt with the brush, then press Enter to finish")

    def on_instance_finished(self,index):
        if not self.capture_mask_prompt:return
        self.capture_mask_prompt=False; self.mask_prompt=self.masks[index]["mask"].copy(); self.mask_prompt_source="Drawn Mask Prompt"
        answer=QMessageBox.question(self,"Mask Prompt","Also save the drawn Mask Prompt as a formal instance?\nChoose No to use it only as a prompt.",QMessageBox.Yes|QMessageBox.No)
        if answer==QMessageBox.No:self.masks.pop(index); self.view.masks=self.masks; self.view.selected=-1; self.refresh_masks()
        self.view.exemplar_mask=self.mask_prompt; self.mask_prompt_status.setText(f"Mask Prompt: {self.mask_prompt_source}"); self.view.redraw(); self.update_run_button()

    def clear_mask_prompt(self):
        self.mask_prompt=None; self.mask_prompt_source=None; self.capture_mask_prompt=False; self.view.exemplar_mask=None; self.view.redraw(); self.mask_prompt_status.setText("Mask Prompt: not set"); self.update_run_button()

    def refresh_masks(self):
        current=self.view.selected; self.mask_list.blockSignals(True); self.mask_list.clear()
        for i,mask in enumerate(self.masks):
            area=float(mask.get("area",-1)) if not mask.get("_geometry_dirty",False) else -1
            if area<0:area=float(np.count_nonzero(mask["mask"]));mask["area"]=area
            category_id=int(mask.get("category_id",1)); category=self.project.category(category_id); name=(category or {}).get("name","Unknown")
            self.mask_list.addItem(f"Instance {i+1} | Label: {name} | Area {area:.0f}")
        self.mask_list.blockSignals(False)
        if 0<=current<len(self.masks):self.mask_list.setCurrentRow(current)

    def new_category(self):
        name,ok=QInputDialog.getText(self,"New Category","Category name:")
        if not ok:return
        supercategory,ok=QInputDialog.getText(self,"New Category","Supercategory:",text=name)
        if not ok:return
        try:
            item=self.project.add_category(name,supercategory); self.refresh_categories(item["id"]); self.statusBar().showMessage(f"Created Category {item['id']} | {item['name']}")
        except Exception as exc:self.error(exc)

    def manage_categories(self):
        self.guard_save(); dialog=QDialog(self); dialog.setWindowTitle("Manage Categories"); dialog.resize(650,420)
        layout=QVBoxLayout(dialog); table=QTableWidget(0,4); table.setHorizontalHeaderLabels(["ID","Category","Supercategory","Instance Count"]); table.setSelectionBehavior(QAbstractItemView.SelectRows); table.setEditTriggers(QAbstractItemView.NoEditTriggers); layout.addWidget(table)
        def refill(select_id=None):
            table.setRowCount(0)
            for row,item in enumerate(self.project.categories()):
                table.insertRow(row); values=[item["id"],item.get("name",""),item.get("supercategory",""),self.project.category_usage(int(item["id"]))]
                for col,value in enumerate(values):table.setItem(row,col,QTableWidgetItem(str(value)))
                if int(item["id"])==select_id:table.selectRow(row)
        def selected_id():
            row=table.currentRow(); return int(table.item(row,0).text()) if row>=0 else None
        def add(): self.new_category(); refill(self.current_category_id())
        def rename():
            category_id=selected_id()
            if category_id is None:return
            item=self.project.category(category_id); name,ok=QInputDialog.getText(dialog,"Rename Category","Category name:",text=item["name"])
            if not ok:return
            sup,ok=QInputDialog.getText(dialog,"Rename Category","Supercategory:",text=item.get("supercategory",name))
            if ok:
                try:self.project.rename_category(category_id,name,sup); refill(category_id)
                except Exception as exc:self.error(exc)
        def remove():
            category_id=selected_id()
            if category_id is None:return
            try:self.project.delete_category(category_id); refill()
            except Exception as exc:self.error(exc)
        def merge():
            source=selected_id()
            if source is None:return
            choices=[f"{c['id']} | {c['name']}" for c in self.project.categories() if int(c["id"])!=source]
            target_text,ok=QInputDialog.getItem(dialog,"Merge Category",f"Reassign every Category {source} instance to:",choices,0,False)
            if not ok:return
            target=int(target_text.split("|",1)[0].strip())
            if QMessageBox.question(dialog,"Confirm Merge",f"Category {source} will be deleted and its instances reassigned to Category {target}. Continue?")!=QMessageBox.Yes:return
            for mask in self.masks:
                if int(mask.get("category_id",0))==source:mask["category_id"]=target
            self.project.merge_category(source,target); refill(target)
        buttons=QHBoxLayout()
        for text,slot in [("New",add),("Rename",rename),("Delete Unused",remove),("Merge into Another Category",merge)]:
            button=QPushButton(text); button.clicked.connect(slot); buttons.addWidget(button)
        close=QPushButton("Close"); close.clicked.connect(dialog.accept); buttons.addWidget(close); layout.addLayout(buttons); refill(); dialog.exec(); self.refresh_categories(); self.refresh_masks()

    def merge_masks(self):
        rows=sorted({item.row() for item in self.mask_list.selectedIndexes()})
        if len(rows)<2:self.statusBar().showMessage("Select at least two instances in the list");return
        base=self.masks[rows[0]]; category=self.project.category(int(base.get("category_id",1))) or {"name":"Unknown"}
        message=f"Merge {len(rows)} masks into one instance?\nThe first instance's Category {base.get('category_id',1)} | {category['name']} and noun phrase will be retained."
        if QMessageBox.question(self,"Confirm Instance Merge",message,QMessageBox.Yes|QMessageBox.No)!=QMessageBox.Yes:return
        import numpy as np
        self.push_undo(); base["mask"]=np.logical_or.reduce([self.masks[i]["mask"] for i in rows]); base["_geometry_dirty"]=True; self.masks[:]=[m for i,m in enumerate(self.masks) if i not in rows[1:]]; self.view.selected=rows[0]; self.view.redraw(); self.refresh_masks()

    def error(self,exc):
        QMessageBox.critical(self,"Error",str(exc))

    def closeEvent(self,event):
        if getattr(self,"_batch_process",None) is not None:event.ignore();return
        if self.export_thread and self.export_thread.isRunning():QMessageBox.warning(self,"Export","Please cancel the export and wait for it to stop before closing.");event.ignore();return
        if self.open_thread and self.open_thread.isRunning():QMessageBox.warning(self,"Import Existing Project","Please wait for the project import to finish before closing.");event.ignore();return
        if self.project.dirty:
            answer=QMessageBox.question(self,"Save Changes","Save changes before closing?",QMessageBox.Yes|QMessageBox.No|QMessageBox.Cancel)
            if answer==QMessageBox.Cancel:event.ignore();return
            if answer==QMessageBox.Yes:self.save()
        if (self.inference_process and self.inference_process.state()!=QProcess.NotRunning) or (self.load_process and self.load_process.state()!=QProcess.NotRunning):self.cancel_sam_inference()
        self.sam_thread.quit(); self.sam_thread.wait(3000); event.accept()


if __name__ == "__main__":
    app=QApplication(sys.argv); app.setStyle("Fusion"); window=MainWindow(); window.show(); sys.exit(app.exec())
