from __future__ import annotations

import copy
import csv
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from PySide6.QtCore import QObject, QPointF, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QAction, QColor, QCursor, QIcon, QImage, QKeySequence, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QButtonGroup, QComboBox, QDialog, QFileDialog, QFormLayout,
    QGraphicsPathItem, QGraphicsScene, QGraphicsView, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget,
    QMainWindow, QMenu, QMessageBox, QProgressDialog, QPushButton, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QToolBar,
    QVBoxLayout, QWidget)

from coco_project import CocoProject, mask_geometry
from batch_sam3 import BatchSam3Mixin


COLORS = [(255,75,75),(57,189,248),(75,220,120),(255,196,65),(190,100,255),(255,105,190),(75,230,220)]

THEME_QSS = """
QWidget {
    background: #0b1018;
    color: #eaf0fa;
    font-size: 13px;
}
QMainWindow, QDialog { background: #0b1018; }
QToolBar {
    background: #101824;
    border: 0;
    border-bottom: 1px solid #293348;
    spacing: 5px;
    padding: 5px;
}
QToolBar QToolButton {
    background: #1a2637;
    border: 1px solid #34445e;
    border-radius: 5px;
    padding: 7px 10px;
}
QToolBar QToolButton:hover { background: #23344c; border-color: #46617f; }
QToolBar QToolButton:pressed, QToolBar QToolButton:checked { background: #16c79a; color: #07120f; }
QPushButton {
    background: #1a2637;
    border: 1px solid #34445e;
    border-radius: 6px;
    padding: 8px 12px;
}
QPushButton:hover { background: #23344c; border-color: #46617f; }
QPushButton:pressed, QPushButton:checked { background: #16c79a; color: #07120f; font-weight: 700; }
QPushButton:disabled { background: #111824; color: #667287; border-color: #242e40; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background: #121a28;
    border: 1px solid #303c52;
    border-radius: 5px;
    padding: 6px;
    selection-background-color: #16c79a;
    selection-color: #07120f;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border-color: #16c79a; }
QComboBox QAbstractItemView {
    background: #121a28;
    border: 1px solid #34445e;
    selection-background-color: #16c79a;
    selection-color: #07120f;
}
QListWidget, QTableWidget {
    background: #0f1622;
    alternate-background-color: #121b29;
    border: 1px solid #293348;
    border-radius: 6px;
    gridline-color: #293348;
    outline: 0;
}
QListWidget::item, QTableWidget::item { padding: 5px; }
QListWidget::item:selected, QTableWidget::item:selected {
    background: #16c79a;
    color: #07120f;
}
QHeaderView::section {
    background: #182335;
    color: #dfe8f5;
    border: 0;
    border-right: 1px solid #293348;
    border-bottom: 1px solid #293348;
    padding: 7px;
}
QSplitter::handle { background: #202b3d; }
QSplitter::handle:hover { background: #16c79a; }
QStatusBar { background: #101824; color: #aebbd0; border-top: 1px solid #293348; }
QStatusBar::item { border: 0; }
QScrollBar:vertical { background: #0f1622; width: 12px; margin: 0; }
QScrollBar:horizontal { background: #0f1622; height: 12px; margin: 0; }
QScrollBar::handle { background: #34445e; border-radius: 5px; min-height: 24px; min-width: 24px; }
QScrollBar::handle:hover { background: #46617f; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: none; }
QMessageBox, QInputDialog, QFileDialog { background: #0b1018; }
QToolTip { background: #182335; color: #eaf0fa; border: 1px solid #46617f; padding: 5px; }
"""


class AnnotationView(QGraphicsView):
    changed = Signal()
    selection_requested = Signal(int)
    status = Signal(str)
    box_changed = Signal(object)
    masks_changed = Signal()
    instance_finished = Signal(int)
    small_mask_finished = Signal(int,int)
    tool_reset_requested = Signal()

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self)); self.setBackgroundBrush(QColor("#090d14"))
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.image_item = None; self.array = None; self.masks = []; self.selected = -1; self.show_labels = True
        self.mode = "idle"; self.brush_size = 20; self.polygon = []; self.preview = None; self.drawing = False
        self._middle_panning=False; self._middle_pan_pos=None
        self.sam_box = None; self.box_start = None
        self.exemplar_mask = None
        self._overlay_cache=None; self._overlay_signature=None; self._editing_background=None; self.overlay_item=None; self.active_overlay_item=None
        self._background_key=None; self._hit_map=None; self._hit_signature=None; self._bbox_cache={}
        self._live_redraw_pending=False; self._live_redraw_timer=QTimer(self); self._live_redraw_timer.setSingleShot(True)
        self._live_redraw_timer.setInterval(33); self._live_redraw_timer.timeout.connect(self._flush_live_redraw)
        self.new_mask_metadata = {"category_id":1,"noun_phrase":"particle","annotation_source":"manual"}
        self._update_tool_cursor()

    def set_image(self, array: np.ndarray, masks: list[dict]):
        self.scene().clear(); self.array = array; self.masks = masks; self.selected = -1; self.polygon = []; self.overlay_item=None; self.active_overlay_item=None
        self._overlay_cache=None; self._overlay_signature=None; self._editing_background=None; self._background_key=None
        self._hit_map=None; self._hit_signature=None; self._bbox_cache={}
        h,w=array.shape[:2]
        for entry in masks:
            bbox=entry.get("bbox")
            if bbox and len(bbox)>=4 and not entry.get("_geometry_dirty",False):
                x,y,bw,bh=[int(round(float(value))) for value in bbox[:4]]
                self._bbox_cache[id(entry["mask"])]=(max(0,x),max(0,y),min(w,x+bw),min(h,y+bh)) if bw>0 and bh>0 else None
        rgb = np.ascontiguousarray(array[..., :3]); h,w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888).copy()
        self.image_item = self.scene().addPixmap(QPixmap.fromImage(qimg)); self.image_item.setZValue(0)
        self.redraw(); self.fitInView(self.image_item, Qt.KeepAspectRatio)

    def redraw(self):
        for item in list(self.scene().items()):
            if item is not self.image_item and item is not self.overlay_item and item is not self.active_overlay_item: self.scene().removeItem(item)
        if self.array is None: return
        h,w=self.array.shape[:2]
        # Keep one composite of all instances. Selection changes only add a
        # highlight and therefore no longer rebuild hundreds of SAM3 masks.
        signature=(h,w,*(id(entry["mask"]) for entry in self.masks))
        if signature!=self._overlay_signature:
            self._overlay_cache=self._compose_overlay(); self._overlay_signature=signature
        editing=self.drawing and self._editing_background is not None
        background=self._editing_background if editing else self._overlay_cache
        background_key=("editing",self.selected,signature) if editing else ("base",signature)
        if background_key!=self._background_key:
            rgba=np.ascontiguousarray(background); qimg=QImage(rgba.data,w,h,rgba.strides[0],QImage.Format_RGBA8888).copy(); pixmap=QPixmap.fromImage(qimg)
            if self.overlay_item is None:
                self.overlay_item=self.scene().addPixmap(pixmap); self.overlay_item.setZValue(2)
            else:self.overlay_item.setPixmap(pixmap)
            self._background_key=background_key
        if self.overlay_item is not None:self.overlay_item.setVisible(self.show_labels)
        if self.show_labels:self._redraw_active_overlay()
        elif self.active_overlay_item is not None:self.active_overlay_item.setVisible(False)
        if self.show_labels and self.exemplar_mask is not None:
            contours,_=cv2.findContours(self.exemplar_mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            path=QPainterPath()
            for contour in contours:
                points=contour.reshape(-1,2)
                if not len(points):continue
                path.moveTo(float(points[0,0]),float(points[0,1]))
                for point in points[1:]:path.lineTo(float(point[0]),float(point[1]))
                path.closeSubpath()
            item=QGraphicsPathItem(path); item.setPen(QPen(QColor(190,60,255),4)); item.setZValue(4); self.scene().addItem(item)
        if self.polygon:
            path = QPainterPath(QPointF(*self.polygon[0]))
            for p in self.polygon[1:]: path.lineTo(QPointF(*p))
            self.preview = QGraphicsPathItem(path); self.preview.setPen(QPen(QColor("#ffff00"),2)); self.preview.setZValue(4); self.scene().addItem(self.preview)
        if self.sam_box:
            x0,y0,x1,y1=self.sam_box; path=QPainterPath(); path.addRect(float(x0),float(y0),float(x1-x0),float(y1-y0))
            item=QGraphicsPathItem(path); item.setPen(QPen(QColor("#ffff00"),3)); item.setZValue(5); self.scene().addItem(item)

    def _compose_overlay(self, exclude=-1):
        h,w=self.array.shape[:2]; overlay=np.zeros((h,w,4),dtype=np.uint8)
        for idx,entry in enumerate(self.masks):
            if idx==exclude:continue
            mask=entry["mask"]; box=self._mask_bbox(mask)
            if box is None:continue
            x0,y0,x1,y1=box; crop=mask[y0:y1,x0:x1]; region=overlay[y0:y1,x0:x1]
            region[crop,:3]=COLORS[idx%len(COLORS)]; region[crop,3]=100
        return overlay

    def _editing_overlay(self,selected):
        # Start from the already composed background and repair only the
        # selected instance's bounding box. This preserves overlapping masks
        # without recomposing every full-resolution SAM3 mask.
        if self._overlay_cache is None:return self._compose_overlay(selected)
        active=self.masks[selected]["mask"]; box=self._mask_bbox(active)
        if box is None:return self._overlay_cache
        background=self._overlay_cache.copy(); x0,y0,x1,y1=box
        active_crop=active[y0:y1,x0:x1]; region=background[y0:y1,x0:x1]; region[active_crop]=0
        for idx,entry in enumerate(self.masks):
            if idx==selected:continue
            overlap=entry["mask"][y0:y1,x0:x1] & active_crop
            if overlap.any():region[overlap,:3]=COLORS[idx%len(COLORS)]; region[overlap,3]=100
        return background

    def _commit_editing_overlay(self,selected):
        if self._editing_background is None:return
        background=self._editing_background.copy(); active=self.masks[selected]["mask"]; box=self._mask_bbox(active)
        if box is not None:
            x0,y0,x1,y1=box; active_crop=active[y0:y1,x0:x1]; region=background[y0:y1,x0:x1]
            region[active_crop,:3]=COLORS[selected%len(COLORS)]; region[active_crop,3]=100
            # Masks later in the list are visually on top; restore only their
            # overlap with the edited instance.
            for idx in range(selected+1,len(self.masks)):
                overlap=self.masks[idx]["mask"][y0:y1,x0:x1] & active_crop
                if overlap.any():region[overlap,:3]=COLORS[idx%len(COLORS)]; region[overlap,3]=100
        h,w=self.array.shape[:2]
        self._overlay_cache=background; self._overlay_signature=(h,w,*(id(entry["mask"]) for entry in self.masks))

    def _mask_bbox(self,mask):
        key=id(mask)
        if key not in self._bbox_cache:
            ys,xs=np.nonzero(mask); self._bbox_cache[key]=(int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1) if len(xs) else None
        return self._bbox_cache[key]

    def _redraw_active_overlay(self):
        if self.active_overlay_item is not None:
            self.active_overlay_item.setVisible(False)
        if self.selected<0 or self.selected>=len(self.masks):return
        mask=self.masks[self.selected]["mask"]; box=self._mask_bbox(mask)
        if box is None:return
        x0,y0,x1,y1=box; crop=mask[y0:y1,x0:x1]; overlay=np.zeros((y1-y0,x1-x0,4),np.uint8)
        overlay[crop,:3]=COLORS[self.selected%len(COLORS)]; overlay[crop,3]=155
        if not self.drawing:
            contours,_=cv2.findContours(crop.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay,contours,-1,(*COLORS[self.selected%len(COLORS)],255),2)
        rgba=np.ascontiguousarray(overlay); qimg=QImage(rgba.data,x1-x0,y1-y0,rgba.strides[0],QImage.Format_RGBA8888).copy(); pixmap=QPixmap.fromImage(qimg)
        if self.active_overlay_item is None:
            self.active_overlay_item=self.scene().addPixmap(pixmap); self.active_overlay_item.setZValue(3)
        else:self.active_overlay_item.setPixmap(pixmap)
        self.active_overlay_item.setPos(x0,y0); self.active_overlay_item.setVisible(True)

    def _ensure_hit_map(self):
        h,w=self.array.shape[:2]; signature=(h,w,*(id(entry["mask"]) for entry in self.masks))
        if signature!=self._hit_signature:
            hit_map=np.full((h,w),-1,np.int32)
            for idx,entry in enumerate(self.masks):hit_map[entry["mask"]]=idx
            self._hit_map=hit_map; self._hit_signature=signature

    def set_mode(self, mode: str):
        self.mode = mode; self.setDragMode(QGraphicsView.ScrollHandDrag if mode == "pan" else QGraphicsView.NoDrag)
        self._update_tool_cursor()
        self.status.emit({"idle":"默认光标","pan":"平移/缩放","select":"点击选择实例","polygon":"左键加点，右键或 Enter 完成","brush":"直接绘制新实例；选中实例时修改它；Enter 完成当前实例","erase":"擦除选中实例","sam_box":"拖动绘制 SAM3 正框提示"}[mode])

    def set_labels_visible(self,visible):
        self.show_labels=bool(visible); self.redraw()

    def set_brush_size(self, size: int):
        self.brush_size=int(size); self._update_tool_cursor()

    def _update_tool_cursor(self):
        if self.mode=="pan":cursor=QCursor(Qt.OpenHandCursor)
        elif self.mode in {"idle","select"}:cursor=QCursor(Qt.ArrowCursor)
        elif self.mode=="polygon":cursor=QCursor(Qt.CrossCursor)
        elif self.mode=="sam_box":
            canvas=25; center=canvas//2; pixmap=QPixmap(canvas,canvas); pixmap.fill(Qt.transparent)
            painter=QPainter(pixmap); painter.setRenderHint(QPainter.Antialiasing); painter.setPen(QPen(QColor("#55d8ff"),2))
            painter.drawRect(3,3,canvas-7,canvas-7); painter.drawLine(center,0,center,canvas-1); painter.drawLine(0,center,canvas-1,center); painter.end()
            cursor=QCursor(pixmap,center,center)
        elif self.mode in {"brush","erase"}:
            scale=max(abs(self.transform().m11()),0.05); diameter=max(7,min(61,int(round(self.brush_size*scale))))
            canvas=max(17,diameter+6); pixmap=QPixmap(canvas,canvas); pixmap.fill(Qt.transparent)
            painter=QPainter(pixmap); painter.setRenderHint(QPainter.Antialiasing)
            color=QColor("#f4f8ff") if self.mode=="brush" else QColor("#ff5b5b")
            painter.setPen(QPen(color,2)); margin=3; painter.drawEllipse(margin,margin,canvas-2*margin-1,canvas-2*margin-1)
            center=canvas//2
            if self.mode=="brush":painter.drawPoint(center,center)
            else:
                arm=max(3,min(9,diameter//3)); painter.drawLine(center-arm,center-arm,center+arm,center+arm); painter.drawLine(center-arm,center+arm,center+arm,center-arm)
            painter.end(); cursor=QCursor(pixmap,center,center)
        else:cursor=QCursor(Qt.ArrowCursor)
        self.viewport().setCursor(cursor)

    def wheelEvent(self, event):
        if not event.angleDelta().y():return super().wheelEvent(event)
        cursor=event.position().toPoint(); anchor_before=self.mapToScene(cursor)
        current=abs(self.transform().m11()); requested=1.2 if event.angleDelta().y()>0 else 1/1.2
        target=max(0.05,min(40.0,current*requested)); factor=target/current if current else 1.0
        previous_anchor=self.transformationAnchor(); self.setTransformationAnchor(QGraphicsView.NoAnchor); self.scale(factor,factor)
        anchor_after=self.mapToScene(cursor); delta=anchor_after-anchor_before; self.translate(delta.x(),delta.y())
        self.setTransformationAnchor(previous_anchor); self._update_tool_cursor(); event.accept()

    def image_pos(self, event):
        if self.image_item is None: return None
        p = self.image_item.mapFromScene(self.mapToScene(event.position().toPoint()))
        x,y = int(round(p.x())),int(round(p.y()))
        h,w = self.array.shape[:2]
        return (x,y) if 0 <= x < w and 0 <= y < h else None

    def mousePressEvent(self, event):
        if event.button()==Qt.MiddleButton:
            self._middle_panning=True; self._middle_pan_pos=event.position(); self.viewport().setCursor(QCursor(Qt.ClosedHandCursor)); event.accept(); return
        if event.button()==Qt.RightButton and self.mode in {"brush","erase"}:
            self.finish_current_instance(); event.accept(); return
        pos = self.image_pos(event)
        if pos is None: return super().mousePressEvent(event)
        x,y = pos
        self._ensure_hit_map(); hit=int(self._hit_map[y,x])
        if self.mode=="idle":
            if hit>=0:self.selection_requested.emit(hit)
            else:self.selection_requested.emit(-1)
            return super().mousePressEvent(event)
        # Existing annotations are selectable directly in every interactive
        # mode.  This keeps the image overlay, instance list and label fields
        # synchronized before a user edits or deletes the selected instance.
        if event.button() == Qt.LeftButton and hit>=0 and self.mode in {"pan","select","brush","erase"}:
            if self.selected!=hit:self.selection_requested.emit(hit)
            if self.mode=="pan":return super().mousePressEvent(event)
        elif self.mode=="pan":return super().mousePressEvent(event)
        if self.mode == "select":
            if hit>=0: self.selection_requested.emit(hit)
            else: self.selected=-1; self.selection_requested.emit(-1); self.redraw()
        elif self.mode == "polygon":
            if event.button() == Qt.RightButton: self.finish_polygon()
            else: self.polygon.append((x,y)); self.redraw()
        elif self.mode in {"brush","erase"}:
            if self.selected < 0 and self.mode == "erase": self.status.emit("请先选择一个实例"); return
            created = False
            if self.selected < 0 and self.mode == "brush":
                self.changed.emit(); h,w=self.array.shape[:2]
                item=dict(self.new_mask_metadata); item.update({"id":0,"mask":np.zeros((h,w),bool)})
                self.masks.append(item); self.selected=len(self.masks)-1
                # The new mask is empty, so the existing composite remains
                # valid even though the mask list signature has changed.
                self._overlay_signature=(h,w,*(id(entry["mask"]) for entry in self.masks))
                self.selection_requested.emit(self.selected); self.masks_changed.emit(); created=True
            self._editing_background=self._overlay_cache if created else self._editing_overlay(self.selected)
            self._background_key=None; self.drawing = True
            if not created:self.changed.emit()
            self.paint_at(x,y)
        elif self.mode == "sam_box":
            self.box_start=(x,y); self.sam_box=(x,y,x,y); self.drawing=True; self.redraw()

    def mouseMoveEvent(self,event):
        if self._middle_panning and self._middle_pan_pos is not None:
            delta=event.position()-self._middle_pan_pos; self._middle_pan_pos=event.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value()-int(round(delta.x())))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value()-int(round(delta.y())))
            event.accept(); return
        if self.drawing and self.mode in {"brush","erase"}:
            pos=self.image_pos(event)
            if pos: self.paint_at(*pos)
        elif self.drawing and self.mode=="sam_box":
            pos=self.image_pos(event)
            if pos:
                x0,y0=self.box_start; x1,y1=pos; self.sam_box=(min(x0,x1),min(y0,y1),max(x0,x1),max(y0,y1)); self.redraw()
        else: super().mouseMoveEvent(event)

    def mouseReleaseEvent(self,event):
        if event.button()==Qt.MiddleButton and self._middle_panning:
            self._middle_panning=False; self._middle_pan_pos=None; self._update_tool_cursor(); event.accept(); return
        was_box=self.drawing and self.mode=="sam_box"
        if self.mode in {"brush","erase"}:
            if self.drawing and 0<=self.selected<len(self.masks):self._commit_editing_overlay(self.selected)
            self.drawing=False;self._live_redraw_timer.stop();self._live_redraw_pending=False;self._editing_background=None;self._background_key=None;self._hit_signature=None;self.redraw()
        else:self.drawing=False
        if was_box:self.box_changed.emit(self.sam_box)
        super().mouseReleaseEvent(event)

    def paint_at(self,x,y):
        mask=self.masks[self.selected]["mask"]; radius=max(1,self.brush_size//2); h,w=mask.shape
        x0=max(0,x-radius);x1=min(w,x+radius+1);y0=max(0,y-radius);y1=min(h,y+radius+1)
        yy,xx=np.ogrid[y0-y:y1-y,x0-x:x1-x]; disk=xx*xx+yy*yy<=radius*radius
        region=mask[y0:y1,x0:x1];region[disk]=self.mode=="brush"
        if self.mode=="brush":
            old=self._bbox_cache.get(id(mask)); self._bbox_cache[id(mask)]=(x0,y0,x1,y1) if old is None else (min(old[0],x0),min(old[1],y0),max(old[2],x1),max(old[3],y1))
        if not self._live_redraw_pending:
            self._live_redraw_pending=True;self._live_redraw_timer.start()

    def _flush_live_redraw(self):
        self._live_redraw_pending=False;self.redraw()

    def finish_polygon(self):
        if len(self.polygon) < 3: self.polygon=[]; self.redraw(); return
        self.changed.emit(); h,w=self.array.shape[:2]; mask=np.zeros((h,w),np.uint8)
        cv2.fillPoly(mask,[np.asarray(self.polygon,np.int32)],1)
        item=dict(self.new_mask_metadata); item.update({"id":0,"mask":mask.astype(bool)})
        self.masks.append(item); self.polygon=[]; self.selection_requested.emit(len(self.masks)-1); self.masks_changed.emit(); self.redraw()

    def clear_mask_selection(self):
        self.selected=-1; self.selection_requested.emit(-1); self.redraw()

    def detach_selected_mask(self):
        if not (0<=self.selected<len(self.masks)) or self.array is None:return
        old_mask=self.masks[self.selected]["mask"]; new_mask=old_mask.copy(); self.masks[self.selected]["mask"]=new_mask
        if id(old_mask) in self._bbox_cache:self._bbox_cache[id(new_mask)]=self._bbox_cache[id(old_mask)]
        h,w=self.array.shape[:2]; self._overlay_signature=(h,w,*(id(entry["mask"]) for entry in self.masks)); self._hit_signature=None

    def finish_current_instance(self):
        if self.mode not in {"brush","erase"} or not (0<=self.selected<len(self.masks)):return
        finished=self.selected; finished_item=self.masks[finished]; area=int(finished_item["mask"].sum())
        self.masks_changed.emit(); self.clear_mask_selection(); self.status.emit("当前实例已完成；继续绘制会创建新实例")
        if area<10:self.small_mask_finished.emit(finished,area)
        remaining=next((i for i,item in enumerate(self.masks) if item is finished_item),-1)
        if self.mode=="brush" and remaining>=0:self.instance_finished.emit(remaining)

    def keyPressEvent(self,event):
        if event.key()==Qt.Key_Escape:
            self.polygon=[]; self.drawing=False; self._live_redraw_timer.stop(); self._live_redraw_pending=False; self._editing_background=None; self._overlay_signature=None; self._background_key=None; self._hit_signature=None; self.redraw(); self.tool_reset_requested.emit(); return
        if event.key() in (Qt.Key_Return,Qt.Key_Enter) and self.mode=="polygon":
            self.finish_polygon(); finished=self.selected
            if 0<=finished<len(self.masks) and int(self.masks[finished]["mask"].sum())<10:self.small_mask_finished.emit(finished,int(self.masks[finished]["mask"].sum()))
            self.clear_mask_selection()
        elif event.key() in (Qt.Key_Return,Qt.Key_Enter) and self.mode in {"brush","erase"}:self.finish_current_instance()
        else: super().keyPressEvent(event)


class Sam3Engine(QObject):
    loaded = Signal(str)
    predicted = Signal(object, object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self):
        super().__init__(); self.processor=None; self.device=None

    @Slot(str,str,float)
    def load_model(self, repo: str, checkpoint: str, confidence: float):
        try:
            import time
            started=time.perf_counter(); self.progress.emit("正在导入 PyTorch 和 SAM3（首次约 5–10 秒）……")
            repo_path=Path(repo).resolve(); ckpt=Path(checkpoint).resolve()
            if not (repo_path/"sam3").is_dir(): raise FileNotFoundError(f"SAM3 Python package not found: {repo_path / 'sam3'}")
            if not ckpt.is_file(): raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
            if str(repo_path) not in sys.path: sys.path.insert(0,str(repo_path))
            import torch
            if not torch.cuda.is_available(): raise RuntimeError("CUDA is not available in this SAM3 environment")
            from sam3 import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            self.device="cuda"
            self.progress.emit("正在读取 checkpoint 并加载到 GPU……")
            model=build_sam3_image_model(checkpoint_path=str(ckpt),load_from_HF=False,device="cuda",eval_mode=True)
            self.processor=Sam3Processor(model=model,device="cuda",confidence_threshold=confidence)
            name=torch.cuda.get_device_name(0); mem=torch.cuda.get_device_properties(0).total_memory/(1024**3)
            self.loaded.emit(f"SAM3 已加载：{name} ({mem:.1f} GB)，耗时 {time.perf_counter()-started:.1f} 秒")
        except Exception as exc:self.failed.emit(f"SAM3 加载失败：{type(exc).__name__}: {exc}")

    @Slot(object,str,float,object,str,object)
    def predict(self, rgb, prompt: str, confidence: float, box, mode: str, exemplar):
        try:
            import time
            started=time.perf_counter()
            if self.processor is None: raise RuntimeError("请先加载 SAM3 模型")
            import torch
            self.processor.set_confidence_threshold(confidence)
            full_h,full_w=rgb.shape[:2]
            if box:
                x0,y0,x1,y1=[int(v) for v in box]; x0=max(0,min(x0,full_w-1)); x1=max(x0+1,min(x1+1,full_w)); y0=max(0,min(y0,full_h-1)); y1=max(y0+1,min(y1+1,full_h))
            else:x0,y0,x1,y1=0,0,full_w,full_h
            cropped=rgb[y0:y1,x0:x1]
            self.progress.emit(f"正在提取图像特征（范围 {x1-x0}×{y1-y0}）……")
            with torch.inference_mode(), torch.autocast(device_type="cuda",dtype=torch.bfloat16):
                state=self.processor.set_image(Image.fromarray(cropped))
                if mode=="text":
                    self.progress.emit("正在执行 Text Prompt grounding……"); state=self.processor.set_text_prompt(prompt=prompt,state=state)
                else:
                    if exemplar is None:raise RuntimeError("请先指定或绘制一个 Mask Prompt")
                    exemplar_crop=np.asarray(exemplar,dtype=bool)[y0:y1,x0:x1]
                    if not exemplar_crop.any():raise RuntimeError("Mask Prompt 不在当前 SAM3 框范围内")
                    self.progress.emit("正在执行 Mask Prompt grounding……")
                    state["backbone_out"].update(self.processor.model.backbone.forward_text(["visual"],device=self.device))
                    state["geometric_prompt"]=self.processor.model._get_dummy_prompt()
                    prompt_mask=torch.as_tensor(exemplar_crop,device=self.device,dtype=torch.float32).view(1,1,1,*exemplar_crop.shape)
                    state["geometric_prompt"].append_masks(prompt_mask); state=self.processor._forward_grounding(state)
            cropped_masks=state["masks"].squeeze(1).detach().cpu().numpy().astype(bool)
            masks=np.zeros((len(cropped_masks),full_h,full_w),dtype=bool); masks[:,y0:y1,x0:x1]=cropped_masks
            scores=state["scores"].detach().float().cpu().numpy().tolist()
            self.progress.emit(f"推理完成，正在传回 {len(masks)} 个 Masks……")
            self.predicted.emit(masks,scores)
        except Exception as exc:self.failed.emit(f"SAM3 推理失败：{type(exc).__name__}: {exc}")


class ProjectOpenWorker(QObject):
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, json_path):
        super().__init__(); self.json_path=Path(json_path)

    @Slot()
    def run(self):
        try:
            project=CocoProject(); project.open_json(self.json_path); self.completed.emit(project)
        except Exception as exc:self.failed.emit(f"{type(exc).__name__}: {exc}")


class ExportWorker(QObject):
    progress = Signal(int, int, str)
    completed = Signal(str)
    cancelled = Signal(str)
    failed = Signal(str)

    def __init__(self, data, reviewed, image_paths, output):
        super().__init__(); self.data=copy.deepcopy(data); self.reviewed=set(reviewed); self.image_paths=[Path(path) for path in image_paths]; self.output=Path(output); self.cancel_requested=False

    def request_cancel(self):
        self.cancel_requested=True

    def _check_cancel(self):
        if self.cancel_requested:raise InterruptedError("Export cancelled by user")

    @Slot()
    def run(self):
        marker=self.output/"EXPORT_INCOMPLETE.txt"
        try:
            count=len(self.image_paths); total=max(2,2*count+2); done=0
            training=self.output/"training_dataset"; training_images=training/"images"; auxiliary=self.output/"auxiliary_outputs"
            labels=auxiliary/"instance_labels"; previews=auxiliary/"previews"
            training_images.mkdir(parents=True,exist_ok=True); labels.mkdir(parents=True,exist_ok=True); previews.mkdir(parents=True,exist_ok=True)
            marker.write_text("Export is still running or did not complete. Do not use training_dataset yet.\n",encoding="utf-8")
            self.progress.emit(done,total,"Preparing COCO annotations…")
            temp_json=training/".annotations.tmp"
            temp_json.write_text(json.dumps(self.data,ensure_ascii=False,indent=2),encoding="utf-8")
            os.replace(temp_json,training/"annotations.json"); done+=1; self.progress.emit(done,total,"COCO annotations saved")
            (auxiliary/"project_state.json").write_text(json.dumps({"reviewed_image_ids":sorted(self.reviewed)},ensure_ascii=False,indent=2),encoding="utf-8")
            for index,source in enumerate(self.image_paths,1):
                self._check_cancel(); target=training_images/source.name
                if source.resolve()!=target.resolve():shutil.copy2(source,target)
                done+=1; self.progress.emit(done,total,f"Copying training image {index}/{count}: {source.name}")
            project=CocoProject(); project.data=copy.deepcopy(self.data)
            rows=[]
            for index,(rec,path) in enumerate(zip(self.data["images"],self.image_paths),1):
                self._check_cancel(); rgb=np.asarray(Image.open(path).convert("RGB")); masks=project.masks_for(rec["id"],rgb.shape[1],rgb.shape[0]); label=np.zeros(rgb.shape[:2],np.uint16); overlay=rgb.copy()
                for number,mask in enumerate(masks,1):label[mask["mask"]]=number; color=np.array(COLORS[(number-1)%len(COLORS)]); overlay[mask["mask"]]=(0.55*overlay[mask["mask"]]+0.45*color).astype(np.uint8)
                Image.fromarray(label).save(labels/f"{path.stem}_instances.png"); Image.fromarray(overlay).save(previews/f"{path.stem}_preview.png"); rows.append([path.name,len(masks),rec["id"] in self.reviewed])
                done+=1; self.progress.emit(done,total,f"Rendering labels and preview {index}/{count}: {path.name}")
            self._check_cancel()
            with (auxiliary/"dataset_statistics.csv").open("w",newline="",encoding="utf-8-sig") as handle:writer=csv.writer(handle);writer.writerow(["image","instances","reviewed"]);writer.writerows(rows)
            done=total; self.progress.emit(done,total,"Export completed")
            marker.unlink(missing_ok=True); self.completed.emit(str(training))
        except InterruptedError:
            self.cancelled.emit(str(self.output))
        except Exception as exc:
            try:marker.write_text(f"Export failed: {type(exc).__name__}: {exc}\n",encoding="utf-8")
            except Exception:pass
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class MainWindow(BatchSam3Mixin, QMainWindow):
    load_sam3_requested = Signal(str,str,float)
    predict_sam3_requested = Signal(object,str,float,object,str,object)
    def __init__(self):
        super().__init__(); self.ui_language=getattr(self,"ui_language","zh"); self.setWindowTitle("Particle Label GUI — 2026-08-25"); self.resize(1550,930); self.setStyleSheet(THEME_QSS)
        icon_path=Path(__file__).with_name("BL62_Label.ico")
        if icon_path.is_file():self.setWindowIcon(QIcon(str(icon_path)))
        self.setAcceptDrops(True)
        self.project=CocoProject(); self.index=-1; self.masks=[]; self.current_masks_dirty=False; self.undo_stack=[]; self.redo_stack=[]; self.sam_busy=False; self.sam_loaded_flag=False
        self._sam_replace_requested=False; self._sam_replace_box=None
        self.export_thread=None; self.export_worker=None; self.export_dialog=None
        self.open_thread=None; self.open_worker=None; self.open_dialog=None
        self.prompt_mode="text"; self.mask_prompt=None; self.mask_prompt_source=None; self.capture_mask_prompt=False
        self.settings_path=Path(__file__).with_name("sam3_settings.json")
        self._build(); self.statusBar().showMessage("新建文件夹/图片项目，或打开 COCO annotations.json；可拖拽图片加入当前项目")

    def _build(self):
        toolbar=QToolBar("Tools"); self.addToolBar(toolbar)
        self.tool_actions={}; self.active_tool=None
        for text,mode,key in [("画笔","brush","B"),("多边形","polygon","P"),("橡皮擦","erase","E"),("选择","select","S"),("平移","pan","H"),("SAM3框","sam_box","G")]:
            act=QAction(text,self); act.setShortcut(key); act.setCheckable(True); act.triggered.connect(lambda _,m=mode:self.toggle_tool(m)); toolbar.addAction(act); self.tool_actions[mode]=act
        toolbar.addSeparator(); self.show_labels_action=QAction("隐藏标签",self); self.show_labels_action.setShortcut("L"); self.show_labels_action.setCheckable(True); self.show_labels_action.setChecked(True); self.show_labels_action.triggered.connect(self.toggle_label_visibility); toolbar.addAction(self.show_labels_action)
        toolbar.addSeparator(); toolbar.addWidget(QLabel("笔刷 ")); size=QSpinBox(); size.setRange(1,300); size.setValue(20); size.valueChanged.connect(lambda value:self.view.set_brush_size(value)); toolbar.addWidget(size)
        root=QWidget(); self.setCentralWidget(root); layout=QVBoxLayout(root); top=QHBoxLayout(); layout.addLayout(top)
        for text,slot in [("新建项目",self.new_project),("导入已有项目",self.open_project),("保存 Ctrl+S",self.save),("上一张",lambda:self.navigate(-1)),("下一张",lambda:self.navigate(1)),("标为已审核",self.toggle_review),("导出",self.export_all)]:
            b=QPushButton(text); b.clicked.connect(slot); top.addWidget(b)
            if text=="导出":self.export_btn=b
        top.addStretch(); self.counter=QLabel("未打开项目"); top.addWidget(self.counter)
        split=QSplitter(); layout.addWidget(split,1)
        left=QWidget(); ll=QVBoxLayout(left); ll.addWidget(QLabel("图片（✓ 已审核，● 有标注）")); self.images=QListWidget(); self.images.setSelectionMode(QAbstractItemView.ExtendedSelection); self.images.setContextMenuPolicy(Qt.CustomContextMenu); self.images.customContextMenuRequested.connect(self.show_image_context_menu); self.images.currentRowChanged.connect(self.load_image); ll.addWidget(self.images); split.addWidget(left)
        self.view=AnnotationView(); self.view.changed.connect(self.push_undo); self.view.changed.connect(self.mark_current_dirty); self.view.selection_requested.connect(self.select_mask); self.view.masks_changed.connect(self.refresh_masks); self.view.masks_changed.connect(self.mark_current_dirty); self.view.instance_finished.connect(self.on_instance_finished); self.view.small_mask_finished.connect(self.confirm_small_mask); self.view.tool_reset_requested.connect(self.reset_tool); self.view.box_changed.connect(self.update_roi_status); self.view.status.connect(self.statusBar().showMessage); split.addWidget(self.view)
        right=QWidget(); rl=QVBoxLayout(right)
        rl.addWidget(QLabel("SAM3 GPU 辅助标注")); self.repo_label=QLabel(); self.repo_label.setWordWrap(True); rl.addWidget(self.repo_label)
        choose_repo=QPushButton("选择 SAM3 仓库"); choose_repo.clicked.connect(self.choose_sam_repo); rl.addWidget(choose_repo)
        self.ckpt_label=QLabel(); self.ckpt_label.setWordWrap(True); rl.addWidget(self.ckpt_label)
        choose_ckpt=QPushButton("选择 checkpoint"); choose_ckpt.clicked.connect(self.choose_checkpoint); rl.addWidget(choose_ckpt)
        self.load_sam_btn=QPushButton("加载 SAM3 到 GPU"); self.load_sam_btn.clicked.connect(self.load_sam3); rl.addWidget(self.load_sam_btn)
        self.sam_status=QLabel("SAM3 未加载"); self.sam_status.setWordWrap(True); rl.addWidget(self.sam_status)
        rl.addWidget(QLabel("标签"))
        label_line=QHBoxLayout(); self.category_combo=QComboBox(); self.category_combo.setEditable(False); self.category_combo.currentIndexChanged.connect(self.sync_drawing_defaults); self.prompt=self.category_combo; label_line.addWidget(self.category_combo,1)
        manage_labels=QPushButton("管理标签"); manage_labels.clicked.connect(self.manage_labels); label_line.addWidget(manage_labels); rl.addLayout(label_line)
        rl.addWidget(QLabel("SAM3 提示方式")); prompt_buttons=QHBoxLayout(); self.text_prompt_btn=QPushButton("Text Prompt"); self.mask_prompt_btn=QPushButton("Mask Prompt")
        self.prompt_group=QButtonGroup(self); self.prompt_group.setExclusive(True)
        for button,mode in [(self.text_prompt_btn,"text"),(self.mask_prompt_btn,"mask")]:
            button.setCheckable(True); self.prompt_group.addButton(button); button.clicked.connect(lambda _,m=mode:self.set_prompt_mode(m)); prompt_buttons.addWidget(button)
        self.text_prompt_btn.setChecked(True); rl.addLayout(prompt_buttons)
        confline=QHBoxLayout(); confline.addWidget(QLabel("置信度")); self.confidence=QSpinBox(); self.confidence.setRange(1,99); self.confidence.setValue(50); self.confidence.setSuffix(" %"); confline.addWidget(self.confidence); rl.addLayout(confline)
        self.run_sam_btn=QPushButton("运行 SAM3（文本 + 可选框）"); self.run_sam_btn.clicked.connect(self.run_sam3); self.run_sam_btn.setEnabled(False)
        self.roi_status=QLabel("搜索范围：全图"); rl.addWidget(self.roi_status); clear_box=QPushButton("清除 SAM3 框"); clear_box.clicked.connect(self.clear_sam_box); rl.addWidget(clear_box)
        self.mask_prompt_controls=QWidget(); mp=QVBoxLayout(self.mask_prompt_controls); mp.setContentsMargins(0,0,0,0)
        self.mask_prompt_status=QLabel("Mask Prompt：尚未设置"); self.mask_prompt_status.setWordWrap(True); mp.addWidget(self.mask_prompt_status)
        use_prompt=QPushButton("使用当前选中实例"); use_prompt.clicked.connect(self.use_selected_as_mask_prompt); mp.addWidget(use_prompt)
        draw_prompt=QPushButton("绘制 Mask Prompt"); draw_prompt.clicked.connect(self.draw_mask_prompt); mp.addWidget(draw_prompt)
        clear_prompt=QPushButton("清除 Mask Prompt"); clear_prompt.clicked.connect(self.clear_mask_prompt); mp.addWidget(clear_prompt); self.mask_prompt_controls.setVisible(False); rl.addWidget(self.mask_prompt_controls)
        rl.addWidget(self.run_sam_btn)
        self.batch_sam_btn=QPushButton("对选中图片运行 SAM3"); self.batch_sam_btn.clicked.connect(self.run_sam3_selected); rl.addWidget(self.batch_sam_btn)
        rl.addWidget(QLabel("标注实例（Ctrl 可多选）")); self.mask_list=QListWidget(); self.mask_list.setSelectionMode(QAbstractItemView.ExtendedSelection); self.mask_list.setContextMenuPolicy(Qt.CustomContextMenu); self.mask_list.customContextMenuRequested.connect(self.show_mask_context_menu); self.mask_list.currentRowChanged.connect(self.select_mask); self.mask_list.itemClicked.connect(lambda item:self.select_mask(self.mask_list.row(item))); rl.addWidget(self.mask_list)
        apply_cat=QPushButton("将当前标签分配给选中实例"); apply_cat.clicked.connect(self.apply_labels_to_selected); rl.addWidget(apply_cat)
        delete=QPushButton("删除选中实例 Delete"); delete.clicked.connect(self.delete_mask); rl.addWidget(delete)
        merge=QPushButton("合并所选 Masks 为一个实例"); merge.clicked.connect(self.merge_masks); rl.addWidget(merge)
        split.addWidget(right); split.setSizes([250,930,390]); self._setup_sam3(); self.set_prompt_mode("text")
        for seq,slot in [(QKeySequence.Save,self.save),(QKeySequence.Undo,self.undo),(QKeySequence.Redo,self.redo),(QKeySequence.Delete,self.delete_mask)]:
            a=QAction(self); a.setShortcut(seq); a.triggered.connect(slot); self.addAction(a)
        escape=QAction(self); escape.setShortcut(QKeySequence("Esc")); escape.triggered.connect(self.reset_tool); self.addAction(escape)

    def toggle_tool(self,mode):
        if self.active_tool==mode:self.reset_tool();return
        self.activate_tool(mode)

    def activate_tool(self,mode):
        self.active_tool=mode
        for name,action in self.tool_actions.items():action.setChecked(name==mode)
        self.view.set_mode(mode)

    def reset_tool(self):
        self.active_tool=None
        for action in self.tool_actions.values():action.setChecked(False)
        self.view.polygon=[]; self.view.drawing=False; self.view.clear_mask_selection(); self.view.set_mode("idle")

    def toggle_label_visibility(self,visible):
        self.view.set_labels_visible(visible)
        english=getattr(self,"ui_language","zh")=="en"
        self.show_labels_action.setText(("Hide Labels" if visible else "Show Labels") if english else ("隐藏标签" if visible else "显示标签"))
        message=("Labels shown" if visible else "Labels hidden") if english else ("标签已显示" if visible else "标签已隐藏")
        self.statusBar().showMessage(message,3000)

    def confirm_small_mask(self,index,area):
        if index<0 or index>=len(self.masks):return
        english=getattr(self,"ui_language","zh")=="en"
        title="Small Annotation" if english else "小标注"
        message=(f"This annotation contains only {area} pixel(s), fewer than 10. Delete it?" if english else f"这个标注只有 {area} 个像素，小于10。是否删除？")
        if QMessageBox.question(self,title,message,QMessageBox.Yes|QMessageBox.No,QMessageBox.Yes)==QMessageBox.Yes:
            self.push_undo(); self.masks.pop(index); self.view.masks=self.masks; self.view.selected=-1; self.view.redraw(); self.refresh_masks()
            self.statusBar().showMessage(("Small annotation deleted." if english else "小标注已删除。"),5000)

    def _setup_sam3(self):
        defaults={"repo":r"D:\SAM3p1\sam3","checkpoint":r"D:\SAM3p1\checkpoint_sam3_format.pt"}
        try: defaults.update(json.loads(self.settings_path.read_text(encoding="utf-8")))
        except Exception: pass
        self.sam_repo=defaults["repo"]; self.sam_checkpoint=defaults["checkpoint"]; self._update_sam_labels()
        self.sam_thread=QThread(self); self.sam_engine=Sam3Engine(); self.sam_engine.moveToThread(self.sam_thread)
        self.load_sam3_requested.connect(self.sam_engine.load_model); self.predict_sam3_requested.connect(self.sam_engine.predict)
        self.sam_engine.loaded.connect(self.sam_loaded); self.sam_engine.predicted.connect(self.sam_predicted); self.sam_engine.failed.connect(self.sam_failed); self.sam_engine.progress.connect(self.sam_progress); self.sam_thread.start()

    def _update_sam_labels(self):
        self.repo_label.setText(f"仓库：{self.sam_repo}"); self.ckpt_label.setText(f"Checkpoint：{self.sam_checkpoint}")
        self.settings_path.write_text(json.dumps({"repo":self.sam_repo,"checkpoint":self.sam_checkpoint},ensure_ascii=False,indent=2),encoding="utf-8")
    def choose_sam_repo(self):
        name=QFileDialog.getExistingDirectory(self,"选择包含 sam3 Python 包的仓库",self.sam_repo)
        if name:self.sam_repo=name; self._update_sam_labels()
    def choose_checkpoint(self):
        name,_=QFileDialog.getOpenFileName(self,"选择 SAM3 checkpoint",str(Path(self.sam_checkpoint).parent),"PyTorch checkpoint (*.pt *.pth *.ckpt);;All files (*)")
        if name:self.sam_checkpoint=name; self._update_sam_labels()
    def load_sam3(self):
        if self.sam_busy:return
        self.sam_busy=True; self.load_sam_btn.setEnabled(False); self.sam_status.setText("正在后台加载 SAM3，请稍候……")
        self.load_sam3_requested.emit(self.sam_repo,self.sam_checkpoint,self.confidence.value()/100)
    def sam_progress(self,message):
        self.sam_status.setText(message); self.statusBar().showMessage(message)
    def sam_loaded(self,message):
        self.sam_busy=False; self.sam_loaded_flag=True; self.load_sam_btn.setEnabled(True); self.update_run_button(); self.sam_status.setText(message)
    def sam_failed(self,message):
        self._sam_replace_requested=False; self._sam_replace_box=None
        self.sam_busy=False; self.load_sam_btn.setEnabled(True); self.sam_loaded_flag=self.sam_engine.processor is not None; self.update_run_button(); self.sam_status.setText(message); QMessageBox.critical(self,"SAM3",message)
    def _confirm_sam_replacement(self):
        self._sam_replace_requested=False; self._sam_replace_box=None
        if not self.masks:return True
        english=getattr(self,"ui_language","zh")=="en"; box=self.view.sam_box
        dialog=QMessageBox(self); dialog.setIcon(QMessageBox.Warning); dialog.setWindowTitle("Replace Existing Labels" if english else "替换现有标签")
        if box:
            message=("This image already contains labels. Clear labels intersecting the current SAM3 Box and run SAM3 again? Labels outside the box will be kept."
                     if english else "当前图片已有标签。是否清除与当前 SAM3 Box 相交的标签并重新运行？Box 外的标签会保留。")
        else:
            message=("This image already contains labels. Clear all existing labels and run SAM3 again?"
                     if english else "当前图片已有标签。是否清除全部现有标签并重新运行 SAM3？")
        dialog.setText(message); run_button=dialog.addButton("Clear Labels and Run" if english else "清除标签并运行",QMessageBox.AcceptRole); dialog.addButton(QMessageBox.Cancel); dialog.setDefaultButton(QMessageBox.Cancel); dialog.exec()
        if dialog.clickedButton() is not run_button:return False
        self._sam_replace_requested=True; self._sam_replace_box=tuple(box) if box else None
        return True
    def _apply_pending_sam_replacement(self):
        if not self._sam_replace_requested:return
        box=self._sam_replace_box
        if box:
            x0,y0,x1,y1=[int(v) for v in box]
            self.masks[:]=[item for item in self.masks if not np.asarray(item["mask"],dtype=bool)[max(0,y0):y1+1,max(0,x0):x1+1].any()]
        else:self.masks.clear()
        self._sam_replace_requested=False; self._sam_replace_box=None
    def run_sam3(self):
        if self.view.array is None:return
        prompt=self.prompt.currentText().strip()
        if self.prompt_mode=="text" and not prompt:QMessageBox.warning(self,"SAM3","Text Prompt 不能为空");return
        if self.prompt_mode=="mask" and self.mask_prompt is None:QMessageBox.warning(self,"SAM3","请先选择或绘制一个 Mask Prompt");return
        if not prompt:prompt=(self.project.category(self.current_category_id()) or {"name":"particle"})["name"]
        if not self._confirm_sam_replacement():return
        self.sam_busy=True; self.run_sam_btn.setEnabled(False); self.sam_status.setText("SAM3 正在后台推理……")
        self.predict_sam3_requested.emit(self.view.array.copy(),prompt,self.confidence.value()/100,self.view.sam_box,self.prompt_mode,None if self.mask_prompt is None else self.mask_prompt.copy())
    def sam_predicted(self,masks,scores):
        self.push_undo(); prompt=self.prompt.currentText().strip(); category_id=self.current_category_id()
        self._apply_pending_sam_replacement()
        source="sam3_text" if self.prompt_mode=="text" else "sam3_mask_prompt"
        for mask,score in zip(masks,scores):
            item={"id":0,"category_id":category_id,"noun_phrase":prompt,"score":float(score),"annotation_source":source,"mask":mask}
            if self.prompt_mode=="mask":item["mask_prompt_source"]=self.mask_prompt_source or "drawn_or_selected_mask"
            self.masks.append(item)
        self.view.masks=self.masks; self.view.redraw(); self.refresh_masks(); self.sam_busy=False; self.update_run_button(); self.sam_status.setText(f"推理完成：新增 {len(masks)} 个 Masks（{source}）")
    def update_roi_status(self,box=None):
        box=self.view.sam_box if box is None else box
        if box:
            x0,y0,x1,y1=box; self.roi_status.setText(f"搜索范围：SAM3 框  x={x0}, y={y0}, w={x1-x0}, h={y1-y0}")
        else:self.roi_status.setText("搜索范围：全图")
    def clear_sam_box(self):self.view.sam_box=None; self.view.redraw(); self.update_roi_status(); self.statusBar().showMessage("SAM3 框已清除；搜索范围恢复为全图")

    def set_prompt_mode(self,mode):
        self.prompt_mode=mode; self.text_prompt_btn.setChecked(mode=="text"); self.mask_prompt_btn.setChecked(mode=="mask"); self.mask_prompt_controls.setVisible(mode=="mask"); self.update_run_button()
        self.statusBar().showMessage("Text Prompt 模式" if mode=="text" else "Mask Prompt 模式：请使用选中实例或绘制提示 Mask")
    def update_run_button(self):
        self.update_batch_button()
        ready=self.sam_loaded_flag and not self.sam_busy and (self.prompt_mode=="text" or self.mask_prompt is not None); self.run_sam_btn.setEnabled(ready)
        self.run_sam_btn.setText("运行 SAM3 — Text Prompt" if self.prompt_mode=="text" else "运行 SAM3 — Mask Prompt")
    def use_selected_as_mask_prompt(self):
        index=self.view.selected
        if index<0 or index>=len(self.masks):QMessageBox.warning(self,"Mask Prompt","请先在图像或实例列表中选择一个实例");return
        self.mask_prompt=self.masks[index]["mask"].copy(); self.mask_prompt_source=f"实例 {index+1} / Ann ID {self.masks[index].get('id',0)}"; self.view.exemplar_mask=self.mask_prompt; self.view.redraw(); self.mask_prompt_status.setText(f"Mask Prompt：{self.mask_prompt_source}，面积 {int(self.mask_prompt.sum())} px"); self.update_run_button()
    def draw_mask_prompt(self):
        if self.view.array is None:QMessageBox.warning(self,"Mask Prompt","请先导入图片");return
        self.capture_mask_prompt=True; self.view.selected=-1; self.activate_tool("brush"); self.statusBar().showMessage("请用画笔绘制 Mask Prompt，按 Enter 完成")
    def on_instance_finished(self,index):
        if not self.capture_mask_prompt or index<0 or index>=len(self.masks):return
        self.capture_mask_prompt=False; self.mask_prompt=self.masks[index]["mask"].copy(); self.mask_prompt_source=f"新绘制 Mask / {int(self.mask_prompt.sum())} px"; self.view.exemplar_mask=self.mask_prompt
        answer=QMessageBox.question(self,"Mask Prompt","是否把刚绘制的 Mask Prompt 同时保存为正式实例？\n选择 No 将仅作为提示，不写入标注。",QMessageBox.Yes|QMessageBox.No)
        if answer==QMessageBox.No:self.push_undo(); self.masks.pop(index); self.view.masks=self.masks; self.refresh_masks()
        self.mask_prompt_status.setText(f"Mask Prompt：{self.mask_prompt_source}"); self.view.redraw(); self.update_run_button()
    def clear_mask_prompt(self):
        self.mask_prompt=None; self.mask_prompt_source=None; self.capture_mask_prompt=False; self.view.exemplar_mask=None; self.view.redraw(); self.mask_prompt_status.setText("Mask Prompt：尚未设置"); self.update_run_button()

    def current_category_id(self):
        value=self.category_combo.currentData() if hasattr(self,"category_combo") else 1
        return int(value) if value is not None else 1

    def sync_drawing_defaults(self):
        if not hasattr(self,"category_combo"):return
        category=self.project.category(self.current_category_id())
        label=(category or {}).get("name","particle")
        self.view.new_mask_metadata={"category_id":self.current_category_id(),"noun_phrase":label,"annotation_source":"manual"}

    def refresh_categories(self,preferred_id=None):
        if not hasattr(self,"category_combo"):return
        preferred_id=self.current_category_id() if preferred_id is None else preferred_id
        self.category_combo.blockSignals(True); self.category_combo.clear(); selected=0
        for idx,item in enumerate(self.project.categories()):
            category_id=int(item["id"]); self.category_combo.addItem(item.get("name",""),category_id)
            if category_id==preferred_id:selected=idx
        self.category_combo.setCurrentIndex(selected); self.category_combo.blockSignals(False); self.sync_drawing_defaults()

    def new_category(self):
        name,ok=QInputDialog.getText(self,"新建 Category","Category 名称：")
        if not ok:return
        supercategory,ok=QInputDialog.getText(self,"新建 Category","Supercategory：",text=name)
        if not ok:return
        try:
            item=self.project.add_category(name,supercategory); self.refresh_categories(item["id"]); self.statusBar().showMessage(f"已创建 Category {item['id']} | {item['name']}")
        except Exception as exc:self.error(exc)

    def manage_categories(self):
        self.guard_save(); dialog=QDialog(self); dialog.setWindowTitle("管理 Categories"); dialog.resize(650,420)
        layout=QVBoxLayout(dialog); table=QTableWidget(0,4); table.setHorizontalHeaderLabels(["ID","Category","Supercategory","实例数量"]); table.setSelectionBehavior(QAbstractItemView.SelectRows); table.setEditTriggers(QAbstractItemView.NoEditTriggers); layout.addWidget(table)
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
            item=self.project.category(category_id); name,ok=QInputDialog.getText(dialog,"重命名 Category","Category 名称：",text=item["name"])
            if not ok:return
            sup,ok=QInputDialog.getText(dialog,"重命名 Category","Supercategory：",text=item.get("supercategory",name))
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
            target_text,ok=QInputDialog.getItem(dialog,"合并 Category",f"把 Category {source} 的全部实例重新分配到：",choices,0,False)
            if not ok:return
            target=int(target_text.split("|",1)[0].strip())
            if QMessageBox.question(dialog,"确认合并",f"Category {source} 将被删除，其实例改为 Category {target}。是否继续？")!=QMessageBox.Yes:return
            for mask in self.masks:
                if int(mask.get("category_id",0))==source:mask["category_id"]=target
            self.project.merge_category(source,target); refill(target)
        buttons=QHBoxLayout()
        for text,slot in [("新建",add),("重命名",rename),("删除未使用项",remove),("合并到其他 Category",merge)]:
            button=QPushButton(text); button.clicked.connect(slot); buttons.addWidget(button)
        close=QPushButton("关闭"); close.clicked.connect(dialog.accept); buttons.addWidget(close); layout.addLayout(buttons); refill(); dialog.exec(); self.refresh_categories(); self.refresh_masks()

    def manage_labels(self):
        english=getattr(self,"ui_language","zh")=="en"; self.guard_save(); dialog=QDialog(self); dialog.setWindowTitle("Manage Labels" if english else "管理标签"); dialog.resize(520,380)
        layout=QVBoxLayout(dialog); table=QTableWidget(0,2); table.setHorizontalHeaderLabels((["Label","Instances"] if english else ["标签","实例数量"])); table.setSelectionBehavior(QAbstractItemView.SelectRows); table.setEditTriggers(QAbstractItemView.NoEditTriggers); layout.addWidget(table)
        def refill(select_id=None):
            table.setRowCount(0)
            for row,item in enumerate(self.project.categories()):
                category_id=int(item["id"]); table.insertRow(row); name_item=QTableWidgetItem(str(item.get("name",""))); name_item.setData(Qt.UserRole,category_id); table.setItem(row,0,name_item); table.setItem(row,1,QTableWidgetItem(str(self.project.category_usage(category_id))))
                if category_id==select_id:table.selectRow(row)
        def selected_id():
            row=table.currentRow(); item=table.item(row,0) if row>=0 else None; return int(item.data(Qt.UserRole)) if item else None
        def add():
            name,ok=QInputDialog.getText(dialog,"New Label" if english else "新建标签","Label name:" if english else "标签名称：")
            if ok:
                try:item=self.project.add_category(name,name); self.refresh_categories(item["id"]); refill(item["id"])
                except Exception as exc:self.error(exc)
        def rename():
            category_id=selected_id()
            if category_id is None:return
            old=self.project.category(category_id); name,ok=QInputDialog.getText(dialog,"Rename Label" if english else "重命名标签","Label name:" if english else "标签名称：",text=old["name"])
            if ok:
                try:
                    self.project.rename_category(category_id,name,name)
                    for mask in self.masks:
                        if int(mask.get("category_id",0))==category_id:mask["noun_phrase"]=name
                    refill(category_id); self.refresh_categories(category_id); self.refresh_masks()
                except Exception as exc:self.error(exc)
        def remove():
            category_id=selected_id()
            if category_id is None:return
            try:self.project.delete_category(category_id); refill(); self.refresh_categories()
            except Exception as exc:self.error(exc)
        buttons=QHBoxLayout()
        for text,slot in [(("New" if english else "新建"),add),(("Rename" if english else "重命名"),rename),(("Delete Unused" if english else "删除未使用项"),remove)]:
            button=QPushButton(text); button.clicked.connect(slot); buttons.addWidget(button)
        close=QPushButton("Close" if english else "关闭"); close.clicked.connect(dialog.accept); buttons.addWidget(close); layout.addLayout(buttons); refill(); dialog.exec(); self.refresh_categories(); self.refresh_masks()

    def apply_labels_to_selected(self):
        rows=sorted({item.row() for item in self.mask_list.selectedIndexes()})
        if not rows and 0<=self.view.selected<len(self.masks):rows=[self.view.selected]
        if not rows:self.statusBar().showMessage("请先选择一个或多个实例");return
        self.push_undo(); category_id=self.current_category_id(); category=self.project.category(category_id); label=(category or {}).get("name","particle")
        for row in rows:
            self.masks[row]["category_id"]=category_id; self.masks[row]["noun_phrase"]=label
        self.view.redraw(); self.refresh_masks(); self.statusBar().showMessage(f"已更新 {len(rows)} 个实例的训练标签")

    def guard_save(self):
        if self.index>=0: self.commit_current()
    def mark_current_dirty(self):
        if self.index>=0:
            self.current_masks_dirty=True
            if 0<=self.view.selected<len(self.masks):self.masks[self.view.selected]["_geometry_dirty"]=True
    def new_project(self):
        self._choose_new_project(False)
    def _choose_new_project(self,english=False):
        box=QMessageBox(self); box.setWindowTitle("New Project" if english else "新建项目")
        box.setText("Choose how to add source images." if english else "请选择导入原始图片的方式。")
        folder_button=box.addButton("Select Folder" if english else "选择文件夹",QMessageBox.AcceptRole)
        files_button=box.addButton("Select Images" if english else "选择图片",QMessageBox.ActionRole)
        box.addButton(QMessageBox.Cancel); box.exec()
        try:
            if box.clickedButton() is folder_button:
                folder=QFileDialog.getExistingDirectory(self,"New Project: select an image folder" if english else "新建项目：选择包含原始图片的文件夹")
                if not folder:return
                self.project.new_from_folder(Path(folder))
            elif box.clickedButton() is files_button:
                names,_=QFileDialog.getOpenFileNames(self,"New Project: select one or more images" if english else "新建项目：选择一张或多张图片","","Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp)")
                if not names:return
                self.project.new_from_files([Path(name) for name in names])
            else:return
            self.index=-1; self.masks=[]; self.populate(); self.save()
        except Exception as e:self.error(e)
    def dragEnterEvent(self,event):
        urls=event.mimeData().urls() if event.mimeData().hasUrls() else []
        if any(url.isLocalFile() and Path(url.toLocalFile()).suffix.lower() in {".png",".jpg",".jpeg",".tif",".tiff",".bmp",".webp"} for url in urls):event.acceptProposedAction()
        else:event.ignore()
    def dropEvent(self,event):
        english=getattr(self,"ui_language","zh")=="en"
        paths=[Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()]
        try:
            if self.project.root is None:
                self.project.new_from_files(paths)
                self.index=-1; self.masks=[]; self.populate(); self.project.save()
                self.statusBar().showMessage((f"Created a new project from {len(self.project.image_paths)} dropped image(s)." if english else f"已使用拖入的 {len(self.project.image_paths)} 张图片创建新项目。"),7000)
                event.acceptProposedAction();return
            self.guard_save(); added=self.project.add_images(paths,copy_into_project=True)
            if not added:
                self.statusBar().showMessage("No new supported images were added." if english else "没有添加新的受支持图片。",5000);event.acceptProposedAction();return
            self.project.save(); first=added[0]; self.index=-1; self.masks=[]; self.populate(); self.images.setCurrentRow(first)
            self.statusBar().showMessage((f"Added {len(added)} image(s) to the current project." if english else f"已向当前项目添加 {len(added)} 张图片。"),5000)
            event.acceptProposedAction()
        except Exception as e:self.error(e);event.ignore()
    def open_project(self):
        name,_=QFileDialog.getOpenFileName(self,"导入已有项目：选择 annotations.json","","COCO JSON (*.json)")
        if not name:return
        self._open_project_in_background(Path(name))
    def _open_project_in_background(self,path):
        if self.open_thread and self.open_thread.isRunning():return
        english=getattr(self,"ui_language","zh")=="en"
        self.open_dialog=QProgressDialog("Reading and indexing the COCO project…" if english else "正在读取并索引 COCO 项目……","",0,0,self)
        self.open_dialog.setWindowTitle("Import Existing Project" if english else "导入已有项目"); self.open_dialog.setCancelButton(None); self.open_dialog.setWindowModality(Qt.WindowModal); self.open_dialog.setMinimumDuration(0); self.open_dialog.show()
        self.open_thread=QThread(self); self.open_worker=ProjectOpenWorker(path); self.open_worker.moveToThread(self.open_thread)
        self.open_thread.started.connect(self.open_worker.run); self.open_worker.completed.connect(self._project_opened); self.open_worker.failed.connect(self._project_open_failed)
        self.open_worker.completed.connect(self.open_worker.deleteLater); self.open_worker.failed.connect(self.open_worker.deleteLater)
        self.open_worker.completed.connect(self.open_thread.quit); self.open_worker.failed.connect(self.open_thread.quit); self.open_thread.finished.connect(self._project_open_cleanup); self.open_thread.start()
    def _project_opened(self,project):
        if self.open_dialog:self.open_dialog.close()
        self.project=project; self.index=-1; self.masks=[]; self.undo_stack=[]; self.redo_stack=[]; self.populate()
        count=len(project.image_paths); annotations=len(project.data.get("annotations",[])); english=getattr(self,"ui_language","zh")=="en"
        self.statusBar().showMessage((f"Project imported: {count} images, {annotations} annotations" if english else f"项目导入完成：{count} 张图片，{annotations} 个标注"),7000)
    def _project_open_failed(self,message):
        if self.open_dialog:self.open_dialog.close()
        self.error(message)
    def _project_open_cleanup(self):
        if self.open_thread:self.open_thread.deleteLater()
        self.open_worker=None; self.open_thread=None; self.open_dialog=None
    def remove_from_project(self):
        english=getattr(self,"ui_language","zh")=="en"
        rows=sorted({item.row() for item in self.images.selectedIndexes()})
        if not rows and self.index>=0:rows=[self.index]
        if not rows:
            QMessageBox.information(self,"Remove from Project" if english else "从项目移除","Select one or more images first." if english else "请先选择一张或多张图片。")
            return
        image_ids={int(self.project.data["images"][row]["id"]) for row in rows}
        annotation_count=sum(int(item.get("image_id",-1)) in image_ids for item in self.project.data.get("annotations",[]))
        message=(f"Remove {len(rows)} selected image(s) and {annotation_count} associated annotation(s) from this project?\n\nThe image files will remain on disk."
                 if english else f"从当前项目中移除所选的 {len(rows)} 张图片及其关联的 {annotation_count} 个标注？\n\n磁盘上的图片文件将被保留。")
        if QMessageBox.question(self,"Remove from Project" if english else "从项目移除",message,QMessageBox.Yes|QMessageBox.No,QMessageBox.No)!=QMessageBox.Yes:return
        try:
            self.guard_save(); target=min(rows); removed,annotations=self.project.remove_images(rows); self.project.save()
            self.index=-1; self.masks=[]; self.undo_stack=[]; self.redo_stack=[]
            if self.project.image_paths:
                self.populate(); self.images.setCurrentRow(min(target,len(self.project.image_paths)-1))
            else:
                self.images.clear(); self.counter.setText("No images in project" if english else "项目中没有图片"); self.view.scene().clear(); self.view.array=None; self.mask_list.clear()
            self.statusBar().showMessage((f"Removed {removed} image(s) and {annotations} annotation(s) from the project; files were kept."
                                          if english else f"已从项目移除 {removed} 张图片和 {annotations} 个标注；图片文件已保留。"),7000)
        except Exception as e:self.error(e)
    def show_image_context_menu(self,pos):
        item=self.images.itemAt(pos)
        if item is None:return
        if not item.isSelected():
            self.images.clearSelection(); item.setSelected(True); self.images.setCurrentItem(item)
        english=getattr(self,"ui_language","zh")=="en"
        menu=QMenu(self.images)
        batch_action=menu.addAction("Run SAM3 for Selected Pictures" if english else "对选中图片运行 SAM3"); batch_action.setEnabled(not self.sam_busy and self.prompt_mode=="text"); batch_action.triggered.connect(self.run_sam3_selected)
        remove_action=menu.addAction("Remove from Project" if english else "从项目移除")
        remove_action.triggered.connect(self.remove_from_project)
        menu.exec(self.images.viewport().mapToGlobal(pos))
    def populate(self):
        self.images.blockSignals(True); self.images.clear()
        ann_ids={a["image_id"] for a in self.project.data["annotations"]}
        for rec,path in zip(self.project.data["images"],self.project.image_paths):
            prefix=("✓ " if rec["id"] in self.project.reviewed else "  ")+("● " if rec["id"] in ann_ids else "○ ")
            self.images.addItem(prefix+path.name)
        self.images.blockSignals(False); self.refresh_categories(); self.images.setCurrentRow(0)
    def commit_current(self,force=False):
        if self.index<0 or (not force and not self.current_masks_dirty):return
        rec=self.project.image_record(self.index); self.project.replace_masks(rec["id"],self.masks); self.current_masks_dirty=False
    def load_image(self,index):
        if index<0 or index>=len(self.project.image_paths):return
        if self.index>=0:self.commit_current()
        self.index=index; path=self.project.image_paths[index]
        try:
            rgb=np.asarray(Image.open(path).convert("RGB")); rec=self.project.image_record(index)
            self.masks=self.project.masks_for(rec["id"],rgb.shape[1],rgb.shape[0]); self.undo_stack=[]; self.redo_stack=[]
            self.mask_prompt=None; self.mask_prompt_source=None; self.capture_mask_prompt=False; self.view.exemplar_mask=None; self.view.sam_box=None
            self.view.set_image(rgb,self.masks); self.current_masks_dirty=False; self.mask_prompt_status.setText("Mask Prompt：尚未设置"); self.update_roi_status(); self.update_run_button(); self.refresh_masks(); self.counter.setText(f"{index+1}/{len(self.project.image_paths)}  {path.name}")
        except Exception as e:self.error(e)
    def navigate(self,delta):
        if self.images.count():self.images.setCurrentRow(max(0,min(self.images.count()-1,self.index+delta)))
    def refresh_masks(self):
        current=self.view.selected; self.mask_list.blockSignals(True); self.mask_list.clear()
        for i,m in enumerate(self.masks):
            area=float(m.get("area",-1)) if not m.get("_geometry_dirty",False) else -1
            if area<0:area=float(np.count_nonzero(m["mask"]));m["area"]=area
            category_id=int(m.get("category_id",1)); category=self.project.category(category_id); name=(category or {}).get("name","未知"); phrase=m.get("noun_phrase",name)
            self.mask_list.addItem(f"实例 {i+1} | 标签：{name} | 面积 {area:.0f}")
        self.mask_list.blockSignals(False)
        if 0<=current<len(self.masks):self.mask_list.setCurrentRow(current)
    def select_mask(self,index):
        if index<0 or index>=len(self.masks):
            self.view.selected=-1; self.mask_list.blockSignals(True); self.mask_list.clearSelection(); self.mask_list.setCurrentRow(-1); self.mask_list.blockSignals(False); self.view.redraw(); return
        self.view.selected=index; self.mask_list.blockSignals(True); self.mask_list.setCurrentRow(index); self.mask_list.blockSignals(False); self.view.redraw()
        mask=self.masks[index]; category_id=int(mask.get("category_id",1)); combo_index=self.category_combo.findData(category_id)
        if combo_index>=0:self.category_combo.blockSignals(True); self.category_combo.setCurrentIndex(combo_index); self.category_combo.blockSignals(False)
        self.sync_drawing_defaults()
    def push_undo(self):
        self.mark_current_dirty()
        self.undo_stack.append(self._mask_snapshot()); self.undo_stack=self.undo_stack[-50:]; self.redo_stack.clear()
        self.view.detach_selected_mask()
    def _mask_snapshot(self):
        snapshot=[]
        for item in self.masks:
            saved={key:copy.deepcopy(value) for key,value in item.items() if key!="mask"}; saved["mask"]=item["mask"]; snapshot.append(saved)
        return snapshot
    @staticmethod
    def _restore_snapshot(snapshot):
        return [{key:(value if key=="mask" else copy.deepcopy(value)) for key,value in saved.items()} for saved in snapshot]
    def undo(self):
        if not self.undo_stack:return
        self.mark_current_dirty()
        self.redo_stack.append(self._mask_snapshot()); self.masks[:]=self._restore_snapshot(self.undo_stack.pop()); self.view.masks=self.masks; self.view.selected=-1; self.view.redraw(); self.refresh_masks()
    def redo(self):
        if not self.redo_stack:return
        self.mark_current_dirty()
        self.undo_stack.append(self._mask_snapshot()); self.masks[:]=self._restore_snapshot(self.redo_stack.pop()); self.view.masks=self.masks; self.view.selected=-1; self.view.redraw(); self.refresh_masks()
    def delete_mask(self):
        rows=sorted({item.row() for item in self.mask_list.selectedIndexes()},reverse=True)
        if not rows and 0<=self.view.selected<len(self.masks):rows=[self.view.selected]
        rows=[row for row in rows if 0<=row<len(self.masks)]
        if not rows:return
        self.push_undo()
        for row in rows:self.masks.pop(row)
        self.view.selected=-1; self.view.masks=self.masks; self.view.redraw(); self.refresh_masks()
    def show_mask_context_menu(self,pos):
        item=self.mask_list.itemAt(pos)
        if item is None:return
        if not item.isSelected():
            self.mask_list.clearSelection(); item.setSelected(True); self.mask_list.setCurrentItem(item)
        rows=sorted({entry.row() for entry in self.mask_list.selectedIndexes()})
        if not rows:return
        english=getattr(self,"ui_language","zh")=="en"; menu=QMenu(self.mask_list); action=menu.addAction("Delete Selected Labels" if english else "删除选中标签")
        chosen=menu.exec(self.mask_list.viewport().mapToGlobal(pos))
        if chosen is not action:return
        message=(f"Delete {len(rows)} selected label(s)?" if english else f"删除选中的 {len(rows)} 个标签？")
        if QMessageBox.question(self,"Delete Labels" if english else "删除标签",message,QMessageBox.Yes|QMessageBox.No,QMessageBox.No)==QMessageBox.Yes:self.delete_mask()
    def merge_masks(self):
        rows=sorted({x.row() for x in self.mask_list.selectedIndexes()})
        if len(rows)<2:self.statusBar().showMessage("请在实例列表中选择至少两个实例");return
        base=self.masks[rows[0]]; category=self.project.category(int(base.get("category_id",1))) or {"name":"未知"}
        message=f"将 {len(rows)} 个 Masks 合并为一个实例。\n将保留第一个实例的 Category {base.get('category_id',1)} | {category['name']} 和 Noun phrase。\n是否继续？"
        if QMessageBox.question(self,"确认合并实例",message,QMessageBox.Yes|QMessageBox.No)!=QMessageBox.Yes:return
        self.push_undo(); merged=np.logical_or.reduce([self.masks[i]["mask"] for i in rows]); base["mask"]=merged; base["_geometry_dirty"]=True
        self.masks[:]=[m for i,m in enumerate(self.masks) if i not in rows[1:]]; self.view.selected=rows[0]; self.view.redraw(); self.refresh_masks()
    def toggle_review(self):
        if self.index<0:return
        self.commit_current(); image_id=self.project.image_record(self.index)["id"]
        if image_id in self.project.reviewed:self.project.reviewed.remove(image_id)
        else:self.project.reviewed.add(image_id)
        self.project.dirty=True; self.update_image_status(self.index); self.statusBar().showMessage("审核状态已更新；按 Ctrl+S 保存")
    def update_image_status(self,row):
        if row<0 or row>=self.images.count():return
        rec=self.project.image_record(row); path=self.project.image_paths[row]
        has_annotations=any(ann.get("image_id")==rec["id"] for ann in self.project.data["annotations"])
        prefix=("✓ " if rec["id"] in self.project.reviewed else "  ")+("● " if has_annotations else "○ ")
        self.images.item(row).setText(prefix+path.name)
    def save(self):
        if self.project.json_path is None:return
        try:
            self.commit_current(force=True); self.project.save(); self.update_image_status(self.index); self.statusBar().showMessage(f"已保存：{self.project.json_path}",5000)
            QMessageBox.information(self,"保存完成",f"已保存到：\n{self.project.json_path}")
        except Exception as e:self.error(e)
    def export_all(self):
        if self.project.root is None:return
        if self.export_thread and self.export_thread.isRunning():QMessageBox.information(self,"Export","An export is already running.");return
        try:self.commit_current(force=True);self.project.save()
        except Exception as exc:self.error(exc);return
        english=getattr(self,"ui_language","zh")=="en"
        chosen=QFileDialog.getExistingDirectory(self,"Select export directory" if english else "选择导出目录",str(self.project.root/"exports"))
        if not chosen:return
        out=Path(chosen); out.mkdir(parents=True,exist_ok=True)
        self.export_thread=QThread(self); self.export_worker=ExportWorker(self.project.data,self.project.reviewed,self.project.image_paths,out); self.export_worker.moveToThread(self.export_thread)
        self.export_dialog=QProgressDialog("Preparing export…" if english else "正在准备导出……","Cancel" if english else "取消",0,max(2,2*len(self.project.image_paths)+2),self)
        self.export_dialog.setWindowTitle("Export Progress" if english else "导出进度"); self.export_dialog.setWindowModality(Qt.WindowModal); self.export_dialog.setMinimumDuration(0); self.export_dialog.setAutoClose(False); self.export_dialog.setAutoReset(False)
        self.export_dialog.canceled.connect(lambda:self.export_worker.request_cancel() if self.export_worker else None)
        self.export_thread.started.connect(self.export_worker.run); self.export_worker.progress.connect(self._export_progress)
        self.export_worker.completed.connect(self._export_completed); self.export_worker.cancelled.connect(self._export_cancelled); self.export_worker.failed.connect(self._export_failed)
        self.export_worker.completed.connect(self.export_thread.quit); self.export_worker.cancelled.connect(self.export_thread.quit); self.export_worker.failed.connect(self.export_thread.quit); self.export_thread.finished.connect(self._export_thread_finished)
        self.export_btn.setEnabled(False); self.export_thread.start()

    def _export_progress(self,done,total,message):
        if self.export_dialog:self.export_dialog.setRange(0,total);self.export_dialog.setValue(done);self.export_dialog.setLabelText(message)

    @Slot(str)
    def _export_completed(self,path):self._export_finished("completed",path)

    @Slot(str)
    def _export_cancelled(self,path):self._export_finished("cancelled",path)

    @Slot(str)
    def _export_failed(self,message):self._export_finished("failed",message)

    def _export_finished(self,status,message):
        english=getattr(self,"ui_language","zh")=="en"
        if self.export_dialog:self.export_dialog.close()
        if status=="completed":QMessageBox.information(self,"Export",("Export completed.\nTraining dataset:\n" if english else "导出完成。\n训练数据：\n")+message)
        elif status=="cancelled":QMessageBox.warning(self,"Export",("Export cancelled. Partial files are marked incomplete:\n" if english else "导出已取消。部分文件已标记为不完整：\n")+message)
        else:QMessageBox.critical(self,"Export Error" if english else "导出错误",message)

    def _export_thread_finished(self):
        self.export_btn.setEnabled(True)
        if self.export_worker:self.export_worker.deleteLater()
        if self.export_thread:self.export_thread.deleteLater()
        self.export_worker=None;self.export_thread=None;self.export_dialog=None
    def error(self,e):QMessageBox.critical(self,"错误",str(e))
    def closeEvent(self,event):
        if getattr(self,"_batch_process",None) is not None:event.ignore();return
        if self.export_thread and self.export_thread.isRunning():QMessageBox.warning(self,"Export","Please cancel the export and wait for it to stop before closing." if getattr(self,"ui_language","zh")=="en" else "请先取消导出并等待任务停止后再关闭窗口。");event.ignore();return
        if self.open_thread and self.open_thread.isRunning():QMessageBox.warning(self,"Import Existing Project" if getattr(self,"ui_language","zh")=="en" else "导入已有项目","Please wait for the project import to finish before closing." if getattr(self,"ui_language","zh")=="en" else "请等待项目导入完成后再关闭。");event.ignore();return
        if self.project.dirty:
            ans=QMessageBox.question(self,"保存更改","关闭前保存更改吗？",QMessageBox.Yes|QMessageBox.No|QMessageBox.Cancel)
            if ans==QMessageBox.Cancel:event.ignore();return
            if ans==QMessageBox.Yes:self.save()
        self.sam_thread.quit(); self.sam_thread.wait(3000); event.accept()


if __name__=="__main__":
    app=QApplication(sys.argv); app.setStyle("Fusion"); window=MainWindow(); window.show(); sys.exit(app.exec())
