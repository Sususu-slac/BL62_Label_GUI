import json,re,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
from PySide6.QtGui import QAction
from PySide6.QtCore import QPoint,QPointF,Qt
from PySide6.QtWidgets import QApplication,QLabel,QPushButton
from app import AnnotationView,ExportWorker,MainWindow
from app_en import MainWindow as EnglishMainWindow

class BrushPerformanceTest(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])
 def masks(self,n=100,h=1024,w=1024):
  result=[]
  for i in range(n):
   mask=np.zeros((h,w),bool);y=(i*83)%(h-35)+15;x=(i*131)%(w-35)+15;mask[y-8:y+9,x-8:x+9]=True;result.append({'id':i,'category_id':1,'noun_phrase':'particle','mask':mask})
  return result
 def test_local_brush_preserves_other_masks(self):
  view=AnnotationView();image=np.zeros((512,512,3),np.uint8);masks=self.masks(30,512,512);before=masks[1]['mask'].copy();view.set_image(image,masks);view.selected=0;view.mode='brush';view.drawing=True
  view.paint_at(250,250);view._flush_live_redraw();self.assertTrue(masks[0]['mask'][250,250]);self.assertTrue(np.array_equal(before,masks[1]['mask']))
 def test_500_points_with_100_sam_masks(self):
  view=AnnotationView();image=np.zeros((1024,1024,3),np.uint8);view.set_image(image,self.masks());view.selected=0;view.mode='brush';view.drawing=True
  started=time.perf_counter()
  for i in range(500):view.paint_at(100+i%700,100+(i*3)%700)
  view._flush_live_redraw();elapsed=time.perf_counter()-started
  self.assertLess(elapsed,1.5,f'brush benchmark took {elapsed:.3f}s')
 def test_show_hide_labels_preserves_masks(self):
  view=AnnotationView();image=np.zeros((256,256,3),np.uint8);masks=self.masks(12,256,256);original=[item['mask'].copy() for item in masks];view.set_image(image,masks);view.selected=0;view.redraw()
  self.assertTrue(view.overlay_item.isVisible());self.assertTrue(view.active_overlay_item.isVisible());view.set_labels_visible(False);self.assertFalse(view.overlay_item.isVisible());self.assertFalse(view.active_overlay_item.isVisible());self.assertTrue(all(np.array_equal(a,item['mask']) for a,item in zip(original,masks)));view.set_labels_visible(True);self.assertTrue(view.overlay_item.isVisible());self.assertTrue(view.active_overlay_item.isVisible())
 def test_show_hide_labels_is_responsive(self):
  view=AnnotationView();image=np.zeros((1024,1024,3),np.uint8);view.set_image(image,self.masks(100));view.selected=0;view.redraw();started=time.perf_counter()
  for _ in range(50):view.set_labels_visible(False);view.set_labels_visible(True)
  elapsed=time.perf_counter()-started;self.assertLess(elapsed,1.0,f'100 label visibility changes took {elapsed:.3f}s')
 def test_wheel_zoom_keeps_cursor_on_same_image_point(self):
  view=AnnotationView();view.resize(800,600);view.set_image(np.zeros((1000,1000,3),np.uint8),[]);view.scale(3,3);cursor=QPointF(213,177);before=view.mapToScene(cursor.toPoint())
  event=type('WheelEvent',(),{'position':lambda self:cursor,'angleDelta':lambda self:QPoint(0,120),'accept':lambda self:None})()
  view.wheelEvent(event);after=view.mapToScene(cursor.toPoint());self.assertAlmostEqual(before.x(),after.x(),places=5);self.assertAlmostEqual(before.y(),after.y(),places=5)
 def test_middle_mouse_temporarily_pans_without_changing_tool(self):
  view=AnnotationView();view.set_image(np.zeros((512,512,3),np.uint8),[]);view.set_mode('brush')
  press=type('MouseEvent',(),{'button':lambda self:Qt.MiddleButton,'position':lambda self:QPointF(100,100),'accept':lambda self:None})()
  move=type('MouseEvent',(),{'position':lambda self:QPointF(125,130),'accept':lambda self:None})();release=type('MouseEvent',(),{'button':lambda self:Qt.MiddleButton,'accept':lambda self:None})()
  view.mousePressEvent(press);self.assertTrue(view._middle_panning);self.assertEqual(view.mode,'brush');view.mouseMoveEvent(move);view.mouseReleaseEvent(release);self.assertFalse(view._middle_panning);self.assertEqual(view.mode,'brush')
 def test_english_save_shows_saved_confirmation(self):
  window=EnglishMainWindow();window.project.json_path=Path('annotations.json');window.project.dirty=False
  with patch.object(window,'commit_current'),patch.object(window.project,'save'),patch('app_en.QMessageBox.information') as notice:window.save();notice.assert_called_once_with(window,'Save Complete','Saved.')
  window.project.dirty=False;window.sam_thread.quit();window.sam_thread.wait(3000);window.close()
 def test_sam_box_replacement_keeps_labels_outside_box(self):
  window=MainWindow();inside=np.zeros((100,100),bool);inside[20:30,20:30]=True;outside=np.zeros((100,100),bool);outside[70:80,70:80]=True
  window.masks=[{'mask':inside},{'mask':outside}];window.view.masks=window.masks;window._sam_replace_requested=True;window._sam_replace_box=(10,10,40,40);window._apply_pending_sam_replacement()
  self.assertEqual(len(window.masks),1);self.assertTrue(np.array_equal(window.masks[0]['mask'],outside));window.project.dirty=False;window.sam_thread.quit();window.sam_thread.wait(3000);window.close()
 def test_full_sam_replacement_clears_all_labels_only_on_success(self):
  window=MainWindow();window.masks=self.masks(4,100,100);window.view.masks=window.masks;window._sam_replace_requested=True;window._sam_replace_box=None
  self.assertEqual(len(window.masks),4);window._apply_pending_sam_replacement();self.assertEqual(window.masks,[]);window.project.dirty=False;window.sam_thread.quit();window.sam_thread.wait(3000);window.close()
 def test_packed_undo_redo(self):
  window=MainWindow();window.masks=self.masks(20,256,256);window.view.masks=window.masks;original=window.masks[0]['mask'].copy();window.push_undo();window.masks[0]['mask'][100:110,100:110]=True;window.undo();self.assertTrue(np.array_equal(window.masks[0]['mask'],original));window.redo();self.assertTrue(window.masks[0]['mask'][105,105]);window.close()
 def test_export_contains_current_coco_json(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);images=root/'images';images.mkdir();Image.fromarray(np.zeros((32,32,3),np.uint8)).save(images/'slice.png');out=root/'export'
   window=MainWindow();window.project.new_from_folder(images);window.populate();mask=np.zeros((32,32),bool);mask[8:20,9:21]=True;window.masks=[{'id':0,'category_id':1,'noun_phrase':'particle','annotation_source':'manual','mask':mask}];window.view.masks=window.masks
   with patch('app.QFileDialog.getExistingDirectory',return_value=str(out)),patch('app.QMessageBox.information'),patch('app.QMessageBox.warning'),patch('app.QMessageBox.critical'):
    window.export_all();deadline=time.monotonic()+10
    while window.export_thread is not None and time.monotonic()<deadline:QApplication.processEvents();time.sleep(.01)
    QApplication.processEvents()
   self.assertIsNone(window.export_thread,"background export did not clean up within 10 seconds");training=out/'training_dataset';auxiliary=out/'auxiliary_outputs';data=json.loads((training/'annotations.json').read_text(encoding='utf-8'));self.assertEqual(len(data['annotations']),1);self.assertEqual(data['annotations'][0]['category_id'],1);self.assertTrue((training/'images'/'slice.png').is_file());self.assertTrue((auxiliary/'project_state.json').is_file());self.assertTrue((auxiliary/'instance_labels'/'slice_instances.png').is_file());self.assertTrue((auxiliary/'previews'/'slice_preview.png').is_file());self.assertTrue((auxiliary/'dataset_statistics.csv').is_file());window.project.dirty=False;window.sam_thread.quit();window.sam_thread.wait(3000);window.deleteLater()
 def test_export_cancel_marks_partial_output(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);source=root/'slice.png';Image.fromarray(np.zeros((16,16,3),np.uint8)).save(source);out=root/'out';worker=ExportWorker({'images':[{'id':1,'file_name':'slice.png','width':16,'height':16}],'annotations':[],'categories':[{'id':1,'name':'particle'}]},set(),[source],out);cancelled=[];worker.cancelled.connect(cancelled.append);worker.request_cancel();worker.run();self.assertEqual(cancelled,[str(out)]);self.assertTrue((out/'EXPORT_INCOMPLETE.txt').is_file())
 def test_export_error_is_reported_and_marked(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);out=root/'out';missing=root/'missing.png';worker=ExportWorker({'images':[{'id':1,'file_name':'missing.png','width':16,'height':16}],'annotations':[],'categories':[{'id':1,'name':'particle'}]},set(),[missing],out);errors=[];worker.failed.connect(errors.append);worker.run();self.assertEqual(len(errors),1);self.assertIn('FileNotFoundError',errors[0]);self.assertIn('Export failed', (out/'EXPORT_INCOMPLETE.txt').read_text(encoding='utf-8'))
 def test_english_ui_visible_and_dynamic_text_has_no_chinese(self):
  window=EnglishMainWindow();texts=[widget.text() for widget in window.findChildren(QLabel)+window.findChildren(QPushButton)]+[action.text() for action in window.findChildren(QAction)];window.set_prompt_mode('mask');texts += [window.statusBar().currentMessage(),window.run_sam_btn.text(),window.mask_prompt_status.text()];window.apply_labels_to_selected();texts.append(window.statusBar().currentMessage());chinese=[text for text in texts if re.search(r'[\u4e00-\u9fff]',text)];self.assertEqual(chinese,[]);window.project.dirty=False;window.close()

if __name__=='__main__':unittest.main()
