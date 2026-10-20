"""Auto Box workflow, append/undo safety and external-example ROI handling."""
import contextlib
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
from PySide6.QtCore import QPointF, QProcess, Qt
from PySide6.QtWidgets import QApplication

from app_en import MainWindow
from sam3_inference_process import predict_image
from test_batch_sam3 import FAKE_WORKER


class AutoBoxTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        for i in range(2):Image.fromarray(np.zeros((24,24,3),np.uint8)).save(self.root/f'{i}.png')
        self.worker=self.root/'worker.py';self.worker.write_text("MODE='success'\n"+FAKE_WORKER)
        arguments=QProcess.setArguments
        def args(process,values):return arguments(process,[str(self.worker)]+values[1:])
        self.patches=[patch.object(QProcess,'setArguments',args),patch.object(MainWindow,'_update_sam_labels'),patch.object(MainWindow,'_model_key',return_value=('repo','ckpt',0,0)),patch('resident_sam3.valid_repository',return_value=True),patch('resident_sam3.valid_checkpoint',return_value=True),patch('resident_sam3.QMessageBox.critical')]
        for p in self.patches:p.start()
        self.w=MainWindow();self.w.project.new_from_folder(self.root);self.w.populate()
        m=np.zeros((24,24),bool);m[4:10,5:11]=True
        self.w.masks=[{'id':0,'category_id':1,'noun_phrase':'particle','annotation_source':'manual','mask':m}]
        self.w.view.set_image(self.w.view.array,self.w.masks);self.w.refresh_masks();self.w.mark_current_dirty()
        self.app.processEvents()

    def tearDown(self):
        if self.w.sam_busy:self.w.cancel_resident_task()
        self.w.project.dirty=False;self.w.close()
        for p in reversed(self.patches):p.stop()
        self.temp.cleanup()

    def wait(self):
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            self.app.processEvents();time.sleep(.005)
            if not self.w.sam_busy:return
        self.fail('Auto Box inference did not finish')

    def box(self,box=(2,2,20,20)):
        self.w.view.sam_box=box;self.w.view.box_changed.emit(box);self.wait()

    def test_box_auto_appends_duplicates_without_confirmation_and_undo(self):
        original=self.w.masks[0]['mask'].copy();self.w.activate_tool('sam_box')
        with patch.object(self.w,'_confirm_sam_replacement',side_effect=AssertionError('No confirmation allowed')):
            self.box();self.assertEqual(len(self.w.masks),2)
            self.box();self.assertEqual(len(self.w.masks),3)
        self.assertTrue(np.array_equal(original,self.w.masks[0]['mask']))
        self.assertTrue(np.array_equal(self.w.masks[1]['mask'],self.w.masks[2]['mask']))
        self.w.undo();self.assertEqual(len(self.w.masks),2)
        self.w.redo();self.assertEqual(len(self.w.masks),3)
        self.assertEqual(self.w.view.mode,'sam_box')
        self.assertEqual(self.w._resident_load_count,1)

    def test_instance_is_frozen_survives_deletion_and_multiple_boxes(self):
        self.w.set_prompt_mode('mask');self.w.view.selected=0;self.w.use_selected_as_mask_prompt()
        example=self.w.mask_prompt.copy();self.assertEqual(self.w.view.mode,'sam_box')
        self.w.select_mask(0);self.w.delete_mask();self.assertEqual(len(self.w.masks),0)
        self.assertTrue(np.array_equal(example,self.w.mask_prompt))
        with patch.object(self.w,'_confirm_sam_replacement',side_effect=AssertionError('No confirmation allowed')):
            self.box((13,13,22,22));self.box((0,12,10,22))
        self.assertEqual(len(self.w.masks),2)
        self.assertTrue(np.array_equal(example,self.w.mask_prompt))
        self.assertEqual(self.w.view.mode,'sam_box')
        requests=[json.loads(line) for line in (self.root/'requests.jsonl').read_text().splitlines()]
        self.assertEqual([r['mode'] for r in requests],['instance','instance'])
        self.assertEqual([r['box'] for r in requests],[[13,13,22,22],[0,12,10,22]])

    def test_disabled_multiselected_and_zero_size_boxes_do_not_run(self):
        with patch.object(self.w,'start_resident_single') as run:
            self.w.auto_box_check.setChecked(False);self.box();run.assert_not_called()
            self.w.auto_box_check.setChecked(True);self.box((4,4,4,4));run.assert_not_called()
            self.w.images.selectAll();self.box();run.assert_not_called()

    def test_reference_required_and_cleared_on_switch(self):
        self.w.set_prompt_mode('mask')
        with patch.object(self.w,'start_resident_single') as run:
            self.box();run.assert_not_called()
            self.w.view.selected=0;self.w.use_selected_as_mask_prompt()
            self.w.images.setCurrentRow(1)
            self.assertIsNone(self.w.mask_prompt)
            self.box();run.assert_not_called()

    def test_no_detections_keeps_labels_and_undo_stack(self):
        self.worker.write_text("MODE='empty'\n"+FAKE_WORKER)
        original=self.w.masks[0]['mask'].copy();undos=len(self.w.undo_stack)
        self.box();self.assertEqual(len(self.w.masks),1)
        self.assertEqual(len(self.w.undo_stack),undos)
        self.assertTrue(np.array_equal(original,self.w.masks[0]['mask']))

    def test_ordinary_run_still_confirms_replacement(self):
        with patch.object(self.w,'_confirm_sam_replacement',return_value=False) as confirm,patch.object(self.w,'start_resident_single') as run:
            self.w.run_sam3();confirm.assert_called_once();run.assert_not_called()

    def test_actual_release_updates_box_and_triggers_one_request(self):
        view=self.w.view;view.set_mode('sam_box');view.box_start=(2,3);view.sam_box=(2,3,2,3);view.drawing=True
        event=SimpleNamespace(button=lambda:Qt.LeftButton,position=lambda:QPointF(0,0))
        with patch.object(view,'image_pos',return_value=(20,21)),patch.object(self.w,'start_resident_single') as run:
            # QGraphicsView needs a real Qt event for its base release handler.
            from PySide6.QtGui import QMouseEvent
            from PySide6.QtCore import QEvent
            event=QMouseEvent(QEvent.MouseButtonRelease,QPointF(20,21),QPointF(20,21),Qt.LeftButton,Qt.NoButton,Qt.NoModifier)
            view.mouseReleaseEvent(event);self.app.processEvents()
            self.assertEqual(view.sam_box,(2,3,20,21))
            run.assert_called_once_with('particle',append_only=True)


class Tensor:
    def __init__(self,array):self.array=np.asarray(array)
    def view(self,*shape):return Tensor(self.array.reshape(shape))
    def squeeze(self,axis):return Tensor(np.squeeze(self.array,axis))
    def detach(self):return self
    def cpu(self):return self
    def float(self):return self
    def numpy(self):return self.array


class Processor:
    def __init__(self):
        self.appended=[];self.model=SimpleNamespace(backbone=SimpleNamespace(forward_text=lambda *a,**kw:{}),_get_dummy_prompt=lambda:SimpleNamespace(append_masks=self.appended.append))
    def set_confidence_threshold(self,value):self.confidence=value
    def set_image(self,image):self.shape=(image.height,image.width);return {'backbone_out':{}}
    def result(self,state):
        h,w=self.shape;m=np.zeros((2,1,h,w),bool);m[0]=True;m[1,0,-5:,-5:]=True
        state.update(masks=Tensor(m),scores=Tensor([.9,.8]));return state
    def _forward_grounding(self,state):return self.result(state)
    def set_text_prompt(self,prompt,state):return self.result(state)


class ExternalExampleTest(unittest.TestCase):
    def test_example_outside_roi_is_seen_but_results_stay_inside_roi(self):
        torch=SimpleNamespace(inference_mode=contextlib.nullcontext,autocast=lambda **kw:contextlib.nullcontext(),bfloat16='bf16',float32='f32',as_tensor=lambda array,**kw:Tensor(array))
        processor=Processor();rgb=np.zeros((80,80,3),np.uint8)
        with tempfile.TemporaryDirectory() as temp:
            example=np.zeros((80,80),bool);example[50:55,50:55]=True
            file=Path(temp)/'example.npy';np.save(file,example)
            masks,scores=predict_image(processor,torch,rgb,'particle',.5,(2,2,10,10),'instance',file)
        self.assertEqual(processor.shape,(53,53))
        self.assertTrue(processor.appended[0].array.any())
        self.assertEqual(masks.shape,(1,80,80))
        self.assertEqual(np.count_nonzero(masks),81)
        self.assertTrue(masks[0,2:11,2:11].all())
        self.assertEqual(scores.tolist(),[.9])
