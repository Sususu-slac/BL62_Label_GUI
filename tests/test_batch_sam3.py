"""Batch lifecycle tests using an isolated fake inference process, without a GPU."""
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
from PySide6.QtCore import QProcess
from PySide6.QtWidgets import QApplication, QMessageBox, QPushButton

from app import MainWindow
from app_en import MainWindow as EnglishMainWindow

FAKE_WORKER = '''
import json,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
print(json.dumps({'type':'ready'}),flush=True)
for i,line in enumerate(sys.stdin):
    request=json.loads(line)
    with (Path(__file__).parent/'requests.jsonl').open('a') as log:log.write(json.dumps(request)+'\\n')
    if MODE == 'slow':time.sleep(30)
    if MODE == 'failure' and i==1:sys.exit(1)
    if MODE == 'cancel' and i==1:time.sleep(30)
    path=Path(request['image'])
    if path.suffix=='.npy':h,w=np.load(path).shape[:2]
    else:
        with Image.open(path) as image:w,h=image.size
    masks=np.zeros((1,h,w),bool);masks[:,4:10,5:11]=True
    if MODE == 'empty':masks=masks[:0]
    np.savez(request['output'],masks=masks,scores=np.ones(len(masks)))
    print(json.dumps({'type':'result','id':request['id']}),flush=True)
'''



class BatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for i in range(3):
            Image.fromarray(np.zeros((24,24,3),np.uint8)).save(self.root/f'{i}.png')
        self.window = MainWindow()
        self.window.project.new_from_folder(self.root)
        self.window.populate()
        mask = np.zeros((24,24),bool); mask[15:20,15:20]=True
        for i in range(3):
            rec=self.window.project.image_record(i)
            self.window.project.replace_masks(rec['id'],[{'mask':mask.copy(),'category_id':1,'annotation_source':'manual'}])
            self.window.project.reviewed.add(rec['id'])
        self.window.load_image(0)
        self.original = json.loads(json.dumps(self.window.project.data['annotations']))
        worker=self.root/'fake.py';worker.write_text("MODE='success'\n"+FAKE_WORKER)
        set_arguments=QProcess.setArguments
        def fake_arguments(process,arguments):return set_arguments(process,[str(worker)]+arguments[1:])
        self.patches=[patch.object(QProcess,'setArguments',fake_arguments),patch.object(MainWindow,'_update_sam_labels'),patch('batch_sam3.QMessageBox.information'),patch('batch_sam3.QMessageBox.critical'),patch('resident_sam3.valid_repository',return_value=True),patch('resident_sam3.valid_checkpoint',return_value=True),patch.object(MainWindow,'_model_key',return_value=('repo','checkpoint',0,0))]
        for p in self.patches:p.start()

    def tearDown(self):
        if getattr(self.window,'_batch_process',None) is not None:
            self.window.cancel_selected_batch();self.wait()
        self.window.project.dirty=False;self.window.close()
        for p in self.patches:p.stop()
        self.temp.cleanup()

    def start(self, mode):
        (self.root/'fake.py').write_text('MODE='+repr(mode)+'\n'+FAKE_WORKER)
        self.window._start_selected_batch([0,2],'particle')

    def wait_idle(self):
        deadline=time.monotonic()+15
        while self.window.sam_busy and time.monotonic()<deadline:
            self.app.processEvents();time.sleep(.01)
        self.assertFalse(self.window.sam_busy)

    def wait(self):
        deadline=time.monotonic()+15
        while getattr(self.window,'_batch_process',None) is not None and time.monotonic()<deadline:
            self.app.processEvents();time.sleep(.01)
        self.assertIsNone(self.window._batch_process)
        self.assertFalse(self.window.sam_busy)

    def assert_original(self,row):
        image_id=self.window.project.image_record(row)['id']
        self.assertEqual([a for a in self.original if a['image_id']==image_id],
                         [a for a in self.window.project.data['annotations'] if a['image_id']==image_id])

    def test_success_replaces_only_selected_and_saves(self):
        self.start('success');self.wait()
        self.assertEqual(self.window._batch_completed,{0,1})
        self.assertFalse(self.window._batch_cancelled)
        self.assert_original(1)
        data=json.loads(self.window.project.json_path.read_text())
        self.assertEqual(len(data['annotations']),3)
        for row in (0,2):
            rec=self.window.project.image_record(row)
            ann=next(a for a in data['annotations'] if a['image_id']==rec['id'])
            self.assertEqual(ann['annotation_source'],'sam3_text')
            self.assertNotIn(rec['id'],self.window.project.reviewed)
        self.assertEqual(self.window.masks[0]['annotation_source'],'sam3_text')
        self.window.save()  # Displayed masks must never restore old labels.
        self.assertEqual(sorted(self.window.project.data['annotations'],key=lambda a:a['id']),sorted(data['annotations'],key=lambda a:a['id']))

    def test_failure_keeps_unfinished_labels(self):
        self.start('failure');self.wait()
        self.assertEqual(self.window._batch_completed,{0})
        self.assertTrue(self.window._batch_error)
        self.assert_original(1);self.assert_original(2)

    def test_cancel_keeps_unfinished_labels(self):
        self.start('cancel')
        deadline=time.monotonic()+10
        while not self.window._batch_completed and time.monotonic()<deadline:
            self.app.processEvents();time.sleep(.01)
        self.assertEqual(self.window._batch_completed,{0})
        self.window.cancel_selected_batch();self.wait()
        self.assert_original(1);self.assert_original(2)

    def test_zero_detections_clears_selected_labels(self):
        self.start('empty');self.wait();self.assert_original(1)
        self.assertEqual(len(self.window.project.data['annotations']),1)
        self.assertEqual(self.window.masks,[])

    def test_save_failure_restores_in_memory_labels(self):
        save = self.window.project.save
        calls = []
        def fail_after_start():
            calls.append(1)
            if len(calls)>1:raise OSError("Disk full")
            return save()
        with patch.object(self.window.project,'save',side_effect=fail_after_start):
            self.start('success');self.wait()
        self.assertIn('Disk full',self.window._batch_error)
        self.assertEqual(self.window._batch_completed,set())
        for row in range(3):self.assert_original(row)

    def test_process_start_failure_unlocks_ui(self):
        original=QProcess.setProgram
        def missing(process,program):return original(process,str(self.root/'missing_python'))
        with patch.object(QProcess,'setProgram',missing):
            self.window._start_selected_batch([0,2],'particle');self.wait()
        self.assertTrue(self.window._batch_error)
        self.assertTrue(self.window.load_sam_btn.isEnabled())
        for row in range(3):self.assert_original(row)

    def test_single_and_multiple_batches_reuse_one_model(self):
        self.start('success');self.wait()
        process=self.window._resident_process;pid=process.processId()
        self.assertTrue(self.window.sam_loaded_flag)
        self.assertEqual(self.window._resident_load_count,1)
        self.window._sam_replace_requested=True
        self.window.start_resident_single('particle');self.wait_idle()
        self.assertEqual(self.window._resident_process.processId(),pid)
        self.assertEqual(len(self.window.masks),1)
        self.start('success');self.wait()
        self.assertEqual(self.window._resident_process.processId(),pid)
        self.assertEqual(self.window._resident_load_count,1)
        self.assertTrue(self.window.release_sam_btn.isEnabled())

    def test_release_and_next_run_reload(self):
        self.start('success');self.wait();old_pid=self.window._resident_process.processId()
        self.window.release_sam_btn.click()
        self.assertIsNone(self.window._resident_process)
        self.assertFalse(self.window.sam_loaded_flag)
        self.assertFalse(self.window.release_sam_btn.isEnabled())
        self.assertTrue(self.window.run_sam_btn.isEnabled())
        self.start('success');self.wait()
        self.assertNotEqual(self.window._resident_process.processId(),old_pid)
        self.assertEqual(self.window._resident_load_count,2)

    def test_configuration_change_reloads_and_explicit_load_reuses(self):
        self.window.load_sam3();self.wait_idle()
        old_pid=self.window._resident_process.processId()
        self.window.sam_configuration_changed();self.wait_idle()
        self.assertNotEqual(self.window._resident_process.processId(),old_pid)
        self.assertEqual(self.window._resident_load_count,2)
        pid=self.window._resident_process.processId()
        self.window.load_sam3();self.wait_idle()
        self.assertEqual(self.window._resident_process.processId(),pid)
        self.assertEqual(self.window._resident_load_count,2)

    def test_changed_checkpoint_key_reloads_before_next_request(self):
        self.start('success');self.wait();old_pid=self.window._resident_process.processId()
        with patch.object(MainWindow,'_model_key',return_value=('repo','new-checkpoint',1,1)):
            self.start('success');self.wait()
        self.assertNotEqual(self.window._resident_process.processId(),old_pid)
        self.assertEqual(self.window._resident_load_count,2)

    def test_idle_worker_crash_recovers_on_next_run(self):
        self.start('success');self.wait();process=self.window._resident_process
        process.kill();process.waitForFinished(3000);self.app.processEvents()
        self.assertFalse(self.window.sam_loaded_flag)
        self.assertIsNone(self.window._resident_process)
        self.start('success');self.wait()
        self.assertEqual(self.window._resident_load_count,2)

    def test_single_cancel_keeps_original_labels_and_reloads_later(self):
        (self.root/'fake.py').write_text("MODE='slow'\n"+FAKE_WORKER)
        original_mask=self.window.masks[0]['mask'].copy()
        self.window._sam_replace_requested=True
        self.window.start_resident_single('particle')
        self.assertTrue(self.window.sam_busy)
        self.window.cancel_resident_task();self.wait_idle()
        self.assertFalse(self.window._sam_replace_requested)
        self.assertTrue(np.array_equal(original_mask,self.window.masks[0]['mask']))
        self.assertIsNone(self.window._resident_process)
        for row in range(3):self.assert_original(row)
        self.start('success');self.wait()
        self.assertTrue(self.window.sam_loaded_flag)

    def test_resident_idle_allows_editing_saving_and_image_switching(self):
        self.start('success');self.wait();pid=self.window._resident_process.processId()
        self.assertFalse(self.window.sam_busy)
        self.assertIsNone(getattr(self.window,'_sam_job_dialog',None))
        self.assertIsNone(self.window._batch_dialog)
        self.window.view.selected=0;self.window.push_undo()
        self.window.view.set_mode('brush');self.window.view.drawing=True
        self.window.view.paint_at(12,12);self.window.view._flush_live_redraw()
        self.assertTrue(self.window.masks[0]['mask'][12,12])
        self.window.view.set_mode('erase');self.window.view.paint_at(12,12)
        self.window.view._flush_live_redraw();self.window.view.drawing=False
        self.assertFalse(self.window.masks[0]['mask'][12,12])
        self.window.save();self.window.images.setCurrentRow(1)
        self.assertEqual(self.window.index,1)
        self.window.images.setCurrentRow(0)
        self.assertEqual(self.window.index,0)
        self.assertTrue(self.window.sam_loaded_flag)
        self.assertEqual(self.window._resident_process.processId(),pid)

    def test_close_terminates_resident_worker(self):
        self.start('success');self.wait();process=self.window._resident_process
        self.window.project.dirty=False;self.window.close()
        self.assertEqual(process.state(),QProcess.NotRunning)
        self.assertIsNone(self.window._resident_process)

    def test_changed_prompt_confidence_and_instance_inputs_share_worker(self):
        self.window.load_sam3();self.wait_idle();pid=self.window._resident_process.processId()
        self.window.confidence.setValue(75)
        self.window.view.sam_box=(2,3,20,21)
        self.window.mask_prompt=self.window.masks[0]['mask'].copy()
        self.window.set_prompt_mode('mask')
        self.window.start_resident_single('other prompt');self.wait_idle()
        requests=[json.loads(line) for line in (self.root/'requests.jsonl').read_text().splitlines()]
        self.assertEqual(requests[-1]['confidence'],.75)
        self.assertEqual(requests[-1]['box'],[2,3,20,21])
        self.assertEqual(requests[-1]['mode'],'instance')
        self.assertEqual(requests[-1]['prompt'],'other prompt')
        self.assertIn('exemplar',requests[-1])
        self.assertEqual(self.window._resident_process.processId(),pid)
        self.assertEqual(self.window._resident_load_count,1)

    def test_confirmation_defaults_to_cancel_and_leaves_labels(self):
        self.window.images.selectAll()
        with patch('batch_sam3.valid_repository',return_value=True),patch('batch_sam3.valid_checkpoint',return_value=True),patch.object(QMessageBox,'exec',return_value=0),patch.object(self.window,'_start_selected_batch') as start:
            self.window.run_sam3_selected();start.assert_not_called()
        self.assertEqual(self.window.project.data['annotations'],self.original)

    def test_unified_button_routes_selection_and_updates_count(self):
        self.window.sam_loaded_flag=True
        self.window.images.clearSelection()
        self.window.images.item(0).setSelected(True)
        self.assertTrue(self.window.run_sam_btn.isEnabled())
        self.assertIn('1',self.window.run_sam_btn.text())
        with patch.object(self.window,'run_sam3') as single,patch.object(self.window,'run_sam3_selected') as batch:
            self.window.run_sam_btn.click();single.assert_called_once();batch.assert_not_called()
            self.window.images.item(2).setSelected(True)
            self.assertIn('2',self.window.run_sam_btn.text())
            self.window.run_sam_btn.click();batch.assert_called_once()
        self.window.set_prompt_mode('mask')
        self.assertFalse(self.window.run_sam_btn.isEnabled())
        self.window.set_prompt_mode('text')
        self.assertTrue(self.window.run_sam_btn.isEnabled())
        self.window.images.clearSelection()
        self.assertFalse(self.window.run_sam_btn.isEnabled())

    def test_english_setup_gpu_test_removed(self):
        with patch.object(EnglishMainWindow,'_update_sam_labels'):
            window=EnglishMainWindow()
        self.assertEqual(len([b for b in window.findChildren(QPushButton) if b.text().startswith('Run SAM3 (')]),1)
        self.assertFalse(hasattr(window,'batch_sam_btn'))
        self.assertFalse(hasattr(window,'sam_setup_btn'))
        self.assertFalse(hasattr(window,'configure_sam3'))
        window.project.dirty=False;window.close()


if __name__=='__main__':unittest.main()
