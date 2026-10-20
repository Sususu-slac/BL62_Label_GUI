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
import argparse,json,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('--batch');p.add_argument('--output');a,_=p.parse_known_args()
for i,path in enumerate(json.loads(Path(a.batch).read_text())):
    if MODE == 'failure' and i==1:sys.exit(1)
    if MODE == 'cancel' and i==1:time.sleep(30)
    with Image.open(path) as image:w,h=image.size
    masks=np.zeros((1,h,w),bool);masks[:,4:10,5:11]=True
    if MODE == 'empty':masks=masks[:0]
    np.savez(a.output,masks=masks,scores=np.ones(len(masks)))
    print(json.dumps({'type':'result','index':i}),flush=True)
    if sys.stdin.readline().strip()!='next':break
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
        self.patches=[patch('batch_sam3.QMessageBox.information'),patch('batch_sam3.QMessageBox.critical')]
        for p in self.patches:p.start()

    def tearDown(self):
        if getattr(self.window,'_batch_process',None) is not None:
            self.window.cancel_selected_batch();self.wait()
        self.window.project.dirty=False;self.window.close()
        for p in self.patches:p.stop()
        self.temp.cleanup()

    def start(self, mode):
        worker=self.root/'fake.py'
        worker.write_text('MODE='+repr(mode)+'\n'+FAKE_WORKER)
        original=QProcess.setArguments
        def args(process, arguments):return original(process,[str(worker)]+arguments[1:])
        with patch.object(QProcess,'setArguments',args):
            self.window._start_selected_batch([0,2],'particle')

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
