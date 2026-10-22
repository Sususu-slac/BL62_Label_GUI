"""One cancellable, persistent SAM3 subprocess shared by all GUI inference jobs."""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from PySide6.QtCore import QProcess, Qt
from PySide6.QtWidgets import QMessageBox, QProgressDialog
from sam3_setup import valid_checkpoint, valid_repository


class ResidentSam3Mixin:
    def _model_key(self):
        checkpoint = Path(self.sam_checkpoint).resolve()
        stat = checkpoint.stat()
        return (str(Path(self.sam_repo).resolve()), str(checkpoint), stat.st_size, stat.st_mtime_ns)

    def _resident_request(self, request, success, failure, progress):
        if getattr(self, '_resident_success', None) is not None:
            raise RuntimeError('A SAM3 request is already running')
        if not valid_repository(self.sam_repo) or not valid_checkpoint(self.sam_checkpoint):
            raise ValueError(self._batch_text('Select a valid SAM3 repository and checkpoint first.', '请先选择有效的 SAM3 仓库和 checkpoint。'))
        key = self._model_key()
        process = getattr(self, '_resident_process', None)
        if process is not None and (key != self._resident_key or process.state() == QProcess.NotRunning):
            self._stop_resident()
            process = None
        self._resident_success, self._resident_failure = success, failure
        self._resident_progress = progress
        self._resident_pending = request
        self._resident_counter = getattr(self, '_resident_counter', 0) + 1
        self._resident_request_id = self._resident_counter
        if process is not None and self.sam_loaded_flag:
            self._send_resident_request()
            return
        self._resident_stdout = ''
        self._resident_stderr = ''
        self._resident_key = key
        process = self._resident_process = QProcess(self)
        process.setProgram(str(Path(sys.executable).with_name('python.exe')) if os.name == 'nt' else sys.executable)
        process.setArguments([str(Path(__file__).with_name('sam3_inference_process.py')), '--serve',
            '--repo', self.sam_repo, '--checkpoint', self.sam_checkpoint, '--confidence', '0.5'])
        process.readyReadStandardOutput.connect(lambda: self._resident_read(process))
        process.readyReadStandardError.connect(lambda: self._resident_stderr_read(process))
        process.finished.connect(lambda code, status: self._resident_exited(process, code))
        process.errorOccurred.connect(lambda error: self._resident_process_error(process, error))
        process.start()

    def _send_resident_request(self):
        if self._resident_pending is None:
            self._resident_complete(None)
        else:
            payload = dict(self._resident_pending, id=self._resident_request_id)
            self._resident_process.write((json.dumps(payload, ensure_ascii=True) + '\n').encode('utf-8'))

    def _resident_read(self, process):
        if process is not getattr(self, '_resident_process', None):
            return
        self._resident_stdout += bytes(process.readAllStandardOutput()).decode('utf-8', errors='replace')
        lines = self._resident_stdout.split('\n')
        self._resident_stdout = lines.pop()
        for line in lines:
            if process is not getattr(self, '_resident_process', None):
                break
            try:
                event = json.loads(line)
            except ValueError:
                continue
            kind = event.get('type')
            if kind == 'ready':
                self.sam_loaded_flag = True
                self._resident_load_count = getattr(self, '_resident_load_count', 0) + 1
                self._send_resident_request()
            elif kind == 'progress':
                if getattr(self, '_resident_progress', None):
                    self._resident_progress(event.get('message', 'SAM3…'))
            elif kind in ('result', 'error'):
                if event.get('id') not in (None, self._resident_request_id):
                    self._resident_abort('Unexpected SAM3 result ID')
                    break
                if kind == 'error':
                    self._resident_abort(event.get('message', 'SAM3 inference failed'))
                    break
                self._resident_complete(event)

    def _resident_stderr_read(self, process):
        if process is getattr(self, '_resident_process', None):
            self._resident_stderr = (self._resident_stderr + bytes(process.readAllStandardError()).decode('utf-8', errors='replace'))[-8000:]

    def _resident_complete(self, event):
        callback = self._resident_success
        self._resident_success = self._resident_failure = self._resident_progress = None
        self._resident_pending = None
        if callback:
            callback(event)

    def _resident_process_error(self, process, error):
        if process is getattr(self, '_resident_process', None) and error == QProcess.FailedToStart:
            self._resident_abort(process.errorString())

    def _resident_exited(self, process, code):
        if process is not getattr(self, '_resident_process', None):
            return
        self._resident_stderr_read(process)
        self._resident_abort(self._resident_stderr or f'SAM3 worker exited ({code}); run again to reload.')

    def _stop_resident(self):
        process = getattr(self, '_resident_process', None)
        self._resident_process = None
        self._resident_key = None
        self.sam_loaded_flag = False
        if process is not None:
            if process.state() != QProcess.NotRunning:
                process.kill()
                process.waitForFinished(3000)
            process.deleteLater()

    def _resident_abort(self, message):
        callback = getattr(self, '_resident_failure', None)
        self._resident_success = self._resident_failure = self._resident_progress = None
        self._resident_pending = None
        self._stop_resident()
        if callback:
            callback(message)
        else:
            self.sam_status.setText(message)
            self.update_run_button()

    def release_sam3(self):
        if self.sam_busy:
            return
        self._stop_resident()
        self.update_run_button()
        self.sam_status.setText(self._batch_text('GPU released. SAM3 will reload on the next run.', '已释放 GPU，下次运行时会重新加载 SAM3。'))

    def sam_configuration_changed(self):
        was_loaded = self.sam_loaded_flag
        self._stop_resident()
        self._update_sam_labels()
        self.update_run_button()
        if was_loaded:
            self.load_resident_sam3()

    def _sam_dialog(self, title, cancel):
        dialog = QProgressDialog(title, self._batch_text('Cancel', '取消'), 0, 0, self)
        dialog.setWindowTitle('SAM3')
        dialog.setWindowModality(Qt.ApplicationModal)
        dialog.setMinimumDuration(0)
        dialog.setAutoClose(False)
        dialog.canceled.connect(cancel)
        dialog.show()
        return dialog

    def _sam_job_progress(self, message):
        self.sam_status.setText(message)
        dialog = getattr(self, '_sam_job_dialog', None)
        if dialog is not None:
            dialog.setLabelText(message)

    def load_resident_sam3(self):
        if self.sam_busy:
            return
        self.sam_busy = True
        self.update_run_button()
        self._sam_job_dialog = self._sam_dialog(self._batch_text('Loading SAM3…', '正在加载 SAM3……'), self.cancel_resident_task)
        try:
            self._resident_request(None, self._resident_loaded, self._single_failed, self._sam_job_progress)
        except Exception as exc:
            self._single_failed(str(exc))

    def _resident_loaded(self, _event):
        self._close_sam_job()
        self.sam_status.setText(self._batch_text('SAM3 is resident on GPU and ready.', 'SAM3 已常驻 GPU，可以运行。'))

    def _close_sam_job(self):
        dialog = getattr(self, '_sam_job_dialog', None)
        self._sam_job_dialog = None
        if dialog is not None:
            dialog.canceled.disconnect()
            dialog.close()
            dialog.deleteLater()
        temp = getattr(self, '_single_temp', None)
        self._single_temp = None
        if temp is not None:
            temp.cleanup()
        self.sam_busy = False
        self.update_run_button()

    def cancel_resident_task(self):
        self._resident_abort('cancelled')

    def start_resident_single(self, prompt, append_only=False):
        if self.sam_busy:
            return
        self.sam_busy = True
        self.update_run_button()
        self._single_index = self.index
        self._single_append_only = append_only
        if append_only:
            self._sam_replace_requested = False
            self._sam_replace_box = None
        self._sam_job_dialog = self._sam_dialog(self._batch_text('Running SAM3…', '正在运行 SAM3……'), self.cancel_resident_task)
        try:
            self._single_temp = tempfile.TemporaryDirectory(prefix='bl62_sam3_single_')
            folder = Path(self._single_temp.name)
            image_path = self.project.image_paths[self._single_index]
            image_data = self.image_cache.get(image_path)
            sam_image = str(image_data.sam_path)
            request = {'image': sam_image, 'output': str(folder/'result.npz'),
                'prompt': prompt, 'confidence': self.confidence.value()/100,
                'box': self.view.sam_box, 'mode': 'instance' if self.prompt_mode == 'mask' else 'text'}
            if self.prompt_mode == 'mask':
                np.save(folder/'exemplar.npy', self.mask_prompt, allow_pickle=False)
                request['exemplar'] = str(folder/'exemplar.npy')
            self._resident_request(request, self._single_completed, self._single_failed, self._sam_job_progress)
        except Exception as exc:
            self._single_failed(str(exc))

    def _single_completed(self, _event):
        try:
            if self.index != self._single_index:
                raise ValueError('The current image changed during inference')
            with np.load(Path(self._single_temp.name)/'result.npz', allow_pickle=False) as result:
                masks, scores = result['masks'], result['scores']
                if masks.ndim != 3 or masks.shape[1:] != self.view.array.shape[:2] or len(masks) != len(scores):
                    raise ValueError('SAM3 result dimensions do not match the image')
                if self._single_append_only and not len(masks):
                    self.sam_status.setText(self._batch_text("No masks found; existing labels kept.","未找到实例，原标签已保留。"))
                else:
                    self.sam_predicted(masks, scores.tolist())
            self._close_sam_job()
        except Exception as exc:
            self._single_failed(str(exc))

    def _single_failed(self, message):
        self._sam_replace_requested = False
        self._sam_replace_box = None
        self._close_sam_job()
        if message == 'cancelled':
            self.sam_status.setText(self._batch_text('Cancelled. GPU released; the next run will reload SAM3.', '已取消并释放 GPU；下次运行会重新加载 SAM3。'))
        else:
            self.sam_status.setText(message)
            QMessageBox.critical(self, 'SAM3', message)
