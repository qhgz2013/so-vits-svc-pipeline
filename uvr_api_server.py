# make UVR GUI supports external HTTP requests (for process automation)
# prerequisite:
#   1. UVR 5.5.1 from https://github.com/Anjok07/ultimatevocalremovergui/tree/master
#   2. flask from: pip install flask
# usage:
#   put this file into uvr gui code directory, then run with command: python <this_file>.py
#   for API samples, refers to uvr_api_client.py
# external calls:
#   task enqueue:
#     GET http://{FLASK_HTTP_HOST}:{FLASK_HTTP_PORT}/create?{"input_file":"...","output_file":"...","env":{...}}
#     response: {"final_output_file":null,"request":{"input_file":[...],"output_file":"...",...},"log":"...","state":1,"task_id":"uuid4 string"}
#     NOTE: task_id is required for querying task status
#           state: -1: FAILED, 0: SUCCESS, 1: PENDING, 2: RUNNING
#   task query:
#     GET http://{FLASK_HTTP_HOST}:{FLASK_HTTP_PORT}/query?{"task_id":"uuid4 string"}
#     response: same as task enqueue
#     NOTE: the app maintains an LRU dict for completed tasks, the task results will reserve for a while before erased
from UVR import MainWindow
import UVR
from flask import Flask, request
from typing import *
from enum import IntEnum
import os
from collections import deque
from uuid import uuid4
from threading import Event, Lock, Thread
from io import StringIO
from functools import partial
from time import sleep
import json
from urllib.parse import unquote
import re
from shutil import rmtree
import uvr_api as api
import traceback


app = Flask(__name__)
FINAL_OUTPUT_TYPE = Optional[Union[Dict[str, str], str]]
stem_ptn = re.compile(r'^(.+)_\((\w+)\)(\.\w+)$')
# cspell:disable-next-line
temp_ensemble_dirname = re.compile(r'^Ensembled_Outputs_\d+$')
_default = object()
FLASK_HTTP_HOST = 'localhost'
FLASK_HTTP_PORT = 8090

# queue settings
MAX_QUEUE_SIZE = 100
MAX_FINISHED_TASKS = 5000
MAX_FILES_PER_TASK = 10
# deque op:
# cspell:ignoreRegExp (pop|append)(left|right)


class UVRCallState(IntEnum):
    FAILED = -1
    SUCCESS = 0
    SCHEDULING = 1
    RUNNING = 2


class TaskContext:
    def __init__(self, request: Optional[api.UVRRequest], task_id: Optional[str] = None,
                 task_state: UVRCallState = UVRCallState.SCHEDULING, task_log: str = ''):
        self.request = request
        # the final output file path may be differ from requested file path (as handled by UVR itself)
        self.final_output_file = None
        if task_id is None:
            task_id = uuid4().hex
        self.task_id = task_id
        self.task_wait = Event()
        self.task_log = task_log
        self.task_state = task_state

    def to_json(self):
        return {
            'request': json.loads(api.serialize_request_json(self.request)) if self.request is not None else None,
            'final_output_file': self.final_output_file,
            'task_id': self.task_id,
            'task_log': self.task_log,
            'task_state': self.task_state.value
        }


class FileChangeWatcher:
    def __init__(self, bind_dir: str, recursive: bool = False, include_dirs: bool = False):
        self.bind_dir = bind_dir
        assert os.path.isdir(bind_dir), f'{bind_dir} is not a directory'
        self.recursive = recursive
        self.include_dirs = include_dirs
        self.file_mtime_dict = {}
        self.dir_mtime_dict = {}
        self.update_file_mtime()

    def update_file_mtime(self, file_dict=None, dir_dict=None):
        if file_dict is None:
            file_dict = self.file_mtime_dict
        if dir_dict is None:
            dir_dict = self.dir_mtime_dict
        for path, dirs, files in os.walk(self.bind_dir):
            for file in files:
                p = os.path.join(path, file)
                st = os.stat(p)
                file_dict[p] = st.st_mtime
            if self.include_dirs:
                for d in dirs:
                    p = os.path.join(path, d)
                    st = os.stat(p)
                    dir_dict[p] = st.st_mtime
            if not self.recursive:
                break

    def __enter__(self):
        self.update_file_mtime()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def generate_diff_files(self) -> Tuple[List[str], List[str]]:
        # returns [files], [dirs]
        new_files = {}
        new_dirs = {}
        self.update_file_mtime(new_files, new_dirs)
        diff_files = []
        diff_dirs = []
        for file, mtime in new_files.items():
            old_mtime = self.file_mtime_dict.get(file, None)
            if old_mtime is None or mtime > old_mtime:
                diff_files.append(file)
        if self.include_dirs:
            for d, mtime in new_dirs.items():
                old_mtime = self.dir_mtime_dict.get(d, None)
                if old_mtime is None or mtime > old_mtime:
                    diff_dirs.append(d)
        self.file_mtime_dict = new_files
        self.dir_mtime_dict = new_dirs
        return diff_files, diff_dirs


class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}

    def get(self, key, default=_default):
        try:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        except KeyError as e:
            if default is _default:
                raise e
            return default

    def set(self, key, value):
        try:
            self.cache.pop(key)
        except KeyError:
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
        self.cache[key] = value


def _set_ctx_failed(ctx: TaskContext, ex: str):
    ctx.task_state = UVRCallState.FAILED
    ctx.task_log = ex


class MainWindowOverwrite(MainWindow):
    def __init__(self):
        super().__init__()
        self.task_queue = deque()
        self.task_pending_ctx_dict = {}  # type: Dict[str, TaskContext]
        self.task_finished_ctx_dict = LRUCache(MAX_FINISHED_TASKS)
        self.error_dialog_content = None
        self.task_log = StringIO()
        self.task_log_mutex = Lock()
        self.task_dict_mutex = Lock()
        self.current_running_task_id = None
        self.wait_event = Event()
        # log write
        self._orig_command_Text_write = self.command_Text.write
        self.command_Text.write = self._intercept_log_write
        self.work_thd = Thread(target=self._work_routine, daemon=True, name='UVRWorker')
        self.wait_event.set()  # set to idle
        # self.gui_set_finished = Event()
        # self.gui_set_finished.set()

    def _intercept_log_write(self, text):
        with self.task_log_mutex:
            self._orig_command_Text_write(text)
            # print(f'[log] >>>{text}<<<')
            self.task_log.write(text)

    # cspell:disable-next-line
    def error_dialoge(self, message):
        # ignore error pop up dialog
        if isinstance(message, (tuple, list)):
            message = ':'.join(message)
        # cspell:disable-next-line
        print(f'error_dialoge: {message}')
        self.error_dialog_content = message

    def process_initialize(self):
        self.wait_event.clear()
        return super().process_initialize()

    def process_end(self, error=None):
        if self.current_running_task_id is None:
            # the running task is from gui, not from request queue?
            print('process_end(): no running task id found')
            return super().process_end(error=error)
        # only set to FAILED here, since it is not finished yet
        context = self.task_pending_ctx_dict[self.current_running_task_id]
        if error is not None:
            context.task_state = UVRCallState.FAILED
        context.task_log = self.task_log.getvalue()
        # print(f'task [{context.task_id}] completed: [state {context.task_state}]')
        # print(f'task [{context.task_id}] log:\n{context.task_log}')
        super().process_end(error=None)
        # call finished
        self.wait_event.set()

    # <---- EXTERNAL calls

    def accept_uvr_task_requests(self):
        self.work_thd.start()
        print('accept external requests...')

    def run_uvr_task(self, context: TaskContext):
        input_file = context.request.input_file
        output_file = context.request.output_file
        if len(input_file) <= 0:
            _set_ctx_failed(context, 'Empty input paths')
            return
        elif len(input_file) > MAX_FILES_PER_TASK:
            _set_ctx_failed(context, f'Too many input files, max files per task: {MAX_FILES_PER_TASK}')
            return
        if os.path.exists(output_file):
            if os.path.isdir(output_file):
                # output_file exist -> directory: write to target directory
                output_dir = output_file
                should_rename = False  # if should_rename == False, check target directory and update new filename
            else:
                # output_file exist -> directory: write to parent directory and rename
                output_dir = os.path.abspath(os.path.join(output_file, '..'))
                should_rename = True
        else:
            # output_file not exist, ensure parent dir is presented
            output_dir = os.path.abspath(os.path.join(output_file, '..'))
            should_rename = True
            if not os.path.isdir(output_dir):
                _set_ctx_failed(context, 'output_dir should be a directory')
                return
        # unify path separators
        input_file = [x.replace('\\', '/') for x in input_file]
        output_file = output_file.replace('\\', '/')
        output_dir = output_dir.replace('\\', '/')

        print(f'output_dir: {output_dir}\nshould_rename: {should_rename}')

        # <-- step 0: setting up context
        self.error_dialog_content = None
        self.current_running_task_id = context.task_id
        self.task_log = StringIO()
        watcher = FileChangeWatcher(output_dir, include_dirs=True)

        # <-- step 1: setting up environments
        if not context.request.check_setup_prerequisite(self):
            frame, msg = api.get_last_check_failed_source()
            _set_ctx_failed(context, f'check_setup_prerequisite failed: {msg} ({frame.filename}:{frame.lineno})')
            return
        context.request.setup(self)
        if self.export_path_var.get() != output_dir:
            self.export_path_var.set(output_dir)

        # <-- step 2: run process call
        print(f'task [{context.task_id}]: process_initialize()')
        self.process_initialize()
        # check initialization result
        if self.error_dialog_content is not None:
            _set_ctx_failed(context, self.error_dialog_content)
            return

        # <-- step 3: await task finish
        print(f'task [{context.task_id}]: wait for complete')
        while not self.wait_event.wait(1):
            pass

        # <-- step 4: check saved file and rename if necessary
        if context.task_state != UVRCallState.FAILED:
            diff_files, diff_dirs = watcher.generate_diff_files()
            print(f'diff_files: {diff_files}, diff_dirs: {diff_dirs}')
            if len(diff_files) == 0:
                context.task_state = UVRCallState.FAILED
                context.task_log += 'ERROR: No new file produced'
                return
            output_files = {}
            if should_rename:
                dst_base_name, ext = os.path.splitext(output_file)
                for diff_file in diff_files:
                    match = re.match(stem_ptn, os.path.basename(diff_file))
                    if match is None:
                        print(f'Could not determine file {diff_file}, maybe it is not generated by UVR')
                        continue
                    stem = match.group(2)
                    src_ext = match.group(3)
                    dst_name = f'{dst_base_name}_({stem}){src_ext}'
                    if diff_file != dst_name:
                        if os.path.exists(dst_name):
                            print(f'remove: {dst_name}')
                            os.remove(dst_name)
                        print(f'rename: {diff_file} -> {dst_name}')
                        os.rename(diff_file, dst_name)
                    output_files[stem] = dst_name
            else:
                for diff_file in diff_files:
                    match = re.match(stem_ptn, os.path.basename(diff_file))
                    if match is None:
                        print(f'Could not determine file {diff_file}, maybe it is not generated by UVR')
                        continue
                    stem = match.group(2)
                    output_files[stem] = diff_file
            context.final_output_file = output_files
            print(f'final_output_file: {context.final_output_file}')

        # <-- step 5: cleaning up ensemble outputs
        if not self.is_save_all_outputs_ensemble_var.get():
            for d in diff_dirs:
                match = re.match(temp_ensemble_dirname, os.path.basename(d))
                if match is None:
                    continue
                print(f'removing temporary ensemble directory: {d}')
                rmtree(d)

    def confirm_stop_process(self):
        super().confirm_stop_process()
        if self.is_process_stopped:
            self.current_running_task_id = None
            self.wait_event.set()

    def _work_routine(self):
        while True:
            try:
                with self.task_dict_mutex:
                    ctx = self.task_queue.popleft()
            except IndexError:
                sleep(0.1)
                continue

            if ctx.task_state != UVRCallState.SCHEDULING:
                continue
            try:
                # check process thread is active or not
                # normally this wait loop is skipped if the task is not triggered manually
                first_log = True
                while not self.wait_event.wait(1):
                    if first_log:
                        print('work thread: waiting previous task')
                        first_log = False
                ctx.task_state = UVRCallState.RUNNING
                print(f'Running task [{ctx.task_id}] [input:{ctx.request.input_file}] [output: {ctx.request.output_file}]')
                self.run_uvr_task(ctx)
                if ctx.task_state == UVRCallState.RUNNING:
                    ctx.task_state = UVRCallState.SUCCESS
            except Exception as ex:
                print(ex)
                traceback.print_exc()
                ctx.task_state = UVRCallState.FAILED
                ctx.task_log += str(ex)
            ctx.task_wait.set()
            with self.task_dict_mutex:
                if ctx.task_id in self.task_pending_ctx_dict:
                    del self.task_pending_ctx_dict[ctx.task_id]
                self.task_finished_ctx_dict.set(ctx.task_id, ctx)
            self.current_running_task_id = None

    def _enqueue_task_no_throw(self, request) -> Union[TaskContext, Exception]:
        try:
            ctx = self._enqueue_task(request)
            return ctx
        except Exception as e:
            return e

    def _enqueue_task(self, request):
        if len(self.task_queue) >= MAX_QUEUE_SIZE:
            raise RuntimeError('Task queue full, please retry later')
        ctx = TaskContext(request)
        print(f'Add task {ctx.task_id}')
        with self.task_dict_mutex:
            self.task_pending_ctx_dict[ctx.task_id] = ctx
            self.task_queue.append(ctx)
        return ctx

    def enqueue_task(self, request: api.UVRRequest) -> TaskContext:
        res = self._enqueue_task_no_throw(request)
        if isinstance(res, Exception):
            return TaskContext(request, task_state=UVRCallState.FAILED)
        return res

    def enqueue_task_json(self, request_json: dict) -> TaskContext:
        try:
            request = api.deserialize_request_json(request_json)
            return self.enqueue_task(request)
        except Exception as ex:
            print(ex)
            ctx = TaskContext(request=None, task_state=UVRCallState.FAILED,
                              task_log=f'deserialize request json failed: {ex}')
            return ctx

    def check_task_json(self, request_json: dict) -> TaskContext:
        return self.check_task(request_json['task_id'])

    def check_task(self, task_id: str) -> TaskContext:
        with self.task_dict_mutex:
            ctx = self.task_pending_ctx_dict.get(task_id, None)
            if ctx is not None:
                if ctx.task_id == self.current_running_task_id:
                    ctx.task_log = self.task_log.getvalue()
                return ctx
            ctx = self.task_finished_ctx_dict.get(task_id, None)
            if ctx is not None:
                return ctx
            ctx = TaskContext(request=None, task_id=task_id, task_state=UVRCallState.FAILED, task_log='Task not found')
            return ctx

    def list_tasks(self) -> List[str]:
        """Returns a list of task ids."""
        task_id_set = set()
        with self.task_dict_mutex:
            task_id_set.update(self.task_pending_ctx_dict.keys())
            task_id_set.update(self.task_finished_ctx_dict.cache.keys())
        return list(task_id_set)

    def route_run_requests(self):
        app.add_url_rule('/create', 'create', view_func=lambda: _route_request(self.enqueue_task_json))
        app.add_url_rule('/query', 'query', view_func=lambda: _route_request(self.check_task_json))


def _route_request(uvr_fn):
    q = unquote(request.query_string.decode())
    json_data = json.loads(q)
    res = uvr_fn(json_data)
    if isinstance(res, TaskContext):
        return res.to_json()
    return res


def run_app_non_blocking(*args, **kwargs):
    run_fn = partial(app.run, *args, **kwargs)
    thd = Thread(target=run_fn, daemon=True, name='FlaskThread')
    thd.start()


def main():
    try:
        from ctypes import windll, wintypes
        windll.user32.SetThreadDpiAwarenessContext(wintypes.HANDLE(-1))
    except Exception as e:
        if UVR.OPERATING_SYSTEM == 'Windows':
            print(e)
    uvr = MainWindowOverwrite()
    setattr(UVR, 'root', uvr)
    uvr.update_checkbox_text()
    uvr.accept_uvr_task_requests()
    uvr.route_run_requests()
    run_app_non_blocking(FLASK_HTTP_HOST, FLASK_HTTP_PORT)
    uvr.mainloop()


if __name__ == '__main__':
    main()
