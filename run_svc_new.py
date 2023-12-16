# v2.3.15 branch of https://www.yuque.com/umoubuton/ueupp5
# rev 7: code refactor
import argparse
import os
from glob import glob
import subprocess
import shutil
from itertools import chain
from time import sleep, time
import json
from typing import *
import uvr_api_client as api
import requests
import threading
from warnings import warn


# cspell:ignore vits,adelay,flac,amerge,channelsplit,getpid,ffprobe,aformat,workenv,libmp

def build_vits_args(config, ext: str, spk_name: str, input_file: str, profile_name: str = 'default') -> List[str]:
    vits_cfg = config['vits']['profile'][profile_name]
    if ext.startswith('.'):
        ext = ext[1:]
    vits_py_executable = os.path.join(config['vits']['src_dir'], 'workenv', 'python')
    vits_args = [vits_py_executable, 'inference_main.py',
                 '--model_path', vits_cfg['model_path'],
                 '--config_path', vits_cfg['config_path'],
                 '--clip', str(vits_cfg['clip']),
                 '--clean_names', os.path.basename(input_file)]
    if len(vits_cfg['trans']) > 0:
        vits_args.append('--trans')
        for trans in vits_cfg['trans']:
            vits_args.append(str(trans))
    vits_args.extend(['--spk_list', spk_name])
    if vits_cfg['auto_predict_f0']:
        vits_args.append('--auto_predict_f0')
    if vits_cfg['cluster_model_path'] is not None:
        vits_args.extend(['--cluster_model_path', vits_cfg['cluster_model_path']])
    if vits_cfg['cluster_infer_ratio'] is not None:
        vits_args.extend(['--cluster_infer_ratio', str(vits_cfg['cluster_infer_ratio'])])
    if vits_cfg['linear_gradient'] is not None:
        vits_args.extend(['--linear_gradient', str(vits_cfg['linear_gradient'])])
    if vits_cfg['f0_predictor'] is not None:
        vits_args.extend(['--f0_predictor', vits_cfg['f0_predictor']])
    if vits_cfg['enhance']:
        vits_args.append('--enhance')
    if vits_cfg['shallow_diffusion']:
        vits_args.append('--shallow_diffusion')
    if vits_cfg['use_spk_mix']:
        vits_args.append('--use_spk_mix')
    if vits_cfg['loudness_envelope_adjustment'] is not None:
        vits_args.extend(['--loudness_envelope_adjustment', str(vits_cfg['loudness_envelope_adjustment'])])
    if vits_cfg['feature_retrieval']:
        vits_args.append('--feature_retrieval')
    if vits_cfg['diffusion_model_path'] is not None:
        vits_args.extend(['--diffusion_model_path', vits_cfg['diffusion_model_path']])
    if vits_cfg['diffusion_config_path'] is not None:
        vits_args.extend(['--diffusion_config_path', vits_cfg['diffusion_config_path']])
    if vits_cfg['k_step'] is not None:
        vits_args.extend(['--k_step', str(vits_cfg['k_step'])])
    if vits_cfg['second_encoding']:
        vits_args.append('--second_encoding')
    if vits_cfg['only_diffusion']:
        vits_args.append('--only_diffusion')
    if vits_cfg['slice_db'] is not None:
        vits_args.extend(['--slice_db', str(vits_cfg['slice_db'])])
    if vits_cfg['device'] is not None:
        vits_args.extend(['--device', vits_cfg['device']])
    if vits_cfg['pad_seconds'] is not None:
        vits_args.extend(['--pad_seconds', str(vits_cfg['pad_seconds'])])
    vits_args.extend(['--wav_format', ext])
    if vits_cfg['linear_gradient_retain'] is not None:
        vits_args.extend(['--linear_gradient_retain', str(vits_cfg['linear_gradient_retain'])])
    if vits_cfg['enhancer_adaptive_key'] is not None:
        vits_args.extend(['--enhancer_adaptive_key', str(vits_cfg['enhancer_adaptive_key'])])
    if vits_cfg['f0_filter_threshold'] is not None:
        vits_args.extend(['--f0_filter_threshold', str(vits_cfg['f0_filter_threshold'])])
    return vits_args


class FileRegistration:
    def __init__(self, work_dir: str, keep_temp_file: bool = False) -> None:
        self.work_dir = os.path.join(work_dir, str(os.getpid()))
        os.makedirs(self.work_dir, exist_ok=True)
        self.files = dict()
        self.temp_file = set()
        self.keep_temp_file = keep_temp_file
        self.daemon_thread = threading.Thread(target=self._daemon_thread_callback, name='FileRegistrationDaemon',
                                              daemon=False)
        self.daemon_thread.start()

    def register_file(self, file_id: str, path: str, delete_after_finish: bool = False) -> str:
        self.files[file_id] = path
        if delete_after_finish:
            self.temp_file.add(file_id)
        return path

    def register_tmp_file_in_work_dir(self, file_id: str, ext: str, delete_after_finish: bool = True) -> str:
        return self.register_file(file_id, os.path.join(self.work_dir, file_id + ext), delete_after_finish)

    def get_file(self, file_id: str) -> Optional[str]:
        return self.files.get(file_id, None)

    def __contains__(self, key: str) -> bool:
        return key in self.files

    def _daemon_thread_callback(self):
        while threading.main_thread().is_alive():
            sleep(0.5)
        if not self.keep_temp_file:
            self.delete_temp_file()

    def delete_temp_file(self):
        for file_id in self.temp_file:
            if file_id in self.files and os.path.exists(self.files[file_id]):
                os.remove(self.files[file_id])
        if os.path.exists(self.work_dir):
            shutil.rmtree(self.work_dir)


def split_audio_channel(config: dict, file_id: str, file_reg: FileRegistration, ext: Optional[str] = None):
    # ffmpeg -i stereo.wav -filter_complex "[0:a]channelsplit=channel_layout=stereo[left][right]"
    # -map "[left]" left.wav -map "[right]" right.wav
    file_in = file_reg.get_file(file_id)
    assert file_in is not None
    if ext is None:
        ext = os.path.splitext(file_in)[1]
    file_out_l = file_reg.register_tmp_file_in_work_dir(file_id + '#L', ext)
    file_out_r = file_reg.register_tmp_file_in_work_dir(file_id + '#R', ext)
    ffmpeg_args = [config['ffmpeg']['ffmpeg_path'], '-y', '-i', file_in,
                   '-filter_complex', '[0]channelsplit=channel_layout=stereo[l][r]',
                   '-map', '[l]', file_out_l, '-map', '[r]', file_out_r]
    print(f'Running FFMpeg with args: {ffmpeg_args}')
    proc = subprocess.Popen(ffmpeg_args)
    if proc.wait() != 0:
        raise RuntimeError(f'FFMpeg exited with return code {proc.returncode}')


class ConfigCache:
    def __init__(self) -> None:
        self.cache = dict()

    def get_config(self, file: str) -> dict:
        if file not in self.cache:
            with open(file, 'r', encoding='utf8') as f:
                data = json.load(f)
            self.cache[file] = data
        return self.cache[file]


def execute_uvr_task(config: dict, input_file: str, output_file: Union[str, Dict[str, str]], env: dict, work_dir: str) -> bool:
    task_output_file = output_file if isinstance(output_file, str) else work_dir
    req = {'input_file': input_file, 'output_file': task_output_file, 'env': env}
    s = requests.session()
    url = config['uvr']['api_url']
    rsp = s.get(f'{url}create', params=json.dumps(req, separators=(',', ':')))
    if not rsp.ok:
        return False
    task = api.deserialize_response_json(rsp.json())
    log_idx = 0
    while task.task_state > 0:
        sleep(1)
        rsp = s.get(f'{url}query', params=json.dumps({'task_id': task.task_id}, separators=(',', ':')))
        if not rsp.ok:
            return False
        task = api.deserialize_response_json(rsp.json())
        if len(task.task_log) > log_idx:
            print(task.task_log[log_idx:], end='')
            log_idx = len(task.task_log)
    print('')
    if task.task_state == api.UVRCallState.SUCCESS:
        if isinstance(output_file, str):
            final_file = next(iter(task.final_output_file.values()))
            if final_file != output_file:
                shutil.move(final_file, output_file)
        else:
            for stem, final_file in task.final_output_file.items():
                if stem not in output_file:
                    print(f'output_id for stem "{stem}" is not defined')
                    os.remove(final_file)
                    continue  # remove unnecessary output (not required in config)
                if final_file != output_file[stem]:
                    shutil.move(final_file, output_file[stem])
        return True


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


def check_diff_file(new_files: List[str], ext) -> str:
    new_sounds = list(filter(lambda x: x.endswith(ext), new_files))
    if len(new_sounds) > 1:
        raise RuntimeError('Found multiple sounds updated during inference')
    elif len(new_sounds) == 0:
        raise RuntimeError('No sound file changed in VITS result directory')
    return new_sounds[0]


def execute_vits_task(config: dict, file_reg: FileRegistration, ext: str, profile_name: str = 'default'):
    sp = config['down_mix']['separate_process']
    vocal_proc_list = config['down_mix']['vocal']
    for vocal_proc_entry in vocal_proc_list:
        input_id = vocal_proc_entry['id']
        vits_profile = vocal_proc_entry.get('profile', profile_name)
        temp_id = str(int(time()))
        proc_file_ids = set()
        vits_output_dir = os.path.join(config['vits']['src_dir'], 'results')
        watcher = FileChangeWatcher(vits_output_dir)
        for spk_name, mix_cfg_list in vocal_proc_entry['spk_list'].items():
            for mix_cfg in mix_cfg_list:
                if sp:
                    channel_list = mix_cfg["channel"]
                    if isinstance(channel_list, str):
                        channel_list = [channel_list]
                    assert all(channel in {'L', 'R'} for channel in channel_list)
                    src_path_list = [file_reg.get_file(f'{input_id}#{channel}') for channel in channel_list]
                    vits_in_file_id_list = [f'vits_in_{input_id}_{spk_name}_{temp_id}#{channel}' for channel in channel_list]
                    vits_out_file_id_list = [f'vits_out_{input_id}_{spk_name}#{channel}' for channel in channel_list]
                else:
                    src_path_list = [file_reg.get_file(input_id)]
                    vits_in_file_id_list = [f'vits_in_{input_id}_{spk_name}_{temp_id}']
                    vits_out_file_id_list = [f'vits_out_{input_id}_{spk_name}']
                for src_path, vits_in_file_id, vits_out_file_id in zip(src_path_list, vits_in_file_id_list, vits_out_file_id_list):
                    if vits_in_file_id in proc_file_ids:
                        continue  # already processed, duplicated task
                    proc_file_ids.add(vits_in_file_id)
                    assert src_path is not None
                    # copy to "raw" dir in so-vits-svc
                    new_src_path = os.path.join(config['vits']['src_dir'], 'raw', f'{vits_in_file_id}{ext}')
                    shutil.copy(src_path, new_src_path)
                    file_reg.register_file(vits_in_file_id, new_src_path, delete_after_finish=True)
                    with watcher:
                        vits_args = build_vits_args(config, ext, spk_name, new_src_path, profile_name=vits_profile)
                        print(f'Running VITS with args: {vits_args}')
                        proc = subprocess.Popen(vits_args, cwd=config['vits']['src_dir'])
                        if proc.wait() != 0:
                            raise RuntimeError(f'VITS exited with return code {proc.returncode}')
                        new_file = check_diff_file(watcher.generate_diff_files()[0], ext)
                        renamed_file = file_reg.register_tmp_file_in_work_dir(vits_out_file_id, ext)
                        shutil.move(new_file, renamed_file)


def down_mix(config: dict, file_reg: FileRegistration, ext: str, metadata: Optional[dict] = None):
    sp = config['down_mix']['separate_process']
    output_path = file_reg.get_file('output')
    inst_file_weight = {file_reg.get_file(x['id']): x['weight'] for x in config['down_mix']['inst']}
    assert output_path is not None
    spk_channel_dict = {'L': {}, 'R': {}}
    for vocal_proc_entry in config['down_mix']['vocal']:
        input_id = vocal_proc_entry['id']
        for spk_name, mix_cfg_list in vocal_proc_entry['spk_list'].items():
            for mix_cfg in mix_cfg_list:
                key_name = f'{input_id}_{spk_name}'
                if sp:
                    channel_list = mix_cfg["channel"]
                    if isinstance(channel_list, str):
                        channel_list = [channel_list]
                    assert all(channel in {'L', 'R'} for channel in channel_list)
                    for channel in channel_list:
                        vits_out_file_id = f'vits_out_{input_id}_{spk_name}#{channel}'
                        spk_channel_dict[channel][key_name] = {'file': file_reg.get_file(vits_out_file_id),
                                                               'weight': mix_cfg['weight'], 'delay': mix_cfg['delay']}
                else:
                    vits_out_file_id = f'vits_out_{input_id}_{spk_name}'
                    obj = {'file': file_reg.get_file(vits_out_file_id), 'weight': mix_cfg['weight'],
                           'delay': mix_cfg['delay']}
                    spk_channel_dict['L'][key_name] = obj
                    spk_channel_dict['R'][key_name] = obj
    # ffmpeg note: stream labels in filter graph can be used only once
    # ffmpeg -y -i <input1> ... -i <inst_file> -filter_complex <filter_cmd> -map [ao] -ac 2 <extra_args> <output>
    ffmpeg_args = ['ffmpeg', '-y']
    filter_cmd = []
    channel_join_nodes = []
    channel_cnt_layout_mapper = {1: 'mono', 2: 'stereo'}
    channel_cnt_layout_mapper_rev = {v: k for k, v in channel_cnt_layout_mapper.items()}
    i = 0
    for channel, spk_dict in spk_channel_dict.items():
        merge_input_nodes = []
        mix_weight = []
        for spk_name, spk_data in spk_dict.items():
            ffmpeg_args.extend(['-i', spk_data['file']])
            delay = spk_data['delay']
            if delay > 0:
                # for adelay filter, ffmpeg need to specify aformat at first, otherwise amerge will throw a format error
                fmt_json = get_metadata(config, spk_data['file'])
                sample_fmt = fmt_json['streams'][0]['sample_fmt']
                sample_rate = fmt_json['streams'][0]['sample_rate']
                channels = fmt_json['streams'][0].get('channels', None)
                if channels is None:
                    layout = fmt_json['streams'][0].get('channel_layout', None)
                    if layout is not None and layout.lower() in channel_cnt_layout_mapper_rev:
                        channels = channel_cnt_layout_mapper_rev[layout.lower()]
                if channels == 1 and not sample_fmt.endswith('p'):
                    sample_fmt += 'p'  # a buggy fix for s16 -> s16p in mono stream?
                layout = channel_cnt_layout_mapper[channels]
                filter_cmd.append(f'[{i}]aformat=sample_fmts={sample_fmt}:'
                                  f'sample_rates={sample_rate}:channel_layouts={layout}[s{i}]')
                filter_cmd.append(f'[s{i}]adelay={delay}[t{i}]')
                merge_input_nodes.append(f'[t{i}]')
            else:
                merge_input_nodes.append(f'[{i}]')
            i += 1
            mix_weight.append(spk_data['weight'])
        n_nodes = len(merge_input_nodes)
        # e.g., [a0][1]amerge=inputs=2,pan=mono|c0=0.5*c0+0.5*c1[vl]
        mix_channels = "+".join(f'{mix_weight[x]}*c{x}' for x in range(n_nodes))
        filter_cmd.append(f'{"".join(merge_input_nodes)}amerge=inputs={n_nodes},pan=mono|c0={mix_channels}[v{channel}]')
        channel_join_nodes.append(f'[v{channel}]')
    # join left and right vocal channel: vl, vr -> vo
    filter_cmd.append(f'{"".join(channel_join_nodes)}join=inputs=2:channel_layout=stereo[vo]')
    # merge vocal and instrumental track
    if len(inst_file_weight) > 0:
        inst_track_id_list = []
        inst_id_offset = i
        inst_c0_mix_str_list = []
        inst_c1_mix_str_list = []
        for i, (inst_file_path, mix_weight) in enumerate(inst_file_weight.items()):
            inst_track_id_list.append(f'[{i + inst_id_offset}]')
            ffmpeg_args.extend(['-i', inst_file_path])
            inst_c0_mix_str_list.append(f'{mix_weight}*c{(i+1)*2}')
            inst_c1_mix_str_list.append(f'{mix_weight}*c{(i+1)*2+1}')
        inst_track_id = ''.join(inst_track_id_list)
        inst_c0_mix_str, inst_c1_mix_str = '+'.join(inst_c0_mix_str_list), '+'.join(inst_c1_mix_str_list)
        filter_cmd.append(f'[vo]{inst_track_id}amerge=inputs={len(inst_file_weight)+1},'
                          f'pan=stereo|c0=c0+{inst_c0_mix_str}|c1=c1+{inst_c1_mix_str}[ao]')

    # add cover if exist
    cover_id_offset = -1
    if 'input_cover' in file_reg:
        cover_file = file_reg.get_file('input_cover')
        if os.path.isfile(cover_file):
            ffmpeg_args.extend(['-i', cover_file])
            cover_id_offset = inst_id_offset + len(inst_file_weight)
    ffmpeg_args += ['-filter_complex', ';'.join(filter_cmd), '-map', '[ao]' if len(inst_file_weight) > 0 else '[vo]', '-ac', '2']

    if cover_id_offset != -1:
        ffmpeg_args += ['-map', f'{cover_id_offset}:0', '-disposition:v', 'attached_pic']
    # metadata
    if metadata is not None and 'tags' in metadata.get('format', {}):
        tags = metadata['format']['tags']
        tags = {k.lower(): v for k, v in tags.items()}
        ffmpeg_tag_args = ['-id3v2_version', '4']  # use v2.4
        cfg_dump = json.dumps(config, separators=(',', ':'))
        ffmpeg_tag_args.extend(['-metadata', f'comment=Created by UVR pipeline {cfg_dump}'])
        candidate_keys = {'title', 'album', 'artist', 'track'}
        for key in candidate_keys:
            if key in tags:
                ffmpeg_tag_args.append('-metadata')
                ffmpeg_tag_args.append(f'{key}={tags[key]}')
        ffmpeg_args.extend(ffmpeg_tag_args)
    # extra arguments for output stream (s32 is required since s16 will overflow during mix)
    # -c:a pcm_s32le for wav
    # -c:a flac -sample_fmt s32 for flac
    # -c:a libmp3lame -sample_fmt s32p -b:a 320k for mp3
    if ext == '.wav':
        ffmpeg_args += ['-c:a', 'pcm_s32le']
    elif ext == '.flac':
        ffmpeg_args += ['-c:a', 'flac', '-sample_fmt', 's32']
    elif ext == '.mp3':
        ffmpeg_args += ['-c:a', 'libmp3lame', '-sample_fmt', 's32p', '-b:a', '320k']
    else:
        print(f'No available extra arguments for wav format "{ext}"')
    ffmpeg_args.append(output_path)
    print('Running FFMpeg with args', ffmpeg_args)
    proc = subprocess.Popen(ffmpeg_args)
    if proc.wait() != 0:
        raise RuntimeError(f'FFMpeg exited with return code {proc.returncode}')


def execute_uvr_pipeline(config: dict, file_reg: FileRegistration, ext: str):
    api_url = config['uvr']['api_url']
    if not api_url.endswith('/'):
        api_url = api_url + '/'
    pipeline_list = config['uvr']['pipeline'].copy()  # type: list
    uvr_config_cache = ConfigCache()
    while len(pipeline_list) > 0:
        for pipeline in pipeline_list:
            if all((x in file_reg if isinstance(x, str) else all(y in file_reg for y in x.values()))
                    for x in pipeline['output_id']):
                # outputs are already generated
                pipeline_list.pop(0)
                break
            if len(pipeline['input_id']) != len(pipeline['output_id']):
                raise ValueError(f'input_id[len={len(pipeline["input_id"])}] and output_id[len={len(pipeline["output_id"])}] length mismatch in config')
            config_source = pipeline['config_source']
            if config_source == 'file':
                uvr_config = uvr_config_cache.get_config(pipeline['config_file'])
            elif config_source == 'data':
                uvr_config = pipeline['config_data']
            else:
                raise ValueError(f'invalid config_source "{config_source}"')
            uvr_config['output_format'] = ext[1:].upper()
            should_pop_and_continue = True
            for input_id, output_id in zip(pipeline['input_id'], pipeline['output_id']):
                if input_id not in file_reg:
                    input_id_split = input_id.split('#')
                    # fallback mode: file_id#channel
                    if len(input_id_split) != 2 or input_id_split[0] not in file_reg or input_id_split[1] not in {'L', 'R'}:
                        should_pop_and_continue = False
                        break  # could not meet the input condition
                    if not config['down_mix']['separate_process']:
                        config['down_mix']['separate_process'] = True
                        warn('down_mix.separate_process is set to true due to "file_id#channel" extended mode in UVR pipeline')
                    split_audio_channel(config, input_id_split[0], file_reg, ext)
                output_id_list = [output_id] if isinstance(output_id, str) else list(output_id.values())
                if all(e in file_reg for e in output_id_list):
                    continue
                input_path = file_reg.get_file(input_id)
                if isinstance(output_id, str):
                    output_path = file_reg.register_tmp_file_in_work_dir(output_id, ext)
                else:
                    output_path = {k: file_reg.register_tmp_file_in_work_dir(v, ext) for k, v in output_id.items()}
                if not execute_uvr_task(config, input_path, output_path, uvr_config, file_reg.work_dir):
                    raise RuntimeError(f'Failed to execute UVR task: [input_path={input_path}] [output_path={output_path}]')
            if should_pop_and_continue:
                pipeline_list.pop(0)
                break


def parse_dot_args_to_dict(d: Union[list, dict], key: str, value: str) -> bool:
    key_split = key.split('.')
    for sect in key_split[:-1]:
        if len(sect) == 0:
            continue
        if isinstance(d, dict) and sect in d:
            d = d[sect]
        elif isinstance(d, list) and key.isdecimal() and 0 <= int(key) < len(d):
            d = d[int(key)]
        else:
            return False
    last_sect = key_split[-1]
    if isinstance(d, dict):
        d[last_sect] = value
        return True
    elif isinstance(d, list):
        d[int(last_sect)] = value
        return True
    return False


def get_metadata(config: dict, path: str) -> Dict[str, Any]:
    ffprobe = config['ffmpeg']['ffprobe_path']
    probe_args = [ffprobe, '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', path]
    print(f'Running ffprobe with args: {probe_args}')
    proc = subprocess.Popen(probe_args, stdout=subprocess.PIPE)
    output = proc.communicate()[0].decode()
    if proc.wait() != 0:
        raise RuntimeError(f'FFMpeg exited with return code {proc.returncode}')
    fmt_json = json.loads(output)
    return fmt_json


def extract_cover(config: dict, audio_file: str, output_cover_file: str, audio_metadata: Optional[dict] = None) -> bool:
    if audio_metadata is None:
        audio_metadata = get_metadata(audio_file)
    streams = audio_metadata['streams']
    video_streams = [stream for stream in streams if stream['codec_type'] == 'video']
    if len(video_streams) == 0:
        return False
    if len(video_streams) > 1:
        print('Multiple cover found in audio file, use the first one')
    ffmpeg = config['ffmpeg']['ffmpeg_path']
    ffmpeg_args = [ffmpeg, '-i', audio_file, '-an', '-vcodec', 'copy', output_cover_file]
    print(f'Running ffmpeg with args: {ffmpeg_args}')
    proc = subprocess.Popen(ffmpeg_args)
    if proc.wait() != 0:
        raise RuntimeError(f'FFMpeg exited with return code {proc.returncode}')
    return True


def main():
    AVAILABLE_EXT = {'.mp3', '.flac', '.wav'}
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='Config file for vits pipeline', default='vits_config.json', type=str)
    parser.add_argument('-p', '--profile', default='default', help='VITS argument profile name', type=str)
    parser.add_argument('-k', '--keep', help='Keep temporarily generated files', action='store_true')
    args, remains = parser.parse_known_args()

    with open(args.config, 'r', encoding='utf8') as f:
        config = json.load(f)

    i = 0
    while i < len(remains):
        if remains[i].startswith('--') and i + 1 < len(remains) and not remains[i + 1].startswith('--'):
            if not parse_dot_args_to_dict(config, remains[i][2:], remains[i + 1]):
                raise ValueError(f'Failed to parse argument "{remains[i]}" with value "{remains[i + 1]}" to config')
            i += 2
        else:
            i += 1

    # append file name if file.output is a directory
    if 'output' in config['file'] and 'input' in config['file'] and os.path.isdir(config['file']['output']):
        output = os.path.join(config['file']['output'], os.path.basename(config['file']['input']))
        output_base, output_ext = os.path.splitext(output)
        # use .flac instead of .mp3 if output is specified as an existing directory when input is .mp3
        if output_ext.lower() == '.mp3':
            output = output_base + '.flac'
        config['file']['output'] = output

    ext = config['sys']['intermediate_file_ext'].lower()
    final_ext = os.path.splitext(config['file']['output'])[1].lower()
    if final_ext == '':
        raise ValueError('extension for output file is empty')
    if final_ext not in AVAILABLE_EXT:
        raise ValueError(f'unsupported extension "{final_ext}"')
    if ext not in AVAILABLE_EXT:
        raise ValueError(f'unsupported extension "{ext}"')

    if config['ffmpeg']['ffmpeg_path'] is None:
        config['ffmpeg']['ffmpeg_path'] = shutil.which('ffmpeg')
    if config['ffmpeg']['ffmpeg_path'] is None:
        raise RuntimeError('ffmpeg not found')
    if config['ffmpeg']['ffprobe_path'] is None:
        config['ffmpeg']['ffprobe_path'] = shutil.which('ffprobe')
    if config['ffmpeg']['ffprobe_path'] is None:
        raise RuntimeError('ffprobe not found')

    if args.profile not in config['vits']['profile']:
        raise ValueError(f'Profile "{args.profile}" not found in vits.profile section')

    file_reg = FileRegistration(config['sys']['work_dir'], args.keep)
    for file_id, path in config['file'].items():
        file_reg.register_file(file_id, path)

    # process input metadata
    input_metadata = None
    if 'input' in file_reg:
        input_metadata = get_metadata(config, file_reg.get_file('input'))
        cover_path = file_reg.register_tmp_file_in_work_dir('input_cover', '.jpg')
        extract_cover(config, file_reg.get_file('input'), cover_path, input_metadata)

    # execute UVR tasks
    if config['uvr']['enable']:
        execute_uvr_pipeline(config, file_reg, ext)

    # split vocal channels
    if config['down_mix']['separate_process']:
        for vocal_entry in config['down_mix']['vocal']:
            vocal_id = vocal_entry['id']
            if f'{vocal_id}#L' not in file_reg or f'{vocal_id}#R' not in file_reg:
                split_audio_channel(config, vocal_id, file_reg, ext)

    # run vits
    execute_vits_task(config, file_reg, ext, profile_name=args.profile)
    # mix final audio
    down_mix(config, file_reg, final_ext, input_metadata)


if __name__ == '__main__':
    main()
