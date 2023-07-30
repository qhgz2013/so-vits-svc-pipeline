# v2.3.11 branch of https://www.yuque.com/umoubuton/ueupp5
# rev 6: support v2.3.11 features, add UVR config (-uc) option
# rev 5: add separate process mode: run inference for left and right channel separately to obtain stereo output
# rev 4: add chorus mode: running inference for multiple speakers and mixing all output streams
# rev 3: add UVR automation pipeline
import argparse
import os
from glob import glob
import subprocess
import shutil
from itertools import chain
from time import sleep
import json
from typing import *
import uvr_api_client as api
import requests

# cspell:ignore vits,adelay,flac,amerge

VOLUME_MULTIPLIER = 1.0
generated_files_to_cleanup= []


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


def get_vocal_and_inst_files(search_path):
    valid_filters = ['*.wav', '*.flac', '*.mp3']
    files = filter(lambda x: search_path in x, chain(*tuple(glob(x) for x in valid_filters)))
    vocal_file = inst_file = None
    for file in files:
        if 'vocal' in file.lower():
            vocal_file = file
        elif 'instrument' in file.lower():
            inst_file = file
    if vocal_file is None:
        raise RuntimeError('Could not detect vocal file')
    if inst_file is None:
        raise RuntimeError('Could not detect instrumental file')
    print('Vocal file:', vocal_file)
    print('Inst file:', inst_file)
    return vocal_file, inst_file


def build_vits_args(args, remains, vits_py_executable, spk_name) -> List[str]:
    vits_args = [vits_py_executable, 'inference_main.py', '-m', args.model, '-c', args.config, '-n', 'tmp_svc.wav', 
                '-t', str(args.trans), '-s', spk_name, '-wf', args.wav_format]
    if args.cluster_infer_ratio is not None:
        vits_args.extend(['-cr', str(args.cluster_infer_ratio)])
    if args.cluster_model_path is not None:
        vits_args.extend(['-cm', args.cluster_model_path])
    if args.slice_db is not None:
        vits_args.extend(['-sd', str(args.slice_db)])
    if args.shallow_diffusion is not None and args.shallow_diffusion is True:
        vits_args.append('-shd')
    if args.only_diffusion is not None and args.only_diffusion is True:
        vits_args.append('-od')
    if args.second_encoding is not None and args.second_encoding is True:
        vits_args.append('-se')
    vits_args.extend(remains)
    return vits_args


def check_diff_file(new_files: List[str], args) -> str:
    new_sounds = list(filter(lambda x: x.endswith(args.wav_format), new_files))
    if len(new_sounds) > 1:
        raise RuntimeError('Found multiple sounds updated during inference')
    elif len(new_sounds) == 0:
        raise RuntimeError('No sound file changed in VITS result directory')
    return new_sounds[0]


def execute_uvr_task(args) -> List[api.TaskContext]:
    s = requests.session()
    req = {'input_file': args.input, 'output_file': args.output}
    if args.uvr_config is not None and os.path.isfile(args.uvr_config):
        with open(args.uvr_config, 'r', encoding='utf8') as f:
            envs = json.load(f)
    if not isinstance(envs, list):
        envs = [envs]
    ret_ctx = []
    for env in envs:
        req['env'] = env
        print(f'Executing UVR task: {req}')
        resp = s.get(f'http://{args.uvr_host}:{args.uvr_port}/create', params=json.dumps(req, separators=(',', ':')))
        if resp.ok:
            run_ctx = api.deserialize_response_json(resp.json())
            task_id = run_ctx.task_id
            log_idx = 0
            while run_ctx.task_state > 0:  # pending and running
                sleep(1)
                resp = s.get(f'http://{args.uvr_host}:{args.uvr_port}/query',
                                params=json.dumps({'task_id': task_id}, separators=(',', ':')))
                if not resp.ok:
                    break
                run_ctx = api.deserialize_response_json(resp.json())
                if len(run_ctx.task_log) > log_idx:
                    # show UVR task log
                    print(run_ctx.task_log[log_idx:], end='')
                    log_idx = len(run_ctx.task_log)
            print('')
            ret_ctx.append(run_ctx)
    return ret_ctx


def main():
    parser = argparse.ArgumentParser()
    # if uvr-http-service is set and available, --input refers to the source audio file, UVR will be run automatically
    # otherwise, --input refers to the extracted vocal and instrumental audio files, with name {input}_(Vocal).{ext} and {input}_(Instrumental).{ext}
    parser.add_argument('-i', '--input', type=str, help='Input file, for files with name "1_いきものがかり - SAKURA_(Instrumental).wav", just type "いきものがかり - SAKURA"', required=True)
    parser.add_argument('-o', '--output', type=str, help='Output file', default=None)
    parser.add_argument('-vits', '--vits_path', type=str, help='Path to so-vits-svc source', required=True)
    parser.add_argument('-m', '--model', type=str, help='Checkpoint path for VITS model', required=True, default='logs/44k/G_64000.pth')
    parser.add_argument('-c', '--config', type=str, help='Config path for VITS model', required=True, default='logs/44k/config.json')
    parser.add_argument('-t', '--trans', type=int, default=0)

    # new feature for chorus mode (rev 4): use "|" to separate the speaker for left and right channel, use "+" to mix multiple speakers, and use ":" to add offset (in ms, relative to instrumental track)
    # for example: s1+s2:40|s1+s3:-60  L channel: s1 and s2 (delay 40 ms), R channel: s1 and s3 (delay -60 ms)
    parser.add_argument('-s', '--spk_list', type=str, default=None, help='Leave none to detect from config')

    # for unambiguous of '-c' and '-s'
    parser.add_argument('-cm', '--cluster_model_path', type=str, default=None)
    parser.add_argument('-cr', '--cluster_infer_ratio', type=float, default=None)
    parser.add_argument('-sd', '--slice_db', type=int, default=None)
    parser.add_argument('-se', '--second_encoding', action='store_true', default=None)
    parser.add_argument('-shd', '--shallow_diffusion', action='store_true', default=None)
    parser.add_argument('-wf', '--wav_format', type=str, default='flac')
    parser.add_argument('-od', '--only_diffusion', action='store_true', default=None)

    # new options (rev 3): add support for uvr
    parser.add_argument('-up', '--uvr_port', type=int, default=-1, help='leave -1 to disable uvr automation')
    parser.add_argument('-uh', '--uvr_host', type=str, default='localhost')
    parser.add_argument('--keep', action='store_true', help='Keep UVR files (generated from UVR automation process)')

    # new option (rev 5):
    parser.add_argument('-sp', '--separate_process', action='store_true',
                        help='Run SO-VITS-SVC separately for two audio channels (L and R)')
    
    # new option (rev 6):
    parser.add_argument('-uc', '--uvr_config', type=str, default=None,
                        help='Path for uvr_config.json (e.g., uvr_config_sample.json)')

    args, remains = parser.parse_known_args()

    # determine output file path
    input_name = args.input
    if '.' in input_name:
        input_name = os.path.splitext(os.path.basename(input_name))[0]
    if args.output is None:
        args.output = input_name + '.' + args.wav_format
    elif os.path.isdir(args.output):
        args.output = os.path.join(args.output, input_name + '.' + args.wav_format)
    args.output = os.path.abspath(args.output)
    output_dir = os.path.split(args.output)[0]
    print('Output will be saved to:', args.output)
    args.wav_format = args.wav_format.lower()

    print(args, remains)

    # lookup speaker from config.json
    if args.spk_list is None:
        with open(os.path.join(args.vits_path, args.config), 'r', encoding='utf8') as f:
            config = json.load(f)
            speaker_mapping = config['spk']
            if len(speaker_mapping) > 0:
                raise RuntimeError('Multiple speaker detected from config, pls specify it explicitly by "-s" argument')
            args.spk_list = next(speaker_mapping.keys())

    # custom speaker operator
    if '|' in args.spk_list:
        l, r = args.spk_list.split('|')
        l, r = l.split('+'), r.split('+')
    else:
        l = r = args.spk_list.split('+')
    # key: "l" or "r" (left and right channel), value: spk_name[:offset]
    channel_spk_name_dict = {'l': [x.split(':') for x in l], 'r': [x.split(':') for x in r]}
    print('channel->speaker mapper:', channel_spk_name_dict)
    
    # execute UVR
    manual_input_path_mode = True
    uvr_exec = False
    vocal_file, inst_file = None, None
    if args.uvr_port > 0 and len(args.uvr_host) > 0 and os.path.isfile(args.input):
        uvr_exec = True
        tasks = execute_uvr_task(args)
        final_output_files = {}
        for i, task in enumerate(tasks):
            if task.task_state == api.UVRCallState.SUCCESS:
                print(f'UVR task [{i}] executed successfully')
                final_output_files.update(task.final_output_file.items())
            else:
                print(f'UVR task [{i}] failed')
                raise RuntimeError(f'UVR task [{i}] failed')
        if 'Vocals' in final_output_files and 'Instrumental' in final_output_files:
            manual_input_path_mode = False
            vocal_file = final_output_files['Vocals']
            inst_file = final_output_files['Instrumental']
        else:
            raise RuntimeError('Missing output Vocals and/or Instrumental file(s)')
    # non-UVR mode
    if manual_input_path_mode:
        if uvr_exec:
            print('Failed to execute UVR task: fallback to manual input mode')
        vocal_file, inst_file = get_vocal_and_inst_files(args.input)

    # split vocal channels
    if args.separate_process:
        # ffmpeg -i stereo.wav -filter_complex "[0:a]channelsplit=channel_layout=stereo[left][right]"
        # -map "[left]" left.wav -map "[right]" right.wav
        ffmpeg_args = ['ffmpeg', '-y', '-i', vocal_file, '-filter_complex',
                       '[0]channelsplit=channel_layout=stereo[l][r]', '-map', '[l]',
                       f'{vocal_file}_l.{args.wav_format}', '-map', '[r]', f'{vocal_file}_r.{args.wav_format}']
        print(f'Running FFMpeg with args: {ffmpeg_args}')
        proc = subprocess.Popen(ffmpeg_args)
        if proc.wait() != 0:
            raise RuntimeError(f'FFMpeg exited with return code {proc.returncode}')
        vocal_files = {x: f'{vocal_file}_{x}.{args.wav_format}' for x in ['l', 'r']}
        generated_files_to_cleanup.extend(vocal_files.values())
    else:
        vocal_files = None

    target_vocal_path = os.path.join(args.vits_path, 'raw', 'tmp_svc.wav')
    vits_output_dir = os.path.join(args.vits_path, 'results')
    watcher = FileChangeWatcher(vits_output_dir)
    vits_py_executable = os.path.abspath(os.path.join(args.vits_path, 'workenv', 'python'))

    # run vits
    spk_result_dict = {'l': {}, 'r': {}}
    if args.separate_process:
        for channel in ['l', 'r']:
            print(f'cp: {vocal_files[channel]} -> {target_vocal_path}')
            shutil.copy(vocal_files[channel], target_vocal_path)
            for spk_name_offset in channel_spk_name_dict[channel]:
                spk_name = spk_name_offset[0]
                watcher.update_file_mtime()
                vits_args = build_vits_args(args, remains, vits_py_executable, spk_name)
                print(f'Running VITS with args: {vits_args}')
                proc = subprocess.Popen(vits_args, cwd=args.vits_path)
                if proc.wait() != 0:
                    raise RuntimeError(f'VITS exited with return code {proc.returncode}')
                new_file = check_diff_file(watcher.generate_diff_files()[0], args)
                renamed_file = os.path.join(output_dir, f'vits_{spk_name}_{channel}.{args.wav_format}')
                shutil.copy(new_file, renamed_file)
                spk_result_dict[channel][spk_name] = renamed_file
                generated_files_to_cleanup.append(renamed_file)
    else:
        spk_list = set(map(lambda x: x[0], channel_spk_name_dict['l'])) | set(map(lambda x: x[0], channel_spk_name_dict['r']))
        shutil.copy(vocal_file, target_vocal_path)
        for spk_name in spk_list:
            watcher.update_file_mtime()
            vits_args = build_vits_args(args, remains, vits_py_executable, spk_name)
            print(f'Running vits with args: {vits_args}')
            proc = subprocess.Popen(vits_args, cwd=args.vits_path)
            if proc.wait() != 0:
                raise RuntimeError(f'VITS exited with return code {proc.returncode}')
            new_file = check_diff_file(watcher.generate_diff_files()[0], args)
            spk_result_dict['l'][spk_name] = new_file
            spk_result_dict['r'][spk_name] = new_file
    print(spk_result_dict)
    
    # down mix
    # ffmpeg note: stream labels in filter graph can be used only once
    # ffmpeg -y -i <input1> ... -i <inst_file> -filter_complex <filter_cmd> -map [ao] -ac 2 <extra_args> <output>
    ffmpeg_args = ['ffmpeg', '-y']
    filter_cmd = []
    i = 0
    for channel, output_dict in spk_result_dict.items():
        merge_input_nodes = []
        for spk_name_offset_list in channel_spk_name_dict[channel]:
            spk_name = spk_name_offset_list[0]
            ffmpeg_args.extend(['-i', output_dict[spk_name]])
            if len(spk_name_offset_list) > 1:
                delay = spk_name_offset_list[1]
                # for adelay filter, ffmpeg need to specify aformat at first, otherwise amerge will throw a format error
                probe_args = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', 
                              output_dict[spk_name]]
                print(f'Running ffprobe with args: {probe_args}')
                proc = subprocess.Popen(probe_args, stdout=subprocess.PIPE)
                output = proc.communicate()[0].decode()
                if proc.wait() != 0:
                    raise RuntimeError(f'FFMpeg exited with return code {proc.returncode}')
                fmt_json = json.loads(output)
                sample_fmt = fmt_json['streams'][0]['sample_fmt']
                sample_rate = fmt_json['streams'][0]['sample_rate']
                layout = fmt_json['streams'][0]['channel_layout']
                if layout.lower() == 'mono' and not sample_fmt.endswith('p'):
                    sample_fmt += 'p'  # a buggy fix for s16 -> s16p in mono stream?
                filter_cmd.append(f'[{i}]aformat=sample_fmts={sample_fmt}:sample_rates={sample_rate}:channel_layouts={layout}[s{i}]')
                filter_cmd.append(f'[s{i}]adelay={delay}[t{i}]')
                merge_input_nodes.append(f'[t{i}]')
            else:
                merge_input_nodes.append(f'[{i}]')
            i += 1
        n_nodes = len(merge_input_nodes)
        norm_volume_factor = round(VOLUME_MULTIPLIER / n_nodes, 2)
        # e.g., [a0][1]amerge=inputs=2,pan=mono|c0=0.5*c0+0.5*c1[vl]
        mix_channels = "+".join(f'{norm_volume_factor}*c{x}' for x in range(n_nodes))
        filter_cmd.append(f'{"".join(merge_input_nodes)}amerge=inputs={n_nodes},pan=mono|c0={mix_channels}[v{channel}]')
    # join left and right vocal channel: vl, vr -> vo
    filter_cmd.append(f'[vl][vr]join=inputs=2:channel_layout=stereo[vo]')
    inst_track_id = i
    ffmpeg_args.extend(['-i', inst_file])
    # merge vocal and instrumental track
    filter_cmd.append(f'[vo][{inst_track_id}]amerge=inputs=2,pan=stereo|c0=c0+{VOLUME_MULTIPLIER}*c2|c1=c1+{VOLUME_MULTIPLIER}*c3[ao]')
    ffmpeg_args += ['-filter_complex', ';'.join(filter_cmd), '-map', '[ao]', '-ac', '2']
    # extra arguments for output stream (s32 is required since s16 will overflow during mix)
    # -c:a pcm_s32le for wav
    # -c:a flac -sample_fmt s32 for flac
    # -c:a libmp3lame -sample_fmt s32p -b:a 320k for mp3
    if args.wav_format == 'wav':
        ffmpeg_args += ['-c:a', 'pcm_s32le']
    elif args.wav_format == 'flac':
        ffmpeg_args += ['-c:a', 'flac', '-sample_fmt', 's32']
    elif args.wav_format == 'mp3':
        ffmpeg_args += ['-c:a', 'libmp3lame', '-sample_fmt', 's32p', '-b:a', '320k']
    else:
        print(f'No available extra arguments for wav format "{args.wav_format}"')
    ffmpeg_args.append(args.output)
    print('Running FFMpeg with args', ffmpeg_args)
    proc = subprocess.Popen(ffmpeg_args)
    if proc.wait() != 0:
        raise RuntimeError(f'FFMpeg exited with return code {proc.returncode}')
    if not args.keep and uvr_exec:
        os.remove(vocal_file)
        os.remove(inst_file)


def cleanup_generated_files():
    for file in generated_files_to_cleanup:
        if os.path.isfile(file):
            print(f'cleanup generated file: {file}')
            os.remove(file)
    generated_files_to_cleanup.clear()


if __name__ == '__main__':
    main()
    cleanup_generated_files()
