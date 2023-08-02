import os
from typing import *
import argparse
from tqdm import tqdm
import uvr_api_client as api
import json
import requests
from time import sleep
import re
from urllib.parse import quote

default_ext = {'.wav', '.mp3', '.flac'}
output_file_ptn = re.compile(r'^\d+_(.+)_\((Vocals|Instrumental)\)\.(wav|flac|mp3)$')


def _gen_file_list(path, _, files, valid_exts, ignore_names=None):
    for file in files:
        name, ext = os.path.splitext(file)
        if ext.lower() not in valid_exts:
            continue
        if ignore_names is not None and name in ignore_names:
            continue
        yield os.path.join(path, file)


def gen_file_list(input_path, output_dir=None, recursive=True, valid_exts=default_ext):
    if os.path.isfile(input_path):
        yield input_path
        return
    output_files = os.listdir(output_dir)
    match_results_not_filtered = map(lambda x: re.match(output_file_ptn, x), output_files)
    match_results = filter(lambda x: x is not None, match_results_not_filtered)
    output_file_names = map(lambda x: x.group(1), match_results)
    output_file_names = set(output_file_names)
    if recursive:
        for item in os.walk(input_path):
            yield from _gen_file_list(*item, valid_exts=valid_exts, ignore_names=output_file_names)
    else:
        yield from _gen_file_list(*next(os.walk(input_path)), valid_exts=valid_exts, ignore_names=output_file_names)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputs', help='Input audio file/directory for UVR tasks', required=True)
    parser.add_argument('-o', '--output', help='Output directory for converted audio stems')
    parser.add_argument('--no_recursive', action='store_true',
                        help='When -i specifies a directory, do not look up audios from its sub-directories')
    parser.add_argument('-e', '--exts', type=str, nargs='+',
                        help=f'Valid extensions for audios, default: {default_ext}')
    parser.add_argument('-c', '--config', type=str, help='Path for uvr_config.json')
    parser.add_argument('--host', default='localhost', help='Host for UVR-API-server', type=str)
    parser.add_argument('--port', default=8090, help='Port for UVR-API-server', type=int)
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()
    if args.exts is not None:
        args.exts = set(args.exts)
    else:
        args.exts = default_ext
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    if args.config is not None and not os.path.isfile(args.config):
        print(f'Config file "{args.config}" is not a file!')
        args.config = None
    return args


def main():
    args = parse_args()
    files_gen = gen_file_list(args.inputs, args.output, not args.no_recursive, args.exts)
    files = list(files_gen)

    if args.dry_run:
        for file in files:
            print(file)
        print(f'# files: {len(files)}')
        return

    first_run = True
    s = requests.session()

    def _get_url(action: str) -> str:
        return f'http://{args.host}:{args.port}/{action}'

    for file in tqdm(files):
        env = None
        if first_run:
            first_run = False
            if args.config is not None:
                with open(args.config, 'r', encoding='utf8') as f:
                    json_data = json.load(f)
                # only set env in the first request
                env = api.deserialize_env_request_json(json_data)
        request = api.UVRRequest(file, args.output, env)
        request_json = quote(api.serialize_request_json(request))
        resp = s.get(_get_url('create'), params=request_json)
        assert resp.ok, f'HTTP call failed with status code {resp.status_code}'
        task = api.deserialize_response_json(resp.json())
        while task.task_state > 0:
            sleep(1)
            task_id = task.task_id
            resp = s.get(_get_url('query'), params=f'{{"task_id":"{task_id}"}}')
            assert resp.ok, f'HTTP call failed with status code {resp.status_code}'
            task = api.deserialize_response_json(resp.json())
        if task.task_state != api.UVRCallState.SUCCESS:
            print(f'UVR call failed with state: {task.task_state}')
            return


if __name__ == '__main__':
    main()
