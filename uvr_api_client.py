# <---- UVR Public API for client side
from enum import Enum, IntEnum
import json
from typing import *
# cspell:ignore demucs,flac,denoise,mdxnet


class StrEnum(str, Enum):
    pass


# <----- UVR request
class EProcessMethod(StrEnum):
    VR_MODE = 'VR Architecture'
    MDX_MODE = 'MDX-Net'
    DEMUCS_MODE = 'Demucs'
    ENSEMBLE_MODE = 'Ensemble Mode'


class EOutputFormat(StrEnum):
    WAV = 'WAV'
    FLAC = 'FLAC'
    MP3 = 'MP3'


class UVRRequestObject:
    pass


class UVREnvRequest(UVRRequestObject):
    """UVR request class for setting required running environments"""
    def __init__(self, process_method: Union[str, EProcessMethod],
                 model_name: Union[str, List[str]],
                 gpu_conversion: Optional[bool] = None,
                 vocal_only: Optional[bool] = None, inst_only: Optional[bool] = None,
                 sample_mode: bool = False,
                 output_format: Optional[Union[str, EOutputFormat]] = None) -> None:
        super().__init__()
        if isinstance(process_method, str):
            process_method = EProcessMethod(process_method)
        self.process_method = process_method
        self.model_name = model_name
        self.gpu_conversion = gpu_conversion
        self.vocal_only = vocal_only
        self.inst_only = inst_only
        self.sample_mode = sample_mode
        if output_format is not None and isinstance(output_format, str):
            output_format = EOutputFormat(output_format)
        self.output_format = output_format


class SecondaryModelOption(UVRRequestObject):
    def __init__(self, activate: Optional[bool] = None, voc_inst_model: Optional[str] = None,
                 voc_inst_model_scale: Optional[float] = None, other_model: Optional[str] = None,
                 other_model_scale: Optional[float] = None, bass_model: Optional[str] = None,
                 bass_model_scale: Optional[float] = None, drums_model: Optional[str] = None,
                 drums_model_scale: Optional[float] = None) -> None:
        self.activate = activate
        self.voc_inst_model = voc_inst_model
        self.voc_inst_model_scale = voc_inst_model_scale
        self.other_model = other_model
        self.other_model_scale = other_model_scale
        self.bass_model = bass_model
        self.bass_model_scale = bass_model_scale
        self.drums_model = drums_model
        self.drums_model_scale = drums_model_scale


class VRArchAdvancedOption(UVRRequestObject):
    def __init__(self, batch_size: Optional[str] = None, enable_tta: Optional[bool] = None, post_process: Optional[bool] = None,
                 post_process_threshold: Optional[float] = None, high_end_process: Optional[bool] = None) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.enable_tta = enable_tta
        self.post_process = post_process
        self.post_process_threshold = post_process_threshold
        self.high_end_process = high_end_process


class VRArchRequest(UVREnvRequest):
    def __init__(self, model_name: str, window_size: Optional[int] = None, aggression_setting: Optional[int] = None,
                 gpu_conversion: Optional[bool] = None, vocal_only: Optional[bool] = None,
                 inst_only: Optional[bool] = None, sample_mode: Optional[bool] = None,
                 output_format: Optional[EOutputFormat] = None,
                 advanced_option: Optional[VRArchAdvancedOption] = None,
                 secondary_model_option: Optional[SecondaryModelOption] = None) -> None:
        super().__init__(EProcessMethod.VR_MODE, model_name, gpu_conversion, vocal_only, inst_only, sample_mode,
                         output_format)
        self.window_size = window_size
        self.aggression_setting = aggression_setting
        self.advanced_option = advanced_option
        self.secondary_model_option = secondary_model_option


class MDXNetArchAdvancedOption(UVRRequestObject):
    def __init__(self, volume_compensation: Optional[Union[str, float]] = None,
                 shift_conversion_pitch: Optional[int] = None, denoise_output: Optional[str] = None,
                 match_freq_cut_off: Optional[bool] = None, spectral_inversion: Optional[bool] = None) -> None:
        super().__init__()
        self.volume_compensation = volume_compensation
        self.shift_conversion_pitch = shift_conversion_pitch
        self.denoise_output = denoise_output
        self.match_freq_cut_off = match_freq_cut_off
        self.spectral_inversion = spectral_inversion


class MDXNet23ArchOnlyAdvancedOption(UVRRequestObject):
    def __init__(self, batch_size: Optional[Union[str, int]] = None, # overlap: Optional[int] = None,
                 segment_default: Optional[bool] = None, combine_stems: Optional[bool] = None) -> None:
        super().__init__()
        self.batch_size = batch_size
        # self.overlap = overlap
        self.segment_default = segment_default
        self.combine_stems = combine_stems


class MDXNetArchRequest(UVREnvRequest):
    def __init__(self, model_name: str, segment_size: Optional[int] = None, overlap: Optional[Union[float, str]] = None,
                 # common option
                 gpu_conversion: Optional[bool] = None, vocal_only: Optional[bool] = None,
                 inst_only: Optional[bool] = None, sample_mode: Optional[bool] = None,
                 output_format: Optional[EOutputFormat] = None,
                 advanced_option: Optional[MDXNetArchAdvancedOption] = None,
                 mdxnet23_advanced_option: Optional[MDXNet23ArchOnlyAdvancedOption] = None,
                 secondary_model_option: Optional[SecondaryModelOption] = None) -> None:
        super().__init__(EProcessMethod.MDX_MODE, model_name, gpu_conversion, vocal_only, inst_only, sample_mode,
                         output_format)
        self.segment_size = segment_size
        self.overlap = overlap
        self.advanced_option = advanced_option
        self.mdxnet23_advanced_option = mdxnet23_advanced_option
        self.secondary_model_option = secondary_model_option


class DemucsArchAdvancedOption(UVRRequestObject):
    def __init__(self, shifts: Optional[int] = None, overlap: Optional[float] = None,
                 shift_conversion_pitch: Optional[int] = None, split_mode: Optional[bool] = None,
                 combine_stems: Optional[bool] = None, spectral_inversion: Optional[bool] = None) -> None:
        super().__init__()
        self.shifts = shifts
        self.overlap = overlap
        self.shift_conversion_pitch = shift_conversion_pitch
        self.split_mode = split_mode
        self.combine_stems = combine_stems
        self.spectral_inversion = spectral_inversion


class DemucsArchRequest(UVREnvRequest):
    def __init__(self, model_name: str, stem: Optional[str] = 'Vocals', segment: Optional[str] = None,
                 primary_stem_only: Optional[bool] = None, secondary_stem_only: Optional[bool] = None,
                 gpu_conversion: Optional[bool] = None, sample_mode: Optional[bool] = None,
                 output_format: Optional[EOutputFormat] = None,
                 advanced_option: Optional[DemucsArchAdvancedOption] = None,
                 secondary_model_option: Optional[SecondaryModelOption] = None) -> None:
        super().__init__(EProcessMethod.DEMUCS_MODE, model_name, gpu_conversion, None, None, sample_mode, output_format)
        self.stem = stem
        self.primary_stem_only = primary_stem_only
        self.secondary_stem_only = secondary_stem_only
        self.segment = segment
        self.advanced_option = advanced_option
        self.secondary_model_option = secondary_model_option


class EnsembleModeAdvancedOption(UVRRequestObject):
    def __init__(self, save_all_outputs: Optional[bool] = None, append_ensemble_name: Optional[bool] = None) -> None:
        super().__init__()
        self.save_all_outputs = save_all_outputs
        self.append_ensemble_name = append_ensemble_name


class EnsembleModeRequest(UVREnvRequest):
    def __init__(self, model_name: List[str], stem_pair: Optional[str] = None,
                 ensemble_algorithm: Optional[str] = None, primary_stem_only: Optional[bool] = None,
                 secondary_stem_only: Optional[bool] = None, gpu_conversion: Optional[bool] = None,
                 sample_mode: Optional[bool] = None, output_format: Optional[EOutputFormat] = None,
                 advanced_option: Optional[EnsembleModeAdvancedOption] = None) -> None:
        super().__init__(EProcessMethod.ENSEMBLE_MODE, model_name, gpu_conversion, None, None, sample_mode,
                         output_format)
        self.stem_pair = stem_pair
        self.ensemble_algorithm = ensemble_algorithm
        self.primary_stem_only = primary_stem_only
        self.secondary_stem_only = secondary_stem_only
        self.advanced_option = advanced_option


class UVRRequest(UVRRequestObject):
    def __init__(self, input_file: Union[str, List[str]], output_file: str,
                 env: Optional[UVREnvRequest] = None) -> None:
        if isinstance(input_file, str):
            input_file = [input_file]
        elif not isinstance(input_file, list):
            input_file = list(input_file)
        self.input_file = input_file
        self.output_file = output_file
        self.env = env


def _serialize_env_node(action: UVRRequestObject):
    attrs = {}
    for k, v in action.__dict__.items():
        if k.startswith('_') or v is None:
            continue
        if isinstance(v, UVRRequestObject):
            attrs[k] = _serialize_env_node(v)
        else:
            attrs[k] = v
    return attrs


def serialize_request_json(uvr_request: UVRRequest) -> str:
    root = _serialize_env_node(uvr_request)
    return json.dumps(root, separators=(',', ':'))


def deserialize_request_json(request_json: Union[str, Dict[str, Any]]) -> UVRRequest:
    if isinstance(request_json, str):
        request_json = json.loads(request_json)
    else:
        request_json = request_json.copy()
    env = request_json.pop('env', None)
    if env is not None:
        req_type = EProcessMethod(env.pop('process_method'))
        if req_type == EProcessMethod.VR_MODE:
            advanced_opt_cls = VRArchAdvancedOption
            base_cls = VRArchRequest
        elif req_type == EProcessMethod.MDX_MODE:
            advanced_opt_cls = MDXNetArchAdvancedOption
            base_cls = MDXNetArchRequest
        elif req_type == EProcessMethod.DEMUCS_MODE:
            advanced_opt_cls = DemucsArchAdvancedOption
            base_cls = DemucsArchRequest
        elif req_type == EProcessMethod.ENSEMBLE_MODE:
            advanced_opt_cls = EnsembleModeAdvancedOption
            base_cls = EnsembleModeRequest
        else:
            raise ValueError('Invalid process_method')
        advanced_opt_args = env.pop('advanced_option', None)
        secondary_model_args = env.pop('secondary_model_option', None)
        if advanced_opt_args is not None:
            advanced_opt = advanced_opt_cls(**advanced_opt_args)
            env['advanced_option'] = advanced_opt
        if secondary_model_args is not None:
            env['secondary_model_option'] = SecondaryModelOption(**secondary_model_args)
        env = base_cls(**env)
        request_json['env'] = env
    return UVRRequest(**request_json)


# <---- UVR response
class UVRCallState(IntEnum):
    FAILED = -1
    SUCCESS = 0
    SCHEDULING = 1
    RUNNING = 2


class TaskContext:
    def __init__(self, request: Optional[UVRRequest], final_output_file: Optional[str] = None,
                 task_id: Optional[str] = None, task_state: UVRCallState = UVRCallState.FAILED,
                 task_log: str = '', task_progress: int = 0):
        self.request = request
        # the final output file path may be differ from requested file path (as handled by UVR itself)
        self.final_output_file = final_output_file
        self.task_id = task_id
        self.task_log = task_log
        self.task_state = task_state
        self.task_progress = task_progress


def deserialize_response_json(resp: Union[str, dict]) -> TaskContext:
    if isinstance(resp, str):
        resp = json.loads(resp)
    return TaskContext(request=deserialize_request_json(resp['request']),
                       final_output_file=resp['final_output_file'],
                       task_id=resp['task_id'],
                       task_log=resp['task_log'],
                       task_state=UVRCallState(resp['task_state']),
                       task_progress=resp['task_progress'])

# <---- End UVR API


def main():
    # input_path = 'd:/CloudMusic/森永真由美 - 華鳥風月.flac'
    # output_path = 'z:/bbb.flac'
    input_path = 'z:/bbb_(Vocals).flac'
    output_path = 'z:/ccc.flac'

    import requests
    from time import sleep

    s = requests.Session()
    # env = MDXNetArchRequest('MDX23C-InstVoc HQ', gpu_conversion=True, vocal_only=True, inst_only=False, output_format=EOutputFormat.FLAC,
    #                         secondary_model_option=SecondaryModelOption(False))
    env = VRArchRequest('UVR-BVE-4B_SN-44100-1', gpu_conversion=True, vocal_only=False, inst_only=True, output_format=EOutputFormat.FLAC, 
                        secondary_model_option=SecondaryModelOption(False))
    req = UVRRequest(input_path, output_path, env)
    req_json = serialize_request_json(req)
    print(req_json)
    resp = s.get('http://localhost:8090/create', params=req_json)
    assert resp.ok
    data = deserialize_response_json(resp.json())
    print('Task creation:')
    print(resp.json())
    assert data.task_state.value > 0
    task_id = data.task_id

    while True:
        resp = s.get(f'http://localhost:8090/query?{{"task_id":"{task_id}"}}')
        assert resp.ok
        data = deserialize_response_json(resp.json())
        print(resp.json())
        print('Task state:')
        print(data.task_state)

        if data.task_state.value > 0:
            sleep(1)
        else:
            break


if __name__ == '__main__':
    main()
