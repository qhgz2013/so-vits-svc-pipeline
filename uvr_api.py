from enum import Enum
from typing import Optional, List, Union, TYPE_CHECKING, Tuple, Dict, Any
from gui_data import constants
import traceback
import threading
from abc import ABCMeta
from time import sleep
from warnings import warn
import json
import os

if TYPE_CHECKING:
    from uvr_api_server import MainWindowOverwrite
# cspell:ignore demucs,flac,denoise,mdxnet


__all__ = ['EProcessMethod', 'EOutputFormat', 'get_last_check_failed_source', 'UVREnvRequest', 'VRArchAdvancedOption',
           'VRArchRequest', 'MDXNetArchAdvancedOption', 'MDXNet23ArchOnlyAdvancedOption', 'MDXNetArchRequest',
           'DemucsArchAdvancedOption', 'DemucsArchRequest', 'EnsembleModeAdvancedOption', 'EnsembleModeRequest',
           'deserialize_request_json', 'serialize_request_json', 'SecondaryModelOption',
           'deserialize_env_request_json']
_local = threading.local()


class StrEnum(str, Enum):
    pass


class EProcessMethod(StrEnum):
    VR_MODE = 'VR Architecture'
    MDX_MODE = 'MDX-Net'
    DEMUCS_MODE = 'Demucs'
    ENSEMBLE_MODE = 'Ensemble Mode'


class EOutputFormat(StrEnum):
    WAV = 'WAV'
    FLAC = 'FLAC'
    MP3 = 'MP3'


def _record_last_check_failed_source(msg: str):
    frames = traceback.extract_stack()
    _local.uvr_env_request_check_fail_source = frames[-2]
    _local.uvr_env_request_check_fail_msg = msg
    return False


def get_last_check_failed_source() -> Optional[Tuple[traceback.FrameSummary, str]]:
    return getattr(_local, 'uvr_env_request_check_fail_source', None), \
        getattr(_local, 'uvr_env_request_check_fail_msg', '')


class IEnvAutomation(metaclass=ABCMeta):
    def setup(self, uvr: 'MainWindowOverwrite') -> None:
        raise NotImplementedError

    def check_setup_prerequisite(self, uvr: 'MainWindowOverwrite') -> bool:
        raise NotImplementedError


def _run_fn_in_non_main_thread_and_wait(fn: callable):
    if threading.current_thread().ident != threading.main_thread().ident:
        return fn()
    wait_event = threading.Event()
    ret = None

    def _dummy_fn():
        nonlocal ret
        ret = fn()
        wait_event.set()
    t = threading.Thread(target=_dummy_fn)
    t.start()
    while not wait_event.wait(1):
        pass
    t.join()
    return ret


def _perform_tk_value_set(original_value, target_var):
    if original_value is None:
        return
    target_value = target_var.get()
    if isinstance(target_value, bool):
        if not isinstance(original_value, bool):
            raise TypeError(f'Invalid type {type(original_value)!r} from original_value, expected: bool')
        if target_value != original_value:
            target_var.set(original_value)
    elif isinstance(target_value, str):
        str_original_value = str(original_value)
        if str_original_value != target_value:
            target_var.set(str_original_value)
    else:
        raise TypeError(f'Invalid type {type(target_value)!r} from target_var, expected: bool or str')
# TODO: support vocal splitter feature


class UVREnvRequest(IEnvAutomation):
    """UVR request class for setting required running environments"""
    def __init__(self, process_method: Union[str, EProcessMethod],
                 model_name: Union[str, List[str]],
                 gpu_conversion: Optional[bool] = None,
                 vocal_only: Optional[bool] = None, inst_only: Optional[bool] = None,  # vocal/inst models
                 no_echo_only: Optional[bool] = None, echo_only: Optional[bool] = None,  # de-echo models
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
        self.no_echo_only = no_echo_only
        self.echo_only = echo_only
        self.sample_mode = sample_mode
        if output_format is not None and isinstance(output_format, str):
            output_format = EOutputFormat(output_format)
        self.output_format = output_format

    def setup(self, uvr: 'MainWindowOverwrite') -> None:
        """setting UVR GUI to specific environment. Run `check_setup_prerequisite()` first before call this automation function!"""
        if uvr.chosen_process_method_var.get() != self.process_method.value:
            uvr.chosen_process_method_var.set(self.process_method.value)
            uvr.selection_action_process_method(self.process_method.value, from_widget=True, is_from_conv_menu=True)
        self._select_model(uvr)
        if self.gpu_conversion is not None:
            gpu_conversion = self.gpu_conversion and uvr.is_gpu_available
            _perform_tk_value_set(gpu_conversion, uvr.is_gpu_conversion_var)
        _perform_tk_value_set(self.sample_mode, uvr.model_sample_mode_var)
        _perform_tk_value_set(self.output_format.value, uvr.save_format_var)
        self._set_vocal_inst_opt(uvr, self.model_name)

    def check_setup_prerequisite(self, uvr: 'MainWindowOverwrite') -> bool:
        return self.validate_model_name(uvr, self.model_name) and \
            self._check_vocal_inst_opt_prerequisite(uvr, self.model_name) and \
            self._check_echo_opt_prerequisite(uvr, self.model_name)

    def validate_model_name(self, uvr: 'MainWindowOverwrite', model_names: Union[str, List[str]]) -> bool:
        process_method = self.process_method
        if process_method == EProcessMethod.ENSEMBLE_MODE:
            process_method_str = constants.ENSEMBLE_CHECK
        else:
            process_method_str = process_method.value
        if isinstance(model_names, str):
            model_names = [model_names]
        for model_name in model_names:
            model_data = uvr.assemble_model_data(model_name, process_method_str)[0]
            if not model_data.model_status:
                return _record_last_check_failed_source(f'Invalid model name: {model_name}')
        return True

    def _select_model(self, uvr: 'MainWindowOverwrite') -> None:
        raise NotImplementedError

    def _set_vocal_inst_opt(self, uvr: 'MainWindowOverwrite', model_name: str) -> None:
        if self.vocal_only is not None or self.inst_only is not None:
            model_data = uvr.assemble_model_data(model_name, self.process_method.value)[0]
            primary_stem = model_data.primary_stem
            print(f'primary stem: {primary_stem}')
            vocal_only_opt = inst_only_opt = None
            if primary_stem == constants.VOCAL_STEM:
                vocal_only_opt = uvr.is_primary_stem_only_var
                inst_only_opt = uvr.is_secondary_stem_only_var
            elif primary_stem == constants.INST_STEM:
                vocal_only_opt = uvr.is_secondary_stem_only_var
                inst_only_opt = uvr.is_primary_stem_only_var
            _perform_tk_value_set(self.vocal_only, vocal_only_opt)
            _perform_tk_value_set(self.inst_only, inst_only_opt)
        if self.no_echo_only is not None or self.echo_only is not None:
            model_data = uvr.assemble_model_data(model_name, self.process_method.value)[0]
            primary_stem = model_data.primary_stem
            print(f'primary stem: {primary_stem}')
            no_echo_opt = echo_opt = None
            if primary_stem == 'No Echo':
                no_echo_opt = uvr.is_primary_stem_only_var
                echo_opt = uvr.is_secondary_stem_only_var
            elif primary_stem == 'Echo':
                echo_opt = uvr.is_secondary_stem_only_var
                no_echo_opt = uvr.is_primary_stem_only_var
            _perform_tk_value_set(self.no_echo_only, no_echo_opt)
            _perform_tk_value_set(self.echo_only, echo_opt)

    def _check_vocal_inst_opt_prerequisite(self, uvr: 'MainWindowOverwrite', model_name: str) -> bool:
        if self.vocal_only is None and self.inst_only is None:
            return True
        model_data = uvr.assemble_model_data(model_name, self.process_method.value)[0]
        if not model_data.model_status:
            return _record_last_check_failed_source('Invalid model name')
        primary_stem = model_data.primary_stem
        if primary_stem == constants.VOCAL_STEM or primary_stem == constants.INST_STEM:
            return True
        return _record_last_check_failed_source('Could not set vocal_only or inst_only: no matching stems from model')

    def _check_echo_opt_prerequisite(self, uvr: 'MainWindowOverwrite', model_name: str) -> bool:
        if self.echo_only is None and self.no_echo_only is None:
            return True
        model_data = uvr.assemble_model_data(model_name, self.process_method.value)[0]
        if not model_data.model_status:
            return _record_last_check_failed_source('Invalid model name')
        primary_stem = model_data.primary_stem
        if primary_stem in {'No Echo', 'Echo'}:
            return True
        return _record_last_check_failed_source('Could not set no_echo_only or echo_only: no matching stems from model')


class SecondaryModelOption(IEnvAutomation):
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

    def check_setup_prerequisite(self, uvr: 'MainWindowOverwrite') -> bool:
        # TODO impl
        return True

    def setup(self, uvr: 'MainWindowOverwrite') -> None:
        proc_method = EProcessMethod(uvr.chosen_process_method_var.get())
        proc_mapper = {
            EProcessMethod.VR_MODE: uvr.vr_secondary_model_vars,
            EProcessMethod.MDX_MODE: uvr.mdx_secondary_model_vars,
            EProcessMethod.DEMUCS_MODE: uvr.demucs_secondary_model_vars
        }
        model_vars = proc_mapper.get(proc_method, None)
        if model_vars is None:
            warn(f'SecondaryModelOption is not available for process method "{proc_method.value}"')
        if self.voc_inst_model is not None and self.voc_inst_model != model_vars['voc_inst_secondary_model'].get():
            model_vars['voc_inst_secondary_model'].set(self.voc_inst_model)
        if self.voc_inst_model_scale is not None and str(self.voc_inst_model_scale) != model_vars['voc_inst_secondary_model_scale'].get():
            model_vars['voc_inst_secondary_model_scale'].set(str(self.voc_inst_model_scale))
        if self.other_model is not None and self.other_model != model_vars['other_secondary_model'].get():
            model_vars['other_secondary_model'].set(self.other_model)
        if self.other_model_scale is not None and str(self.other_model_scale) != model_vars['other_secondary_model_scale'].get():
            model_vars['other_secondary_model_scale'].set(str(self.other_model_scale))
        if self.bass_model is not None and self.bass_model != model_vars['bass_secondary_model'].get():
            model_vars['bass_secondary_model'].set(self.bass_model)
        if self.bass_model_scale is not None and str(self.bass_model_scale) != model_vars['bass_secondary_model_scale'].get():
            model_vars['bass_secondary_model_scale'].set(str(self.bass_model_scale))
        if self.drums_model is not None and self.drums_model != model_vars['drums_secondary_model_scale'].get():
            model_vars['drums_secondary_model'].set(self.drums_model)
        if self.drums_model_scale is not None and str(self.drums_model_scale) != model_vars['drums_secondary_model_scale'].get():
            model_vars['drums_secondary_model_scale'].set(str(self.drums_model_scale))
        if self.activate is not None and self.activate != model_vars['is_secondary_model_activate'].get():
            model_vars['is_secondary_model_activate'].set(self.activate)


class VRArchAdvancedOption(IEnvAutomation):
    def __init__(self, batch_size: Optional[str] = None, enable_tta: Optional[bool] = None, post_process: Optional[bool] = None,
                 post_process_threshold: Optional[float] = None, high_end_process: Optional[bool] = None) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.enable_tta = enable_tta
        self.post_process = post_process
        self.post_process_threshold = post_process_threshold
        self.high_end_process = high_end_process

    def check_setup_prerequisite(self, uvr: 'MainWindowOverwrite') -> bool:
        return True

    def setup(self, uvr: 'MainWindowOverwrite') -> None:
        _perform_tk_value_set(self.batch_size, uvr.batch_size_var)
        _perform_tk_value_set(self.enable_tta, uvr.is_tta_var)
        _perform_tk_value_set(self.post_process, uvr.is_post_process_var)
        if self.post_process is not None:
            _perform_tk_value_set(self.post_process_threshold, uvr.post_process_threshold_var)
        _perform_tk_value_set(self.high_end_process, uvr.is_high_end_process_var)


class VRArchRequest(UVREnvRequest):
    def __init__(self, model_name: str, window_size: Optional[int] = None, aggression_setting: Optional[int] = None,
                 # common option
                 gpu_conversion: Optional[bool] = None, vocal_only: Optional[bool] = None,
                 inst_only: Optional[bool] = None, 
                 no_echo_only: Optional[bool] = None, echo_only: Optional[bool] = None,  # de-echo models
                 sample_mode: Optional[bool] = None,
                 output_format: Optional[EOutputFormat] = None,
                 advanced_option: Optional[VRArchAdvancedOption] = None,
                 secondary_model_option: Optional[SecondaryModelOption] = None) -> None:
        super().__init__(EProcessMethod.VR_MODE, model_name, gpu_conversion, vocal_only, inst_only, no_echo_only,
                         echo_only, sample_mode, output_format)
        self.window_size = window_size
        self.aggression_setting = aggression_setting
        self.advanced_option = advanced_option
        self.secondary_model_option = secondary_model_option

    def check_setup_prerequisite(self, uvr: 'MainWindowOverwrite') -> bool:
        if not super().check_setup_prerequisite(uvr):
            return False
        if self.advanced_option is not None and not self.advanced_option.check_setup_prerequisite(uvr):
            return False
        if self.secondary_model_option is not None and not self.secondary_model_option.check_setup_prerequisite(uvr):
            return False
        return True

    def _select_model(self, uvr: 'MainWindowOverwrite') -> None:
        uvr.vr_model_var.set(self.model_name)
        uvr.selection_action_models(self.model_name)

    def setup(self, uvr: 'MainWindowOverwrite') -> None:
        super().setup(uvr)
        _perform_tk_value_set(self.window_size, uvr.window_size_var)
        _perform_tk_value_set(self.aggression_setting, uvr.aggression_setting_var)
        if self.advanced_option is not None:
            self.advanced_option.setup(uvr)
        if self.secondary_model_option is not None:
            self.secondary_model_option.setup(uvr)


class MDXNetArchAdvancedOption(IEnvAutomation):
    def __init__(self, volume_compensation: Optional[Union[str, float]] = None,
                 shift_conversion_pitch: Optional[int] = None, denoise_output: Optional[str] = None,
                 match_freq_cut_off: Optional[bool] = None, spectral_inversion: Optional[bool] = None) -> None:
        super().__init__()
        self.volume_compensation = volume_compensation
        self.shift_conversion_pitch = shift_conversion_pitch
        self.denoise_output = denoise_output
        self.match_freq_cut_off = match_freq_cut_off
        self.spectral_inversion = spectral_inversion

    def check_setup_prerequisite(self, uvr: 'MainWindowOverwrite') -> bool:
        if self.denoise_output is not None and self.denoise_output not in constants.MDX_DENOISE_OPTION:
            return _record_last_check_failed_source('Invalid denoise_output')
        return True

    def setup(self, uvr: 'MainWindowOverwrite') -> None:
        _perform_tk_value_set(self.volume_compensation, uvr.compensate_var)
        _perform_tk_value_set(self.shift_conversion_pitch, uvr.semitone_shift_var)
        _perform_tk_value_set(self.denoise_output, uvr.denoise_option_var)
        _perform_tk_value_set(self.match_freq_cut_off, uvr.is_match_frequency_pitch_var)
        _perform_tk_value_set(self.spectral_inversion, uvr.is_invert_spec_var)


class MDXNet23ArchOnlyAdvancedOption(IEnvAutomation):
    def __init__(self, batch_size: Optional[Union[str, int]] = None, # overlap: Optional[int] = None,
                 segment_default: Optional[bool] = None, combine_stems: Optional[bool] = None) -> None:
        super().__init__()
        self.batch_size = batch_size
        # self.overlap = overlap
        self.segment_default = segment_default
        self.combine_stems = combine_stems

    def check_setup_prerequisite(self, uvr: 'MainWindowOverwrite') -> bool:
        return True

    def setup(self, uvr: 'MainWindowOverwrite') -> None:
        _perform_tk_value_set(self.batch_size, uvr.mdx_batch_size_var)
        _perform_tk_value_set(self.segment_default, uvr.is_mdx_c_seg_def_var)
        _perform_tk_value_set(self.combine_stems, uvr.is_mdx23_combine_stems_var)


class MDXNetArchRequest(UVREnvRequest):
    def __init__(self, model_name: str, segment_size: Optional[int] = None, overlap: Optional[Union[float, str]] = None,
                 # common option
                 gpu_conversion: Optional[bool] = None, vocal_only: Optional[bool] = None,
                 inst_only: Optional[bool] = None, sample_mode: Optional[bool] = None,
                 output_format: Optional[EOutputFormat] = None,
                 advanced_option: Optional[MDXNetArchAdvancedOption] = None,
                 mdxnet23_advanced_option: Optional[MDXNet23ArchOnlyAdvancedOption] = None,
                 secondary_model_option: Optional[SecondaryModelOption] = None) -> None:
        super().__init__(EProcessMethod.MDX_MODE, model_name, gpu_conversion, vocal_only, inst_only, None, None,
                         sample_mode, output_format)
        self.segment_size = segment_size
        self.overlap = overlap
        self.advanced_option = advanced_option
        self.mdxnet23_advanced_option = mdxnet23_advanced_option
        self.secondary_model_option = secondary_model_option

    def check_setup_prerequisite(self, uvr: 'MainWindowOverwrite') -> bool:
        if not super().check_setup_prerequisite(uvr):
            return False
        # both segment_size and overlap supports user input
        if self.advanced_option is not None and not self.advanced_option.check_setup_prerequisite(uvr):
            return False
        if self.mdxnet23_advanced_option is not None and not self.mdxnet23_advanced_option.check_setup_prerequisite(uvr):
            return False
        if self.secondary_model_option is not None and not self.secondary_model_option.check_setup_prerequisite(uvr):
            return False
        return True

    def _select_model(self, uvr: 'MainWindowOverwrite') -> None:
        uvr.mdx_net_model_var.set(self.model_name)
        uvr.selection_action_models(self.model_name)

    def setup(self, uvr: 'MainWindowOverwrite') -> None:
        super().setup(uvr)
        _perform_tk_value_set(self.segment_size, uvr.mdx_segment_size_var)
        # both overlap_mdx_var and overlap_mdx23_var are controlled using this variable
        _perform_tk_value_set(self.overlap, uvr.overlap_mdx_var)
        _perform_tk_value_set(self.overlap, uvr.overlap_mdx23_var)
        if self.advanced_option is not None:
            self.advanced_option.setup(uvr)
        if self.mdxnet23_advanced_option is not None:
            self.mdxnet23_advanced_option.setup(uvr)
        if self.secondary_model_option is not None:
            self.secondary_model_option.setup(uvr)


class DemucsArchAdvancedOption(IEnvAutomation):
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

    def check_setup_prerequisite(self, uvr: 'MainWindowOverwrite') -> bool:
        return True

    def setup(self, uvr: 'MainWindowOverwrite') -> None:
        _perform_tk_value_set(self.shift, uvr.shifts_var)
        _perform_tk_value_set(self.overlap, uvr.overlap_var)
        _perform_tk_value_set(self.shift_conversion_pitch, uvr.semitone_shift_var)
        _perform_tk_value_set(self.split_mode, uvr.is_split_mode_var)
        _perform_tk_value_set(self.combine_stems, uvr.is_demucs_combine_stems_var)
        _perform_tk_value_set(self.spectral_inversion, uvr.is_invert_spec_var)


class DemucsArchRequest(UVREnvRequest):
    def __init__(self, model_name: str, stem: Optional[str] = constants.VOCAL_STEM, segment: Optional[str] = None,
                 primary_stem_only: Optional[bool] = None, secondary_stem_only: Optional[bool] = None,
                 # common option
                 gpu_conversion: Optional[bool] = None, sample_mode: Optional[bool] = None,
                 output_format: Optional[EOutputFormat] = None,
                 advanced_option: Optional[DemucsArchAdvancedOption] = None,
                 secondary_model_option: Optional[SecondaryModelOption] = None) -> None:
        # TODO: pre-process model
        super().__init__(EProcessMethod.DEMUCS_MODE, model_name, gpu_conversion, None, None, None, None, sample_mode,
                         output_format)
        self.stem = stem
        self.primary_stem_only = primary_stem_only
        self.secondary_stem_only = secondary_stem_only
        self.segment = segment
        self.advanced_option = advanced_option
        self.secondary_model_option = secondary_model_option

    def check_setup_prerequisite(self, uvr: 'MainWindowOverwrite') -> bool:
        if not super().check_setup_prerequisite(uvr):
            return False
        if self.stem is not None:
            # logic in update_button_states()
            if constants.DEMUCS_UVR_MODEL in self.model_name:
                all_stems = constants.DEMUCS_2_STEM_OPTIONS
            elif constants.DEMUCS_6_STEM_MODEL in self.model_name:
                all_stems = constants.DEMUCS_6_STEM_OPTIONS
            else:
                all_stems = constants.DEMUCS_4_STEM_OPTIONS
            if self.stem not in all_stems:
                return _record_last_check_failed_source('Invalid stem')
        if self.advanced_option is not None and not self.advanced_option.check_setup_prerequisite(uvr):
            return False
        if self.secondary_model_option is not None and not self.secondary_model_option.check_setup_prerequisite(uvr):
            return False
        return True

    def _select_model(self, uvr: 'MainWindowOverwrite') -> None:
        uvr.demucs_model_var.set(self.model_name)

    def setup(self, uvr: 'MainWindowOverwrite') -> None:
        super().setup(uvr)
        _perform_tk_value_set(self.primary_stem_only, uvr.is_primary_stem_only_var)
        _perform_tk_value_set(self.secondary_stem_only, uvr.is_secondary_stem_only_var)
        _perform_tk_value_set(self.stem, uvr.demucs_stems_var)
        if self.advanced_option is not None:
            self.advanced_option.setup(uvr)
        if self.secondary_model_option is not None:
            self.secondary_model_option.setup(uvr)


class EnsembleModeAdvancedOption(IEnvAutomation):
    def __init__(self, save_all_outputs: Optional[bool] = None, append_ensemble_name: Optional[bool] = None,
                  ensemble_waveforms: Optional[bool] = None) -> None:
        super().__init__()
        self.save_all_outputs = save_all_outputs
        self.append_ensemble_name = append_ensemble_name
        self.ensemble_waveforms = ensemble_waveforms

    def check_setup_prerequisite(self, uvr: 'MainWindowOverwrite') -> bool:
        return True

    def setup(self, uvr: 'MainWindowOverwrite') -> None:
        _perform_tk_value_set(self.save_all_outputs, uvr.is_save_all_outputs_ensemble_var)
        _perform_tk_value_set(self.append_ensemble_name, uvr.is_append_ensemble_name_var)
        _perform_tk_value_set(self.ensemble_waveforms, uvr.is_wav_ensemble_var)


class EnsembleModeRequest(UVREnvRequest):
    def __init__(self, model_name: List[str], stem_pair: Optional[str] = None,
                 ensemble_algorithm: Optional[str] = None, primary_stem_only: Optional[bool] = None,
                 secondary_stem_only: Optional[bool] = None, gpu_conversion: Optional[bool] = None,
                 sample_mode: Optional[bool] = None, output_format: Optional[EOutputFormat] = None,
                 advanced_option: Optional[EnsembleModeAdvancedOption] = None) -> None:
        super().__init__(EProcessMethod.ENSEMBLE_MODE, model_name, gpu_conversion, None, None, None, None, sample_mode,
                         output_format)
        self.stem_pair = stem_pair
        self.ensemble_algorithm = ensemble_algorithm
        self.primary_stem_only = primary_stem_only
        self.secondary_stem_only = secondary_stem_only
        self.advanced_option = advanced_option

    def _check_ensemble_algorithm(self, uvr):
        if self.ensemble_algorithm is not None:
            stem_pair = self.stem_pair if self.stem_pair is not None else uvr.ensemble_main_stem_var.get()
            if stem_pair == constants.CHOOSE_STEM_PAIR:
                return True
            elif stem_pair == constants.FOUR_STEM_ENSEMBLE and self.ensemble_algorithm not in constants.ENSEMBLE_TYPE_4_STEM:
                return _record_last_check_failed_source('Invalid ensemble_algorithm')
            elif self.ensemble_algorithm not in constants.ENSEMBLE_TYPE:
                return _record_last_check_failed_source('Invalid ensemble_algorithm')
        return True

    def check_setup_prerequisite(self, uvr: 'MainWindowOverwrite') -> bool:
        if not super().check_setup_prerequisite(uvr):
            return False
        if len(self.model_name) < 2:
            return _record_last_check_failed_source('Not enough models: at least 2 are required for ensemble mode')
        # model_name is not checked yet, leave a runtime warning message during setup()
        if self.stem_pair is not None and self.stem_pair not in constants.ENSEMBLE_MAIN_STEM:
            return _record_last_check_failed_source('Invalid stem_pair')
        if not self._check_ensemble_algorithm(uvr):
            return False
        if self.advanced_option is not None and not self.advanced_option.check_setup_prerequisite(uvr):
            return False
        return True

    def _select_model(self, uvr: 'MainWindowOverwrite') -> None:
        pass

    def _select_ensemble_models(self, uvr: 'MainWindowOverwrite') -> None:
        while uvr.ensemble_listbox_Option['state'] == 'disabled':
            sleep(1)
        target_models = set(self.model_name)
        selected_models = set()
        uvr.ensemble_listbox_Option.selection_clear(0, 'end')
        for i in range(uvr.ensemble_listbox_Option.size()):
            if uvr.ensemble_listbox_Option.get(i) in target_models:
                uvr.ensemble_listbox_Option.selection_set(i)
                selected_models.add(uvr.ensemble_listbox_Option.get(i))
        if target_models != selected_models:
            warn(f'Target ensemble models are not fully set, requested: {sorted(target_models)}, '
                 f'real: {sorted(selected_models)}', RuntimeWarning)

    def setup(self, uvr: 'MainWindowOverwrite') -> None:
        super().setup(uvr)
        _perform_tk_value_set(self.stem_pair, uvr.ensemble_main_stem_var)
        uvr.selection_action_ensemble_stems(self.stem_pair)
        _perform_tk_value_set(self.ensemble_algorithm, uvr.ensemble_type_var)
        self._select_ensemble_models(uvr)
        _perform_tk_value_set(self.primary_stem_only, uvr.is_primary_stem_only_var)
        _perform_tk_value_set(self.secondary_stem_only, uvr.is_secondary_stem_only_var)
        if self.advanced_option is not None:
            self.advanced_option.setup(uvr)


class UVRRequest(IEnvAutomation):
    def __init__(self, input_file: Union[str, List[str]], output_file: str,
                 env: Optional[UVREnvRequest] = None) -> None:
        if isinstance(input_file, str):
            input_file = [input_file]
        elif not isinstance(input_file, list):
            input_file = list(input_file)
        self.input_file = input_file
        self.output_file = output_file
        self._output_dir = self._find_parent_dir(output_file)
        self.env = env

    @staticmethod
    def _find_parent_dir(path: str) -> Optional[str]:
        parent_dir = os.path.abspath(os.path.join(path, os.path.pardir))
        if not os.path.exists(path):
            return parent_dir
        if os.path.isdir(path):
            return path
        if os.path.isfile(path):
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir)
                return parent_dir
            if not os.path.isdir(parent_dir):
                return None
            return parent_dir

    def check_setup_prerequisite(self, uvr: 'MainWindowOverwrite') -> bool:
        if self._output_dir is None:
            return _record_last_check_failed_source('Invalid output_path: no parent directory found')
        if self.env is not None and not self.env.check_setup_prerequisite(uvr):
            return False
        return True

    def setup(self, uvr: 'MainWindowOverwrite') -> None:
        if self.env is not None:
            self.env.setup(uvr)
        uvr.inputPaths = [x.replace('\\', '/') for x in self.input_file]
        uvr.update_inputPaths()
        uvr.export_path_var.set(self._output_dir.replace('\\', '/'))



def _json_to_dict(request_json: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(request_json, str):
        return json.loads(request_json)
    return request_json.copy()


def deserialize_request_json(request_json: Union[str, Dict[str, Any]]) -> UVRRequest:
    request_json = _json_to_dict(request_json)
    env = request_json.pop('env', None)
    if env is not None:
        request_json['env'] = deserialize_env_request_json(env)
    return UVRRequest(**request_json)


def deserialize_env_request_json(request_json: Union[str, Dict[str, Any]]) -> UVREnvRequest:
    env = _json_to_dict(request_json)
    req_type = EProcessMethod(env.pop('process_method'))
    if req_type == EProcessMethod.VR_MODE:
        advanced_opt_cls = {'advanced_option': VRArchAdvancedOption}
        base_cls = VRArchRequest
    elif req_type == EProcessMethod.MDX_MODE:
        advanced_opt_cls = {'advanced_option': MDXNetArchAdvancedOption,
                            'mdxnet23_advanced_option': MDXNet23ArchOnlyAdvancedOption}
        base_cls = MDXNetArchRequest
    elif req_type == EProcessMethod.DEMUCS_MODE:
        advanced_opt_cls = {'advanced_option': DemucsArchAdvancedOption}
        base_cls = DemucsArchRequest
    elif req_type == EProcessMethod.ENSEMBLE_MODE:
        advanced_opt_cls = {'advanced_option': EnsembleModeAdvancedOption}
        base_cls = EnsembleModeRequest
    else:
        raise ValueError('Invalid process_method')
    secondary_model_args = env.pop('secondary_model_option', None)
    for key, cls in advanced_opt_cls.items():
        opt_args = env.pop(key, None)
        if opt_args is None:
            continue
        env[key] = cls(**opt_args)
    if secondary_model_args is not None:
        env['secondary_model_option'] = SecondaryModelOption(**secondary_model_args)
    return base_cls(**env)


def _serialize_env_node(action: IEnvAutomation):
    attrs = {}
    for k, v in action.__dict__.items():
        if k.startswith('_') or v is None:
            continue
        if isinstance(v, IEnvAutomation):
            attrs[k] = _serialize_env_node(v)
        else:
            attrs[k] = v
    return attrs


def serialize_request_json(uvr_request: Union[UVREnvRequest, UVRRequest]) -> str:
    root = _serialize_env_node(uvr_request)
    return json.dumps(root, separators=(',', ':'))


def _run_main():
    args = {
        'input_file': 'D:/CloudMusic/Noa - 暁に咲く華.flac',
        'output_file': 'Z:/',
        'env': {
            'process_method': EProcessMethod.VR_MODE,
            'model_name': '4_HP-Vocal-UVR',
            'gpu_conversion': True,
            'inst_only': False,
            'vocal_only': False,
            'secondary_model_option': {
                'activate': False
            }
        }
    }
    req = deserialize_request_json(args)
    print(req)
    print(serialize_request_json(req))
    import UVR
    while True:
        uvr = getattr(UVR, 'root', None)
        if uvr is None:
            sleep(1)
            continue
        break
    print('Wait mainloop signal')
    uvr.gui_set_finished.wait()
    # sleep(3)  # delay start
    check = req.check_setup_prerequisite(uvr)
    print(f'check_setup_prerequisite: {check}')
    if check:
        req.setup(uvr)


def _server_main():
    from uvr_api_server import main
    t = threading.Thread(target=_run_main, daemon=True)
    t.start()
    main()


if __name__ == '__main__':
    _server_main()
