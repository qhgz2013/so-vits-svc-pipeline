# so-vits-svc-pipeline

A python CLI pipeline script for running UVR tasks, So-VITS-SVC model inference and ffmpeg merge with end-to-end automation.

## 1. Prerequisite

Before running pipeline script, the following software must be installed first:

1. [UltimateVocalRemoverGUI] v5.6.0
2. [FFMpeg]
3. Latest version of [So-VITS-SVC 4.1] (Chinese document)

### 1.1 UVR GUI

Follow the **Manual Windows Installation** instruction in [UltimateVocalRemoverGUI]. If anaconda is used, try the following command:
```cmd
conda create -n uvr python=3.9
conda activate uvr
pip install -r requirements.txt
```

**Note**: Package `Dora` (which relies on an outdated version of `scikit-learn`) and `pip` (which is already installed) are not necessary for running UVR tasks. If error encounters during installing required packages, try remove the above two lines from requirements.txt.

After installation, run UVR GUI and download required models (optional) via: `python UVR.py`

**Note2**: The root path of UVR GUI is denoted as `uvr_path`.

### 1.2 FFMpeg

Download and extract [FFMpeg], then add the executable files into `PATH` environment. Check this step by opening a terminal (`cmd`) and type `ffmpeg -version`.

### 1.3 So-VITS-SVC 4.1

Here uses a community-maintain variation of [So-VITS-SVC 4.1] for simplifying preprocessing and training. Download and extract all files, the root path of So-VITS-SVC is denoted as `vits_path` later.

## 2. How to use

This repository contains two main modules to run the pipeline: **UVR-http-service** and **pipeline-client**. Each of them is deployed to different directory.

### 2.1 UVR-http-service

Put `uvr_api.py` and `uvr_api_server.py` into `uvr_path` (the root directory where UVR GUI is extracted). Then run `python uvr_api_server.py` instead of the previous command `python UVR.py`.

This time, UVR GUI is able to accept tasks through HTTP protocol (listening on port 8090 by default) and perform the corresponding automation steps (setting up parameters, inputs and outputs).

### 2.2 pipeline-client

To run So-VITS-SVC pipeline, ensure `run_svc.py` and `uvr_api_client.py` are in the same directory first, then run `python run_svc.py <args>` to perform end-to-end automation.

**Argument description**: Most of the arguments will pass to `inference_main.py` of [So-VITS-SVC 4.1] directly without any process, except for the arguments listed below:

| Argument name                | Flag     | Description                                                                                                                                                                                         |
| :--------------------------- | :------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `-i` / `--input`             | new      | Input audio file (if `-up` and `-uh` are specified) or the substring which the output vocal/instrumental files contain. Note: the script only searches files under current working directory (CWD). |
| `-o` / `--output`            | new      | Output directory or file for the final audio.                                                                                                                                                       |
| `-vits` / `--vits_path`      | new      | The root path of So-VITS-SVC.                                                                                                                                                                       |
| `-s` / `--spk_list`          | modified | Use "\|" to separate the speaker for left and right channel, use "+" to mix multiple speakers, and use ":" to add offset (in ms, relative to instrumental track).                                   |
| `-up` / `--uvr_port`         | new      | HTTP listening port of UVR-http-service, leave -1 to disable UVR automation (default: -1).                                                                                                          |
| `-uh` / `--uvr_host`         | new      | HTTP listening host of UVR-http-service (default: localhost).                                                                                                                                       |
| `--keep`                     | new      | Keep UVR generated audio files.                                                                                                                                                                     |
| `-sp` / `--separate_process` | new      | Run So-VITS-SVC separately for two channels (FL and FR) in vocal audio.                                                                                                                             |
| `-uc` / `--uvr_config`       | new      | Path for uvr_config.json, which is used to run UVR tasks (default: None).                                                                                                                           |

A basic example:
```cmd
python run_svc.py -vits D:\so-vits-svc_2.3.11.1 -m logs/44k/G_466000.pth -c logs/44k/config.json -t 0 -s "banbenzhenling" -cr 0 -up 8090 -uc uvr_config_sample.json -o "Z:\" -i "D:\CloudMusic\Noa - 暁に咲く華.flac"
```
The pipeline script will execute:
1. Split vocal and instrumental stems of input audio `D:\CloudMusic\Noa - 暁に咲く華.flac` by UVR.
2. Run So-VITS-SVC inference for vocal stem to match target speaker `banbenzhenling`.
3. Merge new vocal stem and instrumental stem together, then save to directory `Z:\`.

Looks quite simple, doesn't it?

A more complex and common example I used in practice:

```cmd
python run_svc.py -vits D:\so-vits-svc_2.3.11.1 -m logs/44k/G_466000.pth -c logs/44k/config.json -t 0 -s "banbenzhenling|chuanchenglingzi:40" -dm logs/44k/diffusion/model_460000.pt -dc logs/44k/diffusion/config.yaml -lea 1 -cm logs/44k/feature_and_index.pkl -cr 0.5 -fr -sd -30 -f0p crepe -shd -sp -up 8090 -uc uvr_config_sample.json -o "Z:\" -i "D:\CloudMusic\Noa - 暁に咲く華.flac"
```
This example runs So-VITS-SVC inference using extra **shallow diffusion**, **loudness alignment** and **feature retrieval** features with f0 predictor **crepe**. The left channel of the final audio converts to speaker `banbenzhenling` (坂本 真綾, Sakamoto Maaya) and the right channel converts to speaker `chuanchenglingzi` (川澄 綾子, Kawasumi Ayako) with 40ms delay.  
The output audio will be saved to: `Z:\Noa - 暁に咲く華.flac`

**NOTE**: UVR-http-service must be run before pipeline-client if `--uvr_port` and `--uvr_host` is set.


### 2.3 Dive into high-quality pipeline

This section will be done in future.

## 3. Roadmap

- [x] Add `--separate_process`.
- [x] Refactor UVR-http-service to provide per-task UVR environment setup.
- [x] Use different configs to obtain high-quality vocal and instrumental stems separately. (reuse `-uc` by specifying vocal only and inst only environments, refers to `uvr_config_sep_sample.json` for detail)
- [ ] Customize volume mix factor for multiple speakers (by argument `-s spk1*<factor1>+spk2*<factor2>`).

[UltimateVocalRemoverGUI]: https://github.com/Anjok07/ultimatevocalremovergui#manual-windows-installation
[So-VITS-SVC 4.1]: https://www.yuque.com/umoubuton/ueupp5/sdahi7m5m6r0ur1r
[FFMpeg]: https://ffmpeg.org/download.html
