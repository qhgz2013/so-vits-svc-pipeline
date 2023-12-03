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

To run So-VITS-SVC pipeline, ensure `run_svc_new.py` and `uvr_api_client.py` are in the same directory first, then run `python run_svc_new.py [-c vits_config.json] [-p default] [-k]` to perform end-to-end automation.

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
