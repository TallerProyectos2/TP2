# Conda Activate Loads TP2 Runtime Env

- Date: `2026-04-13`
- Machine: `tp2-EPC`
- Scope:
  - make `conda activate tp2` sufficient for TP2 car and inference scripts
  - remove the need for manual `source /home/tp2/.config/tp2/*.env`
  - keep secrets machine-local and outside the repository

## Applied Change

- Installed Conda activation hook:
  - `/home/tp2/miniforge3/envs/tp2/etc/conda/activate.d/tp2-runtime.sh`
- Installed Conda deactivate hook:
  - `/home/tp2/miniforge3/envs/tp2/etc/conda/deactivate.d/tp2-runtime.sh`
- Ran Conda shell initialization for:
  - `zsh`
  - `bash`
- The activation hook loads:
  - `/home/tp2/.config/tp2/inference.env`
  - fallback `/home/tp2/.config/tp2/coche-jetson.env`
- The hook sets runtime defaults for:
  - `DISPLAY`
  - `PYTHONNOUSERSITE`
  - `TP2_INFERENCE_MODE`
  - `TP2_INFERENCE_TARGET`
  - `ROBOFLOW_LOCAL_API_URL`
  - `ROBOFLOW_WORKSPACE`
  - `ROBOFLOW_WORKFLOW`

## Runtime Evidence

- Zsh interactive activation:
  - command: `zsh -lic 'conda activate tp2 && python ...'`
  - `CONDA_PREFIX`: `/home/tp2/miniforge3/envs/tp2`
  - `TP2_ACTIVE_ENV_FILE`: `/home/tp2/.config/tp2/inference.env`
  - `ROBOFLOW_LOCAL_API_URL`: `http://100.115.99.8:9001`
  - `TP2_INFERENCE_MODE`: `local`
  - `TP2_INFERENCE_TARGET`: `workflow`
  - `ROBOFLOW_WORKSPACE`: `1-v8mk1`
  - `ROBOFLOW_WORKFLOW`: `custom-workflow-2`
  - `DISPLAY`: `:1`
  - `PYTHONNOUSERSITE`: `1`
  - `ROBOFLOW_API_KEY_LEN`: `20`
- Bash interactive activation:
  - `BASH_CONDA_PREFIX`: `/home/tp2/miniforge3/envs/tp2`
  - `BASH_ROBOFLOW_LOCAL_API_URL`: `http://100.115.99.8:9001`
  - `BASH_ROBOFLOW_API_KEY_LEN`: `20`
- Inference validation after only `conda activate tp2`:
  - script: `/home/tp2/TP2_red4G/servicios/inferencia.py`
  - API URL: `http://100.115.99.8:9001`
  - detections: `1`
  - detected class: `stop sign`
- Live car runtime restarted using only:
  - `conda activate tp2`
  - `python -u coche.py`
- Live process evidence:
  - UDP bind: `172.16.0.1:20001`
  - startup log: `Inference: enabled (local/workflow) endpoint=http://100.115.99.8:9001`

## Result

Operators can now use the normal Conda flow on EPC:

```bash
conda activate tp2
cd /home/tp2/TP2_red4G/servicios
python -u coche.py
```

No manual `source` of TP2 env files is required. Secrets remain stored only in machine-local files.
