# Jetson Orin Nano Setup Runbook

## Scope

This document explains how to prepare a `NVIDIA Jetson Orin Nano Developer Kit` from zero to the point where it can run the TP2 inference workload as an inference-only node.

Last validated integrated lab state (`2026-03-26`):

- Jetson service: `tp2-roboflow-inference.service`
- Reachable endpoint from EPC: `http://100.115.99.8:9001`
- EPC keeps the live UDP control loop on `172.16.0.1:20001`
- Secrets are stored only on the machine, outside the repository

Current model target (`2026-04-14`):

- EPC calls the Jetson endpoint with `TP2_INFERENCE_TARGET=model`.
- Active Roboflow model ID: `tp2-g4-2026/2`.
- Evidence: `docs/logs/validations/2026-04-14-jetson-local-roboflow-model-tp2-g4-2026-2.md`

Current availability note (`2026-04-13`):

- Jetson was temporarily unreachable from EPC over Tailscale and management LAN
- Jetson was later recovered by starting `tp2-roboflow-inference.service`
- EPC -> Jetson inference over `http://100.115.99.8:9001` is validated again
- Evidence:
  - incident: `docs/logs/incidents/2026-04-13-jetson-unreachable.md`
  - recovery: `docs/logs/validations/2026-04-13-jetson-remote-inference-restored.md`

Target operating model:

- Jetson runs inference only.
- EPC remains the integration and control host.
- Car does not call Jetson directly.
- Jetson exposes an HTTP inference endpoint compatible with the current scripts in `servicios/`.
- Default inference port is `9001/TCP`.

This runbook is aligned with:

- `PLAN.md` step 4: integrate Jetson without breaking the current EPC path.
- `ARCHITECTURE.md`: Jetson is inference-only offload.
- `docs/INFERENCE.md`: keep compatibility with `servicios/inferencia.py`.
- `docs/NETWORK.md`: Jetson must be reachable from EPC, not from the car.

## Important Constraint

The repository operating contract forbids firmware updates on project hardware. Because of that, this runbook assumes one of these two cases:

1. Your board already boots the selected JetPack image without requiring a QSPI/bootloader update.
2. You have explicit approval outside this repo workflow to perform the required bootloader update.

If the board refuses to boot a JetPack 6.x SD image because it requires a QSPI/UEFI update, stop there and escalate before changing firmware.

## Recommended Target State

- OS family: Ubuntu via JetPack
- Preferred path: JetPack `6.2.x` if the board already supports it
- Safe network position: wired Ethernet behind your router/switch
- Runtime model: Roboflow Inference on Jetson, called from EPC over HTTP
- Validation client: existing repo script `servicios/inferencia.py`

## Current Lab Identity

- Hostname reference: `tp2-jetson`
- Primary SSH user: `grupo4`
- Current management IP: `192.168.72.127`
- Current Tailscale IP: `100.115.99.8`
- Primary SSH command:

```bash
ssh grupo4@tp2-jetson
```

- Alternate SSH commands:

```bash
ssh grupo4@100.115.99.8
ssh grupo4@192.168.72.127
```

## Source References

The procedure below was checked against these primary sources on `2026-03-10`:

- NVIDIA Jetson Orin Nano Developer Kit getting started:
  - <https://developer.nvidia.com/embedded/learn/get-started-jetson-orin-nano-devkit>
- NVIDIA JetPack install and setup:
  - <https://docs.nvidia.com/jetson/jetpack/install-setup/>
- NVIDIA Jetson AI Lab initial setup guide:
  - <https://www.jetson-ai-lab.com/initial_setup_jon.html>
- Roboflow self-hosted inference on Jetson:
  - <https://docs.roboflow.com/deploy/self-hosted/self-hosted-inference/inference-on-nvidia-jetson>

## Hardware And Files You Need

- 1 Jetson Orin Nano Developer Kit
- Power supply for the kit
- Monitor + keyboard for first boot, or a known-good serial/headless procedure
- 1 microSD card, at least `64 GB`, UHS-I recommended
- Optional NVMe SSD if you want the runtime on faster storage later
- 1 Ethernet cable
- 1 host PC to prepare the SD card
- Your Roboflow API key
- Your Roboflow model id, for example `my-project/3`
- This repository available either on the Jetson or reachable from EPC

## High-Level Deployment Plan

1. Prepare a JetPack image.
2. Boot the Jetson and complete Ubuntu first-boot.
3. Update Ubuntu packages and install base tools.
4. Put the Jetson on the trusted wired network.
5. Install Docker and verify NVIDIA runtime support.
6. Launch Roboflow Inference on the Jetson.
7. Run `servicios/inferencia.py` against the Jetson endpoint.
8. Point the EPC to the Jetson endpoint with environment variables.
9. Keep EPC local inference available as fallback until automatic fallback is implemented.

## 1. Choose The Provisioning Path

### Path A: microSD image

Use this when you want the quickest start and the lowest operational complexity.

This is the best first pass for TP2.

### Path B: SDK Manager to NVMe or USB

Use this when you already know you want the root filesystem on NVMe or need a fully managed flash from an Ubuntu host PC.

For TP2, start with Path A unless you already have a reason to standardize on NVMe boot.

## 2. Flash The Jetson

### Path A: Flash A microSD Card

1. From the NVIDIA getting-started page, download the correct JetPack SD image for `Jetson Orin Nano Developer Kit`.
2. On the host PC, install a flashing utility such as `balenaEtcher`.
3. Insert the microSD card into the host PC.
4. Flash the downloaded image to the microSD card.
5. Safely eject the card.
6. Insert the microSD card into the Jetson.
7. Connect monitor, keyboard, Ethernet, and power.
8. Power on the Jetson.

### If The Board Does Not Boot The Selected Image

Do not force it.

Symptoms usually look like:

- black screen or boot loop with a new JetPack image
- UEFI complaining about incompatible boot content
- the board only boots an older SD image

If that happens:

- stop and confirm whether a QSPI/bootloader update is required
- if a firmware update is required, get explicit approval first because it is outside the repo non-negotiables

### Path B: Flash With SDK Manager

Use this only if you intentionally want host-assisted flashing.

1. Prepare an Ubuntu host PC supported by NVIDIA SDK Manager.
2. Install `SDK Manager`.
3. Put the Jetson in recovery mode.
4. Connect the Jetson to the host PC over USB.
5. In SDK Manager, choose the correct Orin Nano developer kit target.
6. Select the JetPack release you want to deploy.
7. Choose the target storage, typically NVMe if installed.
8. Complete the flash and first-boot setup.

For TP2, once the board is flashed, the rest of the runbook is the same.

## 3. Complete Ubuntu First Boot

On the first boot:

1. Create the admin user.
2. Set the hostname to something stable such as `tp2-jetson`.
3. Set the correct timezone.
4. Join the wired network.
5. Finish the OEM wizard.
6. Reboot once if the wizard requests it.

After login, verify the board identity:

```bash
hostnamectl
uname -a
cat /etc/os-release
```

Check the installed Jetson package baseline:

```bash
dpkg-query --show nvidia-l4t-core
```

If available, also record:

```bash
cat /etc/nv_tegra_release
```

## 4. Update The Base System

Run:

```bash
sudo apt update
sudo apt dist-upgrade -y
sudo apt autoremove -y
sudo reboot
```

After reboot, install the basic tools used in TP2 operations:

```bash
sudo apt install -y \
  git curl wget jq vim htop tmux net-tools \
  python3-pip python3-venv python3-dev \
  docker.io
```

Add your user to Docker:

```bash
sudo usermod -aG docker "$USER"
newgrp docker
```

Verify Docker works:

```bash
docker version
docker info
```

On JetPack systems, `docker info` should show an NVIDIA-capable runtime. If it does not, reboot once and re-check. If it still does not appear, treat that as a provisioning problem and fix the JetPack installation before continuing.

## 5. Put The Jetson On The TP2 Trusted Network

For lab use, connect the Jetson to your router/switch by Ethernet.

Recommended network policy:

- Jetson and EPC on the same trusted management LAN
- DHCP reservation or static IP for the Jetson
- do not expose the Jetson inference port to the whole campus or public network
- allow `9001/TCP` only from EPC and operator hosts if needed

Find the active IP:

```bash
ip -4 addr show
ip route
```

If you use DHCP reservation, note the chosen IP and later document it in `docs/NETWORK.md` and `MACHINES.md` when the deployment becomes real.

Test local network reachability:

```bash
ping -c 4 <gateway_ip>
```

## 6. Apply Minimal Hardening

Enable SSH if needed:

```bash
sudo systemctl enable --now ssh
sudo systemctl status ssh --no-pager
```

Optional but recommended: enable `ufw` and restrict access.

Example:

```bash
sudo ufw allow 22/tcp
sudo ufw allow from <EPC_IP> to any port 9001 proto tcp
sudo ufw enable
sudo ufw status verbose
```

Do not open `9001/TCP` to `Anywhere` unless you have a very good reason.

### Tailscale Without `apt`

If Ubuntu package management is temporarily broken on the Jetson because of unrelated CUDA or NVIDIA dependencies, use the repository helper script that installs the official static ARM64 Tailscale binaries without touching `apt`:

```bash
cd ~/workspace/TP2_red4G
bash scripts/install_tailscale_jetson.sh
```

Optional environment variables supported by the script:

```bash
TAILSCALE_VERSION=1.94.2
TAILSCALE_HOSTNAME=tp2-jetson
TAILSCALE_AUTHKEY=tskey-xxxx
```

## 7. Optional Performance Preparation

Before sustained inference sessions, inspect the available power profiles:

```bash
sudo nvpmodel -q --verbose
```

Select the high-performance mode you actually want to use for the session, then pin clocks:

```bash
sudo nvpmodel -m <performance_mode_id>
sudo jetson_clocks
```

The exact mode id can vary by release and board profile, so query first instead of assuming a fixed number.

Monitor thermals and clocks during validation:

```bash
tegrastats
```

## 8. Start Roboflow Inference On The Jetson

Roboflow documents Jetson-specific inference images by JetPack line. For JetPack `6.2.x`, the documented image is:

```text
roboflow/roboflow-inference-server-jetson-6.2.0:latest
```

If you intentionally stay on an older JetPack line, use the matching Jetson image family documented by Roboflow instead of mixing arbitrary versions.

### One-Shot Manual Start

Start the container:

```bash
docker run --rm \
  --name tp2-roboflow-inference \
  --runtime=nvidia \
  --network=host \
  -e ROBOFLOW_API_KEY='<YOUR_ROBOFLOW_API_KEY>' \
  roboflow/roboflow-inference-server-jetson-6.2.0:latest
```

What this does:

- `--runtime=nvidia` gives the container access to Jetson acceleration
- `--network=host` exposes the HTTP API directly on the Jetson host network
- `ROBOFLOW_API_KEY` lets the inference server download and run your hosted model

Keep this terminal open for the first test.

### Health Check The Server

In another shell on the Jetson:

```bash
curl -fsS http://127.0.0.1:9001/openapi.json | jq '.info.title'
ss -ltnp | grep 9001
```

Expected result:

- HTTP responds successfully
- TCP port `9001` is listening

## 9. Make The Inference Service Persistent

For lab operations, a systemd unit is better than a manually held shell.

Create an environment file:

```bash
sudo install -d -m 0755 /etc/tp2
sudoedit /etc/tp2/jetson-inference.env
```

Put only the secrets you need there:

```bash
ROBOFLOW_API_KEY=<YOUR_ROBOFLOW_API_KEY>
```

Create the service file:

```bash
sudoedit /etc/systemd/system/tp2-roboflow-inference.service
```

Use:

```ini
[Unit]
Description=TP2 Roboflow Inference on Jetson
After=network-online.target docker.service
Wants=network-online.target
Requires=docker.service

[Service]
Type=simple
EnvironmentFile=/etc/tp2/jetson-inference.env
ExecStartPre=-/usr/bin/docker rm -f tp2-roboflow-inference
ExecStart=/usr/bin/docker run --rm --name tp2-roboflow-inference --runtime=nvidia --network=host --env-file /etc/tp2/jetson-inference.env roboflow/roboflow-inference-server-jetson-6.2.0:latest
ExecStop=/usr/bin/docker stop tp2-roboflow-inference
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now tp2-roboflow-inference.service
sudo systemctl status tp2-roboflow-inference.service --no-pager
```

Read logs with:

```bash
journalctl -u tp2-roboflow-inference.service -f
```

## 10. Clone The TP2 Repo On The Jetson

If you want to validate using the exact same client scripts:

```bash
mkdir -p ~/workspace
cd ~/workspace
git clone <YOUR_TP2_REPO_URL> TP2_red4G
cd TP2_red4G
```

Create a Python environment for the local test tools:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  inference inference-sdk uvicorn \
  opencv-python-headless gradio
```

This is only for test tooling on the Jetson. The production inference path can stay containerized.

## 11. Run A Local Validation On The Jetson With Repo Scripts

Use your existing script `servicios/inferencia.py`.

Export the variables:

```bash
cd ~/workspace/TP2_red4G
source .venv/bin/activate

export TP2_INFERENCE_MODE=local
export TP2_INFERENCE_TARGET=model
export ROBOFLOW_LOCAL_API_URL=http://127.0.0.1:9001
export ROBOFLOW_API_KEY='<YOUR_ROBOFLOW_API_KEY>'
export ROBOFLOW_MODEL_ID='<YOUR_PROJECT>/<VERSION>'
export TP2_TEST_IMAGE="$PWD/servicios/test.jpg"
```

Run the check:

```bash
python servicios/inferencia.py
```

Expected result:

- no exception
- JSON output printed to the terminal
- annotated output image created next to the input image, usually `servicios/test_pred.jpg`

If your Roboflow asset is a workflow instead of a direct model deployment, use:

```bash
export TP2_INFERENCE_TARGET=workflow
export ROBOFLOW_WORKSPACE='<YOUR_WORKSPACE>'
export ROBOFLOW_WORKFLOW='<YOUR_WORKFLOW>'
python servicios/inferencia.py
```

## 12. Validate From EPC To Jetson

Once the Jetson passes the local test, move to the real integration path: EPC calling Jetson over HTTP.

On the EPC, keep using the current script contract and point it to the Jetson IP:

```bash
export TP2_INFERENCE_MODE=local
export TP2_INFERENCE_TARGET=model
export ROBOFLOW_LOCAL_API_URL=http://<JETSON_IP>:9001
export ROBOFLOW_API_KEY='<YOUR_ROBOFLOW_API_KEY>'
export ROBOFLOW_MODEL_ID='<YOUR_PROJECT>/<VERSION>'
export TP2_TEST_IMAGE=/home/tp2/servicios_tp2/test.jpg

python3 /home/tp2/servicios_tp2/inferencia.py
```

Validate on EPC:

```bash
curl -fsS http://<JETSON_IP>:9001/openapi.json >/dev/null
ss -tn dst <JETSON_IP>:9001
```

Validate on Jetson:

```bash
sudo systemctl status tp2-roboflow-inference.service --no-pager
journalctl -u tp2-roboflow-inference.service -n 50 --no-pager
```

## 13. Fallback Strategy

Current repo state:

- configuration-based target switching already exists
- automatic fallback is still a pending item in `PLAN.md` step 5

Operationally, that means:

- keep EPC local inference available until Jetson is proven stable
- switch the endpoint by environment variables
- if Jetson fails, point the EPC back to `http://127.0.0.1:9001`

Fallback variables on EPC:

```bash
export TP2_INFERENCE_MODE=local
export ROBOFLOW_LOCAL_API_URL=http://127.0.0.1:9001
```

If the EPC local server is not already running:

```bash
cd /home/tp2/servicios_tp2
ROBOFLOW_API_KEY='<YOUR_ROBOFLOW_API_KEY>' python3 start_local_inference_server.py --host 127.0.0.1 --port 9001
```

## 14. Optional Extra Services On The Jetson

Only add these if they solve a real operational problem.

### `ssh`

Useful for remote management. Safe and normal for this setup.

### `tailscaled`

Useful if you want operator access without depending on the local router path. Keep it limited to management, not data-plane inference from the car.

### `inferencia_gui_web.py`

Useful only for human debugging or demo inspection. It is not required for the control path.

If you run it, bind it conservatively:

```bash
cd ~/workspace/TP2_red4G
source .venv/bin/activate
TP2_INFERENCE_MODE=local \
ROBOFLOW_LOCAL_API_URL=http://127.0.0.1:9001 \
python servicios/inferencia_gui_web.py --host 127.0.0.1 --port 7860
```

Then use an SSH tunnel from the operator laptop if needed.

## 15. Validation Checklist For TP2

Minimum checks before treating the Jetson path as ready:

1. Jetson boots reliably.
2. Jetson has stable Ethernet connectivity.
3. Docker works and NVIDIA runtime is available.
4. `9001/TCP` listens on the Jetson.
5. `GET /openapi.json` responds locally.
6. `servicios/inferencia.py` runs locally on the Jetson against `127.0.0.1:9001`.
7. EPC can reach `http://<JETSON_IP>:9001/openapi.json`.
8. `servicios/inferencia.py` runs from EPC against `http://<JETSON_IP>:9001`.
9. EPC local inference remains available as fallback.

## 16. Troubleshooting

### Jetson boots old images but not JetPack 6.x

Likely cause: bootloader/QSPI mismatch.

Action: stop and escalate before changing firmware.

### `docker info` does not show NVIDIA runtime support

Likely cause: incomplete JetPack provisioning.

Action:

- reboot once
- re-check Docker
- if still broken, repair the JetPack installation before debugging Roboflow

### `curl http://127.0.0.1:9001/openapi.json` fails

Likely causes:

- container not running
- service crashed on startup
- port blocked

Action:

```bash
docker ps -a
journalctl -u tp2-roboflow-inference.service -n 100 --no-pager
ss -ltnp | grep 9001
```

### `inferencia.py` says the local endpoint is not reachable

Likely causes:

- wrong `ROBOFLOW_LOCAL_API_URL`
- service only bound locally and you are testing from EPC
- firewall blocking EPC to Jetson

Action:

- test `curl` against `/openapi.json`
- confirm Jetson IP
- confirm firewall rule

### Roboflow returns auth or model errors

Likely causes:

- wrong `ROBOFLOW_API_KEY`
- wrong `ROBOFLOW_MODEL_ID`
- using `model` when you actually need `workflow`

Action:

- verify the project slug and version in Roboflow
- verify whether your deployment is a model id or a workflow id

### The board overheats or throttles

Action:

- inspect `tegrastats`
- use active cooling
- review the selected power profile

## 17. What To Document After Real Deployment

When the Jetson is actually integrated in the lab, update:

- `MACHINES.md` with the final hostname and access path
- `docs/NETWORK.md` with the final Jetson IP or reserved DHCP lease
- `docs/INFERENCE.md` with the chosen endpoint strategy
- `docs/logs/validations/` with the first successful EPC -> Jetson inference evidence

## 18. Minimal Command Summary

Bring up the Jetson inference service:

```bash
sudo systemctl enable --now tp2-roboflow-inference.service
curl -fsS http://127.0.0.1:9001/openapi.json | jq '.info.title'
```

Test on the Jetson with repo scripts:

```bash
cd ~/workspace/TP2_red4G
source .venv/bin/activate
export TP2_INFERENCE_MODE=local
export TP2_INFERENCE_TARGET=model
export ROBOFLOW_LOCAL_API_URL=http://127.0.0.1:9001
export ROBOFLOW_API_KEY='<YOUR_ROBOFLOW_API_KEY>'
export ROBOFLOW_MODEL_ID='<YOUR_PROJECT>/<VERSION>'
python servicios/inferencia.py
```

Test from EPC to Jetson:

```bash
export TP2_INFERENCE_MODE=local
export TP2_INFERENCE_TARGET=model
export ROBOFLOW_LOCAL_API_URL=http://<JETSON_IP>:9001
export ROBOFLOW_API_KEY='<YOUR_ROBOFLOW_API_KEY>'
export ROBOFLOW_MODEL_ID='<YOUR_PROJECT>/<VERSION>'
python3 /home/tp2/servicios_tp2/inferencia.py
```
