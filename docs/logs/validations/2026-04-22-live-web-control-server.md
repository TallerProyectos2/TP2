# 2026-04-22 - Live web control server

## Objetivo

Implementar y validar localmente el servidor web del runtime del coche para:

- ver video MJPEG en tiempo real;
- exponer snapshot y estado JSON;
- mostrar inferencia y overlays sobre el frame recibido;
- mandar control remoto manual desde navegador;
- aplicar watchdog de control para volver a neutro si el navegador deja de publicar.

## Cambios validados localmente

- `servicios/coche.py` arranca el servidor HTTP integrado en el proceso del runtime.
- Endpoints disponibles:
  - `GET /`
  - `GET /status.json`
  - `GET /video.mjpg`
  - `GET /snapshot.jpg`
  - `POST /control`
  - `POST /control/neutral`
- El control web queda limitado por:
  - `TP2_ENABLE_WEB_CONTROL`
  - `TP2_WEB_CONTROL_TIMEOUT_SEC`
  - `TP2_WEB_CONTROL_MAX_FORWARD`
  - `TP2_WEB_CONTROL_MAX_REVERSE`
- El estado web incluye contadores de paquetes, ultimo cliente, ultimo tipo de paquete, frames de video, estado de inferencia y fuente de control activa.

## Validacion local

Comando de compilacion:

```console
/Users/mario/miniconda3/envs/test/bin/python -m py_compile servicios/coche.py servicios/roboflow_runtime.py
```

Resultado: correcto, sin salida de error.

Servidor local de prueba:

```console
TP2_BIND_IP=127.0.0.1 TP2_BIND_PORT=29001 TP2_WEB_HOST=127.0.0.1 TP2_WEB_PORT=18088 TP2_ENABLE_INFERENCE=0 TP2_ENABLE_OPENCV_WINDOWS=0 /Users/mario/miniconda3/envs/test/bin/python -u servicios/coche.py
```

Salida relevante:

```console
Live web view listening on http://127.0.0.1:18088/
Manual control server listening on 127.0.0.1:29001
Inference: disabled (local/model) endpoint=http://100.115.99.8:9001
```

Prueba HTTP/UDP con frame JPEG sintetico enviado como `I` por UDP:

```json
{
  "control_source": "web",
  "control_status": 200,
  "has_video": true,
  "neutral_source": "web-timeout",
  "neutral_status": 200,
  "packet_total": 1,
  "snapshot": {
    "content_type": "image/jpeg",
    "jpeg_soi": true,
    "status": 200
  },
  "timeout_source": "web-timeout",
  "udp_reply": {
    "bytes": 17,
    "kind": "C",
    "steering": 1.0,
    "throttle": 0.2
  },
  "video_frames": 1
}
```

Prueba con frame corrupto: el servidor registro `bad_image_frames=1`, mantuvo `last_error` en el diagnostico y respondio igualmente al coche con paquete `C` de 17 bytes.

Prueba de snapshot:

```json
{
  "status": 200,
  "content_type": "image/jpeg",
  "jpeg_soi": true,
  "first_hex": "ffd8ffe0"
}
```

Resultado: el servidor local acepta control web, responde al coche por UDP con `C`, publica video/snapshot y vuelve a fuente `web-timeout` despues del timeout.

## Estado remoto observado

Comando:

```console
ops/bin/tp2-status
```

Resumen:

- EPC: `srsepc` activo, `mosquitto` activo, `tp2-car-control.service` activo.
- EPC: UDP `172.16.0.1:20001` escuchando.
- EPC: web `0.0.0.0:8088` escuchando y endpoint web responde.
- UE coche: no confirmado por `tp2-status`.
- eNodeB: link, FPGA y `srsenb` activos.
- Jetson: servicio de inferencia activo y OpenAPI accesible.

Consulta remota:

```console
curl -fsS --max-time 4 http://100.97.19.112:8088/status.json
```

Resultado observado: despliegue remoto anterior responde, pero sin frames de video (`has_video=false`, `video_frames=0`) y con inferencia en espera.

## Notas operativas

No se reinicio `tp2-car-control.service` en EPC durante esta validacion para no interrumpir el runtime activo.

El checkout remoto `/home/tp2/TP2_red4G` estaba limpio, pero divergido respecto a `origin/main` (`ahead 1, behind 4`) mientras el repo local tambien tenia cambios pendientes. Para activar esta version en el laboratorio real hay que resolver/sincronizar esa divergencia y reiniciar solo `tp2-car-control.service` en una ventana controlada.

## Intervencion remota posterior

Despues de observar desde la interfaz del operador que el EPC seguia sirviendo la version antigua, se copio la version nueva de:

- `servicios/coche.py`
- `ops/systemd/epc/tp2-car-control.service`

al checkout remoto `/home/tp2/TP2_red4G`.

La instalacion/reinicio por systemd no pudo completarse porque `sudo` solicito password interactiva. Se termino el proceso antiguo de `tp2-car-control.service`; systemd dejo la unidad `inactive` en vez de relanzarla. Para recuperar el runtime, se arranco manualmente el nuevo `coche.py` como usuario `tp2`:

```console
cd /home/tp2/TP2_red4G/servicios
nohup /home/tp2/miniforge3/bin/conda run --no-capture-output -n tp2 python -u coche.py > /tmp/tp2-car-control-web.log 2>&1 &
```

Validacion remota tras el arranque manual:

- `8088/TCP`: activo.
- `20001/UDP`: activo.
- `GET /` sirve la interfaz nueva `TP2 Live Control`.
- `POST /control` acepta control web y `status.json` refleja `control_source=web` durante publicaciones repetidas.
- `POST /control/neutral` devuelve el control a neutro por watchdog.
- Jetson inference endpoint: `GET http://100.115.99.8:9001/info` responde `Roboflow Inference Server 1.1.2`.

Estado actual de frames reales:

```json
{
  "control_enabled": true,
  "has_video": false,
  "last_client": "172.16.0.2",
  "packet_types": {
    "B": 114
  },
  "video_frames": 0,
  "web_port": 8088
}
```

Se republico `AM-Cloud` en `1/command`; durante 30 segundos posteriores el EPC recibio solo paquetes `B`, no paquetes `I`. Por tanto, la interfaz y el runtime EPC estan activos, pero el coche no esta enviando frames de camara al endpoint UDP.

Acceso al coche:

- `172.16.0.2:22` esta abierto.
- SSH no interactivo fallo para usuarios conocidos `tp2`, `grupo4`, `pi`, `ubuntu`, `artemis`.
- Sin credenciales del coche no se pudo entrar a revisar o arrancar el proceso de camara.
