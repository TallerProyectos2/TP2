# EPC Local Inference Validation

- Date: `2026-03-05`
- Jira: `N/A (MCP atlassian no disponible en esta sesion)`
- Machine: `tp2-EPC`
- Scope: `inferencia.py` con `test.jpg` ejecutada directamente en EPC

## Initial State

- `inferencia.py` existia en `/home/tp2/servicios_tp2/inferencia.py`
- `test.jpg` existia en `/home/tp2/servicios_tp2/test.jpg`
- La dependencia `inference-sdk` no estaba instalada en el EPC
- El script apuntaba a `http://localhost:9001` y fallaba por `Connection refused`

## Applied Change

- Instalado `inference-sdk` para el usuario `tp2` con:
  - `python3 -m pip install --user inference-sdk`
- Instalado `gradio` para GUI web con:
  - `python3 -m pip install --user gradio`
- Actualizado `/home/tp2/servicios_tp2/inferencia.py` para:
  - usar ruta absoluta de imagen basada en `__file__`
  - permitir configuracion por variables de entorno (`ROBOFLOW_API_URL`, `ROBOFLOW_API_KEY`, `ROBOFLOW_WORKSPACE`, `ROBOFLOW_WORKFLOW`, `TP2_TEST_IMAGE`)
  - usar por defecto `https://serverless.roboflow.com`
- Creado `/home/tp2/servicios_tp2/inferencia_gui.py` (GUI de escritorio con seleccion multiple)
- Creado `/home/tp2/servicios_tp2/inferencia_gui_web.py` (GUI web con seleccion multiple y galeria de resultados)
- Creado `/home/tp2/servicios_tp2/start_local_inference_server.py` para levantar Roboflow Inference local en EPC sin Docker (uvicorn + `HttpInterface`)
- Actualizado `inferencia.py` e `inferencia_gui_web.py` para soportar conmutacion:
  - modo: `local` o `cloud`
  - target: `workflow` o `model`
  - endpoints cloud separados por target (`serverless` para workflow, `detect` para model)

## Runtime Evidence

- Ejecucion validada en EPC:
  - `cd /home/tp2/servicios_tp2 && python3 inferencia.py`
- Resultado recibido:
  - clase detectada: `stop sign`
  - confianza: `0.9443966746330261`
  - ejecucion completada sin excepciones
- Visualizacion generada:
  - predicciones dibujadas: `1`
  - imagen anotada: `/home/tp2/servicios_tp2/test_pred.jpg`
  - formato validado: `JPEG 1600x1600`
- Validacion GUI web:
  - `python3 -m py_compile /home/tp2/servicios_tp2/inferencia_gui_web.py` -> `OK`
  - arranque de servicio: `python3 inferencia_gui_web.py --host 127.0.0.1 --port 7861`
  - `ss -ltnp` confirma escucha en `127.0.0.1:7861`
- Validacion conmutacion local/cloud:
  - cloud+workflow (`TP2_INFERENCE_MODE=cloud`, `TP2_INFERENCE_TARGET=workflow`) ejecuta OK y detecta `1` objeto en `test.jpg`
  - local+workflow (`TP2_INFERENCE_MODE=local`) ejecuta OK y detecta `1` objeto en `test.jpg` tras arrancar endpoint local
- Arranque endpoint local EPC:
  - comando: `ROBOFLOW_API_KEY=*** python3 /home/tp2/servicios_tp2/start_local_inference_server.py --host 127.0.0.1 --port 9001`
  - estado: `ss -ltnp` confirma escucha en `127.0.0.1:9001` con PID `49707`
  - contrato HTTP: `GET /openapi.json` responde `200` y expone rutas de workflows (`/{workspace_name}/workflows/{workflow_id}`)
- Nota de entorno:
  - `tkinter` no esta instalado en el EPC (`ModuleNotFoundError`), por lo que la GUI operativa validada en este estado es la web
  - el comando `inference server start` de `inference-cli` no es util en este EPC porque depende de Docker daemon

## Result

La inferencia con `test.jpg` queda operativa ejecutando el script directamente en el EPC, sin dependencia de Jetson para esta prueba, y ahora deja evidencia visual con bounding box y etiqueta. Ademas, el EPC dispone de una GUI web para procesar una o varias imagenes seleccionadas desde el filesystem del cliente y conmutar entre inferencia local y cloud.
