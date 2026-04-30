# PLAN OPERATIVO Y DE INTEGRACION TP2

## 1. Objetivo actualizado

Consolidar el sistema real con este criterio:

- EPC, eNodeB y coche ya operativos en LTE.
- El EPC es el gateway unico para UDP del coche, control remoto y vista web.
- Jetson es el nodo primario de inferencia local Roboflow.
- El fallback local EPC se conserva solo como perfil operativo opcional.
- La ruta critica actual no depende de ventanas OpenCV ni de scripts car1/car3 genericos.

Este plan toma como base un runtime unico en `servicios/coche.py` y elimina componentes paralelos que no forman parte del flujo automatico.

## 2. Estado real actual (2026-04-22)

- LTE:
  - EPC y eNodeB enlazan por `10.10.10.0/24`.
  - El coche adjunta como UE.
  - Estado documentado anterior: UE del coche fijado en EPC como `901650000052126 -> 172.16.0.2`.
  - Estado vivo observado el `2026-04-27`: HSS tiene `IP_alloc=dynamic` para `901650000052126` y la sesion activa uso `172.16.0.4`.
- Inferencia:
  - Modelo y flujo de inferencia disponibles en Jetson por HTTP Roboflow.
  - Evidencia en `docs/logs/validations/2026-03-05-epc-inferencia-local.md`.
  - Jetson integrada como endpoint Roboflow remoto en `http://100.115.99.8:9001`, validada el `2026-03-26`.
  - Estado live del `2026-04-14`: Jetson activa y validada desde EPC con inferencia directa al modelo Roboflow `tp2-g4-2026/2`; ver `docs/logs/validations/2026-04-14-jetson-local-roboflow-model-tp2-g4-2026-2.md`.
  - Evidencia en `docs/logs/validations/2026-03-26-jetson-remote-inference-epc-control.md`.
- Ficheros operativos en `servicios/`:
  - `servicios/coche.py`
  - `servicios/roboflow_runtime.py`
  - `servicios/inferencia.py`
  - `servicios/start_local_inference_server.py`
  - `servicios/session_replayer.py`
  - `servicios/environment-tp2.yml`
  - `servicios/test.jpg`

## 3. Arquitectura de trabajo (web-runtime first)

## 3.1 Ruta critica actual

1. Coche se conecta por LTE al EPC.
2. Coche envia `I`, `B` y `D` por UDP a `servicios/coche.py` en el EPC.
3. EPC decodifica el ultimo frame y lo publica como MJPEG en `8088/TCP`.
4. EPC envia los frames mas recientes a Jetson para inferencia asincrona y pinta bounding boxes/labels sobre el stream.
5. El navegador permite alternar modo manual/autonomo por HTTP; EPC aplica watchdog en manual o calcula control autonomo desde detecciones frescas.
6. EPC envia `C + steering + throttle` por UDP al coche.

## 3.2 Inferencia actual

- Se ejecuta principalmente en Jetson.
- Dos modos:
  - Jetson (`ROBOFLOW_LOCAL_API_URL=http://100.115.99.8:9001`)
  - local EPC (`ROBOFLOW_LOCAL_API_URL=http://127.0.0.1:9001`) como fallback/perfil opcional
- `inferencia.py` queda como prueba CLI repetible con imagen conocida.

## 3.3 Jetson

- Jetson ya esta integrada como nodo de inferencia adicional, sin mover la ruta de control fuera del EPC.
- La ruta Jetson validada usa el endpoint Roboflow HTTP en `http://100.115.99.8:9001` con `TP2_INFERENCE_TARGET=model` y `ROBOFLOW_MODEL_ID=tp2-g4-2026/2`.
- La disponibilidad live de Jetson debe comprobarse al inicio de cada sesion.
- El primer objetivo pendiente tras esta integracion es validar end-to-end que el coche esta enviando frames `I` al runtime web.

## 4. Cobertura funcional existente en `servicios/`

## 4.1 Ya cubierto

- Runtime web del coche:
- `coche.py` (UDP del coche, MJPEG, inferencia asincrona, control web, modo autonomo y watchdog manual)
- `autonomous_driver.py` (control autonomo determinista con normalizacion de detecciones, tracking temporal, estimacion de distancia por area, FSM de maniobras, cooldowns y filtrado de comandos)
- `lane_detector.py` (deteccion OpenCV de cinta azul/verde sobre alfombra y correccion suave de carril dentro del modo autonomo)
- grabador de sesion/dataset desde `coche.py` para guardar frames, video MP4 anotado, predicciones, flags criticos, estimaciones autonomas y comandos como candidatos de reentrenamiento Roboflow
- replayer offline `session_replayer.py` para visualizar sesiones, filtrar situaciones criticas y relabelar detecciones sin modificar el manifiesto original
- Inferencia y validacion:
  - `inferencia.py` (CLI)
  - `start_local_inference_server.py` (endpoint local)
  - `roboflow_runtime.py` (cliente/helper compartido)

## 4.2 Fuera de la ruta critica actual

- Reintroducir scripts car1/car3 genericos.
- Depender de ventanas OpenCV en el EPC para operar desde MacBook.
- Construccion de pipeline nueva basada en MQTT/DB para habilitar control basico.

Eso puede existir en el futuro como capa adicional, pero no es requisito para seguir avanzando desde el estado actual.

## 5. Plan paso a paso desde hoy

## Paso 0. Congelar baseline actual (completado)

- EPC + eNodeB + coche se consideran base operativa cerrada.
- Mantener:
  - `srsepc` estable
  - `srsenb` estable
  - mapeo UE estable; si se requiere IP fija, restaurar `IP_alloc=172.16.0.2` en una ventana controlada y validar reattach

## Paso 1. Estandarizar ejecucion de scripts en EPC (completado)

- Usar `servicios/` como fuente de verdad de runtime.
- Mantener el endpoint de inferencia local en EPC con `start_local_inference_server.py`.
- Mantener `inferencia.py` como prueba minima de inferencia repetible.

## Paso 2. Cerrar contrato operativo EPC <-> coche por scripts (en curso)

- Script elegido para sesiones normales: `servicios/coche.py`.
- IP/puerto de escucha UDP: `172.16.0.1:20001`.
- Formato de datos esperados: `I`, `B`, `D` con payload `pickle`.
- Formato de control de salida: `C` con `double` giro y `double` acelerador.
- Control manual remoto directo desde navegador con watchdog a neutro.
- Selector web manual/autonomo:
  - manual: navegador publica giro/gas y watchdog vuelve a neutro si deja de publicar.
  - autonomo: EPC decide desde detecciones Roboflow recientes, priorizando señales persistentes y cercanas por area de bounding box, zona izquierda/centro/derecha y estado de maniobra.
  - throttle autonomo: las acciones de avance usan `+0.65` por defecto y la web puede ajustar la velocidad de crucero en vivo; las paradas, ambiguedad o fallbacks por datos obsoletos usan neutro.
  - compensacion de direccion: el envio UDP aplica trim de giro (`TP2_STEERING_TRIM` default `-0.24`) para corregir con mas autoridad el sesgo fisico hacia la izquierda de las ruedas; la web puede ajustar este valor y un pulso periodico a la derecha en vivo sin reiniciar el servicio; los giros autonomos abiertos omiten el trim para mantener bloqueo maximo.
  - asistencia de carril: `coche.py` segmenta las cintas azul/verde en OpenCV, estima el corredor actual entre lineas y suma una correccion limitada al giro solo en acciones autonomas de avance; prefiere el corredor derecho cuando hay varios carriles visibles para recuperar invasiones del carril contrario, reduce gas en recuperacion fuerte y no compite con STOP ni giros abiertos.
  - distancia de decision: el runtime acepta señales algo mas pequeñas/lejanas por defecto para iniciar antes STOP y giros.
  - tracking/FSM: confirma señales desde el primer frame valido por defecto, ejecuta `STOP` inmediato, ejecuta giros a bloqueo maximo como maniobra abierta de 90 grados durante una ventana controlada y aplica cooldown para no repetir la misma señal.
  - fallback: sin frame o inferencia fresca, EPC manda neutro.
  - dataset: la web y el servicio pueden activar grabacion de sesion; el servicio normal arranca con captura por defecto para generar `manifest.jsonl`, `labels.jsonl`, `critical.jsonl` y `session.mp4` antes de curar/reentrenar.
  - inferencia live: `coche.py` usa `inference_sdk` con frames NumPy de OpenCV, sin JPEG temporal en disco.

## Paso 3. Validacion repetible extremo a extremo sobre EPC (pendiente corta)

- Prueba minima:
  - arrancar `srsepc`
  - arrancar `srsenb`
  - confirmar UE del coche por el ultimo attach del IMSI `901650000052126`
- arrancar `tp2-car-control.service` en EPC
  - verificar ida y vuelta UDP con el coche
- Prueba de inferencia:
  - arrancar endpoint local `9001`
  - ejecutar `inferencia.py` con imagen conocida
  - guardar evidencia de salida anotada

## Paso 4. Integrar Jetson sin romper ruta actual (completado)

- Preparar Jetson solo para inferencia.
- Exponer endpoint compatible con cliente actual de inferencia.
- Añadir selector de destino de inferencia:
  - `EPC local` (fallback por defecto)
  - `Jetson` (modo nuevo)
- No mover control UDP del coche fuera del EPC en esta fase.

## Paso 5. Fallback y conmutacion segura EPC <-> Jetson (pendiente)

- Definir timeout de inferencia remota.
- Si Jetson falla:
  - fallback inmediato a inferencia local EPC
  - mantener control del coche sin parada de servicio

## Paso 6. Cierre para demo operativa (pendiente)

- Checklist unica de arranque/parada.
- Capa de arranque automatico basada en `ops/bin/tp2-up`, unidades `systemd` por maquina y orquestacion por Tailscale.
- Vista web live del control del coche en EPC (`8088/TCP`) para ver video e inferencia desde Tailscale sin depender de ventanas OpenCV locales.
- Control remoto desde la misma vista web, con watchdog a neutro si dejan de llegar comandos.
- Evidencias minimas por sesion:
  - attach UE
  - control UDP
  - inferencia (local o Jetson)
  - accion ejecutada por coche

## 6. Criterios de aceptacion por bloque

## LTE y red

- S1 estable entre EPC y eNodeB.
- UE del coche adjunta; si se requiere IP fija, HSS debe tener `IP_alloc=172.16.0.2` y la sesion debe validarlo.

## Control por scripts

- Script seleccionado recibe datos del coche sin reinicios espurios.
- EPC envia comandos de control y el coche reacciona de forma consistente.

## Inferencia EPC

- `inferencia.py` ejecuta sin excepciones.
- Se genera imagen anotada de salida.
- Endpoint local de inferencia accesible cuando se usa modo local.

## Jetson

- Endpoint de inferencia accesible desde EPC.
- Conmutacion EPC/Jetson controlada por configuracion.
- Fallback a EPC local validado.

## 7. Reglas de ejecucion

- No actualizar firmware de ningun componente.
- No mover servicios al eNodeB fuera de radio.
- No sustituir scripts existentes por nuevos servicios si no hay necesidad tecnica real.
- Cualquier cambio de contrato operativo debe actualizar documentos en el mismo task.

## 8. Orden recomendado para siguientes sesiones

1. Verificar que sigue vivo el baseline LTE (EPC+eNodeB+UE).
2. Ejecutar una prueba corta de script de control del coche en EPC.
3. Ejecutar prueba corta de inferencia en EPC.
4. Avanzar solo en integracion Jetson.
5. Repetir validacion completa y registrar evidencia.
