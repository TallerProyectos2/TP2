# PLAN OPERATIVO Y DE INTEGRACION TP2

## 1. Objetivo actualizado

Consolidar el sistema real con este criterio:

- EPC, eNodeB y coche ya operativos en LTE.
- La inferencia y su servicio ya existen en el EPC.
- La integracion pendiente principal es Jetson.
- La ruta critica actual no depende de una API nueva ni de una plataforma nueva de backend.

Este plan toma como base los scripts ya existentes en `servicios/` y elimina el trabajo no necesario de construir componentes paralelos.

## 2. Estado real actual (2026-03-10)

- LTE:
  - EPC y eNodeB enlazan por `10.10.10.0/24`.
  - El coche adjunta como UE.
  - UE del coche fijado en EPC: `901650000052126 -> 172.16.0.2`.
- Inferencia:
  - Modelo y flujo de inferencia disponibles en EPC.
  - Evidencia en `docs/logs/validations/2026-03-05-epc-inferencia-local.md`.
- Scripts operativos en repo:
  - `servicios/inferencia.py`
  - `servicios/start_local_inference_server.py`
  - `servicios/inferencia_gui_web.py`
  - `servicios/car*_cloud_control_server*.py`
  - `servicios/car*_manual_control_server.py`
  - `servicios/artemis_autonomous_car.py`

## 3. Arquitectura de trabajo (script-first)

## 3.1 Ruta critica actual

1. Coche se conecta por LTE al EPC.
2. Coche envia datos por UDP al servidor de control en EPC.
3. EPC procesa imagen/LIDAR con scripts existentes y calcula control.
4. EPC devuelve comando de control por UDP al coche.

## 3.2 Inferencia actual

- Se ejecuta en EPC.
- Dos modos:
  - local (`ROBOFLOW_LOCAL_API_URL`, por defecto `127.0.0.1:9001`)
  - cloud (Roboflow serverless/detect)
- Puede lanzarse por CLI o GUI web.

## 3.3 Jetson (pendiente)

- Se integrara como nodo de inferencia adicional, sin romper el flujo ya validado en EPC.
- El primer objetivo es compatibilidad de contrato para no tocar la ruta de control del coche.

## 4. Cobertura funcional existente en `servicios/`

## 4.1 Ya cubierto

- Control autonomo por vision/LIDAR:
  - `car1_cloud_control_server.py`
  - `car3_cloud_control_server.py`
  - `artemis_autonomous_car.py`
- Control autonomo con cambio de modo en tiempo real:
  - `car1_cloud_control_server_real_time_control.py`
  - `car3_cloud_control_server_real_time_control.py`
- Control manual asistido por teclado:
  - `car1_manual_control_server.py`
  - `car3_manual_control_server.py`
- Inferencia:
  - `inferencia.py` (CLI)
  - `inferencia_gui_web.py` (GUI web)
  - `start_local_inference_server.py` (endpoint local)

## 4.2 Fuera de la ruta critica actual

- Construccion de una API nueva de backend.
- Construccion de pipeline nueva basada en MQTT/DB para habilitar control basico.

Eso puede existir en el futuro como capa adicional, pero no es requisito para seguir avanzando desde el estado actual.

## 5. Plan paso a paso desde hoy

## Paso 0. Congelar baseline actual (completado)

- EPC + eNodeB + coche se consideran base operativa cerrada.
- Mantener:
  - `srsepc` estable
  - `srsenb` estable
  - mapeo UE fijo `172.16.0.2`

## Paso 1. Estandarizar ejecucion de scripts en EPC (completado)

- Usar `servicios/` como fuente de verdad de runtime.
- Mantener el endpoint de inferencia local en EPC con `start_local_inference_server.py`.
- Mantener `inferencia.py` como prueba minima de inferencia repetible.

## Paso 2. Cerrar contrato operativo EPC <-> coche por scripts (en curso)

- Fijar para cada coche:
  - script elegido (manual/autonomo/real_time_control)
  - IP/puerto de escucha UDP
  - formato de datos esperados (`I`, `L`, `B`, `D`)
  - formato de control de salida (`C` con giro y acelerador)
- Documentar el modo recomendado para sesiones normales.

## Paso 3. Validacion repetible extremo a extremo sobre EPC (pendiente corta)

- Prueba minima:
  - arrancar `srsepc`
  - arrancar `srsenb`
  - confirmar UE `172.16.0.2`
  - arrancar script de control en EPC
  - verificar ida y vuelta UDP con el coche
- Prueba de inferencia:
  - arrancar endpoint local `9001`
  - ejecutar `inferencia.py` con imagen conocida
  - guardar evidencia de salida anotada

## Paso 4. Integrar Jetson sin romper ruta actual (pendiente principal)

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
- Evidencias minimas por sesion:
  - attach UE
  - control UDP
  - inferencia (local o Jetson)
  - accion ejecutada por coche

## 6. Criterios de aceptacion por bloque

## LTE y red

- S1 estable entre EPC y eNodeB.
- UE del coche con IP fija `172.16.0.2`.

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
