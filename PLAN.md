# PLAN DE DESPLIEGUE Y PRUEBAS

## 1. Objetivo

Montar y validar un sistema completo de vehiculo conectado con estos cuatro elementos:

- PC `EPC`: core LTE y nodo central de aplicacion
- PC `eNodeB`: acceso radio LTE con `bladeRF`
- `Jetson`: nodo opcional para aceleracion futura (fuera de la ruta critica actual)
- `Coche`: captura de camara, envio de frames y ejecucion de comandos

La estrategia es montar el sistema por capas, validando cada bloque antes de pasar al siguiente:

1. Red LTE estable
2. Servicios base en el EPC
3. Inferencia en el EPC
4. Agente del coche
5. Integracion extremo a extremo
6. Endurecimiento y demo

Restriccion operativa global:

- Bajo ningun concepto se puede actualizar el firmware de ningun componente del sistema durante este proyecto.

---

## 2. Reparto de servicios por maquina

## 2.1 PC EPC

Este equipo es el centro del sistema. Aqui va la red 4G y toda la logica de aplicacion no GPU.

### Servicios de host

- `srsepc`
  - Funciones: `MME + HSS + SPGW`
  - Gestion del registro LTE del UE
  - Asignacion de IP al coche
  - Salida del trafico del UE hacia servicios internos y externos

- `IP forwarding`
  - Permite enrutar trafico entre interfaces

- `NAT`
  - Reglas con `iptables` o `nftables`
  - Permite salida del UE hacia otras redes si hace falta

- `dnsmasq`
  - DNS ligero para el UE
  - Resolucion interna y reenvio externo

- `almacenamiento local`
  - Guardado de frames, logs de pruebas y datos auxiliares
  - Ruta recomendada: `/srv/tp2/`

### Servicios en Docker

- `backend-api`
  - `FastAPI`
  - Recibe frames del coche
  - Recibe telemetria
  - Guarda metadatos
  - Ejecuta inferencia local en el EPC
  - Decide el comando final
  - Publica comandos por MQTT

- `inference-service`
  - API HTTP minima para inferencia local en EPC
  - Endpoint principal: `POST /infer`
  - Endpoint de salud: `GET /health`

- `mosquitto`
  - Broker MQTT
  - Control y estado del coche

- `postgres`
  - Base de datos principal
  - Guarda sesiones, telemetria, detecciones, comandos y eventos

### Puertos relevantes

- `36412/SCTP`: S1-MME
- `2152/UDP`: GTP-U
- `53/TCP,UDP`: DNS
- `8000/TCP`: API del backend
- `9001/TCP`: API de inferencia local en EPC
- `1883/TCP`: MQTT
- `5432/TCP`: PostgreSQL (solo red interna)

---

## 2.2 PC eNodeB

Este equipo solo debe encargarse del acceso radio.

### Servicios

- `srsenb`
  - Conexion S1 con el EPC
  - Exposicion de la celda LTE

- `bladeRF`
  - SDR y sus drivers/herramientas

### Regla de diseño

- No instalar aqui backend
- No instalar base de datos
- No instalar MQTT
- No instalar IA

---

## 2.3 Jetson

La Jetson queda como nodo opcional de apoyo para fases futuras de optimizacion.

### Estado operativo actual

- No participa en la ruta critica de inferencia
- No es requisito para validar el flujo principal EPC + coche

### Uso futuro (opcional)

- Migrar inferencia desde EPC solo si hay necesidad real de aceleracion
- Mantener compatibilidad de API (`GET /health`, `POST /infer`) para una migracion controlada

---

## 2.4 Coche

El coche debe llevar un agente ligero, simple y robusto.

### Procesos logicos

- `camera-capture`
  - Captura frames de la camara

- `frame-uploader`
  - Envia frames por HTTP al EPC

- `mqtt-client`
  - Recibe comandos
  - Publica estado y acuses de recibo

- `movement-adapter`
  - Traduce comandos abstractos a llamadas a vuestros scripts actuales

- `safety-watchdog`
  - Si no llega orden nueva en un tiempo limite, el coche reduce velocidad o se para

### Tecnologias recomendadas

- `Python 3`
- `OpenCV`
- `requests` o `httpx`
- `paho-mqtt`

---

## 3. Arquitectura funcional

Flujo completo:

1. El coche se registra en la red LTE
2. El coche obtiene IP del EPC
3. El coche envia un frame al backend del EPC
4. El backend guarda el frame y registra el evento
5. El backend llama al servicio de inferencia local en el EPC
6. El servicio local responde con la deteccion
7. El backend decide la accion
8. El backend publica el comando por MQTT
9. El coche recibe el comando y ejecuta movimiento
10. El coche envia `ack` y telemetria
11. El backend registra todo en la base de datos

Reglas clave:

- El coche no habla directamente con servicios de inferencia externos
- El eNodeB no ejecuta logica de aplicacion
- El EPC ejecuta inferencia IA en esta etapa
- Las imagenes van por HTTP, no por MQTT
- MQTT se usa para control, estado y telemetria ligera

---

## 4. Plan secuencial de despliegue y pruebas

### Nota de alineacion con Jira

- Este bloque define el orden cronologico global del proyecto.
- Las subtareas por maquina mantienen su propia numeracion dentro de cada padre de Jira (`EPC-Fase 8`, `INT-Fase 11`, etc.).
- Correspondencia nueva con las subtareas anadidas:
  - `TP2-180` se ejecuta dentro de la Fase 2 global
  - `TP2-181`, `TP2-182` y `TP2-183` se cierran dentro de la Fase 10 global
  - `TP2-184` corresponde a la Fase 11 global
  - `TP2-185` corresponde a la Fase 12 global

## Fase 0. Preparacion y congelacion de configuracion

### Objetivo

Dejar cerrados los parametros de red y las decisiones base antes de instalar mas piezas.

### Tareas

- Fijar IPs de cada equipo:
  - `EPC <-> eNodeB`: `10.10.10.1/24` y `10.10.10.2/24`
  - `SGi EPC`: `172.16.0.1/24`
  - `Jetson` (opcional): IP fija en la red donde el EPC pueda alcanzarla

- Dejar documentados:
  - puertos de cada servicio
  - nombres de host
  - rutas de logs
  - version de `srsRAN`
  - version de `JetPack` y `PyTorch` en Jetson (solo si se usa como acelerador futuro)

- Crear estructura base de directorios en el EPC:
  - `/srv/tp2/frames`
  - `/srv/tp2/logs`
  - `/srv/tp2/docker`
  - `/srv/tp2/config`

### Validacion

- Todos los equipos responden por red a quien les corresponde
- Queda definida una tabla simple de IPs y servicios

### No pasar de fase hasta que

- Este cerrada la topologia de red

---

## Fase 1. Red LTE estable (EPC + eNodeB)

### Objetivo

Tener la red LTE funcionando de manera repetible y estable.

### EPC: tareas

- Revisar `epc.conf`
- Confirmar:
  - `mme_bind_addr = 10.10.10.1`
  - `gtpu_bind_addr = 10.10.10.1`
  - `sgi_if_addr = 172.16.0.1`
  - `db_file` con ruta absoluta

- Arrancar:
  - `sudo srsepc /home/tp2/.config/srsran/epc.conf`

### eNodeB: tareas

- Revisar `enb.conf`
- Confirmar:
  - `mme_addr = 10.10.10.1`
  - `gtp_bind_addr = 10.10.10.2`
  - `gtp_advertise_addr = 10.10.10.2`
  - `s1c_bind_addr = 10.10.10.2`

- Validar estado de `bladeRF`
- Arrancar:
  - `sudo srsenb /home/tp2/.config/srsran/enb.conf`

### Pruebas

- Desde EPC:
  - `ping 10.10.10.2`
- En la consola del EPC debe aparecer:
  - `Received S1 Setup Request`
  - `Sending S1 Setup Response`

### No pasar de fase hasta que

- El enlace S1 sea estable
- El eNodeB no falle al arrancar

---

## Fase 2. Provision del UE y politica de attach reproducible

### Objetivo

Dejar el alta del UE cerrada y repetible antes de depender del trafico IP del coche.

### Tareas

- Revisar `user_db.csv`
- Dejar al menos un UE correctamente provisionado
- Documentar los datos minimos necesarios para el alta real del abonado
- Fijar el procedimiento operativo de attach para el coche o para un UE de prueba
- Evitar dependencias de pasos implicitos o configuraciones a mano no documentadas

### Pruebas

- Registrar el coche o un UE de prueba en la red
- Confirmar que el EPC muestra el attach de forma consistente
- Verificar que el procedimiento puede repetirse sin reinterpretar la configuracion

### No pasar de fase hasta que

- Exista al menos un UE que pueda hacer attach de forma repetible
- El procedimiento de alta y attach quede sin ambiguedad operativa

---

## Fase 3. Salida IP del UE (routing, NAT, DNS)

### Objetivo

Garantizar que el coche, una vez registrado como UE, puede llegar a los servicios del EPC.

### EPC: tareas

- Activar `IP forwarding`
- Configurar NAT
- Instalar y configurar `dnsmasq`
- Confirmar que el trafico desde `172.16.0.0/24` puede salir por la interfaz correcta

### Pruebas

- Confirmar que el UE obtiene IP `172.16.0.x`
- Probar:
  - `ping 172.16.0.1`
  - resolucion DNS si se usa
  - acceso HTTP a un servicio simple en el EPC

### No pasar de fase hasta que

- El UE tenga conectividad IP estable con el EPC

---

## Fase 4. Base del backend en el EPC

### Objetivo

Montar el esqueleto de servicios de aplicacion sin depender aun de la IA.

### EPC: tareas

- Instalar `Docker Engine`
- Instalar `docker compose`
- Crear `docker-compose.yml` en `/srv/tp2/docker`

### Servicios iniciales

- `mosquitto`
- `postgres`
- `backend-api` vacio o minimo

### Requisitos del backend en esta fase

- Endpoint `GET /health`
- Endpoint `POST /v1/telemetry`
- Endpoint `POST /v1/frames` que solo reciba y guarde la imagen

### Pruebas

- Desde el EPC:
  - los contenedores arrancan
  - `backend-api` responde en `/health`

- Desde un cliente de prueba:
  - publicar y suscribirse a MQTT
  - insertar un registro de prueba en PostgreSQL
  - subir un frame y que quede guardado en disco

### No pasar de fase hasta que

- El backend reciba un frame y lo guarde correctamente
- MQTT y DB queden operativos

---

## Fase 5. Servicio de inferencia en el EPC

### Objetivo

Tener el modelo IA encapsulado en un servicio independiente y verificable.

### EPC: tareas

- Preparar entorno:
  - `Python 3`
  - `venv`
  - dependencias del modelo

- Implementar `inference-service`
  - `GET /health`
  - `POST /infer`

- Cargar el modelo al arrancar
- Dejar logs claros de:
  - arranque
  - carga de modelo
  - tiempo por inferencia

### Pruebas

- Llamar a `/health` desde el EPC
- Llamar a `/infer` con una imagen local conocida
- Medir tiempo de respuesta

### Validacion funcional

- El servicio devuelve:
  - etiqueta
  - confianza
  - latencia

### No pasar de fase hasta que

- El EPC procese una imagen de prueba de forma repetible

---

## Fase 6. Integracion backend <-> inferencia local en EPC

### Objetivo

Conectar el backend del EPC con el servicio de inferencia local del EPC.

### Tareas

- Definir el endpoint local de inferencia en el EPC
- Abrir el puerto local del `inference-service` si aplica
- Configurar en el backend del EPC:
  - URL local del servicio de inferencia
  - timeout
  - reintentos controlados

- Implementar en `backend-api`:
  - recepcion de frame
  - guardado del frame
  - llamada al servicio de inferencia local
  - persistencia del resultado

### Pruebas

- Subir un frame al backend del EPC
- Verificar que el backend llama al servicio local de inferencia
- Verificar que guarda en DB:
  - ruta del frame
  - resultado de IA
  - confianza
  - latencia

### No pasar de fase hasta que

- El EPC pueda completar la cadena:
  - frame recibido -> inferencia -> resultado guardado

---

## Fase 7. Pipeline de control en el EPC (sin coche aun)

### Objetivo

Implementar la logica de decision y emision de comandos, aunque todavia se pruebe con clientes simulados.

### Tareas

- Definir acciones abstractas:
  - `STOP`
  - `SLOW`
  - `LEFT`
  - `RIGHT`
  - `STRAIGHT`

- Implementar tabla de traduccion:
  - clase detectada -> accion

- Implementar publicacion MQTT:
  - topico `car/1/cmd`

- Implementar almacenamiento en DB:
  - deteccion
  - accion emitida
  - timestamp

### Pruebas

- Subir un frame de prueba
- Verificar que el backend:
  - recibe deteccion
  - decide accion
  - publica comando MQTT

- Comprobar con un cliente MQTT de prueba que el mensaje llega correctamente

### No pasar de fase hasta que

- El backend publique comandos validos por MQTT

---

## Fase 8. Agente del coche (captura + HTTP + MQTT)

### Objetivo

Montar en el coche un agente ligero que pueda integrarse con los scripts existentes.

### Tareas

- Crear proceso `camera-capture`
- Crear proceso `frame-uploader`
- Crear `mqtt-client`
- Crear `movement-adapter`
- Crear `safety-watchdog`

### Requisitos funcionales

- El coche debe poder:
  - enviar frames al EPC
  - recibir comandos desde MQTT
  - ejecutar movimientos usando vuestros scripts
  - parar si no llegan ordenes nuevas a tiempo

### Integracion con scripts existentes

- No reescribir los scripts actuales de movimiento
- Crear una capa adaptadora que invoque las funciones necesarias

### Pruebas

- El coche publica `status`
- El coche recibe un comando MQTT manual
- El coche ejecuta una accion simple
- El coche sube un frame al backend

### No pasar de fase hasta que

- El coche pueda recibir un comando y actuar sin IA

---

## Fase 9. Integracion extremo a extremo

### Objetivo

Validar el sistema completo con todos los bloques conectados.

### Secuencia de arranque recomendada

1. Arrancar `srsepc`
2. Arrancar `srsenb`
3. Confirmar attach del UE y que obtiene IP `172.16.0.x`
4. Arrancar `docker compose` del EPC
5. Arrancar `inference-service` en el EPC
6. Arrancar el agente del coche

### Flujo a verificar

1. El coche envia un frame
2. El EPC lo recibe
3. El EPC lo manda al servicio local de inferencia
4. El servicio local devuelve deteccion
5. El EPC decide accion
6. El EPC publica por MQTT
7. El coche ejecuta el movimiento
8. El coche envia `ack`
9. El EPC registra el evento completo

### Metricas a medir

- Tiempo de subida del frame
- Tiempo de inferencia
- Tiempo de decision
- Tiempo de recepcion del comando
- Latencia total extremo a extremo

### No pasar de fase hasta que

- La cadena completa funcione al menos de forma estable a baja frecuencia

---

## Fase 10. Endurecimiento, supervision y seguridad funcional

### Objetivo

Hacer el sistema utilizable para demostracion, evitando estados peligrosos y dejando los servicios recuperables.

### Tareas

- Implementar limite de frecuencia de frames
  - Recomendado: `2-5 fps`

- Implementar politicas de fallback
  - Si falla la inferencia en EPC: `SLOW` o `STOP`
  - Si falla MQTT: `STOP`
  - Si la confianza es baja: mantener accion segura

- Definir supervision y arranque automatico de servicios clave
  - `srsepc`
  - `mosquitto`
  - `postgres`
  - `backend-api`
  - `inference-service`
  - agente del coche

- Validar el modo degradado seguro del coche
  - timeout de comandos
  - parada segura
  - no reutilizar ordenes antiguas

- Retencion de datos
  - guardar muestras utiles, no todos los frames si satura disco

### Pruebas

- Simular caida del servicio de inferencia en EPC
- Simular perdida temporal de MQTT
- Simular timeout de backend
- Simular reinicio de un servicio critico
- Confirmar que el coche entra en modo seguro

### No pasar de fase hasta que

- El sistema tenga respuestas previsibles ante errores
- Los servicios clave recuperen su estado de forma controlada

---

## Fase 11. Sincronizacion temporal y correlacion de logs

### Objetivo

Hacer comparables los eventos entre EPC y coche para medir y depurar la integracion real.

### Tareas

- Fijar un mecanismo comun de sincronizacion temporal
- Definir el formato de timestamps
- Alinear logs de EPC y coche sobre una misma referencia temporal
- Dejar documentado el criterio de deriva aceptable entre nodos

### Pruebas

- Comparar tiempos entre los nodos y verificar que la deriva esta dentro del margen definido
- Ejecutar una prueba simple y correlacionar el mismo evento en los tres equipos

### No pasar de fase hasta que

- Los eventos puedan compararse entre maquinas sin ambiguedad temporal

---

## Fase 12. Observabilidad minima, trazabilidad y preparacion de demo

### Objetivo

Dejar un procedimiento de arranque limpio y una traza minima del sistema para la presentacion y para el diagnostico final.

### Tareas

- Definir evidencias minimas del pipeline extremo a extremo
- Preparar checklist de arranque
- Preparar checklist de validacion
- Preparar set de imagenes de prueba
- Preparar una demo guiada:
  - arranque de red
  - conexion del coche
  - envio de frame
  - deteccion
  - accion

### Checklist de arranque

1. Arrancar EPC
2. Arrancar eNodeB
3. Verificar S1
4. Verificar attach del UE e IP
5. Arrancar servicios Docker
6. Arrancar `inference-service` en EPC
7. Verificar `/health`
8. Arrancar coche
9. Verificar MQTT y trazas basicas
10. Hacer prueba de frame

### Checklist de cierre

1. Parar el coche
2. Parar servicios de aplicacion
3. Parar `inference-service` en EPC
4. Parar eNodeB
5. Parar EPC
6. Guardar logs y evidencias

### No pasar de fase hasta que

- Exista un procedimiento reproducible de arranque, validacion y cierre
- Una prueba completa deje evidencias suficientes de lo ocurrido

---

## 5. Orden exacto recomendado de implementacion

Para evitar mezclar errores, el orden correcto es:

1. Cerrar topologia, IPs y estructura base
2. Cerrar configuracion LTE (`EPC + eNodeB`)
3. Cerrar provision del UE y politica de attach
4. Confirmar conectividad IP del UE
5. Montar `mosquitto + postgres + backend` en EPC
6. Probar backend sin IA
7. Montar servicio de inferencia en EPC
8. Conectar backend del EPC con su servicio de inferencia local
9. Implementar decision y publicacion MQTT
10. Implementar agente del coche
11. Integrar extremo a extremo
12. Endurecer supervision y seguridad funcional
13. Alinear tiempos y correlacionar logs
14. Dejar trazabilidad minima y preparar demo

Si aparece un fallo en una fase, no avanzar a la siguiente hasta dejar cerrada la anterior.

---

## 6. Entregables tecnicos por fase

### Fase 0

- Topologia cerrada
- Tabla de IPs y servicios
- Estructura base del EPC preparada

### Fase 1

- Red LTE estable
- Configuraciones `epc.conf` y `enb.conf` cerradas

### Fase 2

- UE provisionado
- Procedimiento de attach reproducible

### Fase 3

- UE con IP y salida correcta

### Fase 4

- `docker-compose.yml`
- backend minimo operativo
- MQTT operativo
- DB operativa

### Fase 5

- servicio `/infer` en EPC

### Fase 6

- backend llamando a inferencia local EPC y guardando resultados

### Fase 7

- comando MQTT emitido automaticamente

### Fase 8

- coche recibiendo y ejecutando comandos

### Fase 9

- pipeline completo funcionando

### Fase 10

- supervision de servicios
- fallbacks y seguridad operativa

### Fase 11

- tiempos alineados
- logs correlacionables

### Fase 12

- trazabilidad minima del flujo
- procedimiento reproducible de demo

---

## 7. Criterios de exito

Se considera que el sistema esta listo cuando se cumplan estos puntos:

- La red LTE arranca siempre sin reconfiguracion manual extra
- Existe un procedimiento reproducible de alta y attach del UE
- El coche se registra como UE y obtiene IP
- El coche puede enviar frames al backend del EPC
- El backend puede pedir inferencia al servicio local en EPC
- El servicio de inferencia en EPC devuelve predicciones validas
- El backend genera y publica comandos
- El coche ejecuta comandos usando los scripts existentes
- Los servicios clave recuperan su estado de forma controlada
- El sistema se para de forma segura cuando algo falla
- Los eventos pueden correlacionarse temporalmente entre maquinas
- Quedan registros y evidencias suficientes para demostrar el funcionamiento
