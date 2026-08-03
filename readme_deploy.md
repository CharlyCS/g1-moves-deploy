# Deploy de `g1-moves`

## Avance

Este despliegue ya no corresponde al flujo original de RoboJuDo "tal cual", sino a una adaptación para los bailes de `g1-moves` sobre el robot G1 real. Tomando como referencia el repositorio original `HansZ8/RoboJuDo`, los principales cambios de esta rama son:

- Se ajustó el manejo del tiempo en `RoboJuDo/scripts/run_pipeline.py` usando `time_diff` para mantener la frecuencia objetivo del pipeline.
- Cuando hay `frame drop`, ahora se registra el evento en logs y, si la caída es grande, el sistema intenta pasar a locomoción de respaldo mediante `force_locomotion()` en vez de cortar el proceso inmediatamente.
- El despliegue se orientó a correr en el mismo robot usando `UnitreeCppEnv`, en lugar de depender del flujo típico de ejecución desde una workstation externa.
- Se agregaron configuraciones específicas para bailes y pruebas reales, por ejemplo `g1_daddance`, `g1_salsa_shadow`, `g1_salsa_safe_real` y `g1_woah_safe_real`.
- Se empezó a usar de forma explícita `action_beta=0.8` en las configuraciones de baile. En RoboJuDo ya existía la lógica de suavizado de acciones, pero en esta adaptación se parametrizó y se empezó a utilizar en estas políticas concretas.

## Procedencia de la información y de los assets

La información usada para preparar estas configuraciones, junto con los clips, políticas y archivos asociados de `g1_moves_data`, se obtuvo del dataset público `exptech/g1-moves` en Hugging Face:

- https://huggingface.co/datasets/exptech/g1-moves/tree/main

Ese dataset publica el material de `g1-moves` y organiza el contenido en carpetas como `dance`, `karate`, `bonus` y otros recursos del proyecto. En esta adaptación, `g1_moves_data` se tomó como la fuente de referencia para seleccionar clips, duraciones, nombres de políticas y assets necesarios para el despliegue.

## Cambios importantes frente a RoboJuDo original

### 1. Control del ciclo con `time_diff`

En `RoboJuDo/scripts/run_pipeline.py` se mide el tiempo real de cada iteración:

```python
time_diff = time_end - time_start
time_diff = pipeline.dt - time_diff
```

Con eso:

- si `time_diff > 0`, el proceso duerme el tiempo restante para sostener la frecuencia deseada;
- si `time_diff <= 0`, se reporta `frame drop`;
- si la caída supera `-0.2 s` en robot real, se intenta cambiar a locomoción segura si el pipeline la soporta.

Esto es importante porque en esta adaptación se priorizó mantener vivo el proceso y conservar una vía de recuperación, en vez de salir abruptamente.

Adicionalmente, ante una caída severa de tiempo en robot real, esta rama intenta usar `force_locomotion()` si el pipeline dispone de locomoción de respaldo. Ese comportamiento es parte de la adaptación para baile seguro.

### 2. Despliegue en el mismo robot

El flujo adaptado está pensado para ejecutarse onboard, directamente en el G1, usando:

```python
env_type="UnitreeCppEnv"
net_if="eth0"
```

Esto difiere del uso genérico de RoboJuDo, donde también se contempla correr desde otra computadora conectada por Ethernet. En esta versión, la intención es desplegar desde el propio robot para ganar estabilidad y evitar depender de una máquina externa.

### 3. Configuraciones nuevas para baile y uso explícito de `action_beta`

En `RoboJuDo/robojudo/config/g1/g1_cfg.py` se añadieron configuraciones específicas para esta adaptación. Entre ellas:

- `g1_daddance`
- `g1_salsa_shadow`
- `g1_salsa_safe_real`
- `g1_woah_safe_real`

Las configuraciones `g1_salsa_safe_real` y `g1_woah_safe_real` usan `G1RlLocoMimicPipelineCfg` para ejecutar un baile con locomoción de respaldo. El flujo es:

1. el robot inicia con una política estable de locomoción;
2. `Start` o `r` activa el baile;
3. al terminar el movimiento, vuelve a locomoción;
4. el proceso sigue corriendo y mantiene al robot de pie.

En particular:

- `g1_salsa_safe_real` usa `policy_name="J_Dance2_Salsa"` con `max_timestep=2150`;
- `g1_woah_safe_real` usa `policy_name="J_Dance3_Woah"` con `max_timestep=1799`.

Además, en ambos casos se ajustaron parámetros como:

- `freq=50`
- `motion_fps=60.0`
- `playback_rate=1.0`
- `action_beta=0.8`

Aquí hay un cambio importante respecto al RoboJuDo original: `action_beta` no estaba siendo usado en estas configuraciones de despliegue. Aunque el framework base ya tenía implementada la mezcla/suavizado de acciones dentro de la política, en esta rama se empezó a configurar explícitamente para estabilizar mejor el replay del movimiento sobre robot real.

También aparecen otros ajustes operativos por política, como:

- `without_state_estimator=False`
- `override_robot_anchor_pos=False`
- `use_motion_from_model=True`
- `max_timestep` definido manualmente según la duración real del baile

Esos cambios hacen que la ejecución quede más alineada con los modelos ONNX concretos usados en `g1-moves`.

### 4. Otros cambios encontrados en la carpeta `RoboJuDo`

Además de las configs en `g1_cfg.py`, en esta rama local aparecen otras adaptaciones relevantes:

- Se añadió `scripts/run_g1_moves_robojudo.py`, que funciona como un runner específico para `g1-moves`.
- Ese script carga directamente el par `ONNX + NPZ`, reconstruye la observación de 160 dimensiones usada por `g1-moves` y permite correr tanto en MuJoCo como en robot real.
- En simulación, ese runner inicializa MuJoCo con el primer frame del NPZ para que el arranque coincida con la referencia del clip.
- En robot real, ese runner usa `UnitreeCppEnv`, `enable_odometry=True`, `odometry_type="UNITREE"` y opciones de alineación pensadas para el despliegue onboard.

También se observa una adaptación para exportación de modelos:

- `scripts/policy_to_onnx.py` fue ajustado para trabajar con tareas y rutas del flujo de `g1-moves` y `mjlab`.
- El script incluye imports y rutas locales orientadas a exportar políticas concretas del dataset de bailes a formato ONNX.

En la parte de política y reproducción del movimiento también hay cambios:

- En `robojudo/policy/policy_cfgs.py` se añadieron los campos `motion_fps` y `playback_rate` dentro de `BeyondMimicPolicyCfg`.
- En `robojudo/policy/beyondmimic_policy.py` esos parámetros ya se usan para calcular la velocidad nominal de reproducción según la frecuencia de control real del pipeline.
- Esto permite desacoplar la frecuencia de control del robot de la frecuencia original del motion clip y ajustar mejor el replay de la animación.

En el pipeline hay más ajustes de seguridad y transición:

- En `robojudo/pipeline/rl_pipeline.py` aparece lógica adicional de `blend-in` para pasar de postura inicial a salida de policy de forma progresiva.
- Ese mismo flujo marca `_blend_in_completed` para no confundir el retardo del blend-in con un `frame drop` real.
- En `robojudo/pipeline/rl_loco_mimic_pipeline.py` se añadió `force_locomotion()`, usado por `scripts/run_pipeline.py` para forzar el retorno a locomoción cuando hay retrasos severos.

Finalmente, hay un modo intermedio de validación física:

- `g1_salsa_shadow` deja el entorno real en modo lectura/inferencia (`act=False`) sin enviar objetivos a los motores.
- Esa config sirve para validar observaciones, odometría y alineación del motion antes de habilitar actuación real.

## Dónde están los logs

Los logs del sistema quedan en:

```bash
RoboJuDo/logs/robojudo.log
```

Si el comando se ejecuta desde dentro de la carpeta `RoboJuDo`, entonces la ruta relativa visible será:

```bash
logs/robojudo.log
```

Ese archivo es el que se debe revisar para ver:

- advertencias de `frame drop`;
- errores de ejecución;
- cambios de estado del pipeline;
- mensajes de fallback a locomoción.

Nota: la presencia de `robojudo.log` es parte del mecanismo de logging del proyecto; aquí se documenta porque es el archivo operativo que se revisa durante el despliegue.

## Dónde deben colocarse los bailes

Los modelos ONNX de los bailes deben estar en:

```bash
g1-moves/RoboJuDo/assets/models/g1/beyondmimic
```

La selección del baile depende de `policy_name`, por lo que el archivo debe existir con ese nombre. Ejemplos:

- `J_Dance2_Salsa.onnx`
- `J_Dance3_Woah.onnx`

## Ejecución recomendada

Desde `g1-moves/RoboJuDo`:

```bash
python scripts/run_pipeline.py -c g1_salsa_safe_real
python scripts/run_pipeline.py -c g1_woah_safe_real
```

## Notas operativas

- `A` sigue siendo la parada de emergencia.
- `Select` fuerza locomoción.
- `Start` activa la política mimic.
- En teclado, `r` inicia mimic, `l` vuelve a locomoción y `o` apaga el pipeline.

En resumen, el avance principal de esta rama es que el despliegue dejó de ser una adaptación genérica de RoboJuDo y pasó a ser un flujo más directo para bailar en el mismo G1, con configuraciones nuevas de baile, uso explícito de `action_beta`, fallback de locomoción y ubicación definida para los modelos `beyondmimic`.
