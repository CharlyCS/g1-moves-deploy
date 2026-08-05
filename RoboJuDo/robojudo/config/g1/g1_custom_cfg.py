from robojudo.config import cfg_registry
from robojudo.controller.ctrl_cfgs import (
    JoystickCtrlCfg,  # noqa: F401
    KeyboardCtrlCfg,  # noqa: F401
    UnitreeCtrlCfg,  # noqa: F401
)
from robojudo.pipeline.pipeline_cfgs import (
    RlLocoMimicPipelineCfg,  # noqa: F401
    RlMultiPolicyPipelineCfg,  # noqa: F401
    RlPipelineCfg,  # noqa: F401
)

from .ctrl.g1_beyondmimic_ctrl_cfg import G1BeyondmimicCtrlCfg  # noqa: F401
from .ctrl.g1_motion_ctrl_cfg import (  # noqa: F401
    G1MotionCtrlCfg,
    G1MotionH2HCtrlCfg,
    G1MotionKungfuBotCtrlCfg,
    G1MotionTwistCtrlCfg,
)

from .ctrl.g1_twist_redis_ctrl_cfg import G1TwistRedisCtrlCfg  # noqa: F401
from .env.g1_dummy_env_cfg import G1DummyEnvCfg  # noqa: F401
from .env.g1_mujuco_env_cfg import G1_12MujocoEnvCfg, G1_23MujocoEnvCfg, G1MujocoEnvCfg  # noqa: F401
from .env.g1_real_env_cfg import G1RealEnvCfg, G1UnitreeCfg  # noqa: F401
from .policy.g1_amo_policy_cfg import G1AmoPolicyCfg  # noqa: F401
from .policy.g1_asap_policy_cfg import G1AsapLocoPolicyCfg, G1AsapPolicyCfg  # noqa: F401
from .policy.g1_beyondmimic_policy_cfg import G1BeyondMimicPolicyCfg  # noqa: F401
from .policy.g1_h2h_policy_cfg import G1H2HPolicyCfg  # noqa: F401
from .policy.g1_kungfubot_policy_cfg import G1KungfuBotGeneralPolicyCfg, G1KungfuBotPolicyCfg  # noqa: F401
from .policy.g1_smooth_policy_cfg import G1SmoothPolicyCfg  # noqa: F401
from .policy.g1_twist_policy_cfg import G1TwistPolicyCfg  # noqa: F401
from .policy.g1_unitree_policy_cfg import G1UnitreePolicyCfg, G1UnitreeWoGaitPolicyCfg  # noqa: F401

##### ADDED
from .pipeline.g1_locomimic_pipeline_cfg import G1RlLocoMimicPipelineCfg

# ======================= Define frames======================== #

import numpy as np
from pathlib import Path

def get_motion_frames(policy_name: str) -> int:
    path = Path(
        f"/home/unitree/glexone-ws/g1_moves_data/dance/{policy_name}/training/{policy_name}.npz"
    ).expanduser()

    motion = np.load(path)
    frames = int(motion["joint_pos"].shape[0])
    return frames

# ======================== Custom Configs ======================== #

@cfg_registry.register
class g1_dev(RlPipelineCfg):
    robot: str = "g1"
    env: G1_23MujocoEnvCfg = G1_23MujocoEnvCfg()

    ctrl: list[KeyboardCtrlCfg] = [
        KeyboardCtrlCfg(),
    ]

    policy: G1UnitreePolicyCfg = G1UnitreePolicyCfg()


###### EL REGISTRY PARA EL ONNX "NAME".ONNX
@cfg_registry.register
class g1_daddance_protodance(RlPipelineCfg):
    """
    B_DadDance exportado desde MJLab al formato BeyondMimic ONNX.
    Configuración exclusiva para sim2sim.
    """

    robot: str = "g1"

    env: G1MujocoEnvCfg = G1MujocoEnvCfg(
        update_with_fk=True,
        born_place_align=True,
    )

    ctrl: list[KeyboardCtrlCfg] = [
        KeyboardCtrlCfg(
            triggers={
                "i": "[SIM_REBORN]",
                "o": "[SHUTDOWN]",
                "r": "[MOTION_RESET]",
            }
        ),
    ]

    policy: G1BeyondMimicPolicyCfg = G1BeyondMimicPolicyCfg(
        policy_name= "J_Dance2_Salsa", #"B_DadDance",

        # La ONNX tiene 160 observaciones:
        # incluye anchor position y base_lin_vel.
        without_state_estimator=False,

        # Usar posición y orientación reales del torso simulado.
        override_robot_anchor_pos=False,

        # Leer movimiento directamente de la ONNX.
        use_motion_from_model=True,

        # Leer joints, default pose, KP, KD y action_scale
        # desde la metadata que acabas de verificar.
        use_modelmeta_config=True,

        start_timestep=0,
        max_timestep=2150, #2508
    )

    run_fullspeed: bool = False
    do_safety_check: bool = False


# lectura de motores
@cfg_registry.register
class g1_salsa_shadow(RlPipelineCfg):
    """
    G1 real en modo lectura/inferencia.
    No transmite objetivos a los motores.
    """

    robot: str = "g1"

    env: G1RealEnvCfg = G1RealEnvCfg(
        act=False,

        env_type="UnitreeCppEnv",

        unitree=G1UnitreeCfg(
            net_if="enp7s0",
            enable_odometry=True,
        ),

        odometry_type="UNITREE",

        # Necesario porque el anchor de la ONNX es torso_link.
        update_with_fk=True,
        born_place_align=True,
    )

    ctrl: list[UnitreeCtrlCfg] = [
        UnitreeCtrlCfg(),
    ]

    policy: G1BeyondMimicPolicyCfg = G1BeyondMimicPolicyCfg(
        policy_name="J_Dance2_Salsa",

        # La ONNX tiene 160 observaciones y requiere posición
        # y velocidad lineal estimadas.
        without_state_estimator=False,
        override_robot_anchor_pos=False,

        use_motion_from_model=True,
        use_modelmeta_config=True,

        start_timestep=0,
        max_timestep=2150,
    )

    run_fullspeed: bool = False
    do_safety_check: bool = True


@cfg_registry.register
class g1_daddance_real(G1RlLocoMimicPipelineCfg):
    """
    B_DadDance con política de locomoción de respaldo.

    Flujo:
    1. Inicia usando locomoción estable.
    2. START o R inicia el baile.
    3. Al terminar el movimiento, vuelve automáticamente a locomoción.
    4. El proceso continúa ejecutándose y mantiene al robot de pie.
    """

    dance_name: str = "B_DadDance"

    robot: str = "g1"

    env: G1RealEnvCfg = G1RealEnvCfg(
        env_type="UnitreeCppEnv",

        update_with_fk=True,
        born_place_align=True,

        unitree=G1UnitreeCfg(
            net_if="eth0",
            defer_release=True
        ),
    )

    ctrl: list[UnitreeCtrlCfg | KeyboardCtrlCfg] = [
        UnitreeCtrlCfg(
            combination_init_buttons=[],

            triggers={
                "A": "[SHUTDOWN]",
                "Select": "[POLICY_LOCO]",
                "Start": "[POLICY_MIMIC]",
            },
        ),

        KeyboardCtrlCfg(
            triggers={
                "r": "[POLICY_MIMIC]",
                "l": "[POLICY_LOCO]",
                "o": "[SHUTDOWN]",
            },
        ),
    ]

    # Política que se encarga de mantener equilibrio y locomoción.
    # Usa únicamente una política que ya hayas validado físicamente.
    loco_policy: G1AmoPolicyCfg = G1AmoPolicyCfg(freq = 50) # la frecuencia se puede eliminar

    mimic_policies: list[G1BeyondMimicPolicyCfg] = [
        G1BeyondMimicPolicyCfg(
            policy_name=dance_name,

            ############
            freq=50,
            motion_fps=60.0,
            playback_rate=1.0,

            # Suavizado moderado.
            action_beta=0.8,
            ############

            without_state_estimator=False,

            override_robot_anchor_pos=False,

            use_motion_from_model=True,

            # Usa este campo solamente si el modelo no reporta
            # correctamente el final del movimiento.
            max_timestep=get_motion_frames(dance_name),
        ),
    ]

    do_safety_check: bool = True

@cfg_registry.register
class g1_salsa_real(G1RlLocoMimicPipelineCfg):
    """
    B_DadDance con política de locomoción de respaldo.

    Flujo:
    1. Inicia usando locomoción estable.
    2. START o R inicia el baile.
    3. Al terminar el movimiento, vuelve automáticamente a locomoción.
    4. El proceso continúa ejecutándose y mantiene al robot de pie.
    """

    dance_name: str = "J_Dance2_Salsa"

    robot: str = "g1"

    env: G1RealEnvCfg = G1RealEnvCfg(
        env_type="UnitreeCppEnv",

        update_with_fk=True,
        born_place_align=True,

        unitree=G1UnitreeCfg(
            net_if="eth0",
        ),
    )

    ctrl: list[UnitreeCtrlCfg | KeyboardCtrlCfg] = [
        UnitreeCtrlCfg(
            combination_init_buttons=[],

            triggers={
                "A": "[SHUTDOWN]",
                "Select": "[POLICY_LOCO]",
                "Start": "[POLICY_MIMIC]",
            },
        ),

        KeyboardCtrlCfg(
            triggers={
                "r": "[POLICY_MIMIC]",
                "l": "[POLICY_LOCO]",
                "o": "[SHUTDOWN]",
            },
        ),
    ]

    # Política que se encarga de mantener equilibrio y locomoción.
    # Usa únicamente una política que ya hayas validado físicamente.
    loco_policy: G1AmoPolicyCfg = G1AmoPolicyCfg(freq = 50) # la frecuencia se puede eliminar

    mimic_policies: list[G1BeyondMimicPolicyCfg] = [
        G1BeyondMimicPolicyCfg(
            policy_name=dance_name,

            ############
            freq=50,
            motion_fps=60.0,
            playback_rate=1.0,

            # Suavizado moderado.
            action_beta=0.8,
            ############

            without_state_estimator=False,

            override_robot_anchor_pos=False,

            use_motion_from_model=True,

            # Usa este campo solamente si el modelo no reporta
            # correctamente el final del movimiento.
            max_timestep=get_motion_frames(dance_name),
        ),
    ]

    do_safety_check: bool = True


@cfg_registry.register
class g1_woah_real(G1RlLocoMimicPipelineCfg):
    """
    B_DadDance con política de locomoción de respaldo.

    Flujo:
    1. Inicia usando locomoción estable.
    2. START o R inicia el baile.
    3. Al terminar el movimiento, vuelve automáticamente a locomoción.
    4. El proceso continúa ejecutándose y mantiene al robot de pie.
    """

    dance_name: str = "J_Dance3_Woah"

    robot: str = "g1"

    env: G1RealEnvCfg = G1RealEnvCfg(
        env_type="UnitreeCppEnv",

        update_with_fk=True,
        born_place_align=True,

        unitree=G1UnitreeCfg(
            net_if="eth0",
        ),
    )

    ctrl: list[UnitreeCtrlCfg | KeyboardCtrlCfg] = [
        UnitreeCtrlCfg(
            combination_init_buttons=[],

            triggers={
                "A": "[SHUTDOWN]",
                "Select": "[POLICY_LOCO]",
                "Start": "[POLICY_MIMIC]",
            },
        ),

        KeyboardCtrlCfg(
            triggers={
                "r": "[POLICY_MIMIC]",
                "l": "[POLICY_LOCO]",
                "o": "[SHUTDOWN]",
            },
        ),
    ]

    # Política que se encarga de mantener equilibrio y locomoción.
    # Usa únicamente una política que ya hayas validado físicamente.
    loco_policy: G1AmoPolicyCfg = G1AmoPolicyCfg(freq = 50) # la frecuencia se puede eliminar

    mimic_policies: list[G1BeyondMimicPolicyCfg] = [
        G1BeyondMimicPolicyCfg(
            policy_name=dance_name,

            ############
            freq=50,
            motion_fps=60.0,
            playback_rate=1.0,

            # Suavizado moderado.
            action_beta=0.8,
            ############

            without_state_estimator=False,

            override_robot_anchor_pos=False,

            use_motion_from_model=True,

            # Usa este campo solamente si el modelo no reporta
            # correctamente el final del movimiento.
            max_timestep=get_motion_frames(dance_name)
        ),
    ]

    do_safety_check: bool = True

@cfg_registry.register
class g1_steptouch_real(G1RlLocoMimicPipelineCfg):
    """
    B_DadDance con política de locomoción de respaldo.

    Flujo:
    1. Inicia usando locomoción estable.
    2. START o R inicia el baile.
    3. Al terminar el movimiento, vuelve automáticamente a locomoción.
    4. El proceso continúa ejecutándose y mantiene al robot de pie.
    """

    dance_name: str = "J_Dance0_StepTouch"

    robot: str = "g1"

    env: G1RealEnvCfg = G1RealEnvCfg(
        env_type="UnitreeCppEnv",

        update_with_fk=True,
        born_place_align=True,

        unitree=G1UnitreeCfg(
            net_if="eth0",
        ),
    )

    ctrl: list[UnitreeCtrlCfg | KeyboardCtrlCfg] = [
        UnitreeCtrlCfg(
            combination_init_buttons=[],

            triggers={
                "A": "[SHUTDOWN]",
                "Select": "[POLICY_LOCO]",
                "Start": "[POLICY_MIMIC]",
            },
        ),

        KeyboardCtrlCfg(
            triggers={
                "r": "[POLICY_MIMIC]",
                "l": "[POLICY_LOCO]",
                "o": "[SHUTDOWN]",
            },
        ),
    ]

    # Política que se encarga de mantener equilibrio y locomoción.
    # Usa únicamente una política que ya hayas validado físicamente.
    loco_policy: G1AmoPolicyCfg = G1AmoPolicyCfg(freq = 50) # la frecuencia se puede eliminar

    mimic_policies: list[G1BeyondMimicPolicyCfg] = [
        G1BeyondMimicPolicyCfg(
            policy_name=dance_name,

            ############
            freq=50,
            motion_fps=60.0,
            playback_rate=1.0,

            # Suavizado moderado.
            action_beta=0.8,
            ############

            without_state_estimator=False,

            override_robot_anchor_pos=False,

            use_motion_from_model=True,

            # Usa este campo solamente si el modelo no reporta
            # correctamente el final del movimiento.
            max_timestep=get_motion_frames(dance_name)
        ),
    ]

    do_safety_check: bool = True

@cfg_registry.register
class g1_spiral_real(G1RlLocoMimicPipelineCfg):
    """
    B_DadDance con política de locomoción de respaldo.

    Flujo:
    1. Inicia usando locomoción estable.
    2. START o R inicia el baile.
    3. Al terminar el movimiento, vuelve automáticamente a locomoción.
    4. El proceso continúa ejecutándose y mantiene al robot de pie.
    """

    dance_name: str = "B_SpiralDance"

    robot: str = "g1"

    env: G1RealEnvCfg = G1RealEnvCfg(
        env_type="UnitreeCppEnv",

        update_with_fk=True,
        born_place_align=True,

        unitree=G1UnitreeCfg(
            net_if="eth0",
        ),
    )

    ctrl: list[UnitreeCtrlCfg | KeyboardCtrlCfg] = [
        UnitreeCtrlCfg(
            combination_init_buttons=[],

            triggers={
                "A": "[SHUTDOWN]",
                "Select": "[POLICY_LOCO]",
                "Start": "[POLICY_MIMIC]",
            },
        ),

        KeyboardCtrlCfg(
            triggers={
                "r": "[POLICY_MIMIC]",
                "l": "[POLICY_LOCO]",
                "o": "[SHUTDOWN]",
            },
        ),
    ]

    # Política que se encarga de mantener equilibrio y locomoción.
    # Usa únicamente una política que ya hayas validado físicamente.
    loco_policy: G1AmoPolicyCfg = G1AmoPolicyCfg(freq = 50) # la frecuencia se puede eliminar

    mimic_policies: list[G1BeyondMimicPolicyCfg] = [
        G1BeyondMimicPolicyCfg(
            policy_name=dance_name,

            ############
            freq=50,
            motion_fps=60.0,
            playback_rate=1.0,

            # Suavizado moderado.
            action_beta=0.8,
            ############

            without_state_estimator=False,

            override_robot_anchor_pos=False,

            use_motion_from_model=True,

            # Usa este campo solamente si el modelo no reporta
            # correctamente el final del movimiento.
            max_timestep=get_motion_frames(dance_name)
        ),
    ]

    do_safety_check: bool = True

@cfg_registry.register
class g1_stretch_real(G1RlLocoMimicPipelineCfg):
    """
    B_DadDance con política de locomoción de respaldo.

    Flujo:
    1. Inicia usando locomoción estable.
    2. START o R inicia el baile.
    3. Al terminar el movimiento, vuelve automáticamente a locomoción.
    4. El proceso continúa ejecutándose y mantiene al robot de pie.
    """

    dance_name: str = "B_StretchDance"

    robot: str = "g1"

    env: G1RealEnvCfg = G1RealEnvCfg(
        env_type="UnitreeCppEnv",

        update_with_fk=True,
        born_place_align=True,

        unitree=G1UnitreeCfg(
            net_if="eth0",
        ),
    )

    ctrl: list[UnitreeCtrlCfg | KeyboardCtrlCfg] = [
        UnitreeCtrlCfg(
            combination_init_buttons=[],

            triggers={
                "A": "[SHUTDOWN]",
                "Select": "[POLICY_LOCO]",
                "Start": "[POLICY_MIMIC]",
            },
        ),

        KeyboardCtrlCfg(
            triggers={
                "r": "[POLICY_MIMIC]",
                "l": "[POLICY_LOCO]",
                "o": "[SHUTDOWN]",
            },
        ),
    ]

    # Política que se encarga de mantener equilibrio y locomoción.
    # Usa únicamente una política que ya hayas validado físicamente.
    loco_policy: G1AmoPolicyCfg = G1AmoPolicyCfg(freq = 50) # la frecuencia se puede eliminar

    mimic_policies: list[G1BeyondMimicPolicyCfg] = [
        G1BeyondMimicPolicyCfg(
            policy_name=dance_name,

            ############
            freq=50,
            motion_fps=60.0,
            playback_rate=1.0,

            # Suavizado moderado.
            action_beta=0.8,
            ############

            without_state_estimator=False,

            override_robot_anchor_pos=False,

            use_motion_from_model=True,

            # Usa este campo solamente si el modelo no reporta
            # correctamente el final del movimiento.
            max_timestep=get_motion_frames(dance_name)
        ),
    ]

    do_safety_check: bool = True

@cfg_registry.register
class g1_broadway_real(G1RlLocoMimicPipelineCfg):
    """
    B_DadDance con política de locomoción de respaldo.

    Flujo:
    1. Inicia usando locomoción estable.
    2. START o R inicia el baile.
    3. Al terminar el movimiento, vuelve automáticamente a locomoción.
    4. El proceso continúa ejecutándose y mantiene al robot de pie.
    """

    dance_name: str = "J_Dance4_Broadway"

    robot: str = "g1"

    env: G1RealEnvCfg = G1RealEnvCfg(
        env_type="UnitreeCppEnv",

        update_with_fk=True,
        born_place_align=True,

        unitree=G1UnitreeCfg(
            net_if="eth0",
        ),
    )

    ctrl: list[UnitreeCtrlCfg | KeyboardCtrlCfg] = [
        UnitreeCtrlCfg(
            combination_init_buttons=[],

            triggers={
                "A": "[SHUTDOWN]",
                "Select": "[POLICY_LOCO]",
                "Start": "[POLICY_MIMIC]",
            },
        ),

        KeyboardCtrlCfg(
            triggers={
                "r": "[POLICY_MIMIC]",
                "l": "[POLICY_LOCO]",
                "o": "[SHUTDOWN]",
            },
        ),
    ]

    # Política que se encarga de mantener equilibrio y locomoción.
    # Usa únicamente una política que ya hayas validado físicamente.
    loco_policy: G1AmoPolicyCfg = G1AmoPolicyCfg(freq = 50) # la frecuencia se puede eliminar

    mimic_policies: list[G1BeyondMimicPolicyCfg] = [
        G1BeyondMimicPolicyCfg(
            policy_name=dance_name,

            ############
            freq=50,
            motion_fps=60.0,
            playback_rate=1.0,

            # Suavizado moderado.
            action_beta=0.8,
            ############

            without_state_estimator=False,

            override_robot_anchor_pos=False,

            use_motion_from_model=True,

            # Usa este campo solamente si el modelo no reporta
            # correctamente el final del movimiento.
            max_timestep=get_motion_frames(dance_name)
        ),
    ]

    do_safety_check: bool = True

@cfg_registry.register
class g1_hype_real(G1RlLocoMimicPipelineCfg):
    """
    B_DadDance con política de locomoción de respaldo.

    Flujo:
    1. Inicia usando locomoción estable.
    2. START o R inicia el baile.
    3. Al terminar el movimiento, vuelve automáticamente a locomoción.
    4. El proceso continúa ejecutándose y mantiene al robot de pie.
    """

    dance_name: str = "J_Dance5_Hype"

    robot: str = "g1"

    env: G1RealEnvCfg = G1RealEnvCfg(
        env_type="UnitreeCppEnv",

        update_with_fk=True,
        born_place_align=True,

        unitree=G1UnitreeCfg(
            net_if="eth0",
        ),
    )

    ctrl: list[UnitreeCtrlCfg | KeyboardCtrlCfg] = [
        UnitreeCtrlCfg(
            combination_init_buttons=[],

            triggers={
                "A": "[SHUTDOWN]",
                "Select": "[POLICY_LOCO]",
                "Start": "[POLICY_MIMIC]",
            },
        ),

        KeyboardCtrlCfg(
            triggers={
                "r": "[POLICY_MIMIC]",
                "l": "[POLICY_LOCO]",
                "o": "[SHUTDOWN]",
            },
        ),
    ]

    # Política que se encarga de mantener equilibrio y locomoción.
    # Usa únicamente una política que ya hayas validado físicamente.
    loco_policy: G1AmoPolicyCfg = G1AmoPolicyCfg(freq = 50) # la frecuencia se puede eliminar

    mimic_policies: list[G1BeyondMimicPolicyCfg] = [
        G1BeyondMimicPolicyCfg(
            policy_name=dance_name,

            ############
            freq=50,
            motion_fps=60.0,
            playback_rate=1.0,

            # Suavizado moderado.
            action_beta=0.8,
            ############

            without_state_estimator=False,

            override_robot_anchor_pos=False,

            use_motion_from_model=True,

            # Usa este campo solamente si el modelo no reporta
            # correctamente el final del movimiento.
            max_timestep=get_motion_frames(dance_name)
        ),
    ]

    do_safety_check: bool = True

@cfg_registry.register
class g1_party_real(G1RlLocoMimicPipelineCfg):
    """
    Flujo:
    1. Inicia usando locomoción estable.
    2. START o R inicia el baile.
    3. Al terminar el movimiento, vuelve automáticamente a locomoción.
    4. El proceso continúa ejecutándose y mantiene al robot de pie.
    """

    dance_name: str = "J_Dance7_Party"

    robot: str = "g1"

    env: G1RealEnvCfg = G1RealEnvCfg(
        env_type="UnitreeCppEnv",

        update_with_fk=True,
        born_place_align=True,

        unitree=G1UnitreeCfg(
            net_if="eth0",
        ),
    )

    ctrl: list[UnitreeCtrlCfg | KeyboardCtrlCfg] = [
        UnitreeCtrlCfg(
            combination_init_buttons=[],

            triggers={
                "A": "[SHUTDOWN]",
                "Select": "[POLICY_LOCO]",
                "Start": "[POLICY_MIMIC]",
            },
        ),

        KeyboardCtrlCfg(
            triggers={
                "r": "[POLICY_MIMIC]",
                "l": "[POLICY_LOCO]",
                "o": "[SHUTDOWN]",
            },
        ),
    ]

    # Política que se encarga de mantener equilibrio y locomoción.
    # Usa únicamente una política que ya hayas validado físicamente.
    loco_policy: G1AmoPolicyCfg = G1AmoPolicyCfg(freq = 50) # la frecuencia se puede eliminar

    mimic_policies: list[G1BeyondMimicPolicyCfg] = [
        G1BeyondMimicPolicyCfg(
            policy_name=dance_name,

            ############
            freq=50,
            motion_fps=60.0,
            playback_rate=1.0,

            # Suavizado moderado.
            action_beta=0.8,
            ############

            without_state_estimator=False,

            override_robot_anchor_pos=False,

            use_motion_from_model=True,

            # Usa este campo solamente si el modelo no reporta
            # correctamente el final del movimiento.
            max_timestep=get_motion_frames(dance_name)
        ),
    ]

    do_safety_check: bool = True


@cfg_registry.register
class g1_disco_real(G1RlLocoMimicPipelineCfg):
    """
    B_DadDance con política de locomoción de respaldo.

    Flujo:
    1. Inicia usando locomoción estable.
    2. START o R inicia el baile.
    3. Al terminar el movimiento, vuelve automáticamente a locomoción.
    4. El proceso continúa ejecutándose y mantiene al robot de pie.
    """

    dance_name: str = "J_ShortDance14_Disco"

    robot: str = "g1"

    env: G1RealEnvCfg = G1RealEnvCfg(
        env_type="UnitreeCppEnv",

        update_with_fk=True,
        born_place_align=True,

        unitree=G1UnitreeCfg(
            net_if="eth0",
        ),
    )

    ctrl: list[UnitreeCtrlCfg | KeyboardCtrlCfg] = [
        UnitreeCtrlCfg(
            combination_init_buttons=[],

            triggers={
                "A": "[SHUTDOWN]",
                "Select": "[POLICY_LOCO]",
                "Start": "[POLICY_MIMIC]",
            },
        ),

        KeyboardCtrlCfg(
            triggers={
                "r": "[POLICY_MIMIC]",
                "l": "[POLICY_LOCO]",
                "o": "[SHUTDOWN]",
            },
        ),
    ]

    # Política que se encarga de mantener equilibrio y locomoción.
    # Usa únicamente una política que ya hayas validado físicamente.
    loco_policy: G1AmoPolicyCfg = G1AmoPolicyCfg(freq = 50) # la frecuencia se puede eliminar

    mimic_policies: list[G1BeyondMimicPolicyCfg] = [
        G1BeyondMimicPolicyCfg(
            policy_name=dance_name,

            ############
            freq=50,
            motion_fps=60.0,
            playback_rate=1.0,

            # Suavizado moderado.
            action_beta=0.8,
            ############

            without_state_estimator=False,

            override_robot_anchor_pos=False,

            use_motion_from_model=True,

            # Usa este campo solamente si el modelo no reporta
            # correctamente el final del movimiento.
            max_timestep=get_motion_frames(dance_name)
        ),
    ]

    do_safety_check: bool = True

