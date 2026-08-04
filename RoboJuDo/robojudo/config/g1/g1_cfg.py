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
from .policy.g1_protomotions_tracker_cfg import ProtoMotionsTrackerPolicyCfg  # noqa: F401
from .policy.g1_smooth_policy_cfg import G1SmoothPolicyCfg  # noqa: F401
from .policy.g1_twist_policy_cfg import G1TwistPolicyCfg  # noqa: F401
from .policy.g1_unitree_policy_cfg import G1UnitreePolicyCfg, G1UnitreeWoGaitPolicyCfg  # noqa: F401


##### AGREGADO
from .pipeline.g1_locomimic_pipeline_cfg import G1RlLocoMimicPipelineCfg

# ======================== Basic Configs ======================== #
@cfg_registry.register
class g1(RlPipelineCfg):
    """
    Unitree G1 robot configuration, Unitree Policy, Sim2Sim.
    You can modify to play with other policies and controllers.
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg()
    # env: G1_23MujocoEnvCfg = G1_23MujocoEnvCfg()
    # env: G1_12MujocoEnvCfg = G1_12MujocoEnvCfg()

    ctrl: list[JoystickCtrlCfg | KeyboardCtrlCfg] = [  # note: the ranking of controllers matters
        JoystickCtrlCfg(),
        # KeyboardCtrlCfg(),
    ]

    policy: G1UnitreePolicyCfg = G1UnitreePolicyCfg()
    # policy: G1UnitreeWoGaitPolicyCfg = G1UnitreeWoGaitPolicyCfg()
    # policy: G1AmoPolicyCfg = G1AmoPolicyCfg()

    # run_fullspeed: bool = env.is_sim


@cfg_registry.register
class g1_real(g1):
    """
    Unitree G1 robot, Unitree Policy, Sim2Real.
    To extend the sim2sim config to sim2real, just need to change the env to real env.
    """

    # env: G1DummyEnvCfg = G1DummyEnvCfg()
    env: G1RealEnvCfg = G1RealEnvCfg(
        # env_type="UnitreeEnv",  # For unitree_sdk2py
        env_type="UnitreeCppEnv",  # For unitree_cpp, check README for more details
        unitree=G1UnitreeCfg(
            net_if="enp7s0",  # note: change to your network interface eth0
        ),
    )

    ctrl: list[UnitreeCtrlCfg] = [
        UnitreeCtrlCfg(),
    ]

    do_safety_check: bool = True  # enable safety check for real robot


@cfg_registry.register
class g1_switch(RlMultiPolicyPipelineCfg):
    """
    Example of multi-policy pipeline configuration.
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg()

    ctrl: list[KeyboardCtrlCfg | JoystickCtrlCfg] = [
        # KeyboardCtrlCfg(
        #     triggers_extra={
        #         "Key.tab": "[POLICY_TOGGLE]",
        #     }
        # ),
        JoystickCtrlCfg(
            triggers_extra={
                "RB+Down": "[POLICY_SWITCH],0",
                "RB+Up": "[POLICY_SWITCH],1",
            }
        ),
    ]

    policies: list[G1UnitreePolicyCfg | G1AmoPolicyCfg] = [
        G1UnitreePolicyCfg(),
        G1AmoPolicyCfg(),
    ]


@cfg_registry.register
class g1_locomimic(RlLocoMimicPipelineCfg):
    """
    Example of loco mimic pipeline configuration.
    You can switch between loco and mimic policies during runtime, with interpolation.
    === Check more fancy locomimic examples in g1_loco_mimic_cfg.py ===
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg()

    ctrl: list[KeyboardCtrlCfg | JoystickCtrlCfg] = [
        KeyboardCtrlCfg(
            triggers_extra={
                "]": "[POLICY_LOCO]",
                "[": "[POLICY_MIMIC]",
            }
        ),
        JoystickCtrlCfg(
            triggers_extra={
                "RB+Down": "[POLICY_LOCO]",
                "RB+Up": "[POLICY_MIMIC]",
            }
        ),
    ]

    loco_policy: G1UnitreePolicyCfg = G1UnitreePolicyCfg()
    mimic_policies: list[G1AsapPolicyCfg] = [
        G1AsapPolicyCfg(),
    ]


# ======================== Configs for supported Policy ======================== #


@cfg_registry.register
class g1_h2h(RlPipelineCfg):
    """
    Human2Humanoid
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg()
    ctrl: list[KeyboardCtrlCfg | G1MotionH2HCtrlCfg] = [
        KeyboardCtrlCfg(),
        G1MotionH2HCtrlCfg(),
    ]

    policy: G1H2HPolicyCfg = G1H2HPolicyCfg()


@cfg_registry.register
class g1_beyondmimic(RlPipelineCfg):
    """
    BeyondMimic Policy, support both with and without state estimator.
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg()
    ctrl: list[KeyboardCtrlCfg] = [
        KeyboardCtrlCfg(),
    ]

    policy: G1BeyondMimicPolicyCfg = G1BeyondMimicPolicyCfg(
        policy_name="Jump_wose",
        without_state_estimator=True,
        use_modelmeta_config=True,  # use robot dof config from modelmeta
        use_motion_from_model=True,  # use motion from onnx model
        max_timestep=140,
    )


###### EL REGISTRY PARA EL ONNX "NAME".ONNX
@cfg_registry.register
class g1_daddance(RlPipelineCfg):
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

'''
@cfg_registry.register
class g1_daddance_real(RlPipelineCfg):
    """
    Despliegue físico de B_DadDance.
    """

    robot: str = "g1"

    env: G1RealEnvCfg = G1RealEnvCfg(
        act=True,

        env_type="UnitreeCppEnv",

        unitree=G1UnitreeCfg(
            net_if="enp7s0",
            enable_odometry=True,
        ),

        odometry_type="UNITREE",
        update_with_fk=True,
        born_place_align=True,
    )

    ctrl: list[UnitreeCtrlCfg] = [
        UnitreeCtrlCfg(),
    ]

    policy: G1BeyondMimicPolicyCfg = G1BeyondMimicPolicyCfg(
        policy_name="B_DadDance",

        without_state_estimator=False,
        override_robot_anchor_pos=False,

        use_motion_from_model=True,
        use_modelmeta_config=True,

        start_timestep=0,
        max_timestep=2508,
    )

    run_fullspeed: bool = False
    do_safety_check: bool = True

'''

@cfg_registry.register
class g1_daddance_safe_real(G1RlLocoMimicPipelineCfg):
    """
    B_DadDance con política de locomoción de respaldo.

    Flujo:
    1. Inicia usando locomoción estable.
    2. START o R inicia el baile.
    3. Al terminar el movimiento, vuelve automáticamente a locomoción.
    4. El proceso continúa ejecutándose y mantiene al robot de pie.
    """

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
            policy_name="B_DadDance",

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
            max_timestep=2508,
        ),
    ]

    do_safety_check: bool = True

@cfg_registry.register
class g1_salsa_safe_real(G1RlLocoMimicPipelineCfg):
    """
    B_DadDance con política de locomoción de respaldo.

    Flujo:
    1. Inicia usando locomoción estable.
    2. START o R inicia el baile.
    3. Al terminar el movimiento, vuelve automáticamente a locomoción.
    4. El proceso continúa ejecutándose y mantiene al robot de pie.
    """

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
            policy_name="J_Dance2_Salsa",

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
            max_timestep=2150,
        ),
    ]

    do_safety_check: bool = True


@cfg_registry.register
class g1_woah_safe_real(G1RlLocoMimicPipelineCfg):
    """
    B_DadDance con política de locomoción de respaldo.

    Flujo:
    1. Inicia usando locomoción estable.
    2. START o R inicia el baile.
    3. Al terminar el movimiento, vuelve automáticamente a locomoción.
    4. El proceso continúa ejecutándose y mantiene al robot de pie.
    """

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
            policy_name="J_Dance3_Woah",

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
            max_timestep=1799
        ),
    ]

    do_safety_check: bool = True



@cfg_registry.register
class g1_beyondmimic_with_ctrl(RlPipelineCfg):
    """
    BeyondMimic with External BeyondMimicCtrl as motion source.
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg()
    ctrl: list[KeyboardCtrlCfg | G1BeyondmimicCtrlCfg] = [
        KeyboardCtrlCfg(),
        G1BeyondmimicCtrlCfg(
            motion_name="dance1_subject2",  # you can put your own motion file in assets/motions/g1
        ),
    ]

    policy: G1BeyondMimicPolicyCfg = G1BeyondMimicPolicyCfg(
        policy_name="Dance_wose",
        use_motion_from_model=False,  # use motion from BeyondmimicCtrl instead of the onnx
    )


@cfg_registry.register
class g1_asap(RlPipelineCfg):
    """
    Unitree G1 robot configuration, ASAP Policy, Sim2Sim.
    You can modify to play with other policies and controllers.
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg(forward_kinematic=None, update_with_fk=False, born_place_align=True)

    ctrl: list[JoystickCtrlCfg | KeyboardCtrlCfg] = [  # note: the ranking of controllers matters
        # JoystickCtrlCfg(),
        KeyboardCtrlCfg(triggers={"i": "[SIM_REBORN]", "o": "[SHUTDOWN]", "r": "[MOTION_RESET]"}),
    ]

    policy: G1AsapPolicyCfg = G1AsapPolicyCfg()
    """You can also try other models, from ASAP, RoboMimic, KungfuBot(PBHC)"""
    # policy: G1KungfuBotPolicyCfg = G1KungfuBotPolicyCfg() # KungfuBot horse_squat
    # # fmt: off
    # policy: G1AsapPolicyCfg = G1AsapPolicyCfg(
    #     policy_name="robomimic",
    #     relative_path="dance_0605.onnx",
    #     motion_length_s=18.0,
    #     start_upper_body_dof_pos = [
    #         0, 0, 0,
    #         0.35, 0.18, 0, 0.87,
    #         0.35, -0.18, 0, 0.87,
    #     ],
    # )
    # # fmt: on


@cfg_registry.register
class g1_asap_loco(RlPipelineCfg):
    """
    Unitree G1 robot configuration, ASAP Locomotion Policy, Sim2Sim.
    You can modify to play with other policies and controllers.
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg(forward_kinematic=None, update_with_fk=False, born_place_align=False)

    ctrl: list[JoystickCtrlCfg | KeyboardCtrlCfg] = [  # note: the ranking of controllers matters
        # JoystickCtrlCfg(),
        KeyboardCtrlCfg(
            triggers={
                "i": "[SIM_REBORN]",
                "o": "[SHUTDOWN]",
            }
        ),
    ]

    policy: G1AsapLocoPolicyCfg = G1AsapLocoPolicyCfg()


@cfg_registry.register
class g1_kungfubot2(RlPipelineCfg):
    """
    PBHC KungfuBot2 General Policy
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg()
    ctrl: list[KeyboardCtrlCfg | G1MotionKungfuBotCtrlCfg] = [
        KeyboardCtrlCfg(),
        G1MotionKungfuBotCtrlCfg(
            motion_name="kungfubot/Horse-stance_pose",  # put motion files in assets/motions/g1/phc/kungfubot
        ),
    ]

    policy: G1KungfuBotGeneralPolicyCfg = G1KungfuBotGeneralPolicyCfg(
        policy_name="horse_test_43000",  # this is a test model trained with only one motion
        compatibility_old_version=True,  # for old version of kungfubot general policy (before 2025-11-13 bugfix #68)
    )


@cfg_registry.register
class g1_twist(RlPipelineCfg):
    """
    Unitree G1 robot configuration, TWIST Policy, Sim2Sim.
    TwistRedisCtrl for the original repo of high level motion stream over redis.
    MotionTwistCtrl for built-in motion control.
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg(forward_kinematic=None, update_with_fk=False, born_place_align=False)

    ctrl: list[G1TwistRedisCtrlCfg | G1MotionTwistCtrlCfg] = [  # note: the ranking of controllers matters
        G1TwistRedisCtrlCfg(redis_host="localhost"),  # with hign level motion lib through redis
        # G1MotionTwistCtrlCfg(), # with built-in motion ctrl
    ]

    policy: G1TwistPolicyCfg = G1TwistPolicyCfg()


# ======================== Fancy Example Configs ======================== #


@cfg_registry.register
class g1_switch_beyondmimic(RlMultiPolicyPipelineCfg):
    """
    Switch between multiple BeyondMimic policies. Withour Interpolation.
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg()
    ctrl: list[KeyboardCtrlCfg | JoystickCtrlCfg] = [
        KeyboardCtrlCfg(
            triggers_extra={
                "Key.tab": "[POLICY_TOGGLE]",
                "!": "[POLICY_SWITCH],0",  # note: with shift
                "@": "[POLICY_SWITCH],1",  # note: with shift
                "#": "[POLICY_SWITCH],2",  # note: with shift
                "$": "[POLICY_SWITCH],3",  # note: with shift
            }
        ),
        JoystickCtrlCfg(
            triggers_extra={
                "RB+Down": "[POLICY_SWITCH],0",
                "RB+Left": "[POLICY_SWITCH],1",
                "RB+Up": "[POLICY_SWITCH],2",
                "RB+Right": "[POLICY_SWITCH],3",
            }
        ),
    ]

    policies: list[G1AmoPolicyCfg | G1BeyondMimicPolicyCfg] = [
        G1AmoPolicyCfg(),
        G1BeyondMimicPolicyCfg(policy_name="Violin", without_state_estimator=False, max_timestep=500),
        G1BeyondMimicPolicyCfg(policy_name="Waltz", without_state_estimator=False, max_timestep=850),
        G1BeyondMimicPolicyCfg(policy_name="Dance_wose", without_state_estimator=True),
    ]


# ======================== ProtoMotions Tracker ======================== #


@cfg_registry.register
class g1_protomotions_tracker(RlPipelineCfg):
    """ProtoMotions tracker with cached 50fps motion.

    Uses the standard RoboJuDo G1 MuJoCo environment with ``born_place_align``
    disabled (our policy handles heading alignment itself). ``random_heading``
    is on so we exercise the policy's heading-alignment recompute on each spawn.

    Use ``scripts/run_tracker_pipeline.py`` — it parses ``--onnx-path`` /
    ``--motion-path`` / ``--motion-index``, which the generic ``run_pipeline.py``
    does not.

    Usage::

        python scripts/run_tracker_pipeline.py -c g1_protomotions_tracker \\
            --motion-path assets/motions/g1/g1_bones_seed_mini.pt \\
            --motion-index 0
    """

    robot: str = "g1"
    env: G1MujocoEnvCfg = G1MujocoEnvCfg(
        born_place_align=False,
        random_heading=True,
    )
    ctrl: list[KeyboardCtrlCfg] = [
        KeyboardCtrlCfg(
            triggers={
                "r": "[MOTION_RESET]",
                "i": "[SIM_REBORN]",
                "o": "[SHUTDOWN]",
                "<": "[MOTION_FADE_IN]",
                ">": "[MOTION_FADE_OUT]",
            },
        ),
    ]

    policy: ProtoMotionsTrackerPolicyCfg = ProtoMotionsTrackerPolicyCfg()


@cfg_registry.register
class g1_protomotions_tracker_real(g1_protomotions_tracker):
    """ProtoMotions tracker on real G1 hardware.

    Use ``scripts/run_tracker_pipeline.py`` — it parses ``--onnx-path`` /
    ``--motion-path`` / ``--motion-index``, which the generic ``run_pipeline.py``
    does not.

    Usage::

        python scripts/run_tracker_pipeline.py -c g1_protomotions_tracker_real \\
            --motion-path assets/motions/g1/g1_bones_seed_mini.pt \\
            --motion-index 0
    """

    env: G1RealEnvCfg = G1RealEnvCfg(
        env_type="UnitreeCppEnv",
        unitree=G1UnitreeCfg(
            net_if="eth0",  # note: change to your network interface
        ),
        born_place_align=False,
    )
    ctrl: list[UnitreeCtrlCfg] = [
        UnitreeCtrlCfg(),
    ]
    do_safety_check: bool = True


# TIPS: check g1_loco_mimic_cfg.py for more complex examples
