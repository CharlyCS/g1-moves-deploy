import logging
from collections.abc import Callable
from enum import Enum, auto

import numpy as np

import robojudo.environment
from robojudo.controller import CtrlManager
from robojudo.environment import Environment
from robojudo.pipeline import Pipeline, pipeline_registry
from robojudo.pipeline.pipeline_cfgs import RlLocoMimicPipelineCfg
from robojudo.pipeline.rl_multi_policy_pipeline import PolicyManager, RlMultiPolicyPipeline
from robojudo.pipeline.rl_pipeline import PolicyWrapper
from robojudo.policy import PolicyCfg
from robojudo.utils.progress import ProgressBar
import time


logger = logging.getLogger(__name__)


class PolicyInterpManager(PolicyManager):
    class InterpState(Enum):
        IDLE = auto()
        START = auto()
        IN_PROGRESS = auto()
        END = auto()

    DURATIONS_LOCO_MIMIC = [0, 75, 25]  # [start, in-progress, end] in steps
    DURATIONS_MIMIC_LOCO = [25, 75, 0]  # [start, in-progress, end] in steps

    def __init__(
        self,
        cfg_policy_loco: PolicyCfg,
        cfg_policies: list[PolicyCfg],
        env: Environment,
        loco_dof_pos: np.ndarray | None = None,
        device: str = "cpu",
    ):
        cfg_policies_all = [cfg_policy_loco] + cfg_policies
        super().__init__(cfg_policies_all, env, device)

        self.policy_loco_id = 0
        self.policy_mimic_num = len(cfg_policies)
        assert self.policy_mimic_num > 0, "At least one mimic policy is required for switching."
        self.policy_mimic_ids = list(range(1, self.policy_mimic_num + 1))
        self.policy_mimic_idx = 0

        # Interpolation variables
        self.interp_state = self.InterpState.IDLE
        self.interp_timestep = 0
        self.interp_durations = [20, 40, 20]  # [start, in-progress, end] in steps
        self.interp_pbar = None
        self.interp_callback_start = None
        self.interp_callback_end = None

        self.loco_dof_pos = loco_dof_pos if loco_dof_pos is not None else self.env.default_pos.copy()
        self.override_dof_pos = self.loco_dof_pos.copy()

    def _interpolate_init(
        self,
        get_target_pos: Callable[[], np.ndarray],
        durations: list[int],
        callback_start=None,
        callback_end=None,
    ):
        self.interp_get_target_pos = get_target_pos
        self.interp_durations = durations
        self.interp_callback_start = callback_start
        self.interp_callback_end = callback_end
        self.interp_pbar = ProgressBar("Interpolation", durations[1])

        self.interp_state = self.InterpState.START
        # Starting Tasks
        self.timer.add(self._interpolate_start, delay_steps=durations[0])
        # Ending Tasks
        self.timer.add(self._interpolate_end, delay_steps=sum(durations) + 1)

    def _interpolate_start(self):
        if self.interp_state != self.InterpState.START:
            return
        if self.interp_callback_start is not None:
            self.interp_callback_start()
            self.interp_callback_start = None

        self.interp_start_pos = self.env.dof_pos.copy()
        self.interp_target_pos = self.interp_get_target_pos()
        self.interp_timestep = 0
        self.interp_state = self.InterpState.IN_PROGRESS

        # logger.debug("Interpolation started.")

    def _interpolate_end(self):
        if self.interp_state != self.InterpState.END:
            return
        self.override_dof_pos = self.interp_target_pos.copy()
        if self.interp_pbar:
            self.interp_pbar.close()
            self.interp_pbar = None
        if self.interp_callback_end is not None:
            self.interp_callback_end()
            self.interp_callback_end = None
        self.interp_state = self.InterpState.IDLE

        # logger.debug("Interpolation ended.")

    def _interpolate_step(self):
        if self.interp_state != self.InterpState.IN_PROGRESS:
            return

        if self.interp_pbar:
            self.interp_pbar.set(self.interp_timestep)

        progress = self.interp_timestep / self.interp_durations[1]
        alpha = min(progress, 1.0)
        self.override_dof_pos = (1 - alpha) * self.interp_start_pos + alpha * self.interp_target_pos

        if self.interp_timestep < self.interp_durations[1]:
            self.interp_timestep += 1
        else:
            self.interp_state = self.InterpState.END

    def toggle_mimic_policy(self, delta: int):
        # only switch mimic policy if current policy is locomotion
        if self.current_policy_id != self.policy_loco_id:
            logger.warning("Cannot switch mimic policy when policy is mimic.")
            return

        self.policy_mimic_idx = (self.policy_mimic_idx + delta) % self.policy_mimic_num
        policy_id = self.policy_mimic_ids[self.policy_mimic_idx]
        policy_name = self.policy_by_id(policy_id).name
        logger.info(f"Switch mimic policy to {self.policy_mimic_idx}: {policy_name}")

    # select by policy function
    def select_mimic_policy(self, policy_name: str):
        """
        Selecciona una mimic policy por nombre.

        No inicia la policy inmediatamente.
        Solo modifica policy_mimic_idx para que START
        ejecute posteriormente la policy seleccionada.
        """

        # Solo permitir selección mientras estamos en locomoción.
        if self.current_policy_id != self.policy_loco_id:
            logger.warning(
                "Cannot switch mimic policy when policy is mimic."
            )
            return False

        # Evitar cambiar selección durante una transición.
        if self.interp_state != self.InterpState.IDLE:
            logger.warning(
                "Cannot switch mimic policy during interpolation."
            )
            return False

        policy_name = policy_name.strip()

        for idx, policy_id in enumerate(self.policy_mimic_ids):
            policy = self.policy_by_id(policy_id)

            if policy.name == policy_name:
                self.policy_mimic_idx = idx

                logger.info(
                    f"Selected mimic policy "
                    f"idx={idx}, "
                    f"id={policy_id}, "
                    f"name={policy.name}"
                )

                return True

        available = [
            self.policy_by_id(policy_id).name
            for policy_id in self.policy_mimic_ids
        ]

        logger.error(
            f"Mimic policy '{policy_name}' not found. "
            f"Available policies: {available}"
        )

        return False


    def switch_to_loco(self):
        if self.current_policy_id == self.policy_loco_id and self.interp_state == self.InterpState.IDLE:
            logger.warning("Already in locomotion policy.")
            return
        if self.current_policy_id != self.policy_loco_id:
            self.policy_by_id(self.policy_loco_id).reset()
            self.warmup_policy_indices.add(self.policy_loco_id)
        self._interpolate_init(
            get_target_pos=lambda: self.loco_dof_pos,
            durations=self.DURATIONS_MIMIC_LOCO,
            callback_start=lambda: self.set_policy(self.policy_loco_id),
        )

    def switch_to_mimic(self):
        if self.current_policy_id != self.policy_loco_id:
            logger.warning("Already in mimic policy.")
            return
        policy_mimic_id = self.policy_mimic_ids[self.policy_mimic_idx]
        self.policy_by_id(policy_mimic_id).reset()
        self.warmup_policy_indices.add(policy_mimic_id)
        self._interpolate_init(
            get_target_pos=lambda: self.policy_by_id(policy_mimic_id).get_init_dof_pos(),
            durations=self.DURATIONS_LOCO_MIMIC,
            callback_end=lambda: self.set_policy(policy_mimic_id),
        )

    def step(self, env_data, ctrl_data):
        super().step(env_data, ctrl_data)
        self._interpolate_step()
    

@pipeline_registry.register
class RlLocoMimicPipeline(RlMultiPolicyPipeline):
    cfg: RlLocoMimicPipelineCfg

    @property
    def policy(self) -> PolicyWrapper:
        return self.policy_manager.policy

    def __init__(self, cfg: RlLocoMimicPipelineCfg):
        # Skip RlMultiPolicyPipeline initialization
        Pipeline.__init__(self, cfg=cfg)

        env_class: type[Environment] = getattr(robojudo.environment, self.cfg.env.env_type)
        self.env: Environment = env_class(cfg_env=self.cfg.env, device=self.device)

        self.ctrl_manager = CtrlManager(cfg_ctrls=self.cfg.ctrl, env=self.env, device=self.device)

        # upper body override
        self.num_upper_body_dof = self.cfg.upper_dof_num
        if upper_dof_pos_default := self.cfg.upper_dof_pos_default:
            loco_dof_pos = self.env.default_pos.copy()
            loco_dof_pos[-self.num_upper_body_dof :] = upper_dof_pos_default
            self.loco_dof_pos = loco_dof_pos
        else:
            self.loco_dof_pos = self.env.default_pos
        if override_dof_indices := self.cfg.upper_dof_override_indices:
            self.override_dof_indices = override_dof_indices
        else:
            self.override_dof_indices = list(range(-self.num_upper_body_dof, 0))

        self.policy_manager = PolicyInterpManager(
            cfg_policy_loco=self.cfg.loco_policy,
            cfg_policies=self.cfg.mimic_policies,
            env=self.env,
            loco_dof_pos=self.loco_dof_pos,
            device=self.device,
        )
        self.env.update_dof_cfg(override_cfg=self.policy.cfg_action_dof)
        self.visualizer = self.env.visualizer

        self.freq = self.cfg.loco_policy.freq
        self.dt = 1.0 / self.freq

        self.policy_locomotion_mimic_flag = 0  # 0: locomotion, 1: mimic

        self.self_check()
        self.reset()

    def post_step_callback(self, env_data, ctrl_data, extras, pd_target):
        self.timestep += 1

        commands = ctrl_data.get("COMMANDS", [])

        # Handle policy CALLBACK
        for callback in extras.get("CALLBACK", []):
            match callback:
                case "[MOTION_DONE]":
                    if self.policy_locomotion_mimic_flag == 1:
                        commands.append("[POLICY_LOCO]")
                        logger.info("Mimic motion done, switch to locomotion policy.")

        for command in commands:
            match command:
                case "[SHUTDOWN]":
                    logger.warning("Emergency shutdown!")
                    self.env.shutdown()
                case "[SIM_REBORN]":
                    if hasattr(self.env, "reborn"):
                        logger.warning("Simulation Env reborn!")
                        self.env.reborn()  # pyright: ignore[reportAttributeAccessIssue]
                case cmd if cmd.startswith("[POLICY_SWITCH]"):
                    switch_target = cmd.split(",")[1]
                    if switch_target == "NEXT":
                        self.policy_manager.toggle_mimic_policy(1)
                    elif switch_target == "LAST":
                        self.policy_manager.toggle_mimic_policy(-1)
                case "[POLICY_LOCO]":
                    self.policy_locomotion_mimic_flag = 0
                    self.policy_manager.switch_to_loco()
                case "[POLICY_MIMIC]":
                    self.policy_locomotion_mimic_flag = 1
                    self.policy_manager.switch_to_mimic()

        self.ctrl_manager.post_step_callback(ctrl_data)

        self.policy.post_step_callback(commands)
        if self.visualizer is not None:
            self.policy.debug_viz(self.visualizer, env_data, ctrl_data, extras)

        # # Handle policy switch after step to avoid mid-step change
        self.policy_manager.step(env_data, ctrl_data)

        self.safety_check()
        if self.cfg.debug.log_obs:
            self.debug_logger.log(
                env_data=env_data,
                ctrl_data=ctrl_data,
                extras=extras,
                pd_target=pd_target,
                timestep=self.timestep,
            )

    def step(self, dry_run=False):
        self.env.update()
        env_data = self.env.get_data()
        ctrl_data = self.ctrl_manager.get_ctrl_data(env_data)

        commands = ctrl_data.get("COMMANDS", [])
        if len(commands) > 0:
            logger.info(f"{'=' * 10} COMMANDS {'=' * 10}\n{commands}")

        if self.policy_manager.current_policy_id == self.policy_manager.policy_loco_id:
            ctrl_data["ref_dof_pos"] = self.policy.obs_adapter.fit(self.policy_manager.override_dof_pos)

        obs, extras = self.policy.get_observation(env_data, ctrl_data)

        pd_target = self.policy.get_pd_target(obs)

        if self.policy_manager.current_policy_id == self.policy_manager.policy_loco_id:
            pd_target[self.override_dof_indices] = self.policy_manager.override_dof_pos[self.override_dof_indices]

        if not dry_run:
            self.env.step(pd_target, extras.get("hand_pose", None))
            # logger.debug(pd_target)

        self.post_step_callback(env_data, ctrl_data, extras, pd_target)

    # def prepare(self):
    #     # En este punto:
    #     # - AMO ya está cargado.
    #     # - BeyondMimic ya está cargado.
    #     # - La ONNX ya está lista.
    #     # - Unitree todavía sostiene al robot.

    #     if hasattr(self.env, "activate_control"):
    #         self.env.activate_control()

    #     # Inmediatamente comienza la preparación normal,
    #     # partiendo hacia la postura estable de locomoción.

    #     init_motor_angle = self.loco_dof_pos.copy()
    #     super().prepare(init_motor_angle=init_motor_angle)
    

    #   calculo de salide de AMO para locomocion, aun no corre el comando al robot
    #############################
    def _get_loco_pd_target(self):
        """
        Calcula una salida de AMO utilizando el estado real actual.
        No envía todavía el comando al robot.
        """

        self.env.update()

        env_data = self.env.get_data()
        ctrl_data = self.ctrl_manager.get_ctrl_data(env_data)

        # Referencia usada por la política de locomoción.
        ctrl_data["ref_dof_pos"] = self.policy.obs_adapter.fit(
            self.policy_manager.override_dof_pos
        )

        obs, extras = self.policy.get_observation(
            env_data,
            ctrl_data,
        )

        pd_target = self.policy.get_pd_target(obs)

        # AMO controla activamente piernas y cintura.
        # Los DOF superiores se mantienen en la referencia seleccionada.
        pd_target[self.override_dof_indices] = (
            self.policy_manager.override_dof_pos[
                self.override_dof_indices
            ]
        )

        return np.asarray(pd_target, dtype=np.float64), extras
    #############################

    def prepare(self):
        """
        Toma de control segura:

        1. Unitree AI mantiene el balance mientras se prepara AMO.
        2. Se calcula la primera salida AMO antes de ReleaseMode.
        3. El primer LowCmd ya contiene una salida de locomoción activa.
        4. AMO controla inmediatamente piernas y cintura.
        5. Solo brazos y torso superior se interpolan suavemente.
        6. El robot queda en locomoción con comandos cero hasta START.
        """

        logger.warning(
            "LocoMimic prepare: direct takeover with active AMO balance."
        )

        loco_id = self.policy_manager.policy_loco_id

        # Forzar que la política actual sea AMO.
        if self.policy_manager.current_policy_id != loco_id:
            self.policy_manager.set_policy(loco_id)
        else:
            # Asegurar que los KP/KD correspondan a AMO.
            self.env.update_dof_cfg(
                override_cfg=self.policy.cfg_action_dof
            )

        self.policy_locomotion_mimic_flag = 0

        # Reset mientras Unitree AI todavía mantiene el robot.
        self.reset()

        self.env.update()

        current_dof_pos = np.asarray(
            self.env.dof_pos,
            dtype=np.float64,
        ).copy()

        loco_dof_pos = np.asarray(
            self.loco_dof_pos,
            dtype=np.float64,
        ).copy()

        upper_indices = np.asarray(
            self.override_dof_indices,
            dtype=np.int64,
        )

        # Los brazos y el torso superior comienzan exactamente
        # en la posición física actual para evitar un salto.
        initial_override = loco_dof_pos.copy()
        initial_override[upper_indices] = current_dof_pos[upper_indices]

        self.policy_manager.override_dof_pos = initial_override

        # Calcular AMO antes de liberar el modo Unitree.
        first_pd_target, _ = self._get_loco_pd_target()

        max_initial_delta = float(
            np.max(np.abs(first_pd_target - current_dof_pos))
        )

        logger.warning(
            "First AMO target computed. "
            f"Maximum joint difference: {max_initial_delta:.4f} rad"
        )

        # El primer comando cargado en UnitreeController ya es AMO.
        if hasattr(self.env, "activate_control"):
            self.env.activate_control(
                initial_positions=first_pd_target
            )
        else:
            raise RuntimeError(
                "UnitreeCppEnv does not implement activate_control()"
            )

        # AMO controla desde este momento.
        # Durante un segundo se interpolan únicamente los DOF superiores.
        settle_seconds = 1.0
        settle_steps = max(
            int(settle_seconds * self.freq),
            1,
        )

        logger.warning(
            "AMO active — blending upper body only "
            f"({settle_steps} steps, {settle_seconds:.1f}s)"
        )

        next_step_time = time.perf_counter()

        for step_idx in range(settle_steps):
            alpha = (step_idx + 1) / settle_steps

            upper_override = loco_dof_pos.copy()

            upper_override[upper_indices] = (
                (1.0 - alpha) * current_dof_pos[upper_indices]
                + alpha * loco_dof_pos[upper_indices]
            )

            self.policy_manager.override_dof_pos = upper_override

            # Ejecuta AMO completo:
            # observación, inferencia, PD target y envío.
            self.step()

            next_step_time += self.dt
            remaining = next_step_time - time.perf_counter()

            if remaining > 0:
                time.sleep(remaining)
            else:
                next_step_time = time.perf_counter()

        self.policy_manager.override_dof_pos = loco_dof_pos.copy()

        logger.warning(
            "AMO locomotion active — robot holding balance. "
            "Press START to begin mimic motion."
        )
    
    

    # ADDED TO NOT BREAK THE INTERFACE Force Locomation
    def force_locomotion(self):
        if self.policy_locomotion_mimic_flag != 0:
            logger.warning(
                "Forcing transition to locomotion fallback."
            )

        self.policy_locomotion_mimic_flag = 0
        self.policy_manager.switch_to_loco()

if __name__ == "__main__":
    pass
