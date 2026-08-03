#!/usr/bin/env python3
"""
run_g1_moves_robojudo.py

Ejecuta una policy ONNX de G1 Moves con su trayectoria NPZ usando RoboJuDo.

Ejemplos:

Simulación:
    python scripts/run_g1_moves_robojudo.py --clip-dir ~/Documents/Code_G1/g1_moves_data/dance/B_DadDance --no-joint-limit-clip

Robot real:
    python scripts/run_g1_moves_robojudo.py  --clip-dir ~/Documents/Code_G1/g1_moves_data/dance/B_DadDance --real --net-if enp7s0

En el G1 real:
    - Espera a que termine prepare().
    - Pulsa Y en el control Unitree para enviar [MOTION_RESET].
    - A / Ctrl+C detienen la ejecución según la configuración del controlador.

La observación reproduce exactamente el orden usado por g1-moves:
    29 ref_joint_pos
    29 ref_joint_vel
     3 anchor_pos_body
     6 anchor_ori_body (primeras dos columnas, transpuestas y aplanadas)
     3 base_ang_vel
     3 base_lin_vel
    29 joint_pos - default_pos
    29 joint_vel
    29 last_action
    -----------------
   160 dimensiones
"""

from __future__ import annotations

# Evita fluctuaciones de OpenMP en Jetson/ARM.
import os
import platform

if platform.machine().startswith("aarch64"):
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

import robojudo.policy
from robojudo.controller.ctrl_cfgs import KeyboardCtrlCfg, UnitreeCtrlCfg
from robojudo.pipeline.pipeline_cfgs import RlPipelineCfg
from robojudo.pipeline.rl_pipeline import RlPipeline
from robojudo.policy import Policy, policy_registry
from robojudo.policy.policy_cfgs import PolicyCfg
from robojudo.utils.rotation import TransformAlignment
from robojudo.utils.util_func import matrix_from_quat, subtract_frame_transforms

from robojudo.config.g1.env.g1_env_cfg import G1_29DoF
from robojudo.config.g1.env.g1_real_env_cfg import G1RealEnvCfg, G1UnitreeCfg

# El repositorio original conserva "mujuco" en el nombre del módulo.
try:
    from robojudo.config.g1.env.g1_mujuco_env_cfg import G1MujocoEnvCfg
except ImportError:
    from robojudo.config.g1.env.g1_mujoco_env_cfg import G1MujocoEnvCfg  # type: ignore


LOGGER = logging.getLogger("robojudo.g1_moves")


# Ganancias usadas por el script oficial run_policy.py de g1-moves.
G1_MOVES_KP = [
    40.2, 99.1, 40.2, 99.1, 28.6, 28.6,
    40.2, 99.1, 40.2, 99.1, 28.6, 28.6,
    40.2, 28.6, 28.6,
    14.3, 14.3, 14.3, 14.3, 14.3, 16.8, 16.8,
    14.3, 14.3, 14.3, 14.3, 14.3, 16.8, 16.8,
]

G1_MOVES_KD = [
    2.6, 6.3, 2.6, 6.3, 1.8, 1.8,
    2.6, 6.3, 2.6, 6.3, 1.8, 1.8,
    2.6, 1.8, 1.8,
    0.9, 0.9, 0.9, 0.9, 0.9, 1.1, 1.1,
    0.9, 0.9, 0.9, 0.9, 0.9, 1.1, 1.1,
]


class G1MovesDoF(G1_29DoF):
    """
    Mismo orden articular del G1 mode 15 / 29 DOF.

    El entrenamiento de G1 Moves usa default_joint_pos = 0.
    RoboJuDo sumará default_pos a la salida de la policy; al ser cero,
    las 29 salidas ONNX quedan como objetivos articulares directos.
    """

    default_pos: list[float] | None = [0.0] * 29
    stiffness: list[float] | None = G1_MOVES_KP
    damping: list[float] | None = G1_MOVES_KD


class G1MovesPolicyCfg(PolicyCfg):
    policy_type: str = "G1MovesPolicy"
    robot: str = "g1"
    disable_autoload: bool = True

    onnx_path: str
    motion_path: str

    freq: int = 50
    obs_dof: G1MovesDoF = G1MovesDoF()
    action_dof: G1MovesDoF = G1MovesDoF()

    # No hay escalado por articulación en G1 Moves.
    action_scale: float = 1.0
    action_clip: float | None = None
    action_beta: float = 1.0

    anchor_body_index: int = 0
    loop_motion: bool = False
    motion_speed: float = 1.0
    clip_to_joint_limits: bool = True
    require_state_estimator: bool = True

    @property
    def policy_file(self) -> str:
        return self.onnx_path


@policy_registry.register
class G1MovesPolicy(Policy):
    cfg_policy: G1MovesPolicyCfg

    def __init__(self, cfg_policy: G1MovesPolicyCfg, device: str = "cpu"):
        self._validate_file(cfg_policy.onnx_path, "ONNX")
        self._validate_file(cfg_policy.motion_path, "NPZ")

        self.session = self._create_session(cfg_policy.onnx_path, device)
        self.input_name = self._resolve_input_name()
        self.output_name = self._resolve_output_name()

        motion = np.load(cfg_policy.motion_path)

        required_keys = (
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "fps",
        )
        missing = [key for key in required_keys if key not in motion.files]
        if missing:
            raise KeyError(f"El NPZ no contiene las claves requeridas: {missing}")

        self.ref_joint_pos = np.asarray(motion["joint_pos"], dtype=np.float32)
        self.ref_joint_vel = np.asarray(motion["joint_vel"], dtype=np.float32)
        self.ref_body_pos = np.asarray(motion["body_pos_w"], dtype=np.float32)
        self.ref_body_quat = np.asarray(motion["body_quat_w"], dtype=np.float32)
        self.motion_fps = float(np.asarray(motion["fps"]).reshape(()))
        self.num_frames = int(self.ref_joint_pos.shape[0])

        self._validate_motion(cfg_policy.anchor_body_index)

        # Convención NPZ: wxyz. RoboJuDo/SciPy: xyzw.
        anchor_pos_init = self.ref_body_pos[0, cfg_policy.anchor_body_index]
        anchor_quat_init_xyzw = self.ref_body_quat[0, cfg_policy.anchor_body_index][[1, 2, 3, 0]]

        # Lleva la trayectoria de referencia a un marco local:
        # frame inicial -> x=0, y=0, yaw=0; conserva la altura z.
        self.motion_alignment = TransformAlignment(
            quat=anchor_quat_init_xyzw,
            pos=anchor_pos_init,
            yaw_only=True,
            xy_only=True,
        )

        super().__init__(cfg_policy=cfg_policy, device=device)

        self.anchor_body_index = cfg_policy.anchor_body_index
        self.loop_motion = cfg_policy.loop_motion
        self.motion_speed_default = cfg_policy.motion_speed
        self.clip_to_joint_limits = cfg_policy.clip_to_joint_limits
        self.require_state_estimator = cfg_policy.require_state_estimator

        limits = self.cfg_action_dof.position_limits
        self.position_limits = (
            np.asarray(limits, dtype=np.float32) if limits is not None else None
        )

        self._validate_onnx_signature()
        self._warmup()
        self.reset()

        duration = self.num_frames / self.motion_fps
        LOGGER.info(
            "G1 Moves cargado: %d frames, %.3f FPS, %.2f s, obs=160, actions=29",
            self.num_frames,
            self.motion_fps,
            duration,
        )

    @staticmethod
    def _validate_file(path: str, label: str) -> None:
        if not Path(path).is_file():
            raise FileNotFoundError(f"{label} no encontrado: {path}")

    @staticmethod
    def _provider_order(device: str) -> list[str]:
        device = device.lower()
        if device == "cpu":
            requested = ["CPUExecutionProvider"]
        elif device == "cuda":
            requested = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device == "tensorrt":
            requested = [
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
        else:
            raise ValueError(
                f"Device desconocido: {device}. Usa cpu, cuda o tensorrt."
            )

        available = set(ort.get_available_providers())
        providers = [provider for provider in requested if provider in available]
        if not providers:
            raise RuntimeError(
                "No existe ningún ONNX Runtime provider compatible. "
                f"Disponibles: {sorted(available)}"
            )
        return providers

    def _create_session(self, path: str, device: str) -> ort.InferenceSession:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1

        providers = self._provider_order(device)
        LOGGER.info("ONNX Runtime providers: %s", providers)

        return ort.InferenceSession(
            path,
            sess_options=options,
            providers=providers,
        )

    def _resolve_input_name(self) -> str:
        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            raise RuntimeError(
                "La policy G1 Moves debe tener una sola entrada. "
                f"Entradas detectadas: {[item.name for item in inputs]}"
            )
        return inputs[0].name

    def _resolve_output_name(self) -> str:
        outputs = self.session.get_outputs()
        names = [item.name for item in outputs]
        if "actions" in names:
            return "actions"
        if len(outputs) == 1:
            LOGGER.warning(
                "La salida no se llama 'actions'; se usará '%s'.", outputs[0].name
            )
            return outputs[0].name
        raise RuntimeError(
            "No se pudo identificar la salida de acciones. "
            f"Salidas detectadas: {names}"
        )

    def _validate_onnx_signature(self) -> None:
        input_meta = self.session.get_inputs()[0]
        output_meta = next(
            item for item in self.session.get_outputs() if item.name == self.output_name
        )

        input_shape = input_meta.shape
        output_shape = output_meta.shape

        if len(input_shape) != 2:
            raise RuntimeError(
                f"Entrada ONNX inesperada: {input_meta.name} {input_shape}; "
                "se esperaba [batch, 160]."
            )

        if isinstance(input_shape[-1], int) and input_shape[-1] != 160:
            raise RuntimeError(
                f"La ONNX espera {input_shape[-1]} observaciones, no 160."
            )

        if len(output_shape) != 2:
            raise RuntimeError(
                f"Salida ONNX inesperada: {output_meta.name} {output_shape}; "
                "se esperaba [batch, 29]."
            )

        if isinstance(output_shape[-1], int) and output_shape[-1] != 29:
            raise RuntimeError(
                f"La ONNX produce {output_shape[-1]} acciones, no 29."
            )

    def _validate_motion(self, anchor_body_index: int) -> None:
        if self.ref_joint_pos.ndim != 2 or self.ref_joint_pos.shape[1] != 29:
            raise ValueError(
                f"joint_pos debe tener forma (T, 29), tiene {self.ref_joint_pos.shape}"
            )

        if self.ref_joint_vel.shape != self.ref_joint_pos.shape:
            raise ValueError(
                "joint_vel debe tener la misma forma que joint_pos: "
                f"{self.ref_joint_vel.shape} != {self.ref_joint_pos.shape}"
            )

        if (
            self.ref_body_pos.ndim != 3
            or self.ref_body_pos.shape[0] != self.num_frames
            or self.ref_body_pos.shape[2] != 3
        ):
            raise ValueError(
                f"body_pos_w debe tener forma (T, N, 3), tiene {self.ref_body_pos.shape}"
            )

        if (
            self.ref_body_quat.ndim != 3
            or self.ref_body_quat.shape[0] != self.num_frames
            or self.ref_body_quat.shape[2] != 4
        ):
            raise ValueError(
                "body_quat_w debe tener forma (T, N, 4), "
                f"tiene {self.ref_body_quat.shape}"
            )

        if self.ref_body_pos.shape[1] != self.ref_body_quat.shape[1]:
            raise ValueError("body_pos_w y body_quat_w tienen distinto número de cuerpos")

        if not 0 <= anchor_body_index < self.ref_body_pos.shape[1]:
            raise IndexError(
                f"anchor_body_index={anchor_body_index} fuera de rango; "
                f"el NPZ contiene {self.ref_body_pos.shape[1]} cuerpos."
            )

        if self.motion_fps <= 0:
            raise ValueError(f"FPS inválido en NPZ: {self.motion_fps}")

        arrays = (
            self.ref_joint_pos,
            self.ref_joint_vel,
            self.ref_body_pos,
            self.ref_body_quat,
        )
        if not all(np.isfinite(array).all() for array in arrays):
            raise ValueError("El NPZ contiene NaN o valores infinitos")

    def _warmup(self) -> None:
        obs = np.zeros((1, 160), dtype=np.float32)
        output = self.session.run([self.output_name], {self.input_name: obs})[0]
        if np.asarray(output).shape[-1] != 29:
            raise RuntimeError(
                f"La inferencia de prueba no produjo 29 acciones: {np.asarray(output).shape}"
            )

    def reset(self) -> None:
        self.motion_time = 0.0
        self.play_speed = float(self.motion_speed_default)
        self.flag_motion_done = False
        self._motion_done_emitted = False
        self.last_action = np.zeros(29, dtype=np.float32)

    def _frame_index(self) -> int:
        raw_frame = int(max(self.motion_time, 0.0) * self.motion_fps)

        if self.loop_motion:
            return raw_frame % self.num_frames

        if raw_frame >= self.num_frames:
            self.flag_motion_done = True
            return self.num_frames - 1

        return raw_frame

    @staticmethod
    def _as_vector(
        value: Any,
        size: int,
        name: str,
        *,
        allow_none: bool = False,
    ) -> np.ndarray:
        if value is None:
            if allow_none:
                return np.zeros(size, dtype=np.float32)
            raise RuntimeError(f"RoboJuDo no proporcionó {name}")

        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if array.size != size:
            raise RuntimeError(
                f"{name} debe tener {size} elementos, tiene forma {array.shape}"
            )
        if not np.isfinite(array).all():
            raise RuntimeError(f"{name} contiene NaN o valores infinitos")
        return array

    def get_observation(self, env_data, ctrl_data):
        frame = self._frame_index()

        ref_joint_pos = self.ref_joint_pos[frame]
        ref_joint_vel = self.ref_joint_vel[frame]

        robot_pos = self._as_vector(
            env_data.base_pos,
            3,
            "base_pos",
            allow_none=not self.require_state_estimator,
        )
        robot_quat_xyzw = self._as_vector(env_data.base_quat, 4, "base_quat")
        base_ang_vel = self._as_vector(
            env_data.base_ang_vel, 3, "base_ang_vel"
        )
        base_lin_vel = self._as_vector(
            env_data.base_lin_vel,
            3,
            "base_lin_vel",
            allow_none=not self.require_state_estimator,
        )

        joint_pos = self._as_vector(env_data.dof_pos, 29, "dof_pos")
        joint_vel = self._as_vector(env_data.dof_vel, 29, "dof_vel")

        anchor_pos_w = self.ref_body_pos[frame, self.anchor_body_index].astype(
            np.float64
        )
        anchor_quat_xyzw = self.ref_body_quat[
            frame, self.anchor_body_index
        ][[1, 2, 3, 0]].astype(np.float64)

        anchor_quat_w, anchor_pos_w = self.motion_alignment.align_transform(
            anchor_quat_xyzw,
            anchor_pos_w,
        )

        anchor_pos_body, anchor_quat_body = subtract_frame_transforms(
            robot_pos,
            robot_quat_xyzw,
            anchor_pos_w,
            anchor_quat_w,
        )

        relative_rotation = matrix_from_quat(anchor_quat_body)

        # IMPORTANTE:
        # El script oficial de g1-moves usa:
        #     rot_matrix[:, :2].T.flatten()
        # No usar mat[:, :2].flatten(), porque cambia el orden de los 6 valores.
        anchor_ori_body = relative_rotation[:, :2].T.reshape(-1).astype(np.float32)

        obs = np.concatenate(
            [
                ref_joint_pos,                         # 29
                ref_joint_vel,                         # 29
                anchor_pos_body.astype(np.float32),    # 3
                anchor_ori_body,                       # 6
                base_ang_vel,                          # 3
                base_lin_vel,                          # 3
                joint_pos - self.default_dof_pos,      # 29
                joint_vel,                             # 29
                self.last_action.astype(np.float32),   # 29
            ]
        ).astype(np.float32)

        if obs.shape != (160,):
            raise RuntimeError(f"Observación inválida: {obs.shape}, se esperaba (160,)")

        if not np.isfinite(obs).all():
            raise RuntimeError("La observación contiene NaN o valores infinitos")

        callbacks: list[str] = []
        if self.flag_motion_done and not self._motion_done_emitted:
            callbacks = ["[MOTION_DONE]"]
            self._motion_done_emitted = True

        extras = {
            "motion_frame": frame,
            "motion_time": self.motion_time,
            "anchor_pos_w": anchor_pos_w,
            "anchor_quat_w": anchor_quat_w,
            "robot_anchor_pos_w": robot_pos,
            "robot_anchor_quat_w": robot_quat_xyzw,
            "CALLBACK": callbacks,
        }
        return obs, extras

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        ort_inputs = {
            self.input_name: np.expand_dims(obs, axis=0).astype(np.float32)
        }

        output = self.session.run([self.output_name], ort_inputs)[0]
        actions = np.asarray(output, dtype=np.float32).reshape(-1)

        if actions.shape != (29,):
            raise RuntimeError(
                f"Salida de la policy inválida: {actions.shape}, se esperaba (29,)"
            )

        if not np.isfinite(actions).all():
            raise RuntimeError("La policy produjo NaN o valores infinitos")

        if self.clip_to_joint_limits and self.position_limits is not None:
            clipped = np.clip(
                actions,
                self.position_limits[:, 0],
                self.position_limits[:, 1],
            )
            max_clip = float(np.max(np.abs(clipped - actions)))
            if max_clip > 1e-4:
                LOGGER.warning(
                    "Objetivos ONNX recortados a límites articulares; "
                    "máxima corrección: %.6f rad",
                    max_clip,
                )
            actions = clipped.astype(np.float32)

        # La salida ya es un objetivo articular. No se aplica action_scale.
        self.last_action = actions.copy()
        return actions

    def post_step_callback(self, commands: list[str] | None = None) -> None:
        reset_requested = False

        for command in commands or []:
            match command:
                case "[MOTION_RESET]":
                    reset_requested = True
                case "[MOTION_FADE_IN]":
                    self.play_speed = float(self.motion_speed_default)
                case "[MOTION_FADE_OUT]":
                    self.play_speed = 0.0

        if reset_requested:
            self.reset()
            return

        if self.play_speed > 0.0 and not self.flag_motion_done:
            self.motion_time += self.dt * self.play_speed

    def get_init_dof_pos(self) -> np.ndarray:
        """Primer frame de la referencia, usado por prepare()/blend-in."""
        return self.ref_joint_pos[0].copy()

    def debug_viz(self, visualizer, env_data, ctrl_data, extras) -> None:
        # Se deja vacío para no introducir carga adicional en el robot real.
        return


def resolve_clip_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.clip_dir:
        clip_dir = Path(args.clip_dir).expanduser().resolve()
        clip_name = clip_dir.name
        onnx_path = clip_dir / "policy" / f"{clip_name}_policy.onnx"
        npz_path = clip_dir / "training" / f"{clip_name}.npz"
    elif args.onnx and args.npz:
        onnx_path = Path(args.onnx).expanduser().resolve()
        npz_path = Path(args.npz).expanduser().resolve()
    else:
        raise ValueError(
            "Proporciona --clip-dir o, alternativamente, --onnx y --npz."
        )

    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX no encontrado: {onnx_path}")
    if not npz_path.is_file():
        raise FileNotFoundError(f"NPZ no encontrado: {npz_path}")

    return onnx_path, npz_path


def make_pipeline_cfg(
    args: argparse.Namespace,
    onnx_path: Path,
    npz_path: Path,
) -> RlPipelineCfg:
    policy_cfg = G1MovesPolicyCfg(
        onnx_path=str(onnx_path),
        motion_path=str(npz_path),
        loop_motion=args.loop,
        motion_speed=args.speed,
        anchor_body_index=args.anchor_body_index,
        clip_to_joint_limits=not args.no_joint_limit_clip,
        require_state_estimator=not args.allow_missing_odometry,
    )

    if args.real:
        env_cfg = G1RealEnvCfg(
            env_type="UnitreeCppEnv",
            unitree=G1UnitreeCfg(
                net_if=args.net_if,
                enable_odometry=True,
            ),
            odometry_type="UNITREE",
            born_place_align=True,
            update_with_fk=False,
        )
        ctrl_cfg = [UnitreeCtrlCfg()]
    else:
        env_cfg = G1MujocoEnvCfg(
            born_place_align=True,
            update_with_fk=False,
        )
        ctrl_cfg = [
            KeyboardCtrlCfg(
                triggers={
                    "r": "[MOTION_RESET]",
                    "i": "[SIM_REBORN]",
                    "o": "[SHUTDOWN]",
                }
            )
        ]

    return RlPipelineCfg(
        robot="g1",
        env=env_cfg,
        ctrl=ctrl_cfg,
        policy=policy_cfg,
        device=args.device,
        run_fullspeed=False,
        do_safety_check=args.real,
    )


def initialize_mujoco_from_motion(pipeline: RlPipeline) -> None:
    """
    Inicializa el root y las articulaciones con el primer frame del NPZ,
    igual que run_policy.py de g1-moves.
    """
    policy = pipeline.policy.policy
    env = pipeline.env

    if not isinstance(policy, G1MovesPolicy):
        raise TypeError("La policy activa no es G1MovesPolicy")

    if not hasattr(env, "data"):
        return

    anchor = policy.anchor_body_index
    root_pos = policy.ref_body_pos[0, anchor]
    root_quat_wxyz = policy.ref_body_quat[0, anchor]
    joint_pos = policy.ref_joint_pos[0]

    env.data.qpos[:3] = root_pos
    env.data.qpos[3:7] = root_quat_wxyz
    env.data.qpos[-29:] = joint_pos
    env.data.qvel[:] = 0.0
    env.data.ctrl[:] = 0.0

    import mujoco

    mujoco.mj_forward(env.model, env.data)
    pipeline.reset()

    LOGGER.info("MuJoCo inicializado con el primer frame de la trayectoria")


def run_loop(pipeline: RlPipeline, *, real: bool) -> None:
    stop_requested = False

    def request_stop(signum=None, frame=None):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        if real:
            pipeline.prepare()

        while not stop_requested:
            start = time.perf_counter()
            pipeline.step()
            elapsed = time.perf_counter() - start

            if pipeline.cfg.run_fullspeed:
                continue

            remaining = pipeline.dt - elapsed
            if remaining > 0:
                time.sleep(remaining)
                continue

            if not real:
                continue

            # La versión modificada usada por g1-moves realiza un blend-in
            # bloqueante. Al terminar puede marcar esta bandera; ese tiempo no
            # debe ser interpretado como un bloqueo real del control.
            if getattr(pipeline, "_blend_in_completed", False):
                setattr(pipeline, "_blend_in_completed", False)
                LOGGER.info(
                    "Blend-in terminado; se reinicia la referencia temporal "
                    "del bucle de control"
                )
                continue

            LOGGER.error(
                "Frame drop: %.6f s; pipeline.step tardó %.6f s",
                remaining,
                elapsed,
            )

            if remaining < -0.2:
                LOGGER.critical(
                    "Salida de seguridad por retraso excesivo del ciclo"
                )
                break

    finally:
        if real:
            try:
                pipeline.env.shutdown()
            except Exception:
                LOGGER.exception("Error durante el apagado del entorno Unitree")
        else:
            try:
                pipeline.env.shutdown()
            except Exception:
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ejecutar una policy G1 Moves mediante RoboJuDo"
    )

    source = parser.add_argument_group("archivos")
    source.add_argument(
        "--clip-dir",
        help=(
            "Directorio del clip, por ejemplo "
            "~/Documents/Code_G1/g1_moves_data/dance/B_DadDance"
        ),
    )
    source.add_argument("--onnx", help="Ruta explícita a <clip>_policy.onnx")
    source.add_argument("--npz", help="Ruta explícita a <clip>.npz")

    runtime = parser.add_argument_group("ejecución")
    runtime.add_argument(
        "--real",
        action="store_true",
        help="Desplegar en el G1 real; sin esta opción usa MuJoCo",
    )
    runtime.add_argument(
        "--net-if",
        default="enp7s0",
        help="Interfaz de red de UnitreeCppEnv (predeterminado: eth0)",
    )
    runtime.add_argument(
        "--device",
        choices=("cpu", "cuda", "tensorrt"),
        default="cpu",
        help="Provider preferido de ONNX Runtime",
    )
    runtime.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Multiplicador temporal de la trayectoria",
    )
    runtime.add_argument(
        "--loop",
        action="store_true",
        help="Repetir la trayectoria al llegar al último frame",
    )
    runtime.add_argument(
        "--anchor-body-index",
        type=int,
        default=0,
        help="Índice del pelvis/root en body_pos_w y body_quat_w",
    )
    runtime.add_argument(
        "--no-joint-limit-clip",
        action="store_true",
        help="No recortar los objetivos ONNX a los límites del G1",
    )
    runtime.add_argument(
        "--allow-missing-odometry",
        action="store_true",
        help=(
            "Usar ceros si base_pos/base_lin_vel no están disponibles. "
            "No recomendado para sim2real."
        ),
    )

    args = parser.parse_args()

    if args.speed <= 0:
        parser.error("--speed debe ser mayor que cero")

    if args.clip_dir and (args.onnx or args.npz):
        parser.error("Usa --clip-dir o --onnx/--npz, no ambos")

    if bool(args.onnx) != bool(args.npz):
        parser.error("--onnx y --npz deben proporcionarse juntos")

    return args


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
        ),
        datefmt="%m-%d %H:%M:%S",
    )

    args = parse_args()

    try:
        onnx_path, npz_path = resolve_clip_paths(args)
        LOGGER.info("ONNX: %s", onnx_path)
        LOGGER.info("NPZ:  %s", npz_path)

        cfg = make_pipeline_cfg(args, onnx_path, npz_path)
        pipeline = RlPipeline(cfg=cfg)

        if not args.real:
            initialize_mujoco_from_motion(pipeline)

        run_loop(pipeline, real=args.real)
        return 0

    except KeyboardInterrupt:
        return 130
    except Exception:
        LOGGER.exception("No se pudo ejecutar G1 Moves con RoboJuDo")
        return 1


if __name__ == "__main__":
    sys.exit(main())
