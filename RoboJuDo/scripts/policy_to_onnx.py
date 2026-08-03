import torch  
from pathlib import Path  
from dataclasses import asdict
import os

import mjlab.tasks # CARGA LOS TASKS PARA IDENTIFICAR UNITREE G1 NO ESTIMATION 
import src.tasks # CARGA LOS TASKS PARA IDENTIFICAR UNITREE G1 NO ESTIMATION 
from mjlab.tasks.registry import (
    list_tasks,
    load_env_cfg,
    load_rl_cfg,
    load_runner_cls,
)

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper, MjlabOnPolicyRunner
from mjlab.tasks.tracking.mdp import MotionCommandCfg


#NAME = "model_154000"
NAME = "B_DadDance"
task_id = "Unitree-G1-Tracking-No-State-Estimation"

#motion_file = "src/assets/motions/g1/salsa_basico_g1.npz"
motion_file = "~/Documents/Code_G1/g1_moves_data/dance/B_DadDance/training/B_DadDance.npz"
checkpoint_path = (
    f"logs/rsl_rl/g1_tracking/"
    f"2026-06-04_18-11-10_salsa_agent_from_141500/"
    f"{NAME}.pt"
)

checkpoint_path = "~/Documents/Code_G1/g1_moves_data/dance/B_DadDance/policy/B_DadDance_policy.pt"

device = "cpu"
output_dir = Path(checkpoint_path).parent


# =========================
# VALIDACIONES
# =========================

all_tasks = list_tasks()

if task_id not in all_tasks:
    print("\nERROR: task_id no encontrado:")
    print(f"  {task_id}")

    print("\nTasks disponibles:")
    for t in all_tasks:
        print(f"  - {t}")

    raise SystemExit("\nUsa uno de los nombres exactos de la lista anterior.\n")

checkpoint_path_obj = Path(checkpoint_path)
if not checkpoint_path_obj.exists():
    raise FileNotFoundError(f"No existe el checkpoint: {checkpoint_path}")

motion_path = Path(motion_file).expanduser().resolve()
if not motion_path.exists():
    raise FileNotFoundError(f"No existe el motion_file: {motion_path}")


# =========================
# CARGAR CONFIGS
# =========================

env_cfg = load_env_cfg(task_id)
agent_cfg = load_rl_cfg(task_id)

# Recomendado para exportar: no necesitas 4096 envs
env_cfg.scene.num_envs = 1

# Si es tracking, cargar motion_file igual que en train.py
is_tracking_task = (
    "motion" in env_cfg.commands
    and isinstance(env_cfg.commands["motion"], MotionCommandCfg)
)

if is_tracking_task:
    motion_cmd = env_cfg.commands["motion"]
    motion_cmd.motion_file = str(motion_path)
    print(f"[INFO] Using motion file: {motion_cmd.motion_file}")


# =========================
# CREAR ENV Y RUNNER
# =========================

env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

runner_cls = load_runner_cls(task_id)
if runner_cls is None:
    runner_cls = MjlabOnPolicyRunner

agent_cfg_dict = asdict(agent_cfg)

runner = runner_cls(
    env,
    agent_cfg_dict,
    str(output_dir),
    device,
)


# =========================
# CARGAR CHECKPOINT
# =========================

print(f"[INFO] Loading checkpoint: {checkpoint_path}")
runner.load(str(checkpoint_path_obj))


# =========================
# EXPORTAR ONNX
# =========================

policy_path = str(output_dir) + "/"

if hasattr(runner, "export_motion_policy_to_onnx"):
    motion_onnx_name = f"{NAME}_motion.onnx"
    normal_onnx_name = f"policy_{NAME}.onnx"

    print(f"[INFO] Exporting motion policy: {motion_onnx_name}")
    runner.export_motion_policy_to_onnx(policy_path, motion_onnx_name)

    print(f"[INFO] Exporting policy: {normal_onnx_name}")
    runner.export_policy_to_onnx(policy_path, normal_onnx_name)

else:
    normal_onnx_name = f"policy_{NAME}.onnx"

    print(f"[INFO] Exporting policy: {normal_onnx_name}")
    runner.export_policy_to_onnx(policy_path, normal_onnx_name)


env.close()

print("\nExportación completada.")
print(f"Carpeta de salida: {output_dir}")