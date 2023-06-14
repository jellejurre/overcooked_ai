import numpy as np

from pathlib import Path
import sys

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

import os
from human_aware_rl.imitation.behavior_cloning_tf2 import load_bc_model, BehaviorCloningPolicy, _get_base_ae
from human_aware_rl.rllib.rllib import load_agent_pair, load_trainer, get_agent_from_trainer, RlLibAgent
from overcooked_ai_py.agents.agent import AgentPair
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
import json


def evaluate_mappo_single(path, layout, checkpoint_path):
    ae = AgentEvaluator.from_layout_name(
        {"layout_name": layout, "old_dynamics": True}, {"horizon": 400}
    )
    ap = load_agent_pair(path, "mappo", "mappo", checkpoint_path)
    result = ae.evaluate_agent_pair(ap, 1, 400)
    return result, result["ep_returns"]


def evaluate_ppo_single(path, layout, checkpoint_path):
    ae = AgentEvaluator.from_layout_name(
        {"layout_name": layout, "old_dynamics": True}, {"horizon": 400}
    )
    ap = load_agent_pair(path, "ppo", "ppo", checkpoint_path)
    result = ae.evaluate_agent_pair(ap, 1, 400)
    return result, result["ep_returns"]


def evaluate_mappo_bc_single(path_mappo, path_bc, layout, checkpoint_path):
    ae = AgentEvaluator.from_layout_name(
        {"layout_name": layout, "old_dynamics": True}, {"horizon": 400}
    )
    trainer_mappo = load_trainer(path_mappo, checkpoint_path=checkpoint_path)
    bc_model, bc_params = load_bc_model(path_bc)
    bc_policy = BehaviorCloningPolicy.from_model(
        bc_model, bc_params, stochastic=True
    )
    base_ae = _get_base_ae(bc_params)
    base_env = base_ae.env
    bc_agent = RlLibAgent(bc_policy, 1, base_env.featurize_state_mdp)
    agent_mappo = get_agent_from_trainer(trainer_mappo, policy_id="mappo")
    agent_pair = AgentPair(agent_mappo, bc_agent)
    result = ae.evaluate_agent_pair(agent_pair, 1, 400)
    return result, result["ep_returns"]


def evaluate_ppo_bc_single(path_ppo, path_bc, layout, checkpoint_path):
    ae = AgentEvaluator.from_layout_name(
        {"layout_name": layout, "old_dynamics": True}, {"horizon": 400}
    )
    trainer_ppo = load_trainer(path_ppo, checkpoint_path=checkpoint_path)
    bc_model, bc_params = load_bc_model(path_bc)
    bc_policy = BehaviorCloningPolicy.from_model(
        bc_model, bc_params, stochastic=True
    )
    base_ae = _get_base_ae(bc_params)
    base_env = base_ae.env
    bc_agent = RlLibAgent(bc_policy, 1, base_env.featurize_state_mdp)
    agent_ppo = get_agent_from_trainer(trainer_ppo, policy_id="ppo")
    agent_pair = AgentPair(agent_ppo, bc_agent)
    result = ae.evaluate_agent_pair(agent_pair, 1, 400)
    return result, result["ep_returns"]


def evaluate_bc_single(path_bc, layout):
    ae = AgentEvaluator.from_layout_name(
        {"layout_name": layout, "old_dynamics": True}, {"horizon": 400}
    )
    bc_model, bc_params = load_bc_model(path_bc)
    bc_policy = BehaviorCloningPolicy.from_model(
        bc_model, bc_params, stochastic=True
    )
    base_ae = _get_base_ae(bc_params)
    base_env = base_ae.env
    bc_agent = RlLibAgent(bc_policy, 0, base_env.featurize_state_mdp)
    bc_agent = RlLibAgent(bc_policy, 1, base_env.featurize_state_mdp)

    agent_pair = AgentPair(bc_agent, bc_agent, allow_duplicate_agents=True)

    result = ae.evaluate_agent_pair(agent_pair, 1, 400)
    return result, result["ep_returns"]


def evaluate_mappo_sp(path_mappo, layout):
    l = []
    out = list(filter(lambda x: "checkpoint" in x, os.listdir(path_mappo)))
    for checkpoint_folder in out:
        print(f"{path_mappo}, {checkpoint_folder}, {layout}")
        _, res = evaluate_mappo_single(os.path.join(path_mappo, checkpoint_folder), layout, None)
        # print(f"For checkpoint: {checkpoint_folder}, ", end="")
        # print((np.mean(res), np.std(res) / len(res) ** 0.5))
        l.append((checkpoint_folder, (np.mean(res), np.std(res) / len(res) ** 0.5)))
    return l


def evaluate_ppo_sp(path_ppo, layout):
    l = []
    out = list(filter(lambda x: "checkpoint" in x, os.listdir(path_ppo)))
    for checkpoint_folder in out:
        print(f"{path_ppo}, {checkpoint_folder}, {layout}")
        _, res = evaluate_ppo_single(os.path.join(path_ppo, checkpoint_folder), layout, None)
        # print(f"For checkpoint: {checkpoint_folder}, ", end="")
        # print((np.mean(res), np.std(res) / len(res) ** 0.5))
        l.append((checkpoint_folder, (np.mean(res), np.std(res) / len(res) ** 0.5)))
    return l

def evaluate_mappo_bc(path_mappo, path_bc, layout):
    l = []
    out = list(filter(lambda x: "checkpoint" in x, os.listdir(path_mappo)))
    for checkpoint_folder in out:
        print(f"{path_mappo}, {checkpoint_folder}, {layout}")
        _, res = evaluate_mappo_bc_single(os.path.join(path_mappo, checkpoint_folder), path_bc, layout, None)
        # print(f"For checkpoint: {checkpoint_folder}, ", end="")
        # print((np.mean(res), np.std(res) / len(res) ** 0.5))
        l.append((checkpoint_folder, (np.mean(res), np.std(res) / len(res) ** 0.5)))
    return l

def evaluate_ppo_bc(path_ppo, path_bc, layout):
    l = []
    out = list(filter(lambda x: "checkpoint" in x, os.listdir(path_ppo)))
    for checkpoint_folder in out:
        print(f"{path_ppo}, {checkpoint_folder}, {layout}")
        _, res = evaluate_ppo_bc_single(os.path.join(path_ppo, checkpoint_folder), path_bc, layout, None)
        # print(f"For checkpoint: {checkpoint_folder}, ", end="")
        # print((np.mean(res), np.std(res) / len(res) ** 0.5))
        l.append((checkpoint_folder, (np.mean(res), np.std(res) / len(res) ** 0.5)))
    return l

def evaluate_bc(path_bc, layout):
    _, res = evaluate_bc_single(path_bc, layout)
    return [("0", (np.mean(res), np.std(res) / len(res) ** 0.5))]

if __name__ == '__main__':
    for layout in os.listdir(os.path.join(Path(__file__).parents[0], "models")):
        path_bc = os.path.join(Path(__file__).parents[0], "models", layout, "BC")
        path_ppo = os.path.join(Path(__file__).parents[0], "models", layout, "PPO_SP")
        path_mappo = os.path.join(Path(__file__).parents[0], "models", layout, "MAPPO_SP")
        # print(path_bc)
        # print(path_ppo)
        # print(path_mappo)
        functions = {
            "mappo_sp": lambda bc, ppo, mappo, layout: evaluate_mappo_sp(mappo, layout),
            "ppo_sp": lambda bc, ppo, mappo, layout: evaluate_ppo_sp(ppo, layout),
            "mappo_bc": lambda bc, ppo, mappo, layout: evaluate_mappo_bc(mappo, bc, layout),
            "ppo_bc": lambda bc, ppo, mappo, layout: evaluate_ppo_bc(ppo, bc, layout),
            "bc": lambda bc, ppo, mappo, layout: evaluate_bc(bc, layout)
        }
        for alg in functions.keys():
            if os.path.isfile(os.path.join(Path(__file__).parents[0], "results", f"{layout}_{alg}.txt")):
                continue
            data = functions[alg](path_bc, path_ppo, path_mappo, layout)
            text_file = open(os.path.join(Path(__file__).parents[0], "results", f"{layout}_{alg}.txt"), "w")
            n = text_file.write(json.dumps(data))
            text_file.close()





