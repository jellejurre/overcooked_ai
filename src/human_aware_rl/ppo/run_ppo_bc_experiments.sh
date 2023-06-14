#!/usr/bin/env bash
# This file contains the script to generate the ppo self-play agents for the 5 classic layouts

# This bash script can be run from anywhere

#First we need to create the BC-agents, the agent will be saved in the bc_runs directory under human_aware_rl/imitation/bc_runs directory
python "$(dirname "$0")/../imitation/reproduce_bc.py"

#Now we create the PPO agents trained with BC agents
path="$(dirname "$0")/../imitation/bc_runs/train/coordination_ring"
python ppo_rllib_client.py with  seeds=[0]  layout_name="coordination_ring" clip_param=0.063 gamma=0.977 grad_clip=0.254 kl_coeff=0.176 lmbda=0.6 lr=2.4e-4 num_training_iters=550 old_dynamics=True reward_shaping_horizon=4500000 use_phi=False vf_loss_coeff=0.024 bc_schedule="[(0,0),(300000,0),(3000000,1)]" bc_model_dir=$path results_dir=reproduced_results/exp_results/mappo_bc_coordination_ring centralized_critic=True
python ppo_rllib_client.py with  seeds=[10] layout_name="coordination_ring" clip_param=0.063 gamma=0.977 grad_clip=0.254 kl_coeff=0.176 lmbda=0.6 lr=2.4e-4 num_training_iters=550 old_dynamics=True reward_shaping_horizon=4500000 use_phi=False vf_loss_coeff=0.024 bc_schedule="[(0,0),(300000,0),(3000000,1)]" bc_model_dir=$path results_dir=reproduced_results/exp_results/mappo_bc_coordination_ring centralized_critic=True
python ppo_rllib_client.py with  seeds=[20] layout_name="coordination_ring" clip_param=0.063 gamma=0.977 grad_clip=0.254 kl_coeff=0.176 lmbda=0.6 lr=2.4e-4 num_training_iters=550 old_dynamics=True reward_shaping_horizon=4500000 use_phi=False vf_loss_coeff=0.024 bc_schedule="[(0,0),(300000,0),(3000000,1)]" bc_model_dir=$path results_dir=reproduced_results/exp_results/mappo_bc_coordination_ring centralized_critic=True
python ppo_rllib_client.py with  seeds=[30] layout_name="coordination_ring" clip_param=0.063 gamma=0.977 grad_clip=0.254 kl_coeff=0.176 lmbda=0.6 lr=2.4e-4 num_training_iters=550 old_dynamics=True reward_shaping_horizon=4500000 use_phi=False vf_loss_coeff=0.024 bc_schedule="[(0,0),(300000,0),(3000000,1)]" bc_model_dir=$path results_dir=reproduced_results/exp_results/mappo_bc_coordination_ring centralized_critic=True
python ppo_rllib_client.py with  seeds=[40] layout_name="coordination_ring" clip_param=0.063 gamma=0.977 grad_clip=0.254 kl_coeff=0.176 lmbda=0.6 lr=2.4e-4 num_training_iters=550 old_dynamics=True reward_shaping_horizon=4500000 use_phi=False vf_loss_coeff=0.024 bc_schedule="[(0,0),(300000,0),(3000000,1)]" bc_model_dir=$path results_dir=reproduced_results/exp_results/mappo_bc_coordination_ring centralized_critic=True

path="$(dirname "$0")/../imitation/bc_runs/train/random0"
python ppo_rllib_client.py with  seeds=[0]  layout_name="forced_coordination" clip_param=0.0608 gamma=0.9738 grad_clip=0.3022 kl_coeff=0.2527 lmbda=0.8 lr=2.5e-4 num_training_iters=550 old_dynamics=True reward_shaping_horizon=2500000 use_phi=True vf_loss_coeff=0.009 bc_schedule="[(0,0),(400000,0),(1000000,1)]" bc_model_dir=$path results_dir=reproduced_results/exp_results/mappo_bc_forced_coordination centralized_critic=True
python ppo_rllib_client.py with  seeds=[10] layout_name="forced_coordination" clip_param=0.0608 gamma=0.9738 grad_clip=0.3022 kl_coeff=0.2527 lmbda=0.8 lr=2.5e-4 num_training_iters=550 old_dynamics=True reward_shaping_horizon=2500000 use_phi=True vf_loss_coeff=0.009 bc_schedule="[(0,0),(400000,0),(1000000,1)]" bc_model_dir=$path results_dir=reproduced_results/exp_results/mappo_bc_forced_coordination centralized_critic=True
python ppo_rllib_client.py with  seeds=[20] layout_name="forced_coordination" clip_param=0.0608 gamma=0.9738 grad_clip=0.3022 kl_coeff=0.2527 lmbda=0.8 lr=2.5e-4 num_training_iters=550 old_dynamics=True reward_shaping_horizon=2500000 use_phi=True vf_loss_coeff=0.009 bc_schedule="[(0,0),(400000,0),(1000000,1)]" bc_model_dir=$path results_dir=reproduced_results/exp_results/mappo_bc_forced_coordination centralized_critic=True
python ppo_rllib_client.py with  seeds=[30] layout_name="forced_coordination" clip_param=0.0608 gamma=0.9738 grad_clip=0.3022 kl_coeff=0.2527 lmbda=0.8 lr=2.5e-4 num_training_iters=550 old_dynamics=True reward_shaping_horizon=2500000 use_phi=True vf_loss_coeff=0.009 bc_schedule="[(0,0),(400000,0),(1000000,1)]" bc_model_dir=$path results_dir=reproduced_results/exp_results/mappo_bc_forced_coordination centralized_critic=True
python ppo_rllib_client.py with  seeds=[40] layout_name="forced_coordination" clip_param=0.0608 gamma=0.9738 grad_clip=0.3022 kl_coeff=0.2527 lmbda=0.8 lr=2.5e-4 num_training_iters=550 old_dynamics=True reward_shaping_horizon=2500000 use_phi=True vf_loss_coeff=0.009 bc_schedule="[(0,0),(400000,0),(1000000,1)]" bc_model_dir=$path results_dir=reproduced_results/exp_results/mappo_bc_forced_coordination centralized_critic=True

path="$(dirname "$0")/../imitation/bc_runs/train/cramped_room"
python ppo_rllib_client.py with  seeds=[0]  layout_name="cramped_room" clip_param=0.1543 gamma=0.9777 grad_clip=0.2884 kl_coeff=0.2408 lmbda=0.6 lr=2.69e-4 num_training_iters=500 old_dynamics=True reward_shaping_horizon=4500000 use_phi=False vf_loss_coeff=0.0069 bc_schedule="[(0,0),(200000,0),(6000000,1)]" bc_model_dir=$path results_dir=reproduced_results/exp_results/mappo_bc_cramped_room centralized_critic=True
python ppo_rllib_client.py with  seeds=[10] layout_name="cramped_room" clip_param=0.1543 gamma=0.9777 grad_clip=0.2884 kl_coeff=0.2408 lmbda=0.6 lr=2.69e-4 num_training_iters=500 old_dynamics=True reward_shaping_horizon=4500000 use_phi=False vf_loss_coeff=0.0069 bc_schedule="[(0,0),(200000,0),(6000000,1)]" bc_model_dir=$path results_dir=reproduced_results/exp_results/mappo_bc_cramped_room centralized_critic=True
python ppo_rllib_client.py with  seeds=[20] layout_name="cramped_room" clip_param=0.1543 gamma=0.9777 grad_clip=0.2884 kl_coeff=0.2408 lmbda=0.6 lr=2.69e-4 num_training_iters=500 old_dynamics=True reward_shaping_horizon=4500000 use_phi=False vf_loss_coeff=0.0069 bc_schedule="[(0,0),(200000,0),(6000000,1)]" bc_model_dir=$path results_dir=reproduced_results/exp_results/mappo_bc_cramped_room centralized_critic=True
python ppo_rllib_client.py with  seeds=[30] layout_name="cramped_room" clip_param=0.1543 gamma=0.9777 grad_clip=0.2884 kl_coeff=0.2408 lmbda=0.6 lr=2.69e-4 num_training_iters=500 old_dynamics=True reward_shaping_horizon=4500000 use_phi=False vf_loss_coeff=0.0069 bc_schedule="[(0,0),(200000,0),(6000000,1)]" bc_model_dir=$path results_dir=reproduced_results/exp_results/mappo_bc_cramped_room centralized_critic=True
python ppo_rllib_client.py with  seeds=[40] layout_name="cramped_room" clip_param=0.1543 gamma=0.9777 grad_clip=0.2884 kl_coeff=0.2408 lmbda=0.6 lr=2.69e-4 num_training_iters=500 old_dynamics=True reward_shaping_horizon=4500000 use_phi=False vf_loss_coeff=0.0069 bc_schedule="[(0,0),(200000,0),(6000000,1)]" bc_model_dir=$path results_dir=reproduced_results/exp_results/mappo_bc_cramped_room centralized_critic=True

path="$(dirname "$0")/../imitation/bc_runs/train/random3"
python ppo_rllib_client.py with  seeds=[0]  layout_name="counter_circuit_o_1order" clip_param=0.1235 gamma=0.98 grad_clip=0.2736 kl_coeff=0.2511 lmbda=0.5 lr=2.57e-4 num_training_iters=550 old_dynamics=True reward_shaping_horizon=4500000 use_phi=False vf_loss_coeff=0.0206 bc_schedule="[(0,0),(200000,0),(3000000,1)]" bc_model_dir=$path results_dir=reproduced_results/exp_results/mappo_bc_counter centralized_critic=True
python ppo_rllib_client.py with  seeds=[10] layout_name="counter_circuit_o_1order" clip_param=0.1235 gamma=0.98 grad_clip=0.2736 kl_coeff=0.2511 lmbda=0.5 lr=2.57e-4 num_training_iters=550 old_dynamics=True reward_shaping_horizon=4500000 use_phi=False vf_loss_coeff=0.0206 bc_schedule="[(0,0),(200000,0),(3000000,1)]" bc_model_dir=$path results_dir=reproduced_results/exp_results/mappo_bc_counter centralized_critic=True
python ppo_rllib_client.py with  seeds=[20] layout_name="counter_circuit_o_1order" clip_param=0.1235 gamma=0.98 grad_clip=0.2736 kl_coeff=0.2511 lmbda=0.5 lr=2.57e-4 num_training_iters=550 old_dynamics=True reward_shaping_horizon=4500000 use_phi=False vf_loss_coeff=0.0206 bc_schedule="[(0,0),(200000,0),(3000000,1)]" bc_model_dir=$path results_dir=reproduced_results/exp_results/mappo_bc_counter centralized_critic=True
python ppo_rllib_client.py with  seeds=[30] layout_name="counter_circuit_o_1order" clip_param=0.1235 gamma=0.98 grad_clip=0.2736 kl_coeff=0.2511 lmbda=0.5 lr=2.57e-4 num_training_iters=550 old_dynamics=True reward_shaping_horizon=4500000 use_phi=False vf_loss_coeff=0.0206 bc_schedule="[(0,0),(200000,0),(3000000,1)]" bc_model_dir=$path results_dir=reproduced_results/exp_results/mappo_bc_counter centralized_critic=True
python ppo_rllib_client.py with  seeds=[40] layout_name="counter_circuit_o_1order" clip_param=0.1235 gamma=0.98 grad_clip=0.2736 kl_coeff=0.2511 lmbda=0.5 lr=2.57e-4 num_training_iters=550 old_dynamics=True reward_shaping_horizon=4500000 use_phi=False vf_loss_coeff=0.0206 bc_schedule="[(0,0),(200000,0),(3000000,1)]" bc_model_dir=$path results_dir=reproduced_results/exp_results/mappo_bc_counter centralized_critic=True

path="$(dirname "$0")/../imitation/bc_runs/train/asymmetric_advantages"
python ppo_rllib_client.py with  seeds=[0]  layout_name="asymmetric_advantages" clip_param=0.1245 gamma=0.966 grad_clip=0.2469 kl_coeff=0.2355 lmbda=0.5 lr=2.07e-4 num_training_iters=600 old_dynamics=True reward_shaping_horizon=5000000 use_phi=True vf_loss_coeff=0.0158 bc_schedule="[(0,0),(400000,0),(1000000,1)]" bc_model_dir=$path results_dir=reproduced_results/exp_results/mappo_bc_asymmetric_advantages centralized_critic=True
python ppo_rllib_client.py with  seeds=[10] layout_name="asymmetric_advantages" clip_param=0.1245 gamma=0.966 grad_clip=0.2469 kl_coeff=0.2355 lmbda=0.5 lr=2.07e-4 num_training_iters=600 old_dynamics=True reward_shaping_horizon=5000000 use_phi=True vf_loss_coeff=0.0158 bc_schedule="[(0,0),(400000,0),(1000000,1)]" bc_model_dir=$path results_dir=reproduced_results/exp_results/mappo_bc_asymmetric_advantages centralized_critic=True
python ppo_rllib_client.py with  seeds=[20] layout_name="asymmetric_advantages" clip_param=0.1245 gamma=0.966 grad_clip=0.2469 kl_coeff=0.2355 lmbda=0.5 lr=2.07e-4 num_training_iters=600 old_dynamics=True reward_shaping_horizon=5000000 use_phi=True vf_loss_coeff=0.0158 bc_schedule="[(0,0),(400000,0),(1000000,1)]" bc_model_dir=$path results_dir=reproduced_results/exp_results/mappo_bc_asymmetric_advantages centralized_critic=True
python ppo_rllib_client.py with  seeds=[30] layout_name="asymmetric_advantages" clip_param=0.1245 gamma=0.966 grad_clip=0.2469 kl_coeff=0.2355 lmbda=0.5 lr=2.07e-4 num_training_iters=600 old_dynamics=True reward_shaping_horizon=5000000 use_phi=True vf_loss_coeff=0.0158 bc_schedule="[(0,0),(400000,0),(1000000,1)]" bc_model_dir=$path results_dir=reproduced_results/exp_results/mappo_bc_asymmetric_advantages centralized_critic=True
python ppo_rllib_client.py with  seeds=[40] layout_name="asymmetric_advantages" clip_param=0.1245 gamma=0.966 grad_clip=0.2469 kl_coeff=0.2355 lmbda=0.5 lr=2.07e-4 num_training_iters=600 old_dynamics=True reward_shaping_horizon=5000000 use_phi=True vf_loss_coeff=0.0158 bc_schedule="[(0,0),(400000,0),(1000000,1)]" bc_model_dir=$path results_dir=reproduced_results/exp_results/mappo_bc_asymmetric_advantages centralized_critic=True