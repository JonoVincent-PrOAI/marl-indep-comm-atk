
import gymnasium as gym
#import ma_gym
from common.arguments import common_args, config_args
from runner import Runner
from smaclite.env import smaclite
import smaclite
import numpy as np
import torch
import random
import json
import sys
import os 

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import ma_gym

N_EXPERIMENTS = 1
SMAC_MAPS =  ["2c_vs_64zg", "2s_vs_1sc", "2s3z", "3s_vs_5z", "3s5z_vs3s6z",
				 "3s5z", "10m_vs_11m", "27m_vs_30m", "bane_vs_bane", "corridor", 
				 "MMM", "MMM2"]
HEALING = ['MEDIVAC']

if __name__ == '__main__':
	args = common_args()

	seed = random.randrange(0, 2**32 - 1)
	print("Using seed: ", seed)
	torch.random.manual_seed(seed)
	np.random.seed(seed)
	
	if args.env in SMAC_MAPS:

		env = gym.make(f"smaclite/{args.env}-v0", seed = seed).unwrapped
		map_info = json.load(open(f"../smaclite_maps/{args.env}.json"))

		n_enemies = map_info['num_enemy_units']
		n_agents = map_info['num_allied_units']
		num_healers = sum(
            sum(
                group['units'][t] 
                for t in group['units'].keys()
                if t in HEALING
            )
            for group in map_info['groups']
            if group['faction'] == "ALLY"
        )

		n_actions = 6 + (
            max(n_agents - num_healers, n_enemies)
            if num_healers
            else n_enemies
        )

		ally_state_feat_size = (
					4 + map_info['ally_has_shields'] + map_info['num_unit_types']
				)
		enemy_state_feat_size = (
            3 + map_info['enemy_has_shields'] + map_info['num_unit_types']
        )
		state_size = sum(
            (
                n_agents * ally_state_feat_size,
                n_enemies * enemy_state_feat_size,
                # previous actions
                n_agents * n_actions,
            )
        )
		
		enemy_feat_size = (
            5 + map_info['enemy_has_shields'] + map_info['num_unit_types']
        )
        # aly visible, distance, x, y, health, shield, unit type
		ally_feat_size = (
            5 + map_info['ally_has_shields'] + map_info['num_unit_types']
        )
		obs_size = sum(
            (
                # whether movement in the 4 directions is possible
                4,
                (n_enemies) * enemy_feat_size,
                (n_agents - 1) * ally_feat_size,
                # own health, shield, unit type
                1 + map_info['ally_has_shields'] + map_info['num_unit_types'],
            )
        )

		args.n_agents = n_agents
		args.n_actions = n_actions
		args.state_shape = state_size
		args.obs_shape = obs_size
		args.episode_limit = 150


	elif args.env in ["PredatorPrey"]:
			try:
				gym.register(
					id='PredatorPrey-v0',
					entry_point='ma_gym.envs.predator_prey:PredatorPrey',
					max_episode_steps=100,
				)
			except Exception:
				pass
			# Use .unwrapped to access custom attributes like _max_steps and _penalty
			env = gym.make(
				'ma_gym:PredatorPrey-v0', 
				grid_shape=(51, 51), 
				n_agents=4, 
				n_preys=10, 
				penalty=0.0,
				step_cost= -0.05,
				disable_env_checker=True,
				render_mode = args.render_mode
			).unwrapped			
			args.n_actions = env.action_space[0].n
			args.n_agents = env.n_agents
			
			# Safely compute observation and state shapes dynamically from the environment
			args.obs_shape = env.observation_space[0].shape[0]
			args.state_shape = args.obs_shape * args.n_agents  # or customize based on your runner's global state needs
			
			args.episode_limit = env._max_steps
			print("Env PP with penalty ", env._penalty)
	else:
		raise Exception('Invalid environment: environment not supported!')
		

	print("Environment {} initialized, for {} time steps and evaluating every {} time steps".format(args.env, \
																							args.n_steps, args.evaluate_cycle))

	# load args
	if args.alg == "idql":
		args = config_args(args)
	else:
		raise Exception('No such algorithm!')


	print("CUDA set to", args.cuda)
	print("Communication set to", args.with_comm)
	print("With args:\n", args)

	runner = Runner(env, args)

	# parameterize run according to the number of independent experiments to run, i.e., independent sets of n_epochs over the model; default is 1
	if args.learn:
		runner.run(N_EXPERIMENTS)
