import time
import numpy as np
import torch
import gymnasium as gym
import json
# Alternatively, you can import the native environment class directly:
# from smaclite import SMACliteEnv

# Import your agent configuration classes from your repository framework
from agent.agent import Agents
from common.arguments import common_args, config_args
from attack.jsma import JSMA
HEALING = ['MEDIVAC']
SMAC_MAPS =  ["2c_vs_64zg", "2s_vs_1sc", "2s3z", "3s_vs_5z", "3s5z_vs3s6z",
				 "3s5z", "10m_vs_11m", "27m_vs_30m", "bane_vs_bane", "corridor", 
				 "MMM", "MMM2"]
args = common_args()
print(f"Initializing SMAClite environment for scenario: {args.env}")

# 1. Initialize environment with Gymnasium visual rendering mode
# Option A: Using the default Gymnasium string registry system
if args.env in SMAC_MAPS:
    env = gym.make(f"smaclite:smaclite/{args.env}-v0", render_mode="human", target_agent = args.target_agent).unwrapped

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

elif args.env == 'PredatorPrey':
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
        disable_env_checker=True
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

# 2. Force deterministic action selection (Disable all random exploration)
args.epsilon = 0.05
args.min_epsilon = 0.0

# 3. Instantiate your policy network workers
args = config_args(args)
agents = Agents(args)

if args.target_agent != None:
    jsma = JSMA(agents)

# 5. Visual playback loop
try:
    for episode in range(50):
        if args.env == 'PredatorPrey':
            time.sleep(30/60)
        obs, info = env.reset()
        terminated = False
        truncated = False
        episode_reward = None
        step = 0
        
        # Flush multi-agent RNN hidden tracking vectors if using deep recurrent layers
        if hasattr(agents.policy, 'init_hidden'):
            agents.policy.init_hidden(1)
            
        last_action = np.zeros((args.n_agents, args.n_actions))
        print(f"\n--- Displaying Episode {episode + 1} ---")
        step = 0
        while (not (terminated or truncated)) and step < args.n_steps:
            # Renders the updated unit coordinates to the Pygame UI canvas
            env.render()
            
            # Slow down frame calculations so you can track agent micro-positioning
            time.sleep(0.05) 
            
            # Fetch fresh observation vectors
            current_obs = env.get_agent_obs()
            avail_actions = env.get_avail_actions()
            actions = []
            
            # Independent communication tracking layer support
            all_msgs = []
            if getattr(args, 'with_comm', False):
                all_msgs = agents.get_all_messages(np.array(current_obs), last_action)

            # Map individual policy actions
            for agent_id in range(args.n_agents):

                avail_act = [1] * args.n_actions

                if agent_id == args.target_agent:
                    input_dict = {'current_obs': current_obs[agent_id], 
                                    'last_action': last_action[agent_id], 
                                    'agent_id': agent_id, 
                                    'avail_actions': avail_act, 
                                    'epsilon': 0.0, 
                                    'evaluate': True,
                                    'msg_all': all_msgs
                                    }
                    atk_msgs = jsma.attack(input_dict, target_action=0)

                    action = agents.choose_action(
                        current_obs[agent_id], 
                        last_action[agent_id], 
                        agent_id, 
                        avail_act, 
                        epsilon=0.0, 
                        evaluate=True,
                        msg_all=atk_msgs
                    )                

                else:
                    action = agents.choose_action(
                        obs = current_obs[agent_id], 
                        last_action = last_action[agent_id], 
                        agent_num = agent_id, 
                        avail_actions = avail_act, 
                        epsilon = 0.05, 
                        evaluate = True,
                        msg_all = all_msgs
                    )
                if action == 1:
                    print(agent_id)
                action_onehot = np.zeros(args.n_actions)
                action_onehot[action] = 1
                last_action[agent_id] = action_onehot
                actions.append(int(action))
            # Step the game logic forward
            next_obs, reward, terminated, truncated, info = env.step(actions)

            if step <= 0:
                episode_reward = reward
            else:
                episode_reward += reward
            step += 1
            
        print(f"Episode {episode + 1} Complete | Total Reward: {episode_reward} | Duration: {step} steps")
        
finally:
    env.close()
    print("Pygame renderer terminated cleanly.")