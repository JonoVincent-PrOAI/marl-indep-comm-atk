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
HEALING = ['MEDIVAC']

args = common_args()
print(f"Initializing SMAClite environment for scenario: {args.env}")

# 1. Initialize environment with Gymnasium visual rendering mode
# Option A: Using the default Gymnasium string registry system
env = gym.make(f"smaclite:smaclite/{args.env}-v0", render_mode="human").unwrapped

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

# Option B: Using direct instantiation if your code bypasses gym.make:
# env = SMACliteEnv(map_name=args.map_name, render_mode="human")

# 2. Force deterministic action selection (Disable all random exploration)
args.epsilon = 0.0
args.min_epsilon = 0.0

# 3. Instantiate your policy network workers
args = config_args(args)
agents = Agents(args)

# 5. Visual playback loop
try:
    for episode in range(50):  # Watch 5 consecutive showcase runs
        obs, info = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0
        step = 0
        
        # Flush multi-agent RNN hidden tracking vectors if using deep recurrent layers
        if hasattr(agents.policy, 'init_hidden'):
            agents.policy.init_hidden(1)
            
        last_action = np.zeros((args.n_agents, args.n_actions))
        print(f"\n--- Displaying Episode {episode + 1} ---")
        step = 0
        while (not (terminated or truncated)) and step < args.n_steps:
            step+=1
            # Renders the updated unit coordinates to the Pygame UI canvas
            env.render()
            
            # Slow down frame calculations so you can track agent micro-positioning
            time.sleep(0.05) 
            
            # Fetch fresh observation vectors
            current_obs = env.get_obs()
            avail_actions = env.get_avail_actions()
            actions = []
            
            # Independent communication tracking layer support
            all_msgs = []
            if getattr(args, 'with_comm', False):
                all_msgs = agents.get_all_messages(np.array(current_obs), last_action)
            
            # Map individual policy actions
            for agent_id in range(args.n_agents):
                avail_act = avail_actions[agent_id]
                action = agents.choose_action(
                    current_obs[agent_id], 
                    last_action[agent_id], 
                    agent_id, 
                    avail_act, 
                    epsilon=0.0, 
                    evaluate=True,
                    msg_all=all_msgs
                )
                
                action_onehot = np.zeros(args.n_actions)
                action_onehot[action] = 1
                last_action[agent_id] = action_onehot
                actions.append(int(action))
            
            # Step the game logic forward
            next_obs, reward, terminated, truncated, info = env.step(actions)
            episode_reward += reward
            step += 1
            
        battle_won = info.get('battle_won', False)
        print(f"Episode {episode + 1} Complete | Outcome: {'WIN' if battle_won else 'LOSS'} | Total Reward: {episode_reward:.2f} | Duration: {step} steps")
        
finally:
    env.close()
    print("Pygame renderer terminated cleanly.")