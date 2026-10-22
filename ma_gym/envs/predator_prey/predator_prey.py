import copy
import logging

import pygame
import gymnasium as gym
import numpy as np
from PIL import ImageColor
from gymnasium import spaces
from gymnasium.utils import seeding

from ..utils.action_space import MultiAgentActionSpace
from ..utils.draw import draw_grid, fill_cell, draw_circle, write_cell_text
from ..utils.observation_space import MultiAgentObservationSpace

logger = logging.getLogger(__name__)


class PredatorPrey(gym.Env):

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, grid_shape=(5, 5), n_agents=2, n_preys=1, prey_move_probs=(0.175, 0.175, 0.175, 0.175, 0.3),
                 full_observable=False, penalty=0.0, step_cost=0.0, prey_capture_reward=5, max_steps=200,
                 agent_view_mask=(5, 5), render_mode=None):
        assert len(grid_shape) == 2, 'expected a tuple of size 2 for grid_shape, but found {}'.format(grid_shape)
        assert len(agent_view_mask) == 2, 'expected a tuple of size 2 for agent view mask,' \
                                          ' but found {}'.format(agent_view_mask)
        assert grid_shape[0] > 0 and grid_shape[1] > 0, 'grid shape should be > 0'
        assert 0 < agent_view_mask[0] <= grid_shape[0], 'agent view mask has to be within (0,{}]'.format(grid_shape[0])
        assert 0 < agent_view_mask[1] <= grid_shape[1], 'agent view mask has to be within (0,{}]'.format(grid_shape[1])

        self._grid_shape = grid_shape
        self.n_agents = n_agents
        self.n_preys = n_preys
        self._max_steps = max_steps
        self._step_count = None
        self._steps_beyond_done = None
        self._penalty = penalty
        self._step_cost = step_cost
        self._prey_capture_reward = prey_capture_reward
        self._agent_view_mask = agent_view_mask
        self.render_mode = render_mode

        self.action_space = MultiAgentActionSpace([spaces.Discrete(5) for _ in range(self.n_agents)])
        self.agent_pos = {_: None for _ in range(self.n_agents)}
        self.prey_pos = {_: None for _ in range(self.n_preys)}
        self._prey_alive = None
        self.prey_last_move = [np.array([0, 0]) for _ in range(self.n_preys)] #useed to calculate boid movement 

        self._base_grid = self.__create_grid()  # with no agents
        self._full_obs = self.__create_grid()
        self._agent_dones = [False for _ in range(self.n_agents)]
        self._prey_move_probs = prey_move_probs
        self.viewer = None
        self.full_observable = full_observable

        # agent pos (2), prey (25), step (1)
        mask_size = np.prod(self._agent_view_mask)
        self._obs_high = np.array([1., 1.] + [1.] * mask_size + [1.0], dtype=np.float32)
        self._obs_low = np.array([0., 0.] + [0.] * mask_size + [0.0], dtype=np.float32)
        if self.full_observable:
            self._obs_high = np.tile(self._obs_high, self.n_agents)
            self._obs_low = np.tile(self._obs_low, self.n_agents)
        self.observation_space = MultiAgentObservationSpace(
            [spaces.Box(self._obs_low, self._obs_high) for _ in range(self.n_agents)])

        self._total_episode_reward = None
        self.seed()

    def get_action_meanings(self, agent_i=None):
        if agent_i is not None:
            assert agent_i <= self.n_agents
            return [ACTION_MEANING[i] for i in range(self.action_space[agent_i].n)]
        else:
            return [[ACTION_MEANING[i] for i in range(ac.n)] for ac in self.action_space]

    def action_space_sample(self):
        return [agent_action_space.sample() for agent_action_space in self.action_space]

    def __draw_base_img(self):
        self._base_img = draw_grid(self._grid_shape[0], self._grid_shape[1], cell_size=CELL_SIZE, fill='white')

    def __create_grid(self):
        _grid = [[PRE_IDS['empty'] for _ in range(self._grid_shape[1])] for row in range(self._grid_shape[0])]
        return _grid

    def __init_full_obs(self):
        self._full_obs = self.__create_grid()

        for agent_i in range(self.n_agents):
            while True:
                pos = [self.np_random.integers(0, self._grid_shape[0] - 1),
                       self.np_random.integers(0, self._grid_shape[1] - 1)]
                if self._is_cell_vacant(pos):
                    self.agent_pos[agent_i] = pos
                    break
            self.__update_agent_view(agent_i)

        for prey_i in range(self.n_preys):
            while True:
                pos = [self.np_random.integers(0, self._grid_shape[0] - 1),
                       self.np_random.integers(0, self._grid_shape[1] - 1)]
                if self._is_cell_vacant(pos) and (self._neighbour_agents(pos)[0] == 0):
                    self.prey_pos[prey_i] = pos
                    break
            self.__update_prey_view(prey_i)

        self.__draw_base_img()

    def get_agent_obs(self):
        _obs = []
        for agent_i in range(self.n_agents):
            pos = self.agent_pos[agent_i]
            _agent_i_obs = [pos[0] / (self._grid_shape[0] - 1), pos[1] / (self._grid_shape[1] - 1)]  # coordinates

            # check if prey is in the view area
            _prey_pos = np.zeros(self._agent_view_mask)  # prey location in neighbour
            for row in range(max(0, pos[0] - 2), min(pos[0] + 2 + 1, self._grid_shape[0])):
                for col in range(max(0, pos[1] - 2), min(pos[1] + 2 + 1, self._grid_shape[1])):
                    if PRE_IDS['prey'] in self._full_obs[row][col]:
                        _prey_pos[row - (pos[0] - 2), col - (pos[1] - 2)] = 1  # get relative position for the prey loc.

            _agent_i_obs += _prey_pos.flatten().tolist()  # adding prey pos in observable area
            _agent_i_obs += [self._step_count / self._max_steps]  # adding time
            _obs.append(_agent_i_obs)

        if self.full_observable:
            _obs = np.array(_obs).flatten().tolist()
            _obs = [_obs for _ in range(self.n_agents)]
        return _obs

    def reset(self):
        self._total_episode_reward = [0 for _ in range(self.n_agents)]
        self.agent_pos = {}
        self.prey_pos = {}

        self.__init_full_obs()
        self._step_count = 0
        self._steps_beyond_done = None
        self._agent_dones = [False for _ in range(self.n_agents)]
        self._prey_alive = [True for _ in range(self.n_preys)]

        return self.get_agent_obs(), {}

    def __wall_exists(self, pos):
        row, col = pos
        return PRE_IDS['wall'] in self._base_grid[row, col]

    def is_valid(self, pos):
        return (0 <= pos[0] < self._grid_shape[0]) and (0 <= pos[1] < self._grid_shape[1])

    def _is_cell_vacant(self, pos):
        return self.is_valid(pos) and (self._full_obs[pos[0]][pos[1]] == PRE_IDS['empty'])

    def __update_agent_pos(self, agent_i, move):

        curr_pos = copy.copy(self.agent_pos[agent_i])
        next_pos = None
        if move == 0:  # down
            next_pos = [curr_pos[0] + 1, curr_pos[1]]
        elif move == 1:  # left
            next_pos = [curr_pos[0], curr_pos[1] - 1]
        elif move == 2:  # up
            next_pos = [curr_pos[0] - 1, curr_pos[1]]
        elif move == 3:  # right
            next_pos = [curr_pos[0], curr_pos[1] + 1]
        elif move == 4:  # no-op
            pass
        else:
            raise Exception('Action Not found!')

        if next_pos is not None and self._is_cell_vacant(next_pos):
            self.agent_pos[agent_i] = next_pos
            self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty']
            self.__update_agent_view(agent_i)

    def __next_pos(self, curr_pos, move):
        if move == 0:  # down
            next_pos = [curr_pos[0] + 1, curr_pos[1]]
        elif move == 1:  # left
            next_pos = [curr_pos[0], curr_pos[1] - 1]
        elif move == 2:  # up
            next_pos = [curr_pos[0] - 1, curr_pos[1]]
        elif move == 3:  # right
            next_pos = [curr_pos[0], curr_pos[1] + 1]
        elif move == 4:  # no-op
            next_pos = curr_pos
        return next_pos

    def __update_prey_pos(self, prey_i, move):
        curr_pos = copy.copy(self.prey_pos[prey_i])
        if self._prey_alive[prey_i]:
            next_pos = None
            if move == 0:  # down
                next_pos = [curr_pos[0] + 1, curr_pos[1]]
            elif move == 1:  # left
                next_pos = [curr_pos[0], curr_pos[1] - 1]
            elif move == 2:  # up
                next_pos = [curr_pos[0] - 1, curr_pos[1]]
            elif move == 3:  # right
                next_pos = [curr_pos[0], curr_pos[1] + 1]
            elif move == 4:  # no-op
                pass
            else:
                raise Exception('Action Not found!')

            if next_pos is not None and self._is_cell_vacant(next_pos):
                self.prey_pos[prey_i] = next_pos
                self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty']
                self.__update_prey_view(prey_i)
            else:
                # print('pos not updated')
                pass
        else:
            self._full_obs[curr_pos[0]][curr_pos[1]] = PRE_IDS['empty']

    def __get_boid_move(self, prey_i):
        pos = np.array(self.prey_pos[prey_i])
        grid_max_dim = max(self._grid_shape)
        # Move direction mapping: 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT, 4: NO-OP
        move_vectors = {
            0: np.array([1, 0]),   # DOWN
            1: np.array([0, -1]),  # LEFT
            2: np.array([-1, 0]),  # UP
            3: np.array([0, 1]),   # RIGHT
            4: np.array([0, 0]),   # NO-OP
        }
        avail_actions = self.get_avail_prey_actions(prey_id=prey_i)
        
        alive_prey_indices = [
            j for j in range(self.n_preys) 
            if self._prey_alive[j] and j != prey_i
        ]
        
        predator_positions = [np.array(p_pos) for p_pos in self.agent_pos]
        steering_force = np.array([0.0, 0.0])

        # --- RULE 0: CENTER-ATTRACTION BOUNDARY FORCE ---
        grid_height, grid_width = self._grid_shape
        center = np.array([(grid_height - 1) / 2.0, (grid_width - 1) / 2.0])

        # Distance from center along each axis
        dist_from_center = pos - center

        # Activate attraction only outside a central safe zone (e.g., inner 60% of the grid)
        safe_radius_r = grid_height * 0.2
        safe_radius_c = grid_width * 0.2

        center_pull_weight = 1.5

        # Pull back toward center if beyond safe zone
        if abs(dist_from_center[0]) > safe_radius_r:
            steering_force[0] -= center_pull_weight * np.sign(dist_from_center[0]) * (abs(dist_from_center[0]) - safe_radius_r)

        if abs(dist_from_center[1]) > safe_radius_c:
            steering_force[1] -= center_pull_weight * np.sign(dist_from_center[1]) * (abs(dist_from_center[1]) - safe_radius_c)

        # --- RULE 1: PREDATOR AVOIDANCE (Repulsion) ---
        pred_threat_dist = 3.0
        for pred_pos in predator_positions:
            dist = np.linalg.norm(pos - pred_pos)
            if dist < pred_threat_dist:
                repel_dir = (pos - pred_pos) / (dist + 1e-5)
                steering_force += repel_dir * (10.0 / (dist ** 2))

        # --- RULE 2: SEPARATION (Avoid crowding) ---
        separation_dist = 2.0
        for j in alive_prey_indices:
            other_pos = np.array(self.prey_pos[j])
            dist = np.linalg.norm(pos - other_pos)
            if dist < separation_dist:
                repel_dir = (pos - other_pos) / (dist + 1e-5)
                steering_force += repel_dir * (2 / dist)

        # --- RULE 3: COHESION (Steer toward local flock center) ---
        cohesion_dist = grid_max_dim /5.0
        local_prey_indices = [
            j for j in alive_prey_indices 
            if np.linalg.norm(pos - np.array(self.prey_pos[j])) < cohesion_dist
        ]
        
        if len(local_prey_indices) > 0:
            local_positions = [np.array(self.prey_pos[j]) for j in local_prey_indices]
            center_of_mass = np.mean(local_positions, axis=0)
            cohesion_dir = center_of_mass - pos
            if np.linalg.norm(cohesion_dir) > 0:
                steering_force += (cohesion_dir / np.linalg.norm(cohesion_dir)) * 0.8

        # --- RULE 4: ALIGNMENT (Match heading/velocity of local flock) ---
        if len(local_prey_indices) > 0:
            # Compute average velocity vector of local neighbors
            neighbor_velocities = [self.prey_last_move[j] for j in local_prey_indices]
            avg_velocity = np.mean(neighbor_velocities, axis=0)
            
            if np.linalg.norm(avg_velocity) > 0:
                alignment_dir = avg_velocity / np.linalg.norm(avg_velocity)
                steering_force += alignment_dir * 0.5  # Alignment weight

        # Small tie-breaking noise to prevent static grid deadlocks
        steering_force += self.np_random.uniform(-0.05, 0.05, size=2)

        # --- MAP STEERING VECTOR TO DISCRETE GRID MOVES ---
        best_move = 4
        max_score = -float('inf')

        for move_idx, v in move_vectors.items():
            if avail_actions[move_idx] == 0:
                continue
            next_p = np.array(self.__next_pos(self.prey_pos[prey_i], move_idx))
            
            pred_count, _ = self._neighbour_agents(next_p)
            if pred_count > 0 and move_idx != 4:
                score = -100.0  # Avoid stepping directly into predators
            else:
                score = np.dot(v, steering_force)

            if score > max_score:
                max_score = score
                best_move = move_idx

        # Record chosen vector for the next step's alignment calculation
        self.prey_last_move[prey_i] = move_vectors[best_move]

        return best_move

    def __update_agent_view(self, agent_i):
        self._full_obs[self.agent_pos[agent_i][0]][self.agent_pos[agent_i][1]] = PRE_IDS['agent'] + str(agent_i + 1)

    def __update_prey_view(self, prey_i):
        self._full_obs[self.prey_pos[prey_i][0]][self.prey_pos[prey_i][1]] = PRE_IDS['prey'] + str(prey_i + 1)

    def _neighbour_agents(self, pos):
        # check if agent is in neighbour
        _count = 0
        neighbours_xy = []
        if self.is_valid([pos[0] + 1, pos[1]]) and PRE_IDS['agent'] in self._full_obs[pos[0] + 1][pos[1]]:
            _count += 1
            neighbours_xy.append([pos[0] + 1, pos[1]])
        if self.is_valid([pos[0] - 1, pos[1]]) and PRE_IDS['agent'] in self._full_obs[pos[0] - 1][pos[1]]:
            _count += 1
            neighbours_xy.append([pos[0] - 1, pos[1]])
        if self.is_valid([pos[0], pos[1] + 1]) and PRE_IDS['agent'] in self._full_obs[pos[0]][pos[1] + 1]:
            _count += 1
            neighbours_xy.append([pos[0], pos[1] + 1])
        if self.is_valid([pos[0], pos[1] - 1]) and PRE_IDS['agent'] in self._full_obs[pos[0]][pos[1] - 1]:
            neighbours_xy.append([pos[0], pos[1] - 1])
            _count += 1

        agent_id = []
        for x, y in neighbours_xy:
            agent_id.append(int(self._full_obs[x][y].split(PRE_IDS['agent'])[1]) - 1)
        return _count, agent_id

    def step(self, agents_action):
        assert (self._step_count is not None), \
            "Call reset before using step method."

        self._step_count += 1
        rewards = [self._step_cost for _ in range(self.n_agents)]

        for agent_i, action in enumerate(agents_action):
            if not (self._agent_dones[agent_i]):
                self.__update_agent_pos(agent_i, action)

        for prey_i in range(self.n_preys):
            if self._prey_alive[prey_i]:
                predator_neighbour_count, n_i = self._neighbour_agents(self.prey_pos[prey_i])

                if predator_neighbour_count >= 1:
                    _reward = self._penalty if predator_neighbour_count == 1 else self._prey_capture_reward
                    self._prey_alive[prey_i] = (predator_neighbour_count == 1)

                    for agent_i in range(self.n_agents):
                        rewards[agent_i] += _reward

                prey_move = None
                if self._prey_alive[prey_i]:
                    prey_move = self.__get_boid_move(prey_i)

                self.__update_prey_pos(prey_i, prey_move)

        if (self._step_count >= self._max_steps) or (True not in self._prey_alive):
            for i in range(self.n_agents):
                self._agent_dones[i] = True

        for i in range(self.n_agents):
            self._total_episode_reward[i] += rewards[i]

        if self.render_mode == 'human':
            self.render()

        # Check for episode overflow
        if all(self._agent_dones):
            if self._steps_beyond_done is None:
                self._steps_beyond_done = 0
            else:
                if self._steps_beyond_done == 0:
                    logger.warn(
                        "You are calling 'step()' even though this "
                        "environment has already returned all(done) = True. You "
                        "should always call 'reset()' once you receive "
                        "'all(done) = True' -- any further steps are undefined "
                        "behavior."
                    )
                self._steps_beyond_done += 1

        return self.get_agent_obs(), rewards,  (self._agent_dones), {'prey_alive': self._prey_alive}

    def __get_neighbour_coordinates(self, pos):
        neighbours = []
        if self.is_valid([pos[0] + 1, pos[1]]):
            neighbours.append([pos[0] + 1, pos[1]])
        if self.is_valid([pos[0] - 1, pos[1]]):
            neighbours.append([pos[0] - 1, pos[1]])
        if self.is_valid([pos[0], pos[1] + 1]):
            neighbours.append([pos[0], pos[1] + 1])
        if self.is_valid([pos[0], pos[1] - 1]):
            neighbours.append([pos[0], pos[1] - 1])
        return neighbours

    def render(self, mode='human'):
        assert (self._step_count is not None), \
            "Call reset before using render method."

        import pygame

        img = copy.copy(self._base_img)

        # Calculate predator vision offsets from _agent_view_mask (e.g., [5, 5] -> view range of 2 cells in each direction)
        view_h, view_w = self._agent_view_mask
        half_h = view_h // 2
        half_w = view_w // 2

        # 1. Draw Predator Vision Fields
        for agent_i in range(self.n_agents):
            pos = self.agent_pos[agent_i]
            
            # Determine valid grid bounds within the agent's view mask
            min_row = max(0, pos[0] - half_h)
            max_row = min(self._grid_shape[0], pos[0] + half_h + 1)
            min_col = max(0, pos[1] - half_w)
            max_col = min(self._grid_shape[1], pos[1] + half_w + 1)

            # Fill vision cells
            for r in range(min_row, max_row):
                for c in range(min_col, max_col):
                    fill_cell(
                        img, 
                        [r, c], 
                        cell_size=CELL_SIZE, 
                        fill=AGENT_NEIGHBORHOOD_COLOR, 
                        margin=0.1
                    )

        # 2. Draw Predator Circles & Labels
        for agent_i in range(self.n_agents):
            draw_circle(img, self.agent_pos[agent_i], cell_size=CELL_SIZE, fill=AGENT_COLOR)
            write_cell_text(
                img, 
                text=str(agent_i + 1), 
                pos=self.agent_pos[agent_i], 
                cell_size=CELL_SIZE,
                fill='white', 
                margin=0.4
            )

        # 3. Draw Living Prey
        for prey_i in range(self.n_preys):
            if self._prey_alive[prey_i]:
                draw_circle(img, self.prey_pos[prey_i], cell_size=CELL_SIZE, fill=PREY_COLOR)
                write_cell_text(
                    img, 
                    text=str(prey_i + 1), 
                    pos=self.prey_pos[prey_i], 
                    cell_size=CELL_SIZE,
                    fill='white', 
                    margin=0.4
                )

        img = np.asarray(img)

        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            WINDOW_WIDTH = 800
            WINDOW_HEIGHT = 800

            if self.viewer is None:
                pygame.init()
                self.viewer = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
                pygame.display.set_caption("PredatorPrey-v0")

            # Create surface from raw image array
            surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))

            # Scale surface to match fixed target window dimensions
            scaled_surface = pygame.transform.scale(surface, (WINDOW_WIDTH, WINDOW_HEIGHT))

            self.viewer.blit(scaled_surface, (0, 0))
            pygame.display.flip()

            # Handle window close events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    self.viewer = None

            return self.viewer is not None

    
    def seed(self, n=None):
        self.np_random, seed = seeding.np_random(n)
        return [seed]

    def close(self):
        if self.viewer is not None:
            pygame.quit()
            self.viewer = None

    def get_avail_prey_actions(self, prey_id):
        """
        Returns available actions for a single specified prey.
        Actions: 0: down, 1: left, 2: up, 3: right, 4: no-op (disabled for prey)
        """
        curr_pos = self.prey_pos[prey_id]
        avail_actions = [0, 0, 0, 0, 0]  # No-op (action 4) disabled for prey

        # Check Down (action 0)
        down_pos = [curr_pos[0] + 1, curr_pos[1]]
        if self.is_valid(down_pos) and self._is_cell_vacant(down_pos):
            avail_actions[0] = 1

        # Check Left (action 1)
        left_pos = [curr_pos[0], curr_pos[1] - 1]
        if self.is_valid(left_pos) and self._is_cell_vacant(left_pos):
            avail_actions[1] = 1

        # Check Up (action 2)
        up_pos = [curr_pos[0] - 1, curr_pos[1]]
        if self.is_valid(up_pos) and self._is_cell_vacant(up_pos):
            avail_actions[2] = 1

        # Check Right (action 3)
        right_pos = [curr_pos[0], curr_pos[1] + 1]
        if self.is_valid(right_pos) and self._is_cell_vacant(right_pos):
            avail_actions[3] = 1

        return avail_actions
    
    def get_avail_agent_actions(self, agent_id):
            """
            Returns available actions for a single specified agent.
            Actions: 0: down, 1: left, 2: up, 3: right, 4: no-op (stay)
            """
            curr_pos = self.agent_pos[agent_id]
            avail_actions = [0, 0, 0, 0, 1]  # No-op (action 4) is always available

            # Check Down (action 0)
            down_pos = [curr_pos[0] + 1, curr_pos[1]]
            if self.is_valid(down_pos) and self._is_cell_vacant(down_pos):
                avail_actions[0] = 1

            # Check Left (action 1)
            left_pos = [curr_pos[0], curr_pos[1] - 1]
            if self.is_valid(left_pos) and self._is_cell_vacant(left_pos):
                avail_actions[1] = 1

            # Check Up (action 2)
            up_pos = [curr_pos[0] - 1, curr_pos[1]]
            if self.is_valid(up_pos) and self._is_cell_vacant(up_pos):
                avail_actions[2] = 1

            # Check Right (action 3)
            right_pos = [curr_pos[0], curr_pos[1] + 1]
            if self.is_valid(right_pos) and self._is_cell_vacant(right_pos):
                avail_actions[3] = 1

            return avail_actions

    def get_avail_actions(self):
        """
        Returns available actions for all agents based on grid validity and occupancy.
        """
        return [self.get_avail_agent_actions(agent_i) for agent_i in range(self.n_agents)]


AGENT_COLOR = ImageColor.getcolor('blue', mode='RGB')
AGENT_NEIGHBORHOOD_COLOR = (186, 238, 247)
PREY_COLOR = 'red'

CELL_SIZE = 35

WALL_COLOR = 'black'

ACTION_MEANING = {
    0: "DOWN",
    1: "LEFT",
    2: "UP",
    3: "RIGHT",
    4: "NOOP",
}

PRE_IDS = {
    'agent': 'A',
    'prey': 'P',
    'wall': 'W',
    'empty': '0'
}
