from algos.idql import IDQL
import torch
import numpy as np
from torch.distributions import Categorical

class Agents:
	def __init__(self, args):
		self.n_actions = args.n_actions
		self.n_agents = args.n_agents
		self.state_shape = args.state_shape
		self.obs_shape = args.obs_shape

		# only working with independent learning here for now, but other algos can be easily added
		if args.alg == "idql":
			self.policy = IDQL(args)
			print("IDQL policy initialized")
		else:
			raise Exception("No such algorithm!")

		self.args = args
		

	def choose_action(self, obs, last_action, agent_num, avail_actions, epsilon, maven_z=None, evaluate=False, msg_all=None, return_q=False):
		inputs = obs.copy()
		avail_actions_ind = np.nonzero(avail_actions)[0]

		agent_id = np.zeros(self.n_agents)
		agent_id[agent_num] = 1.

		if self.args.last_action:
			inputs = np.hstack((inputs, last_action))
		if self.args.reuse_network:
			inputs = np.hstack((inputs, agent_id))

		# Read hidden state without mutating original reference
		hidden_state = self.policy.eval_hidden[:, agent_num, :].clone()

		inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
		avail_actions_tensor = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)

		if self.args.cuda:
			inputs = inputs.cuda(device=self.args.cuda_device)
			hidden_state = hidden_state.cuda(device=self.args.cuda_device)
			avail_actions_tensor = avail_actions_tensor.cuda(device=self.args.cuda_device)
			if self.args.with_comm and msg_all is not None:
				msg_all = msg_all.cuda(device=self.args.cuda_device)

		# Forward pass
		if self.args.with_comm:
			q_value, new_hidden = self.policy.eval_rnn(inputs, hidden_state, msg_all, agent_num)
		else:
			q_value, new_hidden = self.policy.eval_rnn(inputs, hidden_state)

		# IF return_q IS TRUE (ATTACK MODE): Do NOT perform in-place mutation on self.policy.eval_hidden!
		if return_q:
			return q_value.squeeze(0)

		# Standard execution mode: Update state in-place
		self.policy.eval_hidden[:, agent_num, :] = new_hidden

		q_value_clone = q_value.clone()
		q_value_clone[avail_actions_tensor == 0.0] = -float("inf")

		if self.args.alg == 'coma':
			action = self._choose_action_from_softmax(q_value_clone.cpu(), avail_actions, epsilon, evaluate)
		else:
			if not evaluate and np.random.uniform() < epsilon:
				action = np.random.choice(avail_actions_ind)
			else:
				action = torch.argmax(q_value_clone).item()

		return action


	# this is to be used during execution; given obs and last action (as per the normal inputs), should generate the messages for all agents
	def get_all_messages(self, obs, last_action=None):
		"""
		Constructs full input features per agent (obs + last_action + agent_id)
		and passes them to commtest.
		"""
		inputs = [torch.tensor(obs, dtype=torch.float32)]

		# 1. Append last_action if configured
		if self.args.last_action and last_action is not None:
			inputs.append(torch.tensor(last_action, dtype=torch.float32))

		# 2. Append agent ID one-hot if configured
		if self.args.reuse_network:
			agent_id_onehot = torch.eye(self.n_agents)
			inputs.append(agent_id_onehot)

		# Concatenate features along the feature dimension (dim=-1 -> shape [n_agents, 37])
		inputs_tensor = torch.cat(inputs, dim=-1)

		if self.args.cuda:
			inputs_tensor = inputs_tensor.cuda(device=self.args.cuda_device)

		# Forward pass through Comm_net
		msgs_agents = self.policy.commtest(inputs_tensor).detach()

		return msgs_agents


	def _get_max_episode_len(self, batch):
		terminated = batch['terminated']
		episode_num = terminated.shape[0]  # number of episode batches inside this batch
		max_episode_len = 0
		for episode_idx in range(episode_num):
			for transition_idx in range(self.args.episode_limit):
				if terminated[episode_idx, transition_idx, 0] == 1:  # TODO: better understand this 3D shape; prob is [buffer_size, episode_limit, ?]; transition idx refers to the trajectory, i.e., passage from one state to another inside this episode
					if transition_idx + 1 >= max_episode_len:
						max_episode_len = transition_idx + 1
					break
		return max_episode_len


	def train(self, batch, train_step, epsilon=None):  
		# different episode has different length, so we need to get max length of the batch
		max_episode_len = self._get_max_episode_len(batch)
		for key in batch.keys():
			batch[key] = batch[key][:, :max_episode_len]  # TODO: see this
		if max_episode_len > 0:
			self.policy.learn(batch, max_episode_len, train_step, epsilon)
		else:
			print('empty batch')

		# savind model
		if train_step > 0 and train_step % self.args.save_cycle == 0:
			self.policy.save_model(train_step)


	def _choose_action_from_softmax(self, inputs, avail_actions, epsilon, evaluate=False):
		# inputs refers to q_value of all actions
		action_num = avail_actions.sum(dim=1, keepdim=True).float().repeat(1, avail_actions.shape[-1])  # sum of avail_actions
		# converts output of actor network into a prob dist with softmax
		prob = torch.nn.functional.softmax(inputs, dim=-1)

		# noise of epsilon
		prob = ((1 - epsilon) * prob + torch.ones_like(prob) * epsilon / action_num)
		prob[avail_actions == 0] = 0.0  # unavailable actions get 0 prob

		# note that after setting the unavaible actions prob to 0, the sum in prob is not 1, but no need to regularize because torch.distributions.categorical will be regularized
		# categorical is not used during training so the probability of the action performed during training needs to be regularized again

		if epsilon == 0 and evaluate:
			action = torch.argmax(prob)
		else:
			action = Categorical(prob).sample().long()

		return action


	def _get_max_episode_len(self, batch):
		terminated = batch['terminated']
		episode_num = terminated.shape[0]  # number of episode batches inside this batch
		max_episode_len = 0
		for episode_idx in range(episode_num):
			for transition_idx in range(self.args.episode_limit):
				if terminated[episode_idx, transition_idx, 0] == 1:  # TODO: better understand this 3D shape; prob is [buffer_size, episode_limit, ?]; transition idx refers to the trajectory, i.e., passage from one state to another inside this episode
					if transition_idx + 1 >= max_episode_len:
						max_episode_len = transition_idx + 1
					break
		return max_episode_len


	def train(self, batch, train_step, epsilon=None):  # epsilon is for coma TODO

	# different episode has different length, so we need to get max length of the batch
		max_episode_len = self._get_max_episode_len(batch)  # inside batch there are several episode batches; as they may have different sizes, gets the bigger
		for key in batch.keys():
			batch[key] = batch[key][:, :max_episode_len]  # TODO: see this
		if max_episode_len > 0:
			self.policy.learn(batch, max_episode_len, train_step, epsilon)
		else:
			print('empty batch')
		# savind model
		if train_step > 0 and train_step % self.args.save_cycle == 0:
			self.policy.save_model(train_step)


