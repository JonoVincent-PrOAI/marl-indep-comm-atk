import torch
import os
from network.base_net import RNN
from network.simple_comm_net import Comm_net
import torch.nn as nn
import numpy as np
import sys

if not "../" in sys.path:
    sys.path.append('../')
if not "./" in sys.path:
    sys.path.append('./')

class IDQL:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        
        # Base input shape per agent
        input_shape = self.obs_shape

        if args.last_action:
            input_shape += self.n_actions
        if args.reuse_network:
            input_shape += self.n_agents

        # Comm_net input shape must match the combined base features per agent!
        if args.with_comm:
            input_comm_shape = input_shape
            
            # Message input size added to total RNN input shape:
            input_shape += (args.n_agents - 1) * args.final_msg_dim
            print("obs shape with comm: ", input_shape)

        self.eval_rnn = RNN(input_shape, args)
        self.target_rnn = RNN(input_shape, args)
        self.args = args

        if args.with_comm:
            self.commtest = Comm_net(input_comm_shape, args)
            self.target_commtest = Comm_net(input_comm_shape, args)

        print(self.args.cuda)
        if self.args.cuda:
            self.eval_rnn.cuda(device=self.args.cuda_device)
            self.target_rnn.cuda(device=self.args.cuda_device)
            if args.with_comm:
                self.commtest.cuda(device=self.args.cuda_device)
                self.target_commtest.cuda(device=self.args.cuda_device)

        self.model_dir = args.model_dir
        if self.args.load_model != None:
            if os.path.exists(self.args.model_dir):
                map_location = 'cuda:0' if self.args.cuda else 'cpu'
                self.eval_rnn.load_state_dict(torch.load(self.args.model_dir, map_location=map_location))
                print('Successfully load the model: {}'.format(self.args.model_dir))
            else:
                raise Exception("No model: " + self.model_dir)


        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())

        if args.with_comm:
            self.target_commtest.load_state_dict(self.commtest.state_dict())
            self.eval_parameters = list(self.eval_rnn.parameters()) + list(self.commtest.parameters())
        else:
            self.eval_parameters = list(self.eval_rnn.parameters())

        if args.optimizer == "RMS":
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=args.lr)


        self.eval_hidden = None
        self.target_hidden = None
        print('Init alg IQL')

    def learn(self, batch, max_episode_len, train_step, epsilon=None):  
        episode_num = batch['obs'].shape[0]
        self.init_hidden(episode_num)
        for key in batch.keys():  
            if key == 'actions':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        u, r, avail_u, avail_u_next, terminated = batch['actions'], batch['reward'].repeat(1, 1, self.n_agents),  batch['avail_actions'], \
                                                  batch['avail_actions_next'], batch['terminated'].repeat(1, 1, self.n_agents)
        mask = (1 - batch["padded"].float()).repeat(1, 1, self.n_agents) 

        q_evals, q_targets = self.get_q_values(batch, max_episode_len)
        if self.args.cuda:
            u = u.cuda(device=self.args.cuda_device)
            r = r.cuda(device=self.args.cuda_device)
            terminated = terminated.cuda(device=self.args.cuda_device)
            mask = mask.cuda(device=self.args.cuda_device)

        q_evals = torch.gather(q_evals, dim=3, index=u).squeeze(3)

        q_targets[avail_u_next == 0.0] = - 9999999
        q_targets = q_targets.max(dim=3)[0]

        targets = r + self.args.gamma * q_targets * (1 - terminated)

        td_error = (q_evals - targets.detach())
        masked_td_error = mask * td_error 

        loss = (masked_td_error ** 2).sum() / mask.sum()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()

        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
            if self.args.with_comm:
                self.target_commtest.load_state_dict(self.commtest.state_dict())
        return loss

    def _get_inputs(self, batch, transition_idx):
        obs = batch['obs'][:, transition_idx]
        obs_next = batch['obs_next'][:, transition_idx]
        u_onehot = batch['actions_onehot']
        episode_num = obs.shape[0]

        inputs, inputs_next = [obs], [obs_next]

        # 1. Append last_action FIRST if configured
        if self.args.last_action:
            if transition_idx == 0:
                last_act = torch.zeros_like(u_onehot[:, transition_idx])
            else:
                last_act = u_onehot[:, transition_idx - 1]
            
            inputs.append(last_act)
            inputs_next.append(u_onehot[:, transition_idx])

        # 2. Append agent ID one-hot if configured
        if self.args.reuse_network:
            agent_id_onehot = torch.eye(self.args.n_agents, device=obs.device).unsqueeze(0).expand(episode_num, -1, -1)
            inputs.append(agent_id_onehot)
            inputs_next.append(agent_id_onehot)

        # 3. Concatenate base features per agent along dim=-1 (feature axis)
        inputs_tensor = torch.cat(inputs, dim=-1)
        inputs_next_tensor = torch.cat(inputs_next, dim=-1)

        all_msgs, all_msgs_next = None, None

        # 4. Generate comm messages from the complete feature tensor
        if self.args.with_comm:
            if self.args.cuda:
                inputs_tensor = inputs_tensor.cuda(device=self.args.cuda_device)
                inputs_next_tensor = inputs_next_tensor.cuda(device=self.args.cuda_device)

            all_msgs = self.commtest(inputs_tensor)
            all_msgs_next = self.target_commtest(inputs_next_tensor)

        return inputs_tensor, inputs_next_tensor, all_msgs, all_msgs_next

    def get_q_values(self, batch, max_episode_len):
        episode_num = batch['obs'].shape[0]
        q_evals, q_targets = [], []
        for transition_idx in range(max_episode_len):
            inputs, inputs_next, all_msgs, all_msgs_next = self._get_inputs(batch, transition_idx)  
            if self.args.cuda:
                inputs = inputs.cuda(device=self.args.cuda_device)
                inputs_next = inputs_next.cuda(device=self.args.cuda_device)
                self.eval_hidden = self.eval_hidden.cuda(device=self.args.cuda_device)
                self.target_hidden = self.target_hidden.cuda(device=self.args.cuda_device)
                if self.args.with_comm:
                    all_msgs = all_msgs.cuda(device=self.args.cuda_device)
                    all_msgs_next = all_msgs_next.cuda(device=self.args.cuda_device)

            if self.args.with_comm:
                q_eval, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden, msgs=all_msgs)  
                q_target, self.target_hidden = self.target_rnn(inputs_next, self.target_hidden, msgs=all_msgs_next)
            else:
                q_eval, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)  
                q_target, self.target_hidden = self.target_rnn(inputs_next, self.target_hidden)

            q_eval = q_eval.view(episode_num, self.n_agents, -1)
            q_target = q_target.view(episode_num, self.n_agents, -1)
            q_evals.append(q_eval)
            q_targets.append(q_target)


        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        return q_evals, q_targets

    def init_hidden(self, episode_num):
        self.eval_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.target_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))

    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.eval_rnn.state_dict(),  self.model_dir + '/' + num + '_rnn_net_params.pkl')