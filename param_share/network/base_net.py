import torch.nn as nn
import torch.nn.functional as F

import torch

'''
Because the RNN is used here, the last hidden_state is required each time. For an episode of data, each obs needs the last hidden_state to select the action.
Therefore, it is not possible to directly and randomly extract a batch of experience input to the neural network, so a batch of episodes is needed here, and the transition of the same position of this batch of episodes is passed in each time.
In this case, the hidden_state can be saved, and the next experience is the next experience
'''

class RNN(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, input_shape, args):
        super(RNN, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)  # rnn gating mechanism; gated recurrent unit
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        self.input_shape = input_shape
        self.msg_dim = args.final_msg_dim

    def forward(self, obs, hidden_state, msgs=None, agent_num=None):
       
        if self.args.with_comm:
            device = "cuda" if self.args.cuda and torch.cuda.is_available() else "cpu"

            if agent_num is not None:
                # Execution mode (single agent call during rollout)
                # Filter out self-message (agent_num)
                idxs = torch.tensor([i for i in range(self.args.n_agents) if i != agent_num], device=device)
                msgs_rec = torch.index_select(msgs, dim=1, index=idxs)
                
                # Reshape & concatenate: [1, (n_agents-1) * msg_dim]
                obs = torch.cat((obs, msgs_rec.reshape(obs.shape[0], -1)), dim=-1)
            else:
                # Training mode (batch processing)
                # Determine batch size dynamically regardless of whether obs is 2D or 3D
                if obs.dim() == 3:
                    ep_num = obs.shape[0]
                    obs_flat = obs
                else:
                    ep_num = obs.shape[0] // self.args.n_agents
                    obs_flat = obs.reshape(ep_num, self.args.n_agents, -1)

                # msgs shape: [ep_num, n_agents, msg_dim]
                msg_dim = msgs.shape[-1]
                
                # Build index matrix to select messages from all OTHER agents
                # For agent i, gather indices [0, ..., i-1, i+1, ..., n_agents-1]
                other_indices = []
                for i in range(self.args.n_agents):
                    other_indices.append([j for j in range(self.args.n_agents) if j != i])
                
                # Index tensor shape: [n_agents, n_agents - 1]
                idx_tensor = torch.tensor(other_indices, device=device)
                
                # Expand index tensor across batch dimension -> [ep_num, n_agents, n_agents - 1]
                idx_expanded = idx_tensor.unsqueeze(0).expand(ep_num, -1, -1)
                
                # Gather other agents' messages cleanly: [ep_num, n_agents, n_agents - 1, msg_dim]
                # Expanding idx_expanded for gather on feature dimension
                idx_gather = idx_expanded.unsqueeze(-1).expand(-1, -1, -1, msg_dim)
                msgs_expanded = msgs.unsqueeze(1).expand(-1, self.args.n_agents, -1, -1)
                
                msgs_other = torch.gather(msgs_expanded, dim=2, index=idx_gather)
                
                # Flatten received messages per agent: [ep_num, n_agents, (n_agents - 1) * msg_dim]
                msgs_rec = msgs_other.reshape(ep_num, self.args.n_agents, -1)

                # Concatenate features with received messages
                obs = torch.cat((obs_flat, msgs_rec), dim=-1)

        # Reshape features for linear FC layer
        x = F.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        
        # Process through GRU/RNN
        x_in = x.reshape(-1, x.shape[-1])
        h = self.rnn(x_in, h_in)
        q = self.fc2(h)

        return q, h


# Critic of Central-V
class Critic(nn.Module):
    def __init__(self, input_shape, args):
        super(Critic, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.critic_dim)
        self.fc2 = nn.Linear(args.critic_dim, args.critic_dim)
        self.fc3 = nn.Linear(args.critic_dim, 1)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q
