import torch
import numpy as np

from .base import BaseModel


class JSMA(BaseModel):
    def __init__(self, model, theta=1.0, gamma=0.1, cuda=True):
        """
        JSMA for full communication message tensor
        https://arxiv.org/abs/1511.07528
        """
        super().__init__(model=model, cuda=cuda)

        self.theta = theta
        self.gamma = gamma

    def attack(self, model_inputs: dict, target_action, is_targeted=True):
        """
        JSMA applied across all message channels/agents in model_inputs['msg_all']
        """
        assert is_targeted is True, ValueError("JSMA must be targeted")

        # Full message tensor across all agents/channels
        messages = model_inputs['msg_all']
        var_messages = messages.clone().detach().requires_grad_(True)
        increasing = self.theta > 0

        # Total features across the entire message tensor
        num_features = int(var_messages.numel())
        shape = var_messages.shape

        # Max iterations based on global feature count
        max_iters = int(np.ceil(num_features * self.gamma / 2.0))

        # Masked search domain across all features
        if increasing:
            domain = torch.lt(var_messages, 0.99)
        else:
            domain = torch.gt(var_messages, -0.99)

        domain = domain.view(-1)

        # Initial forward pass
        output = self.model.choose_action(
            obs=model_inputs['current_obs'],
            last_action=model_inputs['last_action'],
            agent_num=model_inputs['agent_id'],
            avail_actions=model_inputs['avail_actions'],
            epsilon=model_inputs['epsilon'],
            evaluate=model_inputs['evaluate'],
            msg_all=model_inputs['msg_all'],
            return_q=True
        )
        current_pred = torch.argmax(output, dim=-1)
        iters = 0

        while (
            (iters < max_iters)
            and (current_pred != target_action)
            and (domain.sum() != 0)
        ):
            # Calculate Jacobian w.r.t the ENTIRE message tensor
            jacobian = self.compute_jacobian(model_inputs, var_messages, num_features)

            # Saliency map across all message features
            p1, p2 = self.saliency_map(jacobian, target_action, increasing, domain, num_features)

            # Apply modifications across flattened message space
            var_sample_flatten = var_messages.view(-1, num_features).clone().detach()
            var_sample_flatten[0, p1] += self.theta
            var_sample_flatten[0, p2] += self.theta

            # Clamp and reshape back to original tensor dimensions
            new_messages = torch.clamp(var_sample_flatten, min=0.0, max=1.0).view(shape)

            domain[p1] = 0
            domain[p2] = 0

            var_messages = new_messages.clone().detach().requires_grad_(True).to(self.device)

            output = self.model.choose_action(
                obs=model_inputs['current_obs'],
                last_action=model_inputs['last_action'],
                agent_num=model_inputs['agent_id'],
                avail_actions=model_inputs['avail_actions'],
                epsilon=model_inputs['epsilon'],
                evaluate=model_inputs['evaluate'],
                msg_all=var_messages,
                return_q=True
            )
            current_pred = torch.argmax(output, dim=-1)
            iters += 1

        return var_messages.detach()

    def compute_jacobian(self, model_inputs, var_messages, num_features):
        with torch.enable_grad():
            if not var_messages.requires_grad:
                var_messages = var_messages.clone().detach().requires_grad_(True)

            # Forward pass with full perturbed message tensor
            output = self.model.choose_action(
                obs=model_inputs['current_obs'],
                last_action=model_inputs['last_action'],
                agent_num=model_inputs['agent_id'],
                avail_actions=model_inputs['avail_actions'],
                epsilon=model_inputs['epsilon'],
                evaluate=model_inputs['evaluate'],
                msg_all=var_messages,
                return_q=True
            )

            flat_output = output.view(-1)
            num_outputs = flat_output.numel()
            jacobian = torch.zeros((num_outputs, num_features), device=self.device)

            # Compute gradient for each action score w.r.t. entire message tensor
            for i in range(num_outputs):
                grad = torch.autograd.grad(
                    outputs=flat_output[i],
                    inputs=var_messages,
                    retain_graph=(i < num_outputs - 1),
                    create_graph=False,
                    allow_unused=True
                )[0]

                if grad is not None:
                    jacobian[i] = grad.view(-1)

        return jacobian

    @torch.no_grad()
    def saliency_map(self, jacobian, target_label, increasing, search_space, nb_features):
        domain = torch.eq(search_space, 1).float()

        # Sum of all feature derivatives w.r.t. each output class
        all_sum = torch.sum(jacobian, dim=0, keepdim=True)

        target_grad = jacobian[target_label]
        others_grad = all_sum - target_grad

        # Coefficient multiplier mask
        if increasing:
            increase_coef = 2 * (torch.eq(domain, 0)).float().to(self.device).view(-1)
        else:
            increase_coef = -1 * 2 * (torch.eq(domain, 0)).float().to(self.device).view(-1)

        target_tmp = target_grad - increase_coef * torch.max(torch.abs(target_grad))
        alpha = target_tmp.view(1, -1, 1) + target_tmp.view(1, 1, -1)

        others_tmp = others_grad + increase_coef * torch.max(torch.abs(others_grad))
        beta = others_tmp.view(1, -1, 1) + others_tmp.view(1, 1, -1)

        # Zero out diagonal (pairing a feature with itself)
        zero_diagonal = (1 - torch.eye(nb_features, device=self.device)).unsqueeze(0)

        if increasing:
            mask1 = torch.gt(alpha, 0.0)
            mask2 = torch.lt(beta, 0.0)
        else:
            mask1 = torch.lt(alpha, 0.0)
            mask2 = torch.gt(beta, 0.0)

        mask = mask1 * mask2 * zero_diagonal
        saliency_map = alpha * torch.abs(beta) * mask.float()

        max_idx = torch.argmax(saliency_map.view(-1))
        p = torch.div(max_idx, nb_features, rounding_mode="floor")
        q = max_idx % nb_features

        return p.item(), q.item()