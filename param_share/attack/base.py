# -*- coding: utf-8 -*-
# @Description:

#https://github.com/gralliry/Adversarial-Attack-Generation-Techniques/blob/master/attack/base.py

import torch


class BaseModel:
    def __init__(self, model, cuda=True):
        if cuda and not torch.cuda.is_available():
            print("No GPU found, please run without --cuda")
        self.device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
        self.model = model

    def forward(self, model_inputs):
        return self.model(
                            model_inputs['current_obs'], 
                            model_inputs['last_action'],
                            model_inputs['agent_id'],
                            model_inputs['avail_actions'],
                            model_inputs['epsilon'],
                            model_inputs['evaluate'],
                            model_inputs['msg_all']
                            )

    def attack(self, image, target, is_targeted=False):
        raise NotImplementedError