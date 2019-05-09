# -*- coding: utf-8 -*-
"""
RL-based controllers
"""

import torch
from torch import nn

from aw_nas.common import Rollout
from aw_nas.controller.base import BaseController
from aw_nas.controller.rl_networks import BaseRLControllerNet
from aw_nas.controller.rl_agents import BaseRLAgent

class RLController(BaseController, nn.Module):
    NAME = "rl"

    def __init__(self, search_space, device,
                 controller_network_type="anchor_lstm", controller_network_cfg=None,
                 rl_agent_type="pg", rl_agent_cfg=None,
                 independent_cell_group=False):
        """
        Args:
            search_space (aw_nas.SearchSpace): The controller will sample arch seeds
                for this search space.
            device (str): `cuda` or `cpu`
            independent_cell_group (bool): If true, use independent controller and agent
                for each cell group.

        .. warning::
            If `independent_cell_group` is set to true, do not merge anything across
            the graph of different cell groups.
        """
        super(RLController, self).__init__(search_space)
        nn.Module.__init__(self)

        self.device = device
        self.independent_cell_group = independent_cell_group

        # handle cell groups here
        self.controllers = []
        self.agents = []
        rl_agent_cfg = rl_agent_cfg or {}
        controller_network_cfg = controller_network_cfg or {}
        cn_cls = BaseRLControllerNet.get_class_(controller_network_type)
        num_cnet = self.search_space.num_cell_groups if self.independent_cell_group else 1
        self.controllers = [cn_cls(self.search_space, self.device,
                                   **controller_network_cfg) for _ in range(num_cnet)]
        self.agents = [BaseRLAgent.get_class_(rl_agent_type)(cnet, **rl_agent_cfg)\
                       for cnet in self.controllers]

        self.controller = nn.ModuleList(self.controllers)

    def forward(self, n=1): #pylint: disable=arguments-differ
        self.sample(n=n)

    def sample(self, n=1):
        arch_lst = []
        log_probs_lst = []
        entropies_lst = []
        hidden = None
        for i_cg in range(self.search_space.num_cell_groups):
            # sample the arch for cell groups sequentially
            cn_idx = i_cg if self.independent_cell_group else 0
            arch, lprob, ent, hidden = self.controllers[cn_idx].sample(batch_size=n,
                                                                       prev_hidden=hidden)
            hidden = None if self.independent_cell_group else hidden
            arch_lst.append(arch)
            log_probs_lst.append(lprob)
            entropies_lst.append(ent)

        # merge the archs for different cell groups
        arch_lst = zip(*arch_lst)
        log_probs_lst = zip(*log_probs_lst)
        entropies_lst = zip(*entropies_lst)

        rollouts = [Rollout(arch, info={"log_probs": log_probs,
                                        "entropies": entropies},
                            search_space=self.search_space)
                    for arch, log_probs, entropies in zip(arch_lst, log_probs_lst,
                                                          entropies_lst)]
        return rollouts

    def save(self, path):
        for i, (controller, agent) in enumerate(zip(self.controllers, self.agents)):
            agent.save("{}_agent_{}".format(path, i))
            controller.save("{}_net_{}".format(path, i))

    def load(self, path):
        for i, (controller, agent) in enumerate(zip(self.controllers, self.agents)):
            agent.load("{}_agent_{}".format(path, i))
            controller.load("{}_net_{}".format(path, i))

    def step(self, rollouts, optimizer):
        if not self.independent_cell_group:
            # Single controller net and agent for all cell groups
            loss = self.agents[0].step(rollouts, optimizer)
        else:
            # One controller net and agent per cel group
            rollouts_lst = zip(*[self._split_rollout(r) for r in rollouts])
            loss = 0.
            for agent, splited_rollouts in zip(self.agents, rollouts_lst):
                loss += agent.step(splited_rollouts, optimizer)
            loss /= len(self.agents)
        return loss

    def on_epoch_start(self, epoch):
        super(RLController, self).on_epoch_start(epoch)
        [c.on_epoch_start(epoch) for c in self.controllers]
        [a.on_epoch_start(epoch) for a in self.agents]

    def on_epoch_end(self, epoch):
        super(RLController, self).on_epoch_end(epoch)
        [c.on_epoch_end(epoch) for c in self.controllers]
        [a.on_epoch_end(epoch) for a in self.agents]

    @staticmethod
    def _split_rollout(rollout):
        rollouts = []
        for log_prob, ent in zip(rollout.info["log_probs"], rollout.info["entropies"]):
            rollouts.append(Rollout(rollout.arch, info={
                "log_probs": (log_prob,),
                "entropies": (ent,),
            }, search_space=rollout.search_space))
            rollouts[-1].perf = rollout.perf
        return rollouts


#pylint: disable=invalid-name,ungrouped-imports
if __name__ == "__main__":
    import numpy as np
    from aw_nas.common import get_search_space
    search_space = get_search_space(cls="cnn")
    device = "cuda"
    controller = RLController(search_space, device)
    controller_i = RLController(search_space, device,
                                independent_cell_group=True,
                                rl_agent_cfg={"batch_update": False})
    assert len(list(controller.parameters())) == 10
    assert len(list(controller_i.parameters())) == 20
    rollouts = controller.sample(3)
    [r.set_perf(np.random.rand(1)) for r in rollouts]
    optimizer = torch.optim.SGD(controller.parameters(), lr=0.01)
    loss = controller.step(rollouts, optimizer)

    rollouts = controller_i.sample(3)
    [r.set_perf(np.random.rand(1)) for r in rollouts]
    optimizer = torch.optim.SGD(controller_i.parameters(), lr=0.01)
    loss = controller_i.step(rollouts, optimizer)
#pylint: enable=invalid-name,ungrouped-imports
