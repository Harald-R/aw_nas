# -*- coding: utf-8 -*-

import abc

from aw_nas import Component, utils

class BaseObjective(Component):
    REGISTRY = "objective"

    def __init__(self, search_space, schedule_cfg=None):
        super(BaseObjective, self).__init__(schedule_cfg)

        self.search_space = search_space

    @utils.abstractclassmethod
    def supported_data_types(cls):
        pass

    @abc.abstractmethod
    def perf_names(self):
        pass

    @abc.abstractmethod
    def get_perfs(self, inputs, outputs, targets, cand_net):
        pass

    @abc.abstractmethod
    def get_reward(self, inputs, outputs, targets, cand_net):
        pass

    @abc.abstractmethod
    def get_loss(self, inputs, outputs, targets, cand_net,
                 add_controller_regularization=True, add_evaluator_regularization=True):
        pass

    def get_loss_item(self, inputs, outputs, targets, cand_net,
                      add_controller_regularization=True, add_evaluator_regularization=True):
        return self.get_loss(inputs, outputs, targets, cand_net,
                             add_controller_regularization, add_evaluator_regularization).item()
