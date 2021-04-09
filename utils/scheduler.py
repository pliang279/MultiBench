import numpy as np


# %%
class LRCosineAnnealingScheduler():

    def __init__(self, eta_max, eta_min, Ti, Tmultiplier, num_batches_per_epoch):

        self.eta_min = eta_min
        self.eta_max = eta_max
        self.Ti = Ti
        self.Tcur = 0.0
        self.nbpe = num_batches_per_epoch
        self.iteration_counter = 0.0
        self.eta = eta_max
        self.Tm = Tmultiplier

    def _compute_rule(self):
        self.eta = self.eta_min + 0.5 * (self.eta_max - self.eta_min) * (1 + np.cos(np.pi * self.Tcur / self.Ti))
        return self.eta

    def step(self):

        self.Tcur = self.iteration_counter / self.nbpe
        self.iteration_counter = self.iteration_counter + 1.0
        eta = self._compute_rule()

        if eta <= self.eta_min + 1e-10:
            self.Tcur = 0
            self.Ti = self.Ti * self.Tm
            self.iteration_counter = 0

        return eta

    def update_optimizer(self, optimizer):
        state_dict = optimizer.state_dict()
        for param_group in state_dict['param_groups']:
            param_group['lr'] = self.eta
        optimizer.load_state_dict(state_dict)


# %%
class FixedScheduler():

    def __init__(self, lr):
        self.lr = lr

    def step(self):
        return self.lr

    def update_optimizer(self, optimizer):
        state_dict = optimizer.state_dict()
        for param_group in state_dict['param_groups']:
            param_group['lr'] = self.lr
        optimizer.load_state_dict(state_dict)
