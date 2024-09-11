import math
import torch


class CosineAnnealingLR:
    def __init__(self, optimizer: torch.optim.Optimizer, T_max: int, eta_min: float = 0, last_epoch: int = -1):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.last_epoch = last_epoch
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.step()

    def get_lr(self) -> list:
        epoch = self.last_epoch + 1
        return [
            self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * epoch / self.T_max)) / 2
            for base_lr in self.base_lrs
        ]

    def step(self) -> None:
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def get_last_lr(self) -> list:
        return [group['lr'] for group in self.optimizer.param_groups]


class StepLR:
    def __init__(self, optimizer: torch.optim.Optimizer, step_size: int, gamma: float = 0.1, last_epoch: int = -1):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = last_epoch
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.step()

    def get_lr(self) -> list:
        epoch = self.last_epoch + 1
        return [
            base_lr * (self.gamma ** (epoch // self.step_size))
            for base_lr in self.base_lrs
        ]

    def step(self) -> None:
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def get_last_lr(self) -> list:
        return [group['lr'] for group in self.optimizer.param_groups]


class ExponentialLR:
    def __init__(self, optimizer: torch.optim.Optimizer, gamma: float, last_epoch: int = -1):
        self.optimizer = optimizer
        self.gamma = gamma
        self.last_epoch = last_epoch
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.step()

    def get_lr(self) -> list:
        return [
            base_lr * (self.gamma ** (self.last_epoch + 1))
            for base_lr in self.base_lrs
        ]

    def step(self) -> None:
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def get_last_lr(self) -> list:
        return [group['lr'] for group in self.optimizer.param_groups]


class CyclicLR:
    def __init__(self, optimizer: torch.optim.Optimizer, base_lr: float, max_lr: float, step_size_up: int,
                 last_epoch: int = -1):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size_up = step_size_up
        self.last_epoch = last_epoch
        self.base_lrs = [base_lr] * len(optimizer.param_groups)
        self.step()

    def get_lr(self) -> list:
        cycle = math.floor(1 + self.last_epoch / (2 * self.step_size_up))
        x = abs(self.last_epoch / self.step_size_up - 2 * cycle + 1)
        lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x))
        return [lr for _ in self.base_lrs]

    def step(self) -> None:
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def get_last_lr(self) -> list:
        return [group['lr'] for group in self.optimizer.param_groups]

