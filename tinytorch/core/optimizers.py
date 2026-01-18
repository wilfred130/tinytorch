import numpy as np
import os
from typing import List, Tuple, Optional

from tinytorch.core.tensor import Tensor
from tinytorch.core.autograd import enable_autograd

enable_autograd()

# CONSTANTS FOR OPTIMIZER DEFAULTS
DEFAULT_LEARING_RATE_SGD = 0.01
DEFAULT_LEARING_RATE_ADAM = 0.001
DEFAULT_MOMENTUM = 0.9                   # default momentum for SGD
DEFAULT_BETA1 = 0.9                      # first moment decay for Adam
DEFAULT_BETA2 = 0.999                    # second moment decay for Adam
DEFAULT_EPS = 1e-8                       # Epsilon for numerical stability in ADAM
DEFAULT_WEIGHT_DECAY_ADAMW = 0.01        # Default weight decay for AdamW


class Optimizer:
    """Base class for all optimizers"""

    def __init__(self, params: List[Tensor]):
        if not isinstance(params, List):
            params = List(params)

        self.params = params
        self.step_count = 0

    def zero_grad(self):
        for param in self.params:
            param.grad = None  # type: ignore

    def step(self):
        """Updates parameters based on gradients"""

        raise NotImplementedError("Subclasses must implement step()")


class SGD(Optimizer):
    """Foundational optimization that moves params in opposite direction to gradients"""

    def __init__(self, params: List[Tensor], lr: float = DEFAULT_LEARING_RATE_SGD,
                 momentum: float = 0.0, weight_decay: float = 0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        # intialize moment buffer (creates them lazily)
        self.momentum_buffer = [None for _ in self.params]

    def has_momentum(self):
        """Checks if optimizer uses momentum"""
        return self.momentum > 0

    def get_momentum_state(self) -> Optional[List]:
        """Gets momentum buffer for checkpointing"""
        if not self.has_momentum():
            return

        return [buf.copy if buf is not None else None for buf in self.momentum_buffer]

    def set_momentum_state(self, state: Optional[List]) -> None:
        """Restore momentum buffer for checkpointing"""

        if state is None or not self.has_momentum:
            return

        if len(state) != len(self.momentum_buffer):
            raise ValueError(
                f"State length {len(state)} does not match "
                f"optimizer parameters {len(self.momentum_buffer)}"
            )

        for i, buf in enumerate(state):
            if buf is not None:
                self.momentum_buffer[i] = buf.copy()

    def step(self):
        """Perform SGD with update"""
        for i, param in enumerate(self.params):
            if param.grad is None:  # type: ignore
                continue

            grad = param.grad  # type: ignore

            if isinstance(grad, Tensor):
                grad_data = grad.data
            else:
                grad_data = grad

            if self.weight_decay != 0:
                grad_data = grad_data + self.weight_decay * param.data

            if self.momentum != 0:
                if self.momentum_buffer[i] is None:
                    self.momentum_buffer[i] = np.zeros_like(  # type: ignore
                        param.data)

                self.momentum_buffer[i] = self.momentum * \
                    self.momentum_buffer[i] + grad_data  # type: ignore
                grad_data = self.momentum_buffer[i]

            param.data = param.data - self.lr * grad_data  # type: ignore
        self.step_count += 1


class Adam(Optimizer):
    """Adam Optimizer with adaptive learning rates"""

    def __init__(self, params: List[Tensor], lr: float = DEFAULT_LEARING_RATE_ADAM,
                 betas: tuple = (DEFAULT_BETA1, DEFAULT_BETA2), eps: float = DEFAULT_EPS,
                 weight_decay: float = 0.0):
        super().__init__(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # Initailize the moment buffers
        self.m_buffers = [None for _ in self.params]        # first moment
        self.v_buffers = [None for _ in self.params]        # second moment

    def step(self):
        """Performs Adam updates"""

        self.step_count += 1

        for i, param in enumerate(self.params):
            if param.grad is None:  # type: ignore
                continue

            grad = param.grad  # type: ignore

            if isinstance(grad, Tensor):
                grad_data = grad.data

            else:
                grad_data = grad

            if self.weight_decay != 0:
                grad_data = grad_data + self.weight_decay * param.data

            if self.m_buffers[i] is None:
                self.m_buffers[i] = np.zeros_like(param.data)  # type: ignore
                self.v_buffers[i] = np.zeros_like(param.data)  # type: ignore

            self.m_buffers[i] = self.beta1 * \
                self.m_buffers[i] + (1 - self.beta1) * grad_data
            self.v_buffers[i] = self.beta2 * self.v_buffers[i] + \
                (1 - self.beta2) * (grad_data ** 2)

            # compute bias corrections
            bias_correction1 = 1 - self.beta1 ** self.step_count
            bias_correction2 = 1 - self.beta2 ** self.step_count

            m_hat = self.m_buffers[i] / bias_correction1
            v_hat = self.v_buffers[i] / bias_correction2

            param.data = param.data - self.lr * \
                m_hat / (np.sqrt(v_hat) + self.eps)


class AdamW(Optimizer):
    """AdamW optimizer with decoupled weight decay"""

    def __init__(self, params: List[Tensor], lr: float = DEFAULT_LEARING_RATE_ADAM,
                 betas: tuple = (DEFAULT_BETA1, DEFAULT_BETA2), eps: float = DEFAULT_EPS,
                 weight_decay: float = DEFAULT_WEIGHT_DECAY_ADAMW):
        super().__init__(params)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.lr = lr
        self.weight_decay = weight_decay

        # Initialize momentum buffers (lazily)
        self.m_buffers = [None for _ in self.params]
        self.v_buffers = [None for _ in self.params]

    def step(self):
        """Performs AdamW updates"""

        self.step_count += 1

        for i, param in enumerate(self.params):
            if param.grad is None:  # type: ignore
                continue

            grad = param.grad  # type: ignore
            if isinstance(grad, Tensor):
                grad_data = grad.data
            else:
                grad_data = grad

            if self.m_buffers[i] is None:
                self.m_buffers[i] = np.zeros_like(param.data)  # type: ignore
                self.v_buffers[i] = np.zeros_like(param.data)  # type: ignore

            self.m_buffers[i] = self.beta1 * \
                self.m_buffers[i] + (1 - self.beta1) * grad_data
            self.v_buffers[i] = self.beta2 * self.v_buffers[i] + \
                (1 - self.beta2) * (grad_data ** 2)  # type: ignore

            # compute bias corrections
            bias_correction1 = 1 - self.beta1 ** self.step_count
            bias_correction2 = 1 - self.beta2 ** self.step_count

            m_hat = self.m_buffers[i] / bias_correction1
            v_hat = self.v_buffers[i] / bias_correction2

            param.data = param.data - self.lr * \
                m_hat / (np.sqrt(v_hat) + self.eps)

            if self.weight_decay != 0:
                param.data = param.data * (1 - self.weight_decay * self.lr)
