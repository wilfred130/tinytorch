import numpy as np
from torch.core.tensor import Tensor


TORELENCE = 1e-10
__all__ = ['Sigmoid', 'ReLU', 'Tanh', 'GELU', 'Softmax']


class Sigmoid:
    def paramters(self):
        return []

    def forward(self, x: Tensor) -> Tensor:
        z = np.clip(x.data, -500, 500)
        result_data = np.zeros_like(z)

        pos_mask = z >= 0
        result_data[pos_mask] = 1.0 / (1.0 + np.exp(-z[pos_mask]))

        neg_mask = z < 0
        exp_z = np.exp(z[neg_mask])
        result_data[neg_mask] = exp_z / (1.0 + exp_z)
        return Tensor(result_data)

    def backward(self, grad: Tensor) -> None:
        """Computes gradients (implementation module 06)"""
        pass


class ReLU:
    def forward(self, x: Tensor) -> Tensor:
        result_data = np.maximum(0, x.data)
        return Tensor(result_data)

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def backward(self, grad: Tensor) -> None:
        """Computes gradients (implementation in module 06)"""
        pass

    def parameters(self):
        return []


class Tanh:
    def parameters(self):
        return []

    def forward(self, x: Tensor) -> Tensor:
        result_data = np.tanh(x.data)
        return Tensor(result_data)

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def backward(self, grad: Tensor) -> None:
        """Computes gradients (implementation module 06)"""
        pass


class GELU:
    def paramters(self):
        return []

    def forward(self, x: Tensor) -> Tensor:
        sigmoid_part = 1.0 / (1.0 + np.exp(-1.702 * x.data))
        result_data = x.data * sigmoid_part
        return Tensor(result_data)

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def backward(self, grad: Tensor) -> None:
        """Computes gradients(implementation in module 06)"""
        pass


class Softmax:
    def parameters(self):
        return []

    def forward(self, x: Tensor, dim: int = -1) -> Tensor:
        x_data_max = np.max(x.data, axis=dim, keepdims=True)
        x_max = Tensor(x_data_max)
        x_shifted = x - x_max

        exp_values = Tensor(np.exp(x_shifted.data))
        exp_data_sum = np.sum(exp_values.data, axis=dim, keepdims=True)
        exp_sum = Tensor(exp_data_sum)

        result = exp_values / exp_sum
        return result

    def __call__(self, x: Tensor, dim: int = -1):
        return self.forward(x, dim=dim)

    def backward(self, grad: Tensor) -> None:
        """Computes gradients (implementation in module 06)"""
        pass
