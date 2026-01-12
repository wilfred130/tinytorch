import numpy as np
from typing import Optional

from tinytorch.core.tensor import Tensor
from tinytorch.core.activations import ReLU
from tinytorch.core.layers import Linear

# constant for numeric stability
EPSILON = 1e-7


def log_softmax(x: Tensor, dim: int = -1) -> Tensor:
    max_vals = np.max(x.data, axis=dim, keepdims=True)
    shifted = x.data - max_vals
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=dim, keepdims=True))
    result = x.data - max_vals - log_sum_exp
    return Tensor(result)


class MSELoss:

    def __init__(self) -> None:
        pass

    def forward(self, predictions: Tensor, target: Tensor) -> Tensor:
        diff = predictions.data - target.data
        squared_diff = diff ** 2
        mse = np.mean(squared_diff)
        return Tensor(mse)

    def __call__(self, prediction: Tensor, target: Tensor) -> Tensor:
        return self.forward(prediction, target)

    def backward(self, grad: Tensor) -> None:
        """Compute gradients (implementation in module 06)"""
        pass


class CrossEntropyLoss:
    def __init__(self) -> None:
        pass

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        log_probs = log_softmax(logits, dim=-1)

        batch_size = logits.shape[0]
        target_indices = target.data.astype('int')

        selected_probs = log_probs.data[np.arange(batch_size), target_indices]
        cross_entropy = -np.mean(selected_probs)

        return Tensor(cross_entropy)

    def __call__(self, logits: Tensor, target: Tensor) -> Tensor:
        return self.forward(logits, target)

    def backward(self, grads: Tensor) -> None:
        pass


class BinaryCrossEntropyLoss:
    def __init__(self) -> None:
        pass

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        eps = EPSILON
        clamped_preds = np.clip(predictions.data, eps, 1 - eps)

        log_preds = np.log(clamped_preds)
        log_one_minus_pred = np.log(1 - clamped_preds)

        bce_per_sample = -(targets.data * log_preds +
                           (1 - targets.data) * log_one_minus_pred)
        bce = np.mean(bce_per_sample)
        return Tensor(bce)

    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return self.forward(predictions, targets)

    def backward(self, grads: Tensor) -> None:
        pass
