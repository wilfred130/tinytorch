import numpy as np
import os
from typing import List, Optional, Tuple

from tinytorch.core.tensor import Tensor

EPSILON = 1e-7


class Function:

    def __init__(self, *tensors):
        self.saved_tensors = tensors
        self.next_functions = []

        for t in tensors:
            if isinstance(t, Tensor) and t.requires_grad:  # type: ignore
                if getattr(t, '_grad_fn', None) is not None:
                    self.next_functions.append(t._grad_fn)  # type: ignore

    def apply(self, grad_outputs):
        raise NotImplementedError(
            'Each Function must implement apply() method')


class AddBackward(Function):
    """Gradient computation for tensor addition"""

    def apply(self, grad_outputs):
        a, b = self.saved_tensors
        grad_a = grad_b = None

        if isinstance(a, Tensor) and a.requires_grad:  # type: ignore
            grad_a = grad_outputs

        if isinstance(b, Tensor) and b.requires_grad:  # type: ignore
            grad_b = grad_outputs

        return grad_a, grad_b


class MulBackward(Function):
    """Gradient computation for tensor elementwise multiplication"""

    def apply(self, grad_outputs):
        a, b = self.saved_tensors
        grad_a = grad_b = None

        if isinstance(a, Tensor) and a.requires_grad:  # type: ignore
            if isinstance(b, Tensor):
                grad_a = grad_outputs * b.data
            else:
                grad_a = grad_outputs * b

        if isinstance(b, Tensor) and b.requires_grad:  # type: ignore
            grad_b = grad_outputs * a.data

        return grad_a, grad_b


class SubBackward(Function):

    """Gradient computation for tensor subtraction"""

    def apply(self, grad_outputs):
        a, b = self.saved_tensors
        grad_a = grad_b = None

        if isinstance(a, Tensor) and a.requires_grad:  # type: ignore
            grad_a = grad_outputs

        if isinstance(b, Tensor) and b.requires_grad:  # type: ignore
            grad_b = -grad_outputs

        return grad_a, grad_b


class DivBackward(Function):
    """Gradient computation for tensor divisin"""

    def apply(self, grad_outputs):
        a, b = self.saved_tensors
        grad_a = grad_b = None

        if isinstance(a, Tensor) and a.requires_grad:  # type: ignore
            if isinstance(b, Tensor):
                grad_a = grad_outputs / b.data

            else:
                grad_a = grad_outputs / b

        if isinstance(b, Tensor) and b.requires_grad:  # type: ignore
            grad_b = -grad_outputs * a.data / (b.data ** 2)

        return grad_a, grad_b


class MatmulBackward(Function):
    """Gradient computation for matrix multiplication i.e A @ B"""

    def apply(self, grad_outputs):
        a, b = self.saved_tensors
        grad_a = grad_b = None

        if isinstance(a, Tensor) and a.requires_grad:  # type: ignore
            if b.data.ndim >= 2:
                b_T = np.swapaxes(b.data, -2, -1)
            else:
                b_T = b.data.T

            grad_a = np.matmul(grad_outputs, b_T)

        if isinstance(b, Tensor) and b.requires_grad:  # type: ignore
            if a.data.ndim >= 2:
                a_T = np.swapaxes(a.data, -2, -1)
            else:
                a_T = a.data.T

            grad_b = np.matmul(a_T, grad_outputs)

        return grad_a, grad_b


class TransposeBackward(Function):
    """Computes gradients for transpose"""

    def __init__(self, tensor, dim0, dim1):
        super().__init__(tensor)
        self.dim0 = dim0
        self.dim1 = dim1

    def apply(self, grad_outputs):
        x, = self.saved_tensors
        grad_x = None

        if isinstance(x, Tensor) and x.requires_grad:  # type: ignore
            if self.dim0 is None and self.dim1 is None:
                if grad_outputs.ndim < 2:
                    grad_x = grad_outputs.copy()
                else:
                    axes = list(range(grad_outputs.ndim))
                    axes[-2], axes[-1] = axes[-1], axes[-2]
                    grad_x = np.transpose(grad_outputs, axes)
            else:
                # transpose for specific dimensions
                axes = list(range(grad_outputs.ndim))
                axes[self.dim0], axes[self.dim1] = axes[self.dim1], axes[self.dim0]
                grad_x = np.transpose(grad_outputs, axes)
        return (grad_x, )


class PermuteBackward(Function):
    """Gradient computation for arbitary axis permutation"""

    def __init__(self, tensors, axes):
        super().__init__(tensors)
        self.axes = axes
        self.inverse_axes = tuple(np.argsort(axes))

    def apply(self, grad_outputs):
        x, = self.saved_tensors
        grad_x = None

        if isinstance(x, Tensor) and x.requires_grad:  # type: ignore
            grad_x = np.transpose(grad_outputs, self.inverse_axes)
        return (grad_x,)


class EmbeddingBackward(Function):
    """Gradient computation for embedding lookup operation"""

    def __init__(self, weight, indices):
        super().__init__(weight)
        self.indices = indices

    def apply(self, grad_outputs):
        weight, = self.saved_tensors
        grad_weight = None

        if isinstance(weight, Tensor) and weight.requires_grad:  # type: ignore
            grad_weight = np.zeros_like(weight.data)

            indices_flat = self.indices.data.astype(int).flatten()
            grad_output_reshaped = grad_outputs.reshape(
                -1, grad_outputs.shape[-1])

            np.add.at(grad_weight, indices_flat, grad_output_reshaped)

        return (grad_weight, )


class SliceBackward(Function):
    """Gradient computation for tensor slicing/ indexing operation"""

    def __init__(self, tensor, key):
        super().__init__(tensor)
        self.key = key
        self.original_shape = tensor.shape

    def apply(self, grad_outputs):
        tensor, = self.saved_tensors
        grad_input = None

        if isinstance(tensor, Tensor) and tensor.requires_grad:  # type: ignore
            grad_input = np.zeros(self.original_shape, dtype=np.float32)
            grad_input[self.key] = grad_outputs

        return (grad_input, )


class ReshapeBackward(Function):
    """Gradient computation for reshape operations"""

    def __init__(self, tensor, original_shape):
        super().__init__(tensor)
        self.original_shape = original_shape

    def apply(self, grad_outputs):
        x, = self.saved_tensors
        grad_x = None

        if isinstance(x, Tensor) and x.requires_grad:  # type: ignore
            grad_x = grad_outputs.reshape(self.original_shape)

        return (grad_x,)


class SumBackward(Function):
    """Computes gradients for sum operation"""

    def apply(self, grad_outputs):
        tensor, = self.saved_tensors

        if isinstance(tensor, Tensor) and tensor.requires_grad:  # type: ignore
            return (np.ones_like(tensor.data) * grad_outputs,)

        return (None,)


class ReLUBackward(Function):
    """Gradiant computation for ReLU activation"""

    def __init__(self, input_tensor):
        super().__init__(input_tensor)

    def apply(self, grad_outputs):
        tensor, = self.saved_tensors

        if isinstance(tensor, Tensor) and tensor.requires_grad:  # type: ignore
            relu_grad = (tensor.data > 0).astype(np.float32)
            return (grad_outputs * relu_grad,)
        return (None,)


class SigmoidBackward(Function):
    """Gradient computation for sigmoid activation"""

    def __init__(self, input_tensor, output_tensor):
        super().__init__(input_tensor)
        self.output_data = output_tensor.data

    def apply(self, grad_outputs):
        tensor, = self.saved_tensors
        if isinstance(tensor, Tensor) and tensor.requires_grad:  # type: ignore
            sigmoid_grad = self.output_data * (1 - self.output_data)
            return (grad_outputs * sigmoid_grad,)
        return (None,)


class SoftmaxBackward(Function):
    """Gradient Computation for Sotfmax activation"""

    def __init__(self, input_tensor, output_tensor, dim=-1):
        super().__init__(input_tensor)
        self.output_data = output_tensor.data
        self.dim = dim

    def apply(self, grad_outputs):
        tensor, = self.saved_tensors

        if isinstance(tensor, Tensor) and tensor.requires_grad:  # type: ignore
            sum_term = np.sum(grad_outputs * self.output_data,
                              axis=self.dim, keepdims=True)
            grad_x = self.output_data * (grad_outputs - sum_term)
            return (grad_x, )
        return (None,)


class GELUBackward(Function):
    """Gradient computation for GELU Activation"""

    def __init__(self, input_tensor):
        super().__init__(input_tensor)

    def apply(self, grad_outputs):
        tensor, = self.saved_tensors

        if isinstance(tensor, Tensor) and tensor.requires_grad:  # type: ignore
            x = tensor.data

            sqrt_2_over_pi = np.sqrt(2 / np.pi)
            x_cubed = x ** 3
            tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
            tanh_out = np.tanh(tanh_arg)
            sech_squared = 1 - tanh_out**2

            d_tanh_arg = sqrt_2_over_pi * (1 + 0.134145 * x**2)
            gelu_grad = 0.5 * (1 + tanh_out) + 0.5 * x * \
                sech_squared * d_tanh_arg
            return (grad_outputs * gelu_grad,)
        return (None,)


class MSEBackward(Function):
    """Gradient computation for MSE Loss Function"""

    def __init__(self, predictions, target):
        super().__init__(predictions)
        self.target_data = target.data
        self.num_samples = np.size(target.data)

    def apply(self, grad_outputs):
        predictions, = self.saved_tensors

        if isinstance(predictions, Tensor) and predictions.requires_grad:  # type: ignore
            grad = 2 * (predictions.data - self.target_data) / self.num_samples
            return (grad * grad_outputs,)
        return (None,)


class BCEBackward(Function):
    """Gradient compuation for BCE Loss Function"""

    def __init__(self, predictions, target):
        super().__init__(predictions)
        self.target_data = target.data
        self.num_samples = np.size(target.data)

    def apply(self, grad_outputs):
        predictions, = self.saved_tensors

        if isinstance(predictions, Tensor) and predictions.requires_grad:  # type: ignore
            eps = EPSILON
            p = np.clip(predictions.data, eps, 1 - eps)
            y = self.target_data

            grad = (p - y) / (p * (1 - p) * self.num_samples)
            return (grad * grad_outputs, )
        return (None, )


class CEBackward(Function):
    """Gradient computation for Cross Entropy"""

    def __init__(self, logits, targets):
        super().__init__(logits)
        self.target_data = targets.data.astype(int)
        self.batch_size = logits.data.shape[0]
        self.num_classes = logits.data.shape[1]

    def apply(self, grad_outputs):
        logits, = self.saved_tensors

        if isinstance(logits, Tensor) and logits.requires_grad:  # type: ignore
            logits_data = logits.data
            max_logits = np.max(logits_data, axis=1, keepdims=True)
            exp_logits = np.exp(logits_data - max_logits)
            softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            one_hot = np.zeros(
                (self.batch_size, self.num_classes), dtype=np.float32)
            one_hot[np.arange(self.batch_size), self.target_data] = 1.0

            grad = (softmax - one_hot) / self.batch_size
            return (grad_outputs * grad, )
        return (None, )


def enable_autograd(quiet=False):
    """Enables gradient tracking for all Tensor Operations"""

    if hasattr(Tensor, '_autograd_enabled'):
        return

    _original_init = Tensor.__init__

    def gradient_aware_init(self, data, requires_grad=False):
        """Extended init that support gradient tracking"""

        _original_init(self, data)
        self.requires_grad = requires_grad
        self.grad = None

    Tensor.__init__ = gradient_aware_init

    _original_add = Tensor.__add__
    _original_mul = Tensor.__mul__
    _original_sub = Tensor.__sub__
    _original_div = Tensor.__truediv__
    _original_getitem = Tensor.__getitem__

    _original_matmul = Tensor.matmul
    _original_transpose = Tensor.transpose
    _original_reshape = Tensor.reshape

    def _get_requires_grad(tensor):
        """Safely gets requires_grad, defaults to False for pre-autograd"""

        return getattr(tensor, 'requires_grad', False) if isinstance(tensor, Tensor) else False

    def _ensure_grad_attrs(tensor):
        if isinstance(tensor, Tensor):
            if not hasattr(tensor, 'requires_grad'):
                tensor.requires_grad = False  # type: ignore
            if not hasattr(tensor, 'grad'):
                tensor.grad = None  # type: ignore

    def tracked_add(self, other):
        """Addition with gradient tracking"""

        _ensure_grad_attrs(self)

        if not isinstance(other, Tensor):
            other = Tensor(other)
        _ensure_grad_attrs(other)

        result = _original_add(self, other)
        _ensure_grad_attrs(result)

        if _get_requires_grad(self) or _get_requires_grad(other):
            result.requires_grad = True  # type: ignore
            result._grad_fn = AddBackward(self, other)  # type: ignore
        return result

    def track_mul(self, other):
        """Multiplication with gradient tracking"""
        _ensure_grad_attrs(self)

        if not isinstance(other, Tensor):
            other_tensor = Tensor(other)
        else:
            other_tensor = other
        _ensure_grad_attrs(other_tensor)

        result = _original_mul(self, other)
        _ensure_grad_attrs(result)

        if _get_requires_grad(self) or _get_requires_grad(other_tensor):
            result.requires_grad = True  # type: ignore
            result._grad_fn = MulBackward(self, other)  # type: ignore

        return result

    def track_matmul(self, other):
        """Matrix multiplication with gradeint tracking"""
        _ensure_grad_attrs(self)
        _ensure_grad_attrs(other)

        result = _original_matmul(self, other)
        _ensure_grad_attrs(result)

        if _get_requires_grad(self) or _get_requires_grad(other):
            result.requires_grad = True  # type: ignore
            result._grad_fn = MatmulBackward(self, other)  # type: ignore

        return result

    def track_transpose(self, dim0=None, dim1=None):
        """Tensor transpose with gradient tracking"""
        _ensure_grad_attrs(self)

        result = _original_transpose(self, dim0, dim1)
        _ensure_grad_attrs(result)

        if _get_requires_grad(self):
            result.requires_grad = True  # type: ignore
            result._grad_fn = TransposeBackward(  # type: ignore
                self, dim0, dim1)

        return result

    def track_reshape(self, *shape):
        """Reshape with gradient tracking"""
        _ensure_grad_attrs(self)
        original_shape = self.shape

        result = _original_reshape(self, *shape)
        _ensure_grad_attrs(result)

        if _get_requires_grad(self):
            result.requires_grad = True  # type: ignore
            result._grad_fn = ReshapeBackward(  # type: ignore
                self, original_shape)

        return result

    def track_sub(self, other):
        """Tensor subtraction with   gradeint tracking"""
        _ensure_grad_attrs(self)

        if not isinstance(other, Tensor):
            other = Tensor(other)
        _ensure_grad_attrs(self)

        result = _original_sub(self, other)
        _ensure_grad_attrs(result)

        if _get_requires_grad(self) or _get_requires_grad(other):
            result.requires_grad = True  # type: ignore
            result._grad_fn = SubBackward(self, other)  # type: ignore

        return result

    def track_div(self, other):
        """True div with grdient tacking"""

        _ensure_grad_attrs(self)

        if not isinstance(other, Tensor):
            other = Tensor(other)
        _ensure_grad_attrs(other)

        result = _original_div(self, other)
        _ensure_grad_attrs(result)

        if _get_requires_grad(self) or _get_requires_grad(other):
            result.requires_grad = True  # type: ignore
            result._grad_fn = DivBackward(self, other)  # type: ignore

        return result

    def track_getitem(self, key):
        """Indexin/slicing with gradeint tracking"""

        _ensure_grad_attrs(self)
        result = _original_getitem(self, key)
        _ensure_grad_attrs(result)

        if _get_requires_grad(self):
            result.requires_grad = True  # type: ignore
            result._grad_fn = SliceBackward(self, key)  # type: ignore
        return result

    def sum_op(self, axis=None, keepdims=False):
        """Sum operation with gradient tracking"""

        _ensure_grad_attrs(self)
        result_data = np.sum(self.data, axis=axis, keepdims=keepdims)
        result = Tensor(result_data)

        if _get_requires_grad(self):
            result.requires_grad = True  # type: ignore
            result._grad_fn = SumBackward(self)  # type: ignore
        return result

    def backward(self, gradient=None):
        """Computes gradients via backward propagation"""

        _ensure_grad_attrs(self)

        if not _get_requires_grad(self):
            return

        if gradient is None:
            if self.data.size == 1:
                gradient = np.ones_like(self.data)
            else:
                raise ValueError(
                    f'backward() called on non-scaler tensor without gradient argument\n'
                    f'Tensor shape {self.shape}\n'
                    f'for non-scaler outputs, you must provide gradient for the next layer\n'
                    f'Fix: Call backward(gradient) with gradient tensor from the loss function'
                )
        if self.grad is None:
            self.grad = np.zeros_like(self.data)

        # handle broadcasting: sum gradients to match self.grad
        if gradient.shape != self.grad.shape:
            while gradient.ndim > self.grad.ndim:
                gradient = gradient.sum(axis=0)

            for i in range(gradient.ndim):
                if self.grad.shape[i] == 1 and gradient.shape[i] != 1:
                    gradient = gradient.sum(axis=i, keepdims=True)

        self.grad += gradient

        # propagate gardients through the computational graph
        # __grad_fn is set by the autograd enhancement when tensor is created from an operation
        grad_fn = getattr(self, '_grad_fn', None)
        if grad_fn is not None:
            grads = grad_fn.apply(gradient)
            for tensor, grad in zip(grad_fn.saved_tensors, grads):
                if isinstance(tensor, Tensor) and tensor.requires_grad and grad is not None:  # type: ignore
                    tensor.backward(grad)  # type: ignore

    def zero_grad(self):
        """Reset gradients to zero"""
        self.grad = None

    # install enhanced operations
    Tensor.__add__ = tracked_add
    Tensor.__sub__ = track_sub
    Tensor.__mul__ = track_mul
    Tensor.__getitem__ = track_getitem
    Tensor.__truediv__ = track_div
    Tensor.matmul = track_matmul
    Tensor.transpose = track_transpose
    Tensor.reshape = track_reshape
    Tensor.sum = sum_op
    Tensor.backward = backward  # type: ignore
    Tensor.zero_grad = zero_grad  # type: ignore

    # patch activations to accept gradient tracking
    try:
        from tinytorch.core.activations import ReLU, GELU, Sigmoid, Softmax
        from tinytorch.core.losses import BinaryCrossEntropyLoss, CrossEntropyLoss, MSELoss

        # store the original methods
        _original_sigmoid_forward = Sigmoid.forward
        _original_relu_forward = ReLU.forward
        _original_softmax_forward = Softmax.forward
        _original_gelu_forward = GELU.forward
        _original_bce_forward = BinaryCrossEntropyLoss.forward
        _original_ce_forward = CrossEntropyLoss.forward
        _original_mse_forward = MSELoss.forward

        def tracked_sigmoid_forward(self, x):
            """Sigmoid with gradient tracking"""
            result = _original_sigmoid_forward(self, x)
            if x.requires_grad:
                result.requires_grad = True  # type: ignore
                result._grad_fn = SigmoidBackward(x, result)  # type: ignore
            return result

        def tracked_relu_forward(self, x):
            """Relu with gradient tracking"""
            result = _original_relu_forward(self, x)
            if x.requires_grad:
                result.requires_grad = True  # type: ignore
                result._grad_fn = ReLUBackward(x)  # type: ignore
            return result

        def tracked_softmax_forward(self, x, dim=-1):
            """Softmax with gradient tracking"""
            result = _original_softmax_forward(self, x, dim)
            if x.requires_grad:
                result.requires_grad = True  # type: ignore
                result._grad_fn = SoftmaxBackward(x, result)  # type: ignore
            return result

        def tracked_gelu_forward(self, x):
            """GELU with gradeint tracking"""
            result = _original_gelu_forward(self, x)
            if x.requires_grad:
                result.requires_grad = True  # type: ignore
                result._grad_fn = GELUBackward(x, result)  # type: ignore
            return result

        def tracked_bce_forward(self, predictions, targets):
            """BCE with gradient tracking"""
            result = _original_bce_forward(self, predictions, targets)
            if predictions.requires_grad:
                result.requires_grad = True  # type: ignore
                result._grad_fn = BCEBackward(  # type: ignore
                    predictions, targets)
            return result

        def tracked_mse_forward(self, predictions, target):
            """MSE with gradient tracking"""
            result = _original_mse_forward(self, predictions, target)
            if predictions.requires_grad:
                result.requires_grad = True  # type: ignore
                result._grad_fn = MSEBackward(  # type: ignore
                    predictions, target)
            return result

        def tracked_ce_forward(self, logits, target):
            """Cross entropy with gradient tracking"""
            result = _original_ce_forward(self, logits, target)
            if logits.requires_grad:
                result.requires_grad = True  # type: ignore
                result._grad_fn = CEBackward(logits, target)  # type: ignore
            return result

        # Install patched methods
        Sigmoid.forward = tracked_sigmoid_forward
        ReLU.forward = tracked_relu_forward
        Softmax.forward = tracked_softmax_forward
        GELU.forward = tracked_gelu_forward
        BinaryCrossEntropyLoss.forward = tracked_bce_forward
        CrossEntropyLoss.forward = tracked_ce_forward
        MSELoss.forward = tracked_mse_forward

    except ImportError:
        """Activations/losses not available. Happens during module development"""
        pass

    # Mark as enabled
    Tensor._autograd_enabled = True  # type: ignore

    if not quiet:
        print("✅ Autograd enabled! Tensors  now track gradients")
        print("    - Operations build computation graphs")
        print("    - backward() computes gradients")
        print("    - requires_grad= True enbales tracking")


enable_autograd(quiet=True)
