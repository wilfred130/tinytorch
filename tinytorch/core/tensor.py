import numpy as np


class Tensor:
    def __init__(self, data):
        self.data = np.array(data, dtype=np.float32)
        self.size = self.data.size
        self.shape = self.data.shape
        self.dtype = self.data.dtype

    def __repr__(self):
        return f'Tensor(data= {self.data}, shape= {self.shape})'

    def __str__(self):
        return f'Tensor({self.data})'

    def numpy(self):
        return self.data

    def __add__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data + other.data)
        else:
            return Tensor(self.data + other)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data - other.data)
        else:
            return Tensor(self.data - other)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data * other.data)
        else:
            return Tensor(self.data * other)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data / other.data)
        else:
            return Tensor(self.data / other)

    def matmul(self, other):
        if not isinstance(other, Tensor):
            raise TypeError(f'Expected a Tensor, but got {type(other)}')
        if self.shape == () or other.shape == ():
            return Tensor(self.data * other.data)
        if len(self.shape) == 0 or len(other.shape) == 0:
            return Tensor(self.data * other.data)
        if len(self.shape) >= 2 and len(other.shape) >= 2:
            if self.shape[-1] != other.shape[-2]:
                raise ValueError(
                    f"Cannot perform matrix multiplication: {self.shape} @ {other.shape}. "
                    f"Inner dimensions must match: {self.shape[-1]} ≠ {other.shape[-2]}"
                )
        a = self.data
        b = other.data
        if len(a.shape) == 2 and len(b.shape) == 2:
            M, K = a.shape
            K2, N = b.shape
            result_data = np.zeros((M, N), dtype=a.dtype)
            for i in range(M):
                for j in range(N):
                    result_data[i, j] = np.dot(a[i, :], b[:, j])
        else:
            result_data = np.matmul(a, b)
        return Tensor(result_data)

    def __matmul__(self, other):
        return self.matmul(other)

    def transpose(self, dim0=None, dim1=None):
        if dim0 is None and dim1 is None:
            if len(self.shape) == 1:
                return Tensor(self.data.copy())
            else:
                axes = list(range(len(self.shape)))
                axes[-1], axes[-2] = axes[-2], axes[-1]
                transposed_data = np.transpose(self.data, axes)
        else:
            if dim0 is None or dim1 is None:
                raise ValueError(
                    'Both dimensions must be provided'
                )
            axes = list(range(len(self.shape)))
            axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
            transposed_data = np.transpose(self.data, axes)
        return Tensor(transposed_data)

    def __getitem__(self, key):
        result_data = self.data[key]
        if not isinstance(result_data, np.ndarray):
            result_data = np.array(result_data)
        return Tensor(result_data)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            new_shape = tuple(shape[0])
        else:
            new_shape = shape
        if -1 in new_shape:
            if new_shape.count(-1) > 1:
                raise ValueError(
                    'Can only specify one unknown dimension with -1'
                )
            known_size = 1
            unknown_idx = new_shape.index(-1)
            for i, dim in enumerate(new_shape):
                if i != unknown_idx:
                    known_size *= dim
            unknown_size = self.size // known_size
            new_shape = list(new_shape)
            new_shape[unknown_idx] = unknown_size
            new_shape = tuple(new_shape)
        if np.prod(new_shape) != self.size:
            target_size = int(np.prod(new_shape))
            raise ValueError(
                f"Total elements must match: {self.size} ≠ {target_size}"
            )

        reshaped_data = np.reshape(self.data, new_shape)
        return Tensor(reshaped_data)

    def sum(self, axis=None, keepdims=False):
        result = np.sum(self.data, axis=axis, keepdims=keepdims)
        return Tensor(result)

    def mean(self, axis=None, keepdims=False):
        result = np.mean(self.data, axis=axis, keepdims=keepdims)
        return Tensor(result)

    def max(self, axis=None, keepdims=False):
        result = np.max(self.data, axis=axis, keepdims=keepdims)
        return Tensor(result)
