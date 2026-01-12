import numpy as np
from tinytorch.core.tensor import Tensor
from tinytorch.core.activations import ReLU, Sigmoid

XAVIER_SCALE_FACTOR = 1.0
HE_SCALE_FACTOR = 2.0

MIN_DROPOUT_PROB = 0.0
MAX_DROPOUT_PROB = 1.0


class Layer:

    def forward(self, x):
        raise NotImplementedError('Subclass must implement forward()')

    def __call__(self, x, *args, **kwds):
        return self.forward(x, *args, **kwds)

    def parameters(self):
        return []

    def __repr__(self):
        return f'{self.__class__.__name__}()'


class BaseLayer(Layer):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, training=True):
        raise NotImplementedError('Subclass must implement forward()')


class Linear(Layer):

    def __init__(self, in_features, out_features, bias=True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        scale = np.sqrt(XAVIER_SCALE_FACTOR / in_features)
        weights_data = np.random.randn(in_features, out_features) * scale
        self.weight = Tensor(weights_data, requires_grad=True)

        if bias:
            bias_data = np.zeros(out_features)
            self.bias = Tensor(bias_data, requires_grad=True)
        else:
            self.bias = None

    def forward(self, x):
        output = x.matmul(self.weight)
        if self.bias is not None:
            output = output + self.bias
        return output

    def parameters(self):
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params

    def __repr__(self):
        bias_str = f', bias= {self.bias is not None}'
        return f'Linear(in_features= {self.in_features.shape}, \
            out_features= {self.out_features.shape}{bias_str})'


class Dropout(BaseLayer):
    def __init__(self, p=0.5) -> None:
        super().__init__()
        if not MIN_DROPOUT_PROB <= p <= MAX_DROPOUT_PROB:
            raise ValueError(
                f'Dropout probability must be between {MIN_DROPOUT_PROB} and {MAX_DROPOUT_PROB}, got {p}'
            )
        self.p = p

    def forward(self, x, training=True):
        if not training or self.p == MIN_DROPOUT_PROB:
            return x
        if self.p == MAX_DROPOUT_PROB:
            return Tensor(np.zeros_like(x.data))

        keep_p = 1 - self.p
        mask = np.random.random(x.data.shape) < keep_p
        mask_tensor = Tensor(mask.astype(np.float32))

        scale = Tensor(np.array(1.0 / keep_p))

        output = x * mask_tensor * scale
        return output

    def __call__(self, x, training=True):
        return self.forward(x, training)

    def parameters(self):
        return []

    def __repr__(self):
        return f'Dropout(p={self.p})'


class Sequential:
    def __init__(self, *layers) -> None:
        if len(layers) == 1 and isinstance(layers[0], (tuple, list)):
            self.layers = list(layers[0])
        else:
            self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def __repr__(self):
        layer_repr = ''.join([repr(layer) for layer in self.layers])
        return f'Sequential({layer_repr})'
