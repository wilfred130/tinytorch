# TinyTorch

**TinyTorch** is a lightweight, educational deep learning framework inspired by PyTorch.  
It is built from scratch to demonstrate how modern deep learning systems work internally.

TinyTorch focuses on:
- Tensor operations and automatic differentiation
- Core neural network components
- Training pipelines without hidden abstractions
- Mathematical clarity and reproducibility

All modules are accessible via the `tinytorch.core.*` namespace.

---

## Features

- 🔢 Tensor class with automatic differentiation
- 🔗 Computational graph construction
- ⚡ Activation functions
- 🧱 Neural network layers
- 📉 Loss functions
- 📦 Data loading utilities
- 🚀 Optimizers
- 🔁 Training and evaluation loops

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/tinytorch.git
cd tinytorch
```

## Project Structure

```bash
tinytorch/
│
├── core/
│   ├── tensor.py        # Tensor object and autograd engine
│   ├── activations.py  # Activation functions
│   ├── layers.py       # Neural network layers
│   ├── losses.py       # Loss functions
│   ├── optimizers.py   # Optimization algorithms
│   ├── dataloader.py   # Dataset and DataLoader
│   ├── training.py     # Training utilities
│
├── examples/
│   ├── linear_regression.py
│   ├── classification.py
│
├── tests/
├── README.md
└── setup.py

```

## Core modules

```bash
tinytorch.core.tensor
```

The foundation of TinyTorch.

Features:

Multi-dimensional tensors

Reverse-mode automatic differentiation

Gradient tracking

Backpropagation through computational graphs

Example

```bash

from tinytorch.core.tensor import Tensor
from tinytorch.core.autograd import enable_autograd
enable_autograd() # <-- Enables development of computation graph

x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x.sum()
y.backward()

print(x.grad)

```

```bash
tinytorch.core.activations
```

Supported activation functions:

ReLU

Sigmoid

Tanh

Softmax

GELU

```bash
from tinytorch.core.activations import ReLU

relu = ReLU()
output = relu(x)

```

```bash
tinytorch.core.layers
```
Supports

Linear layers

Dropout layers

Layer composition

```bash
from tinytorch.core.layers import Linear

layer = Linear(in_features=3, out_features=2)
output = layer(x)

```

```bash
tinytorch.core.losses
```

Supported loss functions:

Mean Squared Error (MSE)

Cross Entropy Loss

Binary Cross Entropy

```bash
from tinytorch.core.losses import MSELoss

loss_fn = MSELoss()
loss = loss_fn(predictions, targets)

```
```bash
tinytorch.core.optimizers
```
Supported optimizers:

SGD

Adam

AdamW

```bash
from tinytorch.core.optimizers import SGD

optimizer = SGD(model.parameters(), lr=0.01)
optimizer.step()
optimizer.zero_grad()

```
