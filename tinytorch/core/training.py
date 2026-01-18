import numpy as np
import os
from typing import List, Dict, Optional, Tuple
import pickle
from pathlib import Path
import time
import sys

from tinytorch.core.tensor import Tensor
from tinytorch.core.layers import Linear
from tinytorch.core.losses import MSELoss, CrossEntropyLoss
from tinytorch.core.optimizers import SGD, AdamW

# CONSTANT FOR LEARNING RATE SCHEDULING DEFAULTS
DEFAULT_MIN_LR = 0.01
DEFAULT_MAX_LR = 0.1
DEFAULT_TOTAL_EPOCHS = 100


class CosineSchedule:
    """Cosine annealing learning rate schedule"""

    def __init__(self, max_lr: float = DEFAULT_MAX_LR, min_lr: float = DEFAULT_MIN_LR,
                 total_epochs: int = DEFAULT_TOTAL_EPOCHS):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.total_epochs = total_epochs

    def get_lr(self, epoch: int) -> float:
        """Get learning rate for current epoch"""
        if epoch >= self.total_epochs:
            return self.min_lr

        # COSINE ANNEALING FORMULA
        cosine_factor = (1 + np.cos(np.pi * epoch / self.total_epochs)) / 2
        return self.min_lr + (self.max_lr - self.min_lr) * cosine_factor


def clip_grad_norm(parameters: List, max_norm: float = 1.0) -> float:
    """Clip gradients by global norm to prevent gradient explosion"""

    if not parameters:
        return 0.0

    total_norm = 0.0
    for param in parameters:
        if param is not None:
            grad = param.grad

            if isinstance(grad, np.ndarray):
                grad_data = grad
            else:
                grad_data = grad.data

            total_norm += np.sum(grad_data ** 2)
    total_norm = np.sqrt(total_norm)

    # check if clipping is necessary
    if total_norm > max_norm:
        clip_coef = max_norm / total_norm

        for param in parameters:
            if param.grad is not None:
                if isinstance(param.grad, np.ndarray):
                    param.grad = param.grad * clip_coef

                else:
                    param.grad.data = param.grad.data * clip_coef

    return float(total_norm)


class Trainer:
    """Complete training ochestrator for Neural Networks"""

    def __init__(self, model, optimizer, loss_fn, scheduler=None, grad_clip_norm=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.grad_clip_norm = grad_clip_norm

        # Training state
        self.epoch = 0
        self.step = 0
        self.training_mode = True

        # History tracking
        self.history = {
            "training_loss": [],
            "eval_loss": [],
            "learning_rates": []
        }

    def train_epoch(self, dataloader, accumulation_steps):
        """
        Train model for  one epoch

        Args:
            dataloader: Iterator yielding (input, target)
            accumulation_steps: num of accumulated batches before update
        """

        self.training_mode = True
        self.model.training = True

        total_loss = 0.0
        num_batches = 0
        accumulated_loss = 0

        for batch_idx, (input, target) in enumerate(dataloader):

            output = self.model.forward(input)
            loss = self.loss_fn.forward(output, target)

            scaled_loss = loss.data / accumulation_steps
            accumulated_loss += scaled_loss

            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                if self.grad_clip_norm is not None:
                    params = self.model.parameters()
                    clip_grad_norm(params, self.grad_clip_norm)

                self.optimizer.step()
                self.optimizer.zero_grad()
                total_loss += accumulated_loss
                accumulated_loss = 0.0
                self.step += 1
                num_batches += 1

        # Handle remaining gradients
        if accumulated_loss > 0:
            if self.grad_clip_norm is not None:
                params = self.model.parameters()
                clip_grad_norm(params, self.grad_clip_norm)

            self.optimizer.step()
            self.optimizer.zero_grad()
            total_loss += accumulated_loss
            self.step += 1
            num_batches += 1

        avg_loss = total_loss / np.maximum(num_batches, 1)
        self.history['training_loss'].append(avg_loss)

        # update scheduler
        if self.scheduler is not None:
            current_lr = self.scheduler.get_lr(self.epoch)
            self.optimizer.lr = current_lr
            self.history['learning_rates'].append(current_lr)

        self.epoch += 1
        return avg_loss

    def evaluate(self, dataloader):
        """
        Evaluate model on dataset without updating gradients

        Args:
            dataloader: Iterator yielding (input, target) batches

        Returns:
            Avergae loss and accuracy
        """

        self.model.training = False
        self.model_training = False

        total_loss = 0.0
        correct = 0
        total = 0

        for input, target in enumerate(dataloader):
            output = self.model.forward(input)
            loss = self.loss_fn.forward(output, target)

            total_loss += loss.data

            if len(output.shape) > 1:       # Multi-class
                predictions = np.argmax(output, axis=1)
                if len(target.shape) == 1:
                    correct += np.sum(predictions == target.data)
                else:
                    correct += np.sum(predictions ==
                                      np.argmax(target.data, axis=1))
                total += len(predictions)
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        self.history['eval_loss'].append(avg_loss)
        return avg_loss, accuracy
