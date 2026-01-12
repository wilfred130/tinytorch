import numpy as np
import time
import random
import sys
from typing import Iterator, Tuple, List, Optional, Union, Any
from abc import ABC, abstractmethod

from tinytorch.core.tensor import Tensor


class Dataset(ABC):

    @abstractmethod
    def __len__(self) -> int:
        """Returns total number of samples in dataset"""
        pass

    @abstractmethod
    def __getitem__(self, index) -> Any:
        """Returns sample at given index"""
        pass


class TensorDataset(Dataset):

    def __init__(self, *tensors) -> None:
        assert len(tensors) > 0, "Must provide atleast one tensor"
        self.tensors = tensors

        first_size = len(tensors[0].data)
        for i, tensor in enumerate(tensors):
            if len(tensor.data) != first_size:
                raise ValueError(
                    f'All tensors must have same first dimension'
                    f'Tensor {0} : {first_size}, Tensor {i}: {len(tensor.data)}'
                )

    def __len__(self) -> int:
        return len(self.tensors[0].data)

    def __getitem__(self, idx: int) -> Tuple[Tensor, ...]:
        if idx >= len(self) or idx < 0:
            raise IndexError(
                f'Index {idx} out of range of dataset of size {len(self)}')
        return tuple(Tensor(tensor.data[idx]) for tensor in self.tensors)


class DataLoader:

    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = False) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        """Returns number of batches per epoch"""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> None:
        pass
