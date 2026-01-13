import numpy as np
import random
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

    def __len__(self) -> int:
        """Returns number of batches per epoch"""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator:
        indices = list(range(len(self.dataset)))

        if self.shuffle:
            random.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i: i + self.batch_size]
            batch_data = [self.dataset[idx] for idx in batch_indices]

            yield self._collate_batch(batch_data)

    def _collate_batch(self, batch: List[Tuple[Tensor, ...]]):
        if len(batch) == 0:
            return ()

        num_tensors = len(batch[0])
        batched_tensors = []

        for tensor_idx in range(num_tensors):
            tensor_list = [sample[tensor_idx].data for sample in batch]

            batch_data = np.stack(tensor_list, axis=0)
            batched_tensors.append(Tensor(batch_data))

        return tuple(batched_tensors)


class RandomHorizontalFlip:

    def __init__(self, p=0.5) -> None:

        if not 0.0 <= p <= 1.0:
            raise ValueError(f'Probability must be between 0 and 1')

        self.p = p

    def __call__(self, x):
        if np.random.random() < self.p:
            if isinstance(x, Tensor):
                return Tensor(np.flip(x.data, axis=-1).copy())

            else:
                return np.flip(x, axis=-1).copy()
        return x


class RandomCrop:

    def __init__(self, size, padding=4) -> None:
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

        self.padding = padding

    def __call__(self, x):
        is_tensor = isinstance(x, Tensor)
        data = x.data if is_tensor else x

        target_h, target_w = self.size

        if len(data.shape) == 2:
            h, w = data.shape
            padded = np.pad(data, self.padding,
                            mode='constant', constant_values=0)

            top = np.random.randint(0, 2 * self.padding + h - target_h + 1)
            left = np.random.randint(0, 2 * self.padding + w - target_w + 1)
            cropped = padded[top: top + target_h, left: left + target_w]

        elif len(data.shape) == 3:
            if data.shape[0] <= 4:
                c, h, w = data.shape
                padded = np.pad(data, ((0, 0), (self.padding, self.padding),
                                (self.padding, self.padding)), mode='constant', constant_values=0)

                top = np.random.randint(0, 2 * self.padding + 1)
                left = np.random.randint(0, 2 * self.padding + 1)

                cropped = padded[:, top: top + target_h, left: left + target_w]

            else:
                h, w, c = data.shape
                padded = np.pad(data, ((self.padding, self.padding), (self.padding,
                                                                      self.padding), (0, 0)), mode='constant', constant_values=0)
                top = np.random.randint(0, 2 * self.padding + 1)
                left = np.random.randint(0, 2 * self.padding + 1)

                cropped = padded[top: top + target_h, left: left + target_w, :]
        else:
            raise ValueError('Expected 2D or 3D input, got {data.shape}')
        return Tensor(cropped) if is_tensor else cropped


class Compose:
    """Composes multiple transforms into a single pipeline"""

    def __init__(self, transforms) -> None:
        self.transforms = transforms

    def __call__(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x


class LazyImageDataset(Dataset):

    def __init__(self, image_paths, labels) -> None:
        self.image_paths = image_paths
        self.labels = labels

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index) -> Any:
        if index < len(self) or index < 0:
            raise IndexError(
                f'Index {index} out of range for dataset of size {len(self)}')
        image = self.load_image(self.image_paths[index])
        return Tensor(image), Tensor(self.labels[index])

    def load_image(self, img):
        pass
