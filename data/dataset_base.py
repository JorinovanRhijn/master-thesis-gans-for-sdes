import torch
import numpy as np
from typing import Tuple
from abc import ABC, abstractmethod


class DatasetBase(ABC):

    @abstractmethod
    def sample_exact(self) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def make_paths(self):
        raise NotImplementedError

    @abstractmethod
    def generate_train_test(self) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        raise NotImplementedError

    @abstractmethod
    def cdf(self):
        raise NotImplementedError

    @abstractmethod
    def pdf(self):
        raise NotImplementedError
