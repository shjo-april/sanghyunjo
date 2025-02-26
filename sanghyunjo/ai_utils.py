# Copyright (C) 2025 * Ltd. All rights reserved.
# author: Sanghyun Jo <shjo.april@gmail.com>

import torch
import random

import numpy as np

from torch.nn import functional as F

# pip install scikit-learn for PCA
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

"""
from .misc import get_name
from .json_utils import read_json

class Dataset:
    def __init__(self, path):
        self.data_dict = read_json(path)
        self.tag = get_name(path).replace('.json', '')

        self.ignore = self.data_dict['ignore']
        self.class_names = np.asarray(self.data_dict['names'])
        self.num_classes = len(self.class_names)
        self.class_dict = {n: i for i, n in enumerate(self.class_names)}

    def __getitem__(self, key):
        if isinstance(key, int): return str(self.class_names[key])
        elif isinstance(key, str): return self.class_dict[key]
        else: raise ValueError(f'Please check a value (key: {key})')
    
# TODO: Optimizing K-means & PCA
class Clustering:
    def __init__(self, K: int=0, tau: float=0.0, seed: int=0):
        self.K = K
        self.tau = tau
        self.seed = seed

    def get_kmeans(self, affinity):
        return KMeans(n_clusters=self.K, n_init=10, random_state=self.seed).fit(affinity)

    def __call__(self, affinity: torch.Tensor, mode='kmeans') -> np.ndarray:
        ph, pw = affinity.shape[1:]
        affinity = affinity.view(ph * pw, ph * pw).permute(1, 0).float().cpu().numpy()
        
        if 'kmeans' in mode:
            kmeans = self.get_kmeans(affinity)
            return kmeans.labels_.reshape(ph, pw)
        else:
            pca = PCA(n_components=3, random_state=self.seed).fit(affinity)
            pca = pca.transform(affinity).reshape(ph, pw, 3) # RGB

            min_v, max_v = pca.min(axis=(0, 1)), pca.max(axis=(0, 1))
            pca = (pca - min_v) / (max_v - min_v)
            return (pca * 255).astype(np.uint8)
"""

def set_seed(seed: int, device: torch.device = torch.device('cuda:0')) -> torch.Generator:
    """Set the random seed for reproducibility across multiple libraries.

    Args:
        seed (int): The seed value to set.
        device (torch.device, optional): The device for PyTorch generator. Defaults to 'cuda:0'.

    Returns:
        torch.Generator: PyTorch random number generator with the set seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return torch.Generator(device).manual_seed(seed)

def resize_tensor(tensor: torch.Tensor, size: tuple, mode: str = 'nearest') -> torch.Tensor:
    """Resize a tensor to the specified size using interpolation.

    Args:
        tensor (torch.Tensor): Input tensor with shape (C, H, W) or (N, C, H, W).
        size (tuple): Target size as (height, width).
        mode (str, optional): Interpolation mode. Defaults to 'nearest'.

    Returns:
        torch.Tensor: Resized tensor with the same number of dimensions as input.
    """
    is_3d = tensor.ndim == 3
    tensor = tensor.unsqueeze(0) if is_3d else tensor
    resized: torch.Tensor = F.interpolate(tensor, size, mode=mode)
    return resized.squeeze(0) if is_3d else resized

def match_size(tensor: torch.Tensor, target: torch.Tensor, mode: str = 'nearest') -> torch.Tensor:
    """Resize a tensor to match the spatial dimensions of a target tensor.

    Args:
        tensor (torch.Tensor): Input tensor to be resized.
        target (torch.Tensor): Target tensor whose spatial dimensions will be matched.

    Returns:
        torch.Tensor: Resized tensor with the same height and width as the target.
    """
    return resize_tensor(tensor, target.shape[-2:], mode)

def normalize(masks: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Normalize a tensor to the range [eps, 1 - eps].

    Args:
        masks (torch.Tensor): Input tensor to normalize.
        eps (float, optional): Small epsilon value to avoid extreme values. Defaults to 1e-3.

    Returns:
        torch.Tensor: Normalized tensor with values in the range [eps, 1 - eps].
    """
    min_v, max_v = masks.view(masks.shape[0], -1).aminmax(dim=1)
    min_v, max_v = min_v[:, None, None], max_v[:, None, None]
    masks = (masks - min_v) / (max_v - min_v)
    return masks.clamp(min=eps, max=1. - eps)

def count_params(params, unit: float = 1e6) -> float:
    """Count the total number of parameters in a model.

    Args:
        params (iterable): Iterable of model parameters.
        unit (float, optional): Unit scale for parameter count (e.g., 1e6 for millions). Defaults to 1e6.

    Returns:
        float: Total number of parameters in the specified unit.
    """
    return sum(p.numel() for p in params) / unit

def pca_reduce(vectors: np.ndarray, dim: int = 3, seed: int = 0) -> np.ndarray:
    """Reduce the dimensionality of input vectors using PCA.

    Args:
        vectors (np.ndarray): Input array of shape (n_samples, n_features).
        dim (int, optional): Number of principal components to keep. Defaults to 3.
        seed (int, optional): Random seed for reproducibility. Defaults to 0.

    Returns:
        np.ndarray: Transformed array of shape (n_samples, dim).
    """
    pca = PCA(n_components=dim, random_state=seed)
    return pca.fit_transform(vectors)