# Copyright (C) 2025 * Ltd. All rights reserved.
# author: Sanghyun Jo <shjo.april@gmail.com>

import torch
import random

import numpy as np

from torch.nn import functional as F

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
    # pip install scikit-learn
    from sklearn.decomposition import PCA
    pca = PCA(n_components=dim, random_state=seed)
    return pca.fit_transform(vectors)

def masks_to_boxes(masks: np.ndarray) -> np.ndarray:
    """
    Compute bounding boxes from binary masks.

    Args:
        masks (np.array): Boolean or binary masks of shape (N, H, W)

    Returns:
        np.array: Bounding boxes in (N, 4) format: [x_min, y_min, x_max, y_max]
    """
    boxes = []

    for mask in masks:
        # Find non-zero pixels
        ys, xs = np.where(mask > 0)

        if len(xs) == 0 or len(ys) == 0:
            boxes.append([0, 0, 0, 0])  # No object found
        else:
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            boxes.append([x_min, y_min, x_max, y_max])

    return np.array(boxes)

def stack(inputs, dim=0):
    """
    Stacks a list or array of tensors/arrays along a specified dimension.

    Args:
        inputs (Union[List[np.ndarray], np.ndarray, List[torch.Tensor]]): 
            The input collection to be stacked. Can be:
              - A list of NumPy arrays
              - A NumPy array of arrays
              - A list of PyTorch tensors
        dim (int): The dimension along which to stack the inputs. 
            Corresponds to `axis` in NumPy and `dim` in PyTorch.

    Returns:
        Union[np.ndarray, torch.Tensor]: 
            A stacked NumPy array or PyTorch tensor, depending on input type.

    Raises:
        TypeError: If the input is neither a NumPy array nor a list of PyTorch tensors.

    Examples:
        >>> import sanghyunjo.ai_utils as shai
        >>> shai.stack([np.ones((2, 2)), np.zeros((2, 2))], dim=0).shape # (2, 2, 2)
        >>> shai.stack([torch.tensor([1, 2]), torch.tensor([3, 4])], dim=1) # tensor([[1, 3], [2, 4]])
    """
    if isinstance(inputs[0], np.ndarray):
        return np.stack(inputs, axis=dim)
    elif isinstance(inputs[0], torch.Tensor):
        return torch.stack(inputs, dim=dim)
    else:
        raise TypeError("Unsupported input type for stack(). Expected NumPy array or list of torch.Tensors.")
    
def cat(inputs, dim=0):
    """
    Concatenates a sequence of arrays or tensors along an existing dimension.

    Args:
        inputs (Union[List[np.ndarray], List[torch.Tensor]]): 
            A list of arrays or tensors to concatenate. All elements must have the same shape, except in the concatenation dimension.
        dim (int): The dimension along which to concatenate. 
            Corresponds to `axis` in NumPy and `dim` in PyTorch.

    Returns:
        Union[np.ndarray, torch.Tensor]: 
            A concatenated NumPy array or PyTorch tensor, depending on input type.

    Raises:
        TypeError: If the input is neither a list of NumPy arrays nor a list of PyTorch tensors.

    Examples:
        >>> import sanghyunjo.ai_utils as shai
        >>> shai.cat([np.ones((2, 2)), np.zeros((2, 2))], dim=0).shape # (4, 2)
        >>> shai.cat([torch.tensor([[1], [2]]), torch.tensor([[3], [4]])], dim=1) # tensor([[1, 3], [2, 4]])
    """
    if isinstance(inputs[0], np.ndarray):
        return np.concatenate(inputs, axis=dim)
    elif isinstance(inputs[0], torch.Tensor):
        return torch.cat(inputs, dim=dim)
    else:
        raise TypeError("Unsupported input type for cat(). Expected list of NumPy arrays or list of torch.Tensors.")

def minmax(inputs, dim=0):
    """
    Computes the minimum and maximum values along a specified dimension.

    Args:
        inputs (Union[np.ndarray, torch.Tensor]):
            Input array or tensor.
        dim (int): The dimension along which to compute min and max.

    Returns:
        Tuple: (min_values, max_values), both of same type as inputs.

    Raises:
        TypeError: If input is not a NumPy array or PyTorch tensor.

    Examples:
        >>> import sanghyunjo.ai_utils as shai

        >>> shai.minmax(np.array([[1, 2], [3, 4]]), dim=0)
        (array([1, 2]), array([3, 4]))

        >>> shai.minmax(torch.tensor([[1, 2], [3, 4]]), dim=1)
        (tensor([1, 3]), tensor([2, 4]))
    """
    if isinstance(inputs, np.ndarray):
        return np.min(inputs, axis=dim), np.max(inputs, axis=dim)
    elif isinstance(inputs, torch.Tensor):
        return tuple(inputs.aminmax(dim=dim))
    else:
        raise TypeError("Unsupported input type for minmax(). Expected np.ndarray or torch.Tensor.")
    
def normalize(masks: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Normalize a tensor to the range [eps, 1 - eps].

    Args:
        masks (torch.Tensor): Input tensor to normalize.
        eps (float, optional): Small epsilon value to avoid extreme values. Defaults to 1e-3.

    Returns:
        torch.Tensor: Normalized tensor with values in the range [eps, 1 - eps].
    """
    is_2D = len(masks.shape) == 2
    if is_2D:
        masks = masks[None]  # Add a batch dimension if input is 2D

    min_v, max_v = masks.view(masks.shape[0], -1).aminmax(dim=1)
    masks = (masks - min_v[:, None, None]) / (max_v[:, None, None] - min_v[:, None, None])
    masks = masks.clamp(min=eps, max=1 - eps)

    if is_2D:
        masks = masks[0]  # Remove the batch dimension if input was 2D

    return masks
