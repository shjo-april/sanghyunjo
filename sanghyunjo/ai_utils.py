# Copyright (C) 2025 * Ltd. All rights reserved.
# author: Sanghyun Jo <shjo.april@gmail.com>

import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np

from typing import Union

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

def resize(tensor: torch.Tensor, size: Union[tuple, torch.Tensor], mode: str = 'nearest') -> torch.Tensor:
    """
    Resize a tensor to the specified size using interpolation.

    Args:
        tensor (torch.Tensor): Input tensor of shape (C, H, W) or (N, C, H, W).
        size (tuple or torch.Tensor): Target size as (height, width) or a tensor whose shape determines the size.
        mode (str, optional): Interpolation mode to use ('nearest', 'bilinear', etc.). Defaults to 'nearest'.

    Returns:
        torch.Tensor: Resized tensor with the same number of dimensions as the input.

    Notes:
        - If the input is 3D (C, H, W), a batch dimension is temporarily added for interpolation.
        - If `size` is a tensor, its spatial dimensions (last two) are used as the target size.
    """
    is_3d = tensor.ndim == 3
    tensor = tensor.unsqueeze(0) if is_3d else tensor
    if isinstance(size, torch.Tensor):
        size = size.shape[-2:]
    resized: torch.Tensor = F.interpolate(tensor, size, mode=mode)
    return resized.squeeze(0) if is_3d else resized

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

def minmax(inputs, dim: Union[int, tuple], keepdim=False):
    """
    Computes the minimum and maximum values along a specified dimension.

    Args:
        inputs (Union[np.ndarray, torch.Tensor]):
            Input data, must be a NumPy array or PyTorch tensor.
        dim (int or tuple of int): The dimension(s) along which to compute the min and max.

    Returns:
        Tuple: (min_values, max_values) of the same type as the input.

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
        return np.min(inputs, axis=dim, keepdims=keepdim), np.max(inputs, axis=dim, keepdims=keepdim)
    elif isinstance(inputs, torch.Tensor):
        return inputs.amin(dim=dim, keepdim=keepdim), inputs.amax(dim=dim, keepdim=keepdim)
    else:
        raise TypeError("Unsupported input type for minmax(). Expected np.ndarray or torch.Tensor.")
    
def normalize(masks: Union[torch.Tensor, np.ndarray], dim: Union[int, tuple], eps: float = 1e-6) -> Union[torch.Tensor, np.ndarray]:
    """
    Normalize a tensor or array to the range [0, 1] (approximately [eps, 1 - eps] if clamped).

    Args:
        masks (Union[torch.Tensor, np.ndarray]): Input tensor or array to normalize.
        dim (int or tuple): Dimension(s) along which to compute min and max.
        eps (float, optional): Small epsilon to avoid division by zero. Defaults to 1e-6.

    Returns:
        Union[torch.Tensor, np.ndarray]: Normalized output in the same type as input.

    Notes:
        - If input is 2D, it's temporarily expanded to 3D for normalization.
        - Output values are not explicitly clamped, but range is approximately [0, 1].
    """
    is_2D = len(masks.shape) == 2
    if is_2D:
        masks = masks[None]  # Add a batch dimension if input is 2D

    min_v, max_v = minmax(masks, dim=dim, keepdim=True)
    masks = (masks - min_v) / (max_v - min_v + eps)

    if is_2D:
        masks = masks[0]  # Remove the batch dimension if input was 2D

    return masks

def visualize_pca(embs: torch.Tensor, patch_size: int, seed: int = 42) -> np.ndarray:
    """
    Projects embedding (D, H, W) using PCA and returns a stable RGB visualization.
    Applies component sorting, sign alignment, and canonical direction normalization.

    Args:
        embs (torch.Tensor): (D, H, W) input tensor
        patch_size (int): Upscale factor to original resolution
        seed (int): Random seed for PCA stability

    Returns:
        np.ndarray: BGR image
    """
    import numpy as np
    from .cv_utils import denorm, resize_mask
    from sklearn.decomposition import PCA

    D, H, W = embs.shape
    np_embs = embs.reshape(D, H * W).T.cpu().detach().numpy()  # (H*W, D)

    # Step 1: PCA
    reducer = PCA(n_components=3, random_state=seed)
    z = reducer.fit_transform(np_embs)  # (N, 3)

    # Step 2: Fix component order by descending explained variance.
    # order = np.argsort(-reducer.explained_variance_)
    # z = z[:, order]

    # Step 3: Flip component signs so each axis has a positive global mean
    z *= np.where(z.mean(axis=0) >= 0, 1.0, -1.0)

    # Step 4: Canonical alignment - align mean direction to x-axis
    def canonical_align(z):
        z = z.astype(np.float32)
        z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-8
        mean_dir = z.mean(axis=0)
        mean_dir /= np.linalg.norm(mean_dir) + 1e-8
        target = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        axis = np.cross(mean_dir, target)
        angle = np.arccos(np.clip(np.dot(mean_dir, target), -1.0, 1.0))
        if np.linalg.norm(axis) < 1e-6 or angle < 1e-4:
            return z
        axis /= np.linalg.norm(axis)
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ], dtype=np.float32)
        R = np.eye(3, dtype=np.float32) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        return z @ R.T

    z = canonical_align(z)

    # Step 5: Normalize per-channel to [0, 1]
    z_min, z_max = z.min(axis=0), z.max(axis=0)
    z = (z - z_min) / (z_max - z_min + 1e-8)
    z = z.reshape(H, W, 3)

    # z = convert(z, 'lch2bgr')
    z = denorm(z)

    # Step 6: Resize and return (RGB to BGR if using OpenCV-compatible tools)
    return resize_mask(z, scale=patch_size)

class GaussianSmoothing(nn.Module):
    """
    Applies Gaussian smoothing to 1D, 2D, or 3D inputs using depthwise convolution.
    This module is non-trainable and can be used to smooth segmentation masks,
    attention maps, etc.

    Args:
        kernel_size (int or list): Size of the Gaussian kernel per dimension
        sigma (float or list): Standard deviation of the Gaussian kernel per dimension
        dim (int): Number of spatial dimensions (1, 2, or 3)

    Example:
        # Given a 3D attention map of shape [D, H, W], convert to 4D by adding batch dim:
        cross_attn = smoothing(cross_attn[None])[0]
        # Adds batch dimension, applies smoothing, then removes batch dimension
        # Resulting shape remains [D, H, W] after smoothing
    """
    def __init__(self, kernel_size=3, sigma=0.5, dim=2):
        super().__init__()
        self.dim = dim
        kernel_size = [kernel_size] * dim if isinstance(kernel_size, int) else kernel_size
        sigma = [sigma] * dim if isinstance(sigma, float) else sigma

        # Create Gaussian kernel in each dimension using meshgrid
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size],
            indexing='ij'
        )
        kernel = 1
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-((mgrid - mean) ** 2) / (2 * std ** 2))

        kernel /= kernel.sum()  # Normalize the kernel
        self.kernel = kernel.view(1, 1, *kernel.size()).requires_grad_(False)  # (1,1,H,W)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Applies Gaussian smoothing to the input tensor.

        Args:
            inputs: (B, C, H, W) for 2D, or appropriate shape for 1D/3D
        Returns:
            Smoothed tensor of the same shape as input
        """
        kernel = self.kernel.to(inputs.device, dtype=inputs.dtype)
        kernel = kernel.repeat(inputs.size(1), *[1] * (kernel.dim() - 1))  # repeat for each channel

        pad_size = self.kernel.shape[-1] // 2
        padding = [pad_size] * (2 * self.dim)  # symmetric padding

        inputs = F.pad(inputs, padding, mode='reflect')

        conv = [F.conv1d, F.conv2d, F.conv3d][self.dim - 1]
        return conv(inputs, weight=kernel, groups=inputs.size(1))  # depthwise conv

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (B, H, W) — predicted probabilities in [0, 1]
            targets: (B, H, W) — ground truth in {0, 1}
        Returns:
            scalar Dice loss
        """
        inputs = inputs.view(inputs.size(0), -1)   # (B, H*W)
        targets = targets.view(targets.size(0), -1).float()  # (B, H*W)

        intersection = (inputs * targets).sum(dim=1)
        union = inputs.sum(dim=1) + targets.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()