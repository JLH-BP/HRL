"""实现MRT、RZF和功率加权的单层RSMA预编码器。"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constraints import ResourceAllocation

RZF_REGULARIZATION: float = 1.0

__all__ = [
    "RZF_REGULARIZATION",
    "mrt_private_directions",
    "normalized_channel_sum_common_direction",
    "one_layer_rsma_precoders",
    "rzf_private_directions",
]


def _validated_channels(channels: ArrayLike) -> NDArray[np.complex128]:
    """Validate and return a finite nonempty user-row channel matrix."""
    values = np.asarray(channels)
    if values.dtype.kind in {"b", "O", "U", "S"}:
        raise TypeError("channels must contain numeric values.")
    try:
        matrix = np.asarray(channels, dtype=np.complex128)
    except (TypeError, ValueError) as error:
        raise TypeError("channels must contain numeric values.") from error
    if matrix.ndim != 2 or 0 in matrix.shape:
        raise ValueError("channels must have a nonempty two-dimensional (users, antennas) shape.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("channels must contain only finite values.")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms == 0.0):
        raise ValueError("channels must not contain a zero-norm user channel.")
    return matrix


def _normalize_columns(matrix: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """返回一个列式单位范数复矩阵."""
    norms = np.linalg.norm(matrix, axis=0)
    if np.any(~np.isfinite(norms)) or np.any(norms == 0.0):
        raise ValueError("Precoder directions must have finite nonzero column norms.")
    return np.asarray(matrix / norms[np.newaxis, :], dtype=np.complex128)

# 最大传输比
def mrt_private_directions(channels: ArrayLike) -> NDArray[np.complex128]:
    """返回形状为（天线，用户）的单位范数MRT专用方向"""
    matrix = _validated_channels(channels)
    return _normalize_columns(matrix.conj().T)

# 正则化迫零
def rzf_private_directions(channels: ArrayLike) -> NDArray[np.complex128]:
    """返回列式单位范数的固定正则化RZF专用方向。"""
    matrix = _validated_channels(channels)
    user_count = matrix.shape[0]
    gram = matrix @ matrix.conj().T
    system = gram + RZF_REGULARIZATION * np.eye(user_count, dtype=np.complex128)
    try:
        raw_directions = np.linalg.solve(system, matrix).conj().T
    except np.linalg.LinAlgError as error:
        raise ValueError("Unable to solve the regularized zero-forcing system.") from error
    return _normalize_columns(raw_directions)
    

def normalized_channel_sum_common_direction(channels: ArrayLike) -> NDArray[np.complex128]:
    """返回从单位用户信道方向之和得到的单位公共方向.
    公共流的方向是所有用户归一化信道方向的叠加和。"""
    private_directions = mrt_private_directions(channels)
    common_sum = np.sum(private_directions, axis=1)
    common_norm = float(np.linalg.norm(common_sum))
    if not np.isfinite(common_norm) or common_norm == 0.0:
        raise ValueError("Normalized user channel directions cancel in the common beam.")
    return np.asarray(common_sum / common_norm, dtype=np.complex128)


def one_layer_rsma_precoders(channels: ArrayLike,allocation: ResourceAllocation,*,private_scheme: str = "mrt",) -> NDArray[np.complex128]:
    """返回功率加权（天线，用户+ 1）单层RSMA预编码器"""
    matrix = _validated_channels(channels)
    if not isinstance(allocation, ResourceAllocation):
        raise TypeError("allocation must be a ResourceAllocation instance.")
    user_count, antenna_count = matrix.shape
    if allocation.private_powers.shape != (user_count,):
        raise ValueError("allocation private_powers must have one entry per channel user.")
    if allocation.common_rate_allocations.shape != (user_count,):
        raise ValueError("allocation common_rate_allocations must have one entry per channel user.")

    if private_scheme == "mrt":
        private_directions = mrt_private_directions(matrix)
    elif private_scheme == "rzf":
        private_directions = rzf_private_directions(matrix)
    else:
        raise ValueError("private_scheme must be either 'mrt' or 'rzf'.")

    if allocation.common_power == 0.0:
        common_direction = np.zeros(antenna_count, dtype=np.complex128)
    else:
        common_direction = normalized_channel_sum_common_direction(matrix)
    precoders = np.empty((antenna_count, user_count + 1), dtype=np.complex128)
    precoders[:, 0] = np.sqrt(allocation.common_power) * common_direction
    precoders[:, 1:] = private_directions * np.sqrt(allocation.private_powers)[np.newaxis, :]
    return precoders
