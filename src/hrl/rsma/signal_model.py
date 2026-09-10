"""
实现单层RSMA有效增益和SIC级接收功率项。
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["OneLayerRSMASignalTerms", "one_layer_rsma_signal_terms"]


@dataclass(frozen=True, slots=True)
class OneLayerRSMASignalTerms:
    """储存单层RSMA信号项的有效增益和接收功率。
    这些项用于计算每个用户的公共和私有流的SINR。"""

    effective_gains: NDArray[np.complex128]
    received_powers: NDArray[np.float64]
    noise_variance: float
    common_signal_powers: NDArray[np.float64]
    common_interference_powers: NDArray[np.float64]
    private_signal_powers: NDArray[np.float64]
    private_interference_powers: NDArray[np.float64]


def _finite_complex_matrix(value: ArrayLike, name: str) -> NDArray[np.complex128]:
    """Validate and return one finite nonempty two-dimensional complex matrix."""
    values = np.asarray(value)
    if values.dtype.kind in {"b", "O", "U", "S"}:
        raise TypeError(f"{name} must contain numeric values.")
    try:
        matrix = np.asarray(value, dtype=np.complex128)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must contain numeric values.") from error
    if matrix.ndim != 2 or 0 in matrix.shape:
        raise ValueError(f"{name} must have a nonempty two-dimensional shape.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def _positive_noise_variance(value: object) -> float:
    """Validate and return a strictly positive finite real noise variance."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError("noise_variance must be a real-valued scalar.")
    variance = float(value)
    if not np.isfinite(variance) or variance <= 0.0:
        raise ValueError("noise_variance must be finite and greater than zero.")
    return variance


def one_layer_rsma_signal_terms(
    channels: ArrayLike, precoders: ArrayLike, *, noise_variance: float
) -> OneLayerRSMASignalTerms:
    """ 
    计算单层RSMA信号项的有效增益和接收功率。
    公共解码将每个私有流视为干扰。
    私人解码假设理想的SIC，因此公共流已被删除，只有其他私有流仍然作为干扰存在
    """
    channel_matrix = _finite_complex_matrix(channels, "channels")
    if np.any(np.linalg.norm(channel_matrix, axis=1) == 0.0):
        raise ValueError("channels must not contain a zero-norm user channel.")
    precoder_matrix = _finite_complex_matrix(precoders, "precoders")
    user_count, antenna_count = channel_matrix.shape
    if precoder_matrix.shape != (antenna_count, user_count + 1):
        raise ValueError(
            "precoders must have shape (channel antennas, channel users + 1)."
        )
    noise_variance = _positive_noise_variance(noise_variance)

    effective_gains = np.asarray(channel_matrix @ precoder_matrix, dtype=np.complex128)  # 有效增益
    received_powers = np.asarray(np.abs(effective_gains) ** 2, dtype=np.float64)         # 接收功率
    private_received_powers = received_powers[:, 1:]    # 私有流接收功率（不包括公共流）
    private_signal_powers = np.asarray(np.diag(private_received_powers), dtype=np.float64) # 每个用户的私有流接收功率
    private_total_powers = np.sum(private_received_powers, axis=1)    # # 每个用户收到的所有私有流功率之和

    return OneLayerRSMASignalTerms(
        effective_gains=effective_gains,
        received_powers=received_powers,
        noise_variance=noise_variance,
        common_signal_powers=np.asarray(received_powers[:, 0], dtype=np.float64),
        common_interference_powers=np.asarray(private_total_powers, dtype=np.float64),
        private_signal_powers=private_signal_powers,
        private_interference_powers=np.asarray(
            private_total_powers - private_signal_powers, dtype=np.float64
        ),
    )
