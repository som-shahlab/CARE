"""Learn-Then-Test helpers: Hoeffding-Bentkus p-value and fixed-sequence testing."""
import numpy as np

from care.calibration import hb_pvalue  # single source of truth  # noqa: F401


def fixed_sequence(losses: np.ndarray, order, alpha: float, delta: float) -> list:
    """Test cells in `order`; return the certified prefix (stop at first non-rejection).

    losses: (n_docs, n_cells) per-document losses on the certification data.
    """
    n = losses.shape[0]
    risk = losses.mean(0)
    certified = []
    for c in order:
        if hb_pvalue(risk[c], n, alpha) <= delta:
            certified.append(c)
        else:
            break
    return certified


def scalar_ltt(losses: np.ndarray, alpha: float, delta: float) -> int:
    """LTT on a monotone 1D family ordered permissive -> restrictive; returns last certified index."""
    certified = fixed_sequence(losses, range(losses.shape[1]), alpha, delta)
    return certified[-1] if certified else 0


def crc_largest(losses: np.ndarray, alpha: float) -> int:
    """Scalar CRC: largest index whose bound (n/(n+1)) R + 1/(n+1) <= alpha (as in paper/scripts)."""
    n = losses.shape[0]
    ok = np.where(n / (n + 1) * losses.mean(0) + 1 / (n + 1) <= alpha)[0]
    return int(ok[-1]) if len(ok) else 0
