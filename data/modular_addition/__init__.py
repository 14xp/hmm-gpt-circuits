import numpy as np

def soft_modular_addition_matrix(p: int, k:int) -> np.ndarray:
    """Creates a transition matrix for the soft modular addition process."""
    assert 0<= k < p, "k must be in the range [0, p-1]"

    sin = np.sin(2 * np.pi * k / p)
    cos = np.cos(2 * np.pi * k / p)


    return np.array(
        [
            [      0, 1/p*cos, 1/p*sin, 1/p,        0,       0,   0],
            [      0,       0,       0,   0,  1/p*cos, 1/p*sin,   0],
            [      0,       0,       0,   0, -1/p*sin, 1/p*cos,   0],
            [      0,       0,       0,   0,        0,       0, 1/p],
            [1/p*cos,       0,       0,   0,        0,       0,   0],
            [1/p*sin,       0,       0,   0,        0,       0,   0],
            [    1/p,       0,       0,   0,        0,       0,   0]
        ]
                    )


def soft_modular_addition(p: int) -> np.ndarray:
    """Creates the transition tensor for the soft modular addition process."""

    return np.array(
        [soft_modular_addition_matrix(p, k) for k in range(p)]
                    )