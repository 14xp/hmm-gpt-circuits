import numpy as np


def bloch(a: float, b: float) -> np.ndarray:
    """Creates a transition matrix for the Bloch ball process."""
    g = 1/(2 * np.sqrt(a**2 + b**2))


    return np.array(
        [
            [
                [       1/4,                  0, 2*a*b*g**2],
                [         0, (a**2 - b**2)*g**2,          0],
                [2*a*b*g**2,                  0,        1/4],
            ],
            [
                [        1/4,                  0, -2*a*b*g**2],
                [          0, (a**2 - b**2)*g**2,           0],
                [-2*a*b*g**2,                  0,         1/4],
            ],
            [
                [       1/4, 2*a*b*g**2,                  0],
                [2*a*b*g**2,        1/4,                  0],
                [         0,          0, (a**2 - b**2)*g**2],
            ],
            [
                [        1/4, -2*a*b*g**2,                 0],
                [-2*a*b*g**2,         1/4,                 0],
                [          0,          0, (a**2 - b**2)*g**2],
            ],
        ]
    )