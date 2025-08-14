import numpy as np


def spiral(scale_array: np.ndarray, angle_array: np.ndarray, x_scale: float) -> np.ndarray:
    """Creates a transition matrix for the Spiral Process."""
    
    scale_cw = scale_array[0]
    scale_ccw = scale_array[1]
    
    angle_cw = angle_array[0]
    angle_ccw = angle_array[1]
    
    # Clockwise rotation matrix (sequence value 0)
    clockwise_matrix = scale_cw * np.array([
        [x_scale * np.cos(angle_cw), -x_scale * np.sin(angle_cw)],
        [np.sin(angle_cw), np.cos(angle_cw)]
    ])
    
    # Counter-clockwise rotation matrix (sequence value 1)
    counter_clockwise_matrix = scale_ccw * np.array([
        [np.cos(angle_ccw), -np.sin(angle_ccw)],
        [np.sin(angle_ccw), np.cos(angle_ccw)]
    ])
    
    transition_matrix = np.zeros((2, 2, 2))
    transition_matrix[0] = counter_clockwise_matrix
    transition_matrix[1] = clockwise_matrix
    
    return transition_matrix