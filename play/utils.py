import numpy as np
import matplotlib.pyplot as plt

def uniform_centered_projection(states_array: np.ndarray) -> np.ndarray:
    """Projects belief states onto 2D plane with uniform distribution at origin.
    
    This projection:
    1. Centers belief states by subtracting the uniform distribution [1/3, 1/3, 1/3]
    2. Projects onto the 2D tangent space of the 2-simplex
    
    The uniform distribution [1/3, 1/3, 1/3] maps to [0, 0] in the projected space.
    
    Args:
        states_array: Array of shape (n_states, 3) containing belief states
        
    Returns:
        Array of shape (n_states, 2) with projected coordinates
    """
    # Center by subtracting uniform distribution
    uniform = np.array([1/3, 1/3, 1/3])
    centered_states = states_array - uniform
    
    # Create orthonormal basis for 2-simplex tangent space
    # The simplex normal is [1, 1, 1], so we need 2 orthogonal vectors to it
    
    # First basis vector: [1, -1, 0] normalized
    v1 = np.array([1, -1, 0]) / np.sqrt(2)
    
    # Second basis vector: [1, 1, -2] normalized  
    # This is orthogonal to both [1, 1, 1] and v1
    v2 = np.array([1, 1, -2]) / np.sqrt(6)
    
    # Project centered states onto the 2D basis
    proj_coord1 = centered_states @ v1
    proj_coord2 = centered_states @ v2
    
    # Rotate by 180 degrees (negate both coordinates)
    proj_coord1 = -proj_coord1
    proj_coord2 = -proj_coord2
    
    return np.column_stack([proj_coord1, proj_coord2])


def plot_belief_states(belief_states_dict: dict, max_depth: int):
    """Performs PCA, eigenbasis, or uniform-centered analysis on belief states and creates a visualization.
    
    Args:
        belief_states_dict: Dictionary of belief states by depth
        transition_matrix: Transition matrix (only needed if use_pca=False and use_uniform_centered=False)
        max_depth: Maximum depth for analysis
        constrained: Whether belief states are constrained
        use_pca: If True, use PCA; if False, use eigenbasis projection (ignored if use_uniform_centered=True)
        use_uniform_centered: If True, use uniform-centered projection; overrides use_pca
        
    Returns:
        Analysis results (PCA, eigenbasis, or uniform-centered projection data)
    """
    # Collect all belief states from all depths
    all_states = []
    depth_labels = []
    
    for depth in range(min(max_depth + 1, len(belief_states_dict))):
        if depth in belief_states_dict:
            for state in belief_states_dict[depth]:
                all_states.append(np.array(state))
                depth_labels.append(depth)
    
    # Convert to numpy array
    states_array = np.array(all_states)
    print(f"Total belief states collected: {len(all_states)}")
    print(f"State dimensions: {states_array.shape}")
    

    # Perform uniform-centered projection
    states_projected = uniform_centered_projection(states_array)
    
    print("Using uniform-centered projection")
    print(f"Uniform distribution [1/3, 1/3, 1/3] maps to origin [0, 0]")
    
    xlabel = 'First Coordinate (tangent to 2-simplex)'
    ylabel = 'Second Coordinate (tangent to 2-simplex)'
    method_name = "Uniform-Centered Projection"
    analysis_results = (states_projected, states_array, depth_labels)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Use belief state entries as RGB colors
    colors = states_array  # Each row is [r, g, b] belief state
    
    # Create scatter plot
    scatter = plt.scatter(states_projected[:, 0], states_projected[:, 1], 
                         c=colors, s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    
    # Add colorbar explanation
    plt.figtext(0.02, 0.02, 'Color represents belief state: Red=State 0, Green=State 1, Blue=State 2', 
                fontsize=10, style='italic')
    
    plt.tight_layout()
    
    # Save to plots directory
    plot_path = f'plots/belief_states.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved as '{plot_path}'")
    plt.show()
    
    return analysis_results