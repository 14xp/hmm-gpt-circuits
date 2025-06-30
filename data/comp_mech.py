import numpy as np
from collections import deque

def stationary_distribution(transition_matrix: np.ndarray) -> np.ndarray:
    """Calculates the stationary distribution of the transition matrix."""
    # Compute eigenvalues and left eigenvectors
    # For left eigenvectors, transpose the matrix first
    eigenvals, eigenvecs = np.linalg.eig(np.transpose(transition_matrix.sum(axis=0)))
    
    # Find indices where eigenvalues are approximately 1
    indices = np.where(np.isclose(np.abs(eigenvals), 1.0, atol=1e-15))[0]
    
    if len(indices) == 0:
        print("Warning: No stationary distribution exists (no eigenvalue 1 found).")
        return None
    
    # Extract the eigenvector corresponding to eigenvalue 1
    eigen_vec = eigenvecs[:, indices[0]]
    
    # Ensure real values and normalize
    stationary = eigen_vec.real / np.sum(eigen_vec.real)
    
    return stationary

def belief_update(transition_matrix: np.ndarray, observation: int, belief: np.ndarray) -> np.ndarray:
    """Updates the belief state using the transition matrix."""
    transition = transition_matrix[observation]
    updated = belief @ transition
    return updated / np.sum(updated)

def belief_to_token_distribution(transition_matrix: np.ndarray, belief: np.ndarray) -> np.ndarray:
    """Converts belief state to token distribution."""
    # Compute the token distribution by contracting belief with transition matrix and marginalizing over output states
    return np.einsum('i,kij->k', belief, transition_matrix)

def sample_tokens(transition_matrix: np.ndarray, initial_belief: np.ndarray, n_samples: int, n_tokens: int, seed: int = 42) -> np.ndarray:
    """Samples token sequences given a transition matrix and initial belief.
    
    Args:
        transition_matrix: Transition matrix of shape (n_vocab, n_states, n_states)
        initial_belief: Initial belief state of shape (n_states,)
        n_samples: Number of independent samples to generate
        n_tokens: Number of tokens to generate per sample
        seed: Random seed for reproducibility (default: 42)
        
    Returns:
        Array of shape (n_samples, n_tokens) containing sampled token sequences
    """
    rng = np.random.default_rng(seed)
    n_vocab = transition_matrix.shape[0]
    
    samples = []
    
    for sample_idx in range(n_samples):
        current_belief = initial_belief
        bos_token = n_vocab # Assuming the last token is the BOS token
        sample_tokens = [bos_token]
        
        for token_idx in range(n_tokens - 1):
            # Get token distribution from current belief
            token_dist = belief_to_token_distribution(transition_matrix, current_belief)
            
            # Sample a token from the distribution
            token = rng.choice(n_vocab, p=token_dist)
            sample_tokens.append(token)
            
            # Update belief with the sampled token
            current_belief = belief_update(transition_matrix, token, current_belief)
        
        samples.append(np.array(sample_tokens))
    
    return np.array(samples)


def belief_state(transition_matrix: np.ndarray, observations: np.ndarray, initial_belief: np.ndarray) -> np.ndarray:
    """Returns the belief state after a sequence of observations, given an initial state."""
    current_belief = initial_belief
    for obs in observations:
        current_belief = belief_update(transition_matrix, obs, current_belief)

    return current_belief

def final_token_distribution(transition_matrix: np.ndarray, observations: np.ndarray, initial_belief: np.ndarray) -> np.ndarray:
    """Returns the token distribution after a sequence of observations, given an initial belief state."""
    # Get the final belief state after all observations
    final_belief = belief_state(transition_matrix, observations, initial_belief)
    
    # Convert final belief state to token distribution
    return belief_to_token_distribution(transition_matrix, final_belief)

def token_distribution(transition_matrix: np.ndarray, observations: np.ndarray, initial_belief: np.ndarray) -> np.ndarray:
    """Returns the token distribution for each observation given the initial belief state."""
    current_belief = initial_belief
    token_distributions = []
    
    for obs in observations:
        # Update belief with current observation
        current_belief = belief_update(transition_matrix, obs, current_belief)
        
        # Get token distribution from updated belief
        token_dist = belief_to_token_distribution(transition_matrix, current_belief)
        token_distributions.append(token_dist)
    
    return np.array(token_distributions)

def constrained_belief_update(transition_matrix: np.ndarray, observation: int, belief: np.ndarray, initial_belief: np.ndarray) -> np.ndarray:
    """Constrained updates of the belief state using the transition matrix."""
    full_transition = transition_matrix.sum(axis=0)
    conditioned_transition = transition_matrix[observation] / np.sum(transition_matrix[observation], axis=1, keepdims=True) 
    correction = initial_belief @ conditioned_transition - initial_belief
    return belief @ full_transition + correction

def enumerate_belief_states(initial_belief: np.ndarray, transition_matrix: np.ndarray, max_depth: int, constrained: bool = False) -> dict:
    """Enumerates all possible belief states up to a given depth.
    
    Args:
        initial_belief: Starting belief state
        transition_matrix: Transition matrix for belief updates
        max_depth: Maximum depth to explore
        constrained: If True, use constrained belief update; if False, use regular belief update
        
    Returns:
        Dictionary with depth as keys and sets of belief states as values
    """
    
    # Get number of possible observations from transition matrix shape
    n_observations = transition_matrix.shape[0]
    
    # Initialize result structure
    initial_tuple = tuple(float(x) for x in initial_belief)
    belief_states_by_depth = {0: {initial_tuple}}
    
    # Queue for BFS: (belief_state, current_depth)
    queue = deque([(initial_belief, 0)])
    visited = {initial_tuple}
    
    while queue:
        current_belief, depth = queue.popleft()
        
        if depth >= max_depth:
            continue
            
        next_depth = depth + 1
        if next_depth not in belief_states_by_depth:
            belief_states_by_depth[next_depth] = set()
        
        # Explore all possible observations
        for obs in range(n_observations):
            if constrained:
                new_belief = constrained_belief_update(transition_matrix, obs, current_belief, initial_belief)
            else:
                new_belief = belief_update(transition_matrix, obs, current_belief)
            new_belief_tuple = tuple(float(x) for x in new_belief)
            
            # Add to results for this depth
            belief_states_by_depth[next_depth].add(new_belief_tuple)
            
            # Add to queue if not visited (to continue exploring)
            if new_belief_tuple not in visited:
                visited.add(new_belief_tuple)
                queue.append((new_belief, next_depth))
    
    # Convert tuples back to numpy arrays for easier use
    result = {}
    for depth, states in belief_states_by_depth.items():
        result[depth] = [np.array(state) for state in states]
    
    return result
