import numpy as np
import matplotlib.pyplot as plt

def create_spiral_transition_tensor(angle_step=np.pi/6, scale_factor=None, spiral_type="custom"):
    """
    Create a transition tensor that produces a spiral when applied repeatedly.
    
    Args:
        angle_step: Rotation angle per step in radians (default: 30 degrees)
        scale_factor: Scaling factor per step (if None, calculated based on spiral_type)
        spiral_type: "custom", "golden", or "logarithmic"
    
    Returns:
        transition_tensor: 3D array of shape (num_observations, 2, 2)
    """
    if scale_factor is None:
        if spiral_type == "golden":
            # Golden spiral: radius grows by φ per quarter turn
            # φ^(1/4) per 90° = φ^(angle_step/90°) per step
            phi = (1 + np.sqrt(5)) / 2  # Golden ratio ≈ 1.618
            scale_factor = phi ** (np.degrees(angle_step) / 90)
        elif spiral_type == "logarithmic":
            # Logarithmic spiral: constant angle between radius and tangent
            # This creates a more uniform spiral
            scale_factor = 1.0  # No scaling, pure rotation
        else:
            scale_factor = 1.05  # Default custom scaling
    
    # Create rotation matrix with scaling
    cos_theta = scale_factor * np.cos(angle_step)
    sin_theta = scale_factor * np.sin(angle_step)
    
    # Transition tensor: 
    # observation 0: clockwise rotation + scaling
    # observation 1: counter-clockwise rotation + scaling
    transition_tensor = np.array([
        [[cos_theta, -sin_theta],      # observation 0: clockwise
         [sin_theta,  cos_theta]],
        [[cos_theta,  sin_theta],      # observation 1: counter-clockwise
         [-sin_theta, cos_theta]]
    ])
    
    return transition_tensor

def iterate_state(state, transition_tensor, observation):
    """Apply transition tensor to current state."""
    return transition_tensor[observation] @ state

def iterate_sequence(state, transition_tensor, sequence):
    """Iterate through sequence and return state history."""
    state_history = []
    current_state = state.copy()
    
    for observation in sequence:
        current_state = iterate_state(current_state, transition_tensor, observation)
        state_history.append(current_state.copy())
    
    return state_history

def plot_spiral(states, title="Spiral State History"):
    """Plot the spiral trajectory."""
    plt.figure(figsize=(8, 8))
    plt.plot(states[:, 0], states[:, 1], 'b-o', linewidth=2, markersize=6)
    plt.plot(states[0, 0], states[0, 1], 'go', markersize=10, label='Start')
    plt.plot(states[-1, 0], states[-1, 1], 'ro', markersize=10, label='End')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()

def main():
    # Define parameters
    angle_step = np.pi/6  # 30 degrees per step
    
    # Set up initial state and sequence
    initial_state = np.array([0.0, 1.0])
    sequence = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 10 steps
    
    # For demonstration, let's use a shorter sequence to see more variation
    demo_sequence = [0, 0, 0, 0, 0]  # 5 steps for clearer visualization
    
    # Create different types of spirals
    print("=== SPIRAL COMPARISON ===\n")
    
    # 1. Custom spiral (original)
    custom_tensor = create_spiral_transition_tensor(angle_step, scale_factor=1.05, spiral_type="custom")
    custom_history = iterate_sequence(initial_state, custom_tensor, sequence)
    
    # 2. Golden spiral
    golden_tensor = create_spiral_transition_tensor(angle_step, spiral_type="golden")
    golden_history = iterate_sequence(initial_state, golden_tensor, sequence)
    
    # 3. Logarithmic spiral (pure rotation)
    log_tensor = create_spiral_transition_tensor(angle_step, spiral_type="logarithmic")
    log_history = iterate_sequence(initial_state, log_tensor, sequence)
    
    # Generate all possible binary sequences of length len(sequence)
    def generate_all_sequences(length):
        """Generate all possible binary sequences of given length."""
        if length == 0:
            return [[]]
        prev_sequences = generate_all_sequences(length - 1)
        return [[0] + seq for seq in prev_sequences] + [[1] + seq for seq in prev_sequences]
    
    all_sequences = generate_all_sequences(len(demo_sequence))
    print(f"Generated {len(all_sequences)} different trajectories")
    
    # Calculate all trajectories for each spiral type
    all_custom_trajectories = []
    all_golden_trajectories = []
    all_log_trajectories = []
    
    for seq in all_sequences:
        all_custom_trajectories.append(iterate_sequence(initial_state, custom_tensor, seq))
        all_golden_trajectories.append(iterate_sequence(initial_state, golden_tensor, seq))
        all_log_trajectories.append(iterate_sequence(initial_state, log_tensor, seq))
    
    # Print parameters for golden spiral
    phi = (1 + np.sqrt(5)) / 2
    golden_scale = phi ** (np.degrees(angle_step) / 90)
    print(f"Golden ratio (φ): {phi:.6f}")
    print(f"Golden scale factor per step: {golden_scale:.6f}")
    print(f"Angle step: {np.degrees(angle_step):.1f}°")
    print(f"Initial state: {initial_state}")
    print(f"Sequence length: {len(sequence)}")
    print()
    
    # Print state histories
    print("=== STATE HISTORIES ===")
    
    print("\nCustom spiral (scale=1.05):")
    for i, state in enumerate(custom_history):
        print(f"Step {i+1}: {state}")
    
    print("\nGolden spiral:")
    for i, state in enumerate(golden_history):
        print(f"Step {i+1}: {state}")
    
    print("\nLogarithmic spiral (pure rotation):")
    for i, state in enumerate(log_history):
        print(f"Step {i+1}: {state}")
    
    print(f"\nGenerated {len(all_sequences)} different trajectories")
    print("Sample sequences:")
    for i, seq in enumerate(all_sequences[:5]):  # Show first 5 sequences
        print(f"  Sequence {i}: {seq}")
    if len(all_sequences) > 5:
        print(f"  ... and {len(all_sequences) - 5} more")
    
    # Debug: Show some trajectory examples
    print("\nSample trajectory endpoints:")
    for i in range(min(5, len(all_sequences))):
        seq = all_sequences[i]
        custom_end = all_custom_trajectories[i][-1] if all_custom_trajectories[i] else initial_state
        golden_end = all_golden_trajectories[i][-1] if all_golden_trajectories[i] else initial_state
        log_end = all_log_trajectories[i][-1] if all_log_trajectories[i] else initial_state
        print(f"  Sequence {seq}: Custom={np.linalg.norm(custom_end):.4f}, Golden={np.linalg.norm(golden_end):.4f}, Log={np.linalg.norm(log_end):.4f}")
    
    # Visualize all spirals with all trajectories colored by accumulated counts
    plt.figure(figsize=(15, 5))
    
    # Custom spiral
    plt.subplot(1, 3, 1)
    # Plot all trajectories with color coding based on accumulated percentage of 1's
    for i, (seq, trajectory) in enumerate(zip(all_sequences, all_custom_trajectories)):
        states = np.array([initial_state] + trajectory)
        
        # Color each point by the percentage of 1's accumulated up to that point
        for j, state in enumerate(states):
            if j == 0:  # Initial state
                color = plt.cm.RdYlBu(0.5)  # Neutral color for start
            else:
                # Calculate percentage of 1's up to this point (j-1 observations)
                observations_so_far = seq[:j]
                percent_1s = sum(observations_so_far) / len(observations_so_far) if len(observations_so_far) > 0 else 0.5
                color = plt.cm.RdYlBu(percent_1s)
            
            # Plot point with appropriate color
            plt.plot(state[0], state[1], 'o', color=color, markersize=3, alpha=0.8)
            
            # Connect points with lines (using average color of adjacent points)
            if j > 0:
                prev_percent = sum(seq[:j-1]) / len(seq[:j-1]) if j-1 > 0 else 0.5
                curr_percent = sum(seq[:j]) / len(seq[:j]) if j > 0 else 0.5
                avg_percent = (prev_percent + curr_percent) / 2
                line_color = plt.cm.RdYlBu(avg_percent)
                plt.plot([states[j-1, 0], state[0]], [states[j-1, 1], state[1]], 
                        color=line_color, alpha=0.4, linewidth=1)
    
    # Mark start and highlight extreme cases
    plt.plot(initial_state[0], initial_state[1], 'ko', markersize=10, label='Start')
    all_0_end = np.array([initial_state] + all_custom_trajectories[0])[-1]  # [0,...,0]
    all_1_end = np.array([initial_state] + all_custom_trajectories[-1])[-1]  # [1,...,1]
    plt.plot(all_0_end[0], all_0_end[1], 'ro', markersize=8, label='All 0\'s')
    plt.plot(all_1_end[0], all_1_end[1], 'bo', markersize=8, label='All 1\'s')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Custom Spiral - All Trajectories\n(Red=more 0\'s, Blue=more 1\'s)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Golden spiral
    plt.subplot(1, 3, 2)
    for i, (seq, trajectory) in enumerate(zip(all_sequences, all_golden_trajectories)):
        states = np.array([initial_state] + trajectory)
        
        for j, state in enumerate(states):
            if j == 0:  # Initial state
                color = plt.cm.RdYlBu(0.5)  # Neutral color for start
            else:
                observations_so_far = seq[:j]
                percent_1s = sum(observations_so_far) / len(observations_so_far) if len(observations_so_far) > 0 else 0.5
                color = plt.cm.RdYlBu(percent_1s)
            
            plt.plot(state[0], state[1], 'o', color=color, markersize=3, alpha=0.8)
            
            if j > 0:
                prev_percent = sum(seq[:j-1]) / len(seq[:j-1]) if j-1 > 0 else 0.5
                curr_percent = sum(seq[:j]) / len(seq[:j]) if j > 0 else 0.5
                avg_percent = (prev_percent + curr_percent) / 2
                line_color = plt.cm.RdYlBu(avg_percent)
                plt.plot([states[j-1, 0], state[0]], [states[j-1, 1], state[1]], 
                        color=line_color, alpha=0.4, linewidth=1)
    
    plt.plot(initial_state[0], initial_state[1], 'ko', markersize=10, label='Start')
    all_0_end = np.array([initial_state] + all_golden_trajectories[0])[-1]
    all_1_end = np.array([initial_state] + all_golden_trajectories[-1])[-1]
    plt.plot(all_0_end[0], all_0_end[1], 'ro', markersize=8, label='All 0\'s')
    plt.plot(all_1_end[0], all_1_end[1], 'bo', markersize=8, label='All 1\'s')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Golden Spiral - All Trajectories\n(Red=0% 1\'s, Blue=100% 1\'s)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Logarithmic spiral
    plt.subplot(1, 3, 3)
    for i, (seq, trajectory) in enumerate(zip(all_sequences, all_log_trajectories)):
        states = np.array([initial_state] + trajectory)
        
        for j, state in enumerate(states):
            if j == 0:  # Initial state
                color = plt.cm.RdYlBu(0.5)  # Neutral color for start
            else:
                observations_so_far = seq[:j]
                percent_1s = sum(observations_so_far) / len(observations_so_far) if len(observations_so_far) > 0 else 0.5
                color = plt.cm.RdYlBu(percent_1s)
            
            plt.plot(state[0], state[1], 'o', color=color, markersize=3, alpha=0.8)
            
            if j > 0:
                prev_percent = sum(seq[:j-1]) / len(seq[:j-1]) if j-1 > 0 else 0.5
                curr_percent = sum(seq[:j]) / len(seq[:j]) if j > 0 else 0.5
                avg_percent = (prev_percent + curr_percent) / 2
                line_color = plt.cm.RdYlBu(avg_percent)
                plt.plot([states[j-1, 0], state[0]], [states[j-1, 1], state[1]], 
                        color=color, alpha=0.4, linewidth=1)
    
    plt.plot(initial_state[0], initial_state[1], 'ko', markersize=10, label='Start')
    all_0_end = np.array([initial_state] + all_log_trajectories[0])[-1]
    all_1_end = np.array([initial_state] + all_log_trajectories[-1])[-1]
    plt.plot(all_0_end[0], all_0_end[1], 'ro', markersize=8, label='All 0\'s')
    plt.plot(all_1_end[0], all_1_end[1], 'bo', markersize=8, label='All 1\'s')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Logarithmic Spiral - All Trajectories\n(Red=0% 1\'s, Blue=100% 1\'s)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()
    
    # Add a colorbar to show the mapping
    fig, ax = plt.subplots(figsize=(6, 1))
    norm = plt.Normalize(0, 1)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu, norm=norm)
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', aspect=40)
    cbar.set_label('Percentage of 1\'s accumulated up to each point (Red=0% 1\'s, Blue=100% 1\'s)')
    ax.set_title('Color Mapping for Trajectories')
    plt.show()
    
    # Analysis
    print("\n=== ANALYSIS ===")
    print(f"Custom spiral final radius: {np.linalg.norm(custom_history[-1]):.4f}")
    print(f"Golden spiral final radius: {np.linalg.norm(golden_history[-1]):.4f}")
    print(f"Logarithmic spiral final radius: {np.linalg.norm(log_history[-1]):.4f}")
    
    # Show some statistics about all trajectories
    print(f"\n=== TRAJECTORY STATISTICS ===")
    custom_radii = [np.linalg.norm(traj[-1]) for traj in all_custom_trajectories]
    golden_radii = [np.linalg.norm(traj[-1]) for traj in all_golden_trajectories]
    log_radii = [np.linalg.norm(traj[-1]) for traj in all_log_trajectories]
    
    print(f"Custom spiral radius range: {min(custom_radii):.4f} to {max(custom_radii):.4f}")
    print(f"Golden spiral radius range: {min(golden_radii):.4f} to {max(golden_radii):.4f}")
    print(f"Logarithmic spiral radius range: {min(log_radii):.4f} to {max(log_radii):.4f}")
    
    # Calculate growth per quarter turn for golden spiral
    steps_per_quarter = int(90 / np.degrees(angle_step))  # 3 steps for 30° steps
    if len(golden_history) >= steps_per_quarter:
        quarter_turn_radius = np.linalg.norm(golden_history[steps_per_quarter - 1])
        growth_factor = quarter_turn_radius / np.linalg.norm(initial_state)
        print(f"\nGolden spiral growth per quarter turn: {growth_factor:.6f}")
        print(f"Expected φ: {phi:.6f}")
        print(f"Difference: {abs(growth_factor - phi):.6f}")

if __name__ == "__main__":
    main() 