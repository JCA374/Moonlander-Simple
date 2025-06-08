import numpy as np

class ImprovedRewardShaper:
    """
    Anti-hovering reward shaper that forces landing attempts
    """
    def __init__(self):
        self.prev_distance = None
        self.hover_penalty_time = 0
        self.altitude_penalty_time = 0
        
    def reset(self):
        self.prev_distance = None
        self.hover_penalty_time = 0
        self.altitude_penalty_time = 0
        
    def shape_reward(self, state, action, reward, done, step):
        """
        Reward shaping that heavily penalizes hovering and rewards landing
        
        State: [x, y, vel_x, vel_y, angle, angular_vel, leg1_contact, leg2_contact]
        """
        x, y, vel_x, vel_y, angle, angular_vel, leg1_contact, leg2_contact = state
        
        # Start with original reward
        shaped_reward = reward
        
        # 1. MASSIVE LANDING BONUSES (unchanged)
        if leg1_contact and leg2_contact and done:
            if abs(x) < 0.3:  # Perfect landing
                shaped_reward += 300
            elif abs(x) < 0.5:  # Good landing
                shaped_reward += 200
            else:  # Any successful landing
                shaped_reward += 100
        elif leg1_contact and leg2_contact:  # Partial bonus while landing
            shaped_reward += 10
                
        # 2. ANTI-HOVERING SYSTEM
        velocity_magnitude = np.sqrt(vel_x**2 + vel_y**2)
        
        # Detect hovering: moderate/high altitude + very low velocity
        if y > 0.3 and velocity_magnitude < 0.2:
            self.hover_penalty_time += 1
            # Escalating penalties for hovering
            if self.hover_penalty_time > 20:
                shaped_reward -= 1  # Start penalizing
            if self.hover_penalty_time > 50:
                shaped_reward -= 3  # Increase penalty
            if self.hover_penalty_time > 80:
                shaped_reward -= 5  # Strong penalty
            if self.hover_penalty_time > 120:
                shaped_reward -= 10  # Severe penalty
        else:
            self.hover_penalty_time = 0  # Reset if not hovering
            
        # 3. TIME PRESSURE - discourage long episodes
        if step > 200:  # After 200 steps
            shaped_reward -= 0.5  # Small per-step penalty
        if step > 300:  # After 300 steps
            shaped_reward -= 1.0  # Larger penalty
            
        # 4. ALTITUDE PENALTIES - discourage staying high
        if y > 1.0:  # High altitude
            shaped_reward -= 1  # Per step penalty
        elif y > 0.8:  # Moderately high
            shaped_reward -= 0.5
            
        # 5. FUEL EFFICIENCY - stronger penalties for engine use
        if action != 0:  # Any engine firing
            shaped_reward -= 0.2  # Increased fuel penalty
            
        # 6. APPROACH BONUS (moderate)
        distance_from_center = abs(x)
        if self.prev_distance is not None:
            distance_improvement = self.prev_distance - distance_from_center
            shaped_reward += distance_improvement * 1  # Moderate bonus for progress
            
        # 7. VELOCITY CONTROL (encourage descent)
        if y < 0.5:  # Close to ground
            if velocity_magnitude < 0.3:  # Controlled approach
                shaped_reward += 3
            elif velocity_magnitude > 1.0:  # Too fast
                shaped_reward -= 8
        else:  # High altitude
            if vel_y < -0.1:  # Encourage descent
                shaped_reward += 1
            elif vel_y > 0.1:  # Penalize going up
                shaped_reward -= 2
                
        # 8. ANGLE STABILITY
        if abs(angle) > 0.4:  # Too tilted
            shaped_reward -= 1
            
        self.prev_distance = distance_from_center
        
        return shaped_reward