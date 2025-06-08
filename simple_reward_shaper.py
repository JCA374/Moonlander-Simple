import numpy as np

class SimpleRewardShaper:
    """
    Simplified reward shaping focused on essential behaviors only
    """
    def __init__(self):
        self.prev_distance = None
        
    def reset(self):
        self.prev_distance = None
        
    def shape_reward(self, state, action, reward, done, step):
        """
        Simple reward shaping with minimal modifications to original reward
        
        State: [x, y, vel_x, vel_y, angle, angular_vel, leg1_contact, leg2_contact]
        """
        x, y, vel_x, vel_y, angle, angular_vel, leg1_contact, leg2_contact = state
        
        # Start with original reward
        shaped_reward = reward
        
        # Only add small bonuses/penalties to guide learning
        
        # 1. Landing bonus - HUGE reward for successful landing
        if leg1_contact and leg2_contact and done:  # Only count completed successful landings
            if abs(x) < 0.3:  # Near landing pad
                shaped_reward += 300  # Massive bonus for perfect landing
            elif abs(x) < 0.5:  # Decent landing
                shaped_reward += 200  # Large bonus for good landing
            else:
                shaped_reward += 100  # Medium bonus for any successful landing
        elif leg1_contact and leg2_contact:  # Partial bonus while landing
            shaped_reward += 10
                
        # 2. Approach bonus - small reward for getting closer to center
        distance_from_center = abs(x)
        if self.prev_distance is not None:
            distance_improvement = self.prev_distance - distance_from_center
            shaped_reward += distance_improvement * 2  # Small bonus for progress
            
        # 3. Velocity control near ground - encourage soft landings
        if y < 0.3:  # Close to ground
            velocity_magnitude = np.sqrt(vel_x**2 + vel_y**2)
            if velocity_magnitude < 0.3:  # Slow and controlled
                shaped_reward += 5  # Reward for controlled approach
            elif velocity_magnitude > 1.0:  # Too fast
                shaped_reward -= 10  # Penalty for dangerous approach
                
        # 4. Angle stability - small penalty for extreme tilting
        if abs(angle) > 0.5:  # Very tilted
            shaped_reward -= 0.5
        
        self.prev_distance = distance_from_center
        
        return shaped_reward