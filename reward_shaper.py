import numpy as np

class RewardShaper:
    def __init__(self):
        self.prev_distance = None
        self.prev_velocity = None
        self.hover_count = 0
        self.low_altitude_time = 0
        
    def reset(self):
        self.prev_distance = None
        self.prev_velocity = None
        self.hover_count = 0
        self.low_altitude_time = 0
        
    def shape_reward(self, state, action, reward, done, step):
        """
        Custom reward shaping to encourage landing and discourage hovering
        
        State: [x, y, vel_x, vel_y, angle, angular_vel, leg1_contact, leg2_contact]
        """
        x, y, vel_x, vel_y, angle, angular_vel, leg1_contact, leg2_contact = state
        
        shaped_reward = reward
        
        # Landing zone is between flags at x=0 (approximately -0.4 to 0.4)
        distance_from_center = abs(x)
        
        # Height above ground (y=0 is ground level)
        altitude = y
        
        # Total velocity magnitude
        velocity_magnitude = np.sqrt(vel_x**2 + vel_y**2)
        
        # 1. Landing bonus - reward successful landings
        if leg1_contact and leg2_contact and not done:
            if distance_from_center < 0.3 and altitude < 0.2:
                shaped_reward += 50  # Big bonus for stable landing
            elif distance_from_center < 0.5:
                shaped_reward += 20  # Smaller bonus for landing in zone
                
        # 2. Approach bonus - reward getting closer to landing zone (reduced)
        if self.prev_distance is not None:
            distance_improvement = self.prev_distance - distance_from_center
            shaped_reward += distance_improvement * 2  # Reduced from 10
            
        # 3. Velocity penalties - discourage high speeds near ground
        if altitude < 0.5:
            if velocity_magnitude > 0.5:
                shaped_reward -= velocity_magnitude * 5
            else:
                shaped_reward += (0.5 - velocity_magnitude) * 2  # Reward slow approach
                
        # 4. Anti-hovering penalty (reduced)
        if altitude > 0.2 and velocity_magnitude < 0.1:
            self.hover_count += 1
            if self.hover_count > 30:  # Increased threshold
                shaped_reward -= 0.5  # Reduced penalty
        else:
            self.hover_count = 0
            
        # 5. Fuel efficiency - penalize unnecessary engine use (reduced)
        if action != 0:  # Any engine firing
            shaped_reward -= 0.03  # Reduced from 0.1
            
        # 6. Angle penalty - discourage tilting too much (reduced)
        if abs(angle) > 0.3:  # Increased threshold
            shaped_reward -= abs(angle) * 2  # Reduced penalty
            
        # 7. Progress reward - encourage descent when far from ground (reduced)
        if altitude > 1.0:
            if vel_y < 0:  # Moving downward
                shaped_reward += 0.2  # Reduced from 1
            else:  # Moving upward when high
                shaped_reward -= 0.1  # Reduced from 0.5
                
        # 8. Time penalty for low altitude hovering (reduced)
        if altitude < 0.5 and velocity_magnitude < 0.3:
            self.low_altitude_time += 1
            if self.low_altitude_time > 100:  # Increased threshold
                shaped_reward -= 0.2  # Reduced penalty
        else:
            self.low_altitude_time = 0
            
        # 9. Early landing bonus (reduced)
        if done and leg1_contact and leg2_contact:
            time_bonus = max(0, (500 - step) * 0.02)  # Reduced bonus
            shaped_reward += time_bonus
            
        # Update tracking variables
        self.prev_distance = distance_from_center
        self.prev_velocity = velocity_magnitude
        
        return shaped_reward