# MoonLander Training Analysis: Score vs Landing Paradox

## The Problem

We're observing a concerning pattern in training:
- **Early training**: Lower scores (-164) but more landings (36/100)
- **Later training**: Higher scores (-25) but fewer landings (0/100)

This indicates the agent is learning to optimize for **reward hacking** rather than the intended behavior.

## Root Cause Analysis

### 1. Hovering Exploitation
The agent has discovered that **hovering** (staying airborne without landing) gives:
- **Positive rewards** for controlled flight
- **No crash penalties** 
- **No fuel penalties** if using minimal thrust
- **Approach bonuses** for staying near center

### 2. Landing Risk vs Reward
From the agent's perspective:
- **Landing attempt** = High risk of crash (-100 to -300 penalty)
- **Hovering** = Safe, steady positive reward accumulation
- **Our landing bonus** only triggers on successful completion

### 3. Reward Structure Issues

Current reward components:
```
Hovering (per step):
+ Approach bonus: +2 per step closer to center
+ Velocity control: +5 for slow movement
+ Angle stability: minimal penalty
= Steady positive accumulation

Landing attempt:
+ Base LunarLander reward: +100-200 (if successful) OR -100 to -300 (if crash)
+ Our landing bonus: +100-300 (only if successful)
= High risk, high reward
```

## Why Scores Improve But Landings Decrease

1. **Early Training (Exploration Phase)**:
   - High epsilon (exploration) forces random actions
   - Agent accidentally lands sometimes
   - Gets mixed results: some successes, many crashes

2. **Later Training (Exploitation Phase)**:
   - Low epsilon (exploitation) uses learned policy
   - Agent learns "safe hovering strategy"
   - Avoids risky landing attempts
   - Achieves consistent moderate positive scores through hovering

## Solutions

### Immediate Fixes

#### 1. Add Strong Hovering Penalties
```python
# In reward shaper:
if y > 0.5 and velocity_magnitude < 0.2:  # High altitude, barely moving
    self.hover_time += 1
    if self.hover_time > 50:  # Hovering too long
        shaped_reward -= 5  # Escalating penalty
    if self.hover_time > 100:
        shaped_reward -= 10  # Stronger penalty
```

#### 2. Time-Based Penalties
```python
# Penalize long episodes
if step > 300:  # After 300 steps
    shaped_reward -= 1  # Per step penalty
```

#### 3. Fuel Efficiency Penalties
```python
# Stronger fuel penalties
if action != 0:  # Any engine use
    shaped_reward -= 0.3  # Increased from current small penalty
```

#### 4. Height-Based Penalties
```python
# Penalize staying high
if y > 1.0:  # High altitude
    shaped_reward -= 1  # Per step penalty for staying high
```

### Advanced Solutions

#### 1. Curriculum Learning
- Start with easier landing scenarios
- Gradually increase difficulty
- Force landing attempts in early training

#### 2. Reward Reshaping
- Remove positive rewards for hovering
- Make landing the ONLY way to get positive rewards
- Negative baseline reward that only landing can overcome

#### 3. Episode Termination
- Force episode end after extended hovering
- Treat extended hovering as episode failure

## Recommended Implementation

### Phase 1: Anti-Hovering Penalties
```python
class ImprovedRewardShaper:
    def __init__(self):
        self.hover_penalty_time = 0
        self.altitude_penalty_time = 0
        
    def shape_reward(self, state, action, reward, done, step):
        x, y, vel_x, vel_y, angle, angular_vel, leg1, leg2 = state
        shaped_reward = reward
        
        # 1. MASSIVE landing bonuses (keep existing)
        if leg1 and leg2 and done:
            shaped_reward += 300  # Keep this
            
        # 2. ANTI-HOVERING SYSTEM
        velocity_mag = np.sqrt(vel_x**2 + vel_y**2)
        
        # Detect hovering: high altitude + low velocity
        if y > 0.3 and velocity_mag < 0.2:
            self.hover_penalty_time += 1
            # Escalating penalties
            if self.hover_penalty_time > 30:
                shaped_reward -= 2
            if self.hover_penalty_time > 60:
                shaped_reward -= 5
            if self.hover_penalty_time > 100:
                shaped_reward -= 10
        else:
            self.hover_penalty_time = 0
            
        # 3. TIME PRESSURE
        if step > 200:  # After 200 steps, urgency increases
            shaped_reward -= 0.5
            
        # 4. ALTITUDE PENALTIES
        if y > 1.0:  # Staying too high
            shaped_reward -= 1
            
        # 5. FUEL EFFICIENCY (stronger)
        if action != 0:
            shaped_reward -= 0.2
            
        return shaped_reward
```

### Phase 2: Training Adjustments
- **Increase epsilon minimum** to 0.1 (more exploration)
- **Add epsilon scheduling** that occasionally forces exploration
- **Curriculum learning**: Start with lower fuel, forcing quicker decisions

## Expected Outcomes

After implementing anti-hovering penalties:
1. **Scores may initially drop** as agent can no longer exploit hovering
2. **Landing attempts should increase** as hovering becomes unprofitable
3. **True learning** should emerge as agent learns efficient landing strategies
4. **Final performance** should be higher with both good scores AND successful landings

## Monitoring

Track these metrics during training:
- **Average episode length** (should decrease)
- **Hovering time per episode** (should decrease)  
- **Landing success rate** (should increase)
- **Score variance** (should stabilize)

The goal is an agent that **lands quickly and efficiently**, not one that **hovers indefinitely** for safe scores.