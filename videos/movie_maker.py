import gymnasium as gym
import numpy as np
import cv2
import os
import glob
from dqn_agent import DQNAgent
from reward_shaper import RewardShaper


class SimpleVideoGenerator:
    """Simple video generator using existing code"""

    def __init__(self):
        # Easy settings to change
        self.speed_multiplier = 1.5  # 3x speed
        self.fps = 30

        # Model descriptions - EDIT THESE!
        self.descriptions = {
            'moonlander_best.pth': 'Best Model',
            'moonlander_final.pth': 'Final Model',
            'moonlander_checkpoint_0.pth': 'Before Training',
            'moonlander_checkpoint_5000.pth': 'Early Training',
            'moonlander_checkpoint_10000.pth': 'Mid Training',
            'moonlander_checkpoint_15000.pth': 'Late Training',
            'moonlander_checkpoint_20000.pth': 'Advanced Training',
            'moonlander_checkpoint_25000.pth': 'Complete Training',
        }

    def add_text_overlay(self, frame, model_name, score, description):
        """Add text and score bar to frame"""
        height, width = frame.shape[:2]

        # Add semi-transparent black bars for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 80), (0, 0, 0), -1)
        cv2.rectangle(overlay, (0, height-80), (width, height), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # Model name at top with shadow effect
        shadow_offset = 2
        cv2.putText(frame, f"Model: {model_name}", (12, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3)  # Shadow
        cv2.putText(frame, f"Model: {model_name}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # Description at bottom with shadow
        cv2.putText(frame, description, (12, height-22),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)  # Shadow
        cv2.putText(frame, description, (10, height-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        # Fancier score bar in top right
        bar_width = 250
        bar_height = 40
        bar_x = width - bar_width - 20
        bar_y = 20

        # Rounded rectangle background (simulate with multiple rectangles)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                      (40, 40, 40), -1)
        cv2.rectangle(frame, (bar_x-2, bar_y-2), (bar_x + bar_width+2, bar_y + bar_height+2),
                      (255, 255, 255), 2)  # White border

        # Score fill with gradient effect
        normalized_score = (score + 100) / 400
        normalized_score = np.clip(normalized_score, 0, 1)
        fill_width = int((bar_width - 4) * normalized_score)

        # Color gradient based on score
        if score < -50:
            color = (60, 60, 255)  # Bright red
        elif score < 0:
            color = (100, 100, 255)  # Light red
        elif score < 100:
            color = (0, 200, 255)  # Orange
        elif score < 200:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 255, 100)  # Bright green

        if fill_width > 0:
            cv2.rectangle(frame, (bar_x + 2, bar_y + 2),
                          (bar_x + 2 + fill_width, bar_y + bar_height - 2),
                          color, -1)

        # Score text with better formatting
        score_text = f"{score:.0f}"
        text_size = cv2.getTextSize(
            score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = bar_x + (bar_width - text_size[0]) // 2
        text_y = bar_y + (bar_height + text_size[1]) // 2

        # Score text with shadow
        cv2.putText(frame, score_text, (text_x + 1, text_y + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
        cv2.putText(frame, score_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Add "SCORE" label above bar
        cv2.putText(frame, "SCORE", (bar_x + bar_width//2 - 30, bar_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return frame

    def record_model(self, model_path):
        """Record one episode of a model"""
        env = gym.make('LunarLander-v2', render_mode='rgb_array')
        agent = DQNAgent(8, 4)

        try:
            agent.load(model_path)
            agent.epsilon = 0
        except:
            print(f"Could not load {model_path}")
            env.close()
            return None

        model_name = os.path.basename(model_path)
        description = self.descriptions.get(model_name, "Training Model")

        state, _ = env.reset()
        frames = []
        total_reward = 0

        print(f"Recording {model_name}: {description}")

        for step in range(1000):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            total_reward += reward

            # Get frame and add overlay
            frame = env.render()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame, (960, 720))  # Higher resolution
            frame = self.add_text_overlay(
                frame, model_name, total_reward, description)

            frames.append(frame)
            state = next_state

            if done:
                # Hold last frame for 2 seconds
                for _ in range(60):
                    frames.append(frame)
                break

        env.close()
        return frames

    def create_video(self, output_name='moonlander_models.mp4', models_folder='model_to_video'):
        """Create video of all models"""
        # Find all models
        model_files = sorted(glob.glob(os.path.join(models_folder, '*.pth')))

        # Sort to show in order: checkpoints by number (low to high), then final, then best
        def sort_key(path):
            name = os.path.basename(path)
            if 'best' in name:
                return 999999  # Best goes last
            if 'final' in name:
                return 999998  # Final goes second to last
            if 'checkpoint' in name:
                try:
                    num = int(name.split('_')[1].split('.')[0])
                    return num  # Sort checkpoints by their number
                except:
                    return 999997
            if 'improving' in name:
                try:
                    num = int(name.split('_')[1].split('.')[0])
                    return num  # Sort improving models by their number
                except:
                    return 999997
            return 999997

        model_files.sort(key=sort_key)

        if not model_files:
            print("No models found in models/ folder")
            return

        # Setup video writer
        first_frames = self.record_model(model_files[0])
        if not first_frames:
            return

        height, width = first_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_name, fourcc, self.fps, (width, height))

        # Write all models
        for model_path in model_files:
            frames = self.record_model(model_path)
            if frames:
                # Apply speed by skipping frames
                skip = int(self.speed_multiplier)
                for i in range(0, len(frames), skip):
                    out.write(frames[i])

        out.release()
        print(f"\nVideo saved as {output_name}")
        print(f"Speed: {self.speed_multiplier}x")


# Simple usage
if __name__ == "__main__":
    generator = SimpleVideoGenerator()

    # Easy to customize
    generator.speed_multiplier = 2.0  # Change speed here

    # Edit descriptions here
    generator.descriptions = {
        'moonlander_best.pth': '🏆 Best Performance',
        'moonlander_final.pth': '🎓 Final Model',
        'moonlander_checkpoint_0.pth': '🎲 Random Agent',
        'moonlander_checkpoint_5000.pth': '📈 Learning to Land',
        'moonlander_checkpoint_10000.pth': '🚀 Getting Better',
        'moonlander_checkpoint_15000.pth': '🎯 Good Control',
        'moonlander_checkpoint_20000.pth': '⭐ Expert Level',
    }

    # Specify the folder containing models
    generator.create_video('moonlander_showcase.mp4',
                           models_folder='model_to_video')
