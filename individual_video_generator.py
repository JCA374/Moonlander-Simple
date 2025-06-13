import gymnasium as gym
import numpy as np
import cv2
import os
import glob
from dqn_agent import DQNAgent


def create_individual_videos():
    """Create separate video for each model in model_test_video folder"""

    # Settings
    speed = 2.0  # 2x speed
    fps = 30
    folder = 'model_test_video'

    # Find all models
    model_files = glob.glob(os.path.join(folder, '*.pth'))

    if not model_files:
        print(f"No models found in {folder}/")
        return

    print(f"Found {len(model_files)} models to process")

    for model_path in model_files:
        # Setup
        model_name = os.path.basename(model_path)
        video_name = model_name.replace('.pth', '.mp4')
        video_path = os.path.join(folder, video_name)

        print(f"\nProcessing {model_name}...")

        # Load environment and agent
        env = gym.make('LunarLander-v2', render_mode='rgb_array')
        agent = DQNAgent(8, 4)

        try:
            agent.load(model_path)
            agent.epsilon = 0
        except:
            print(f"Could not load {model_name}, skipping")
            env.close()
            continue

        # Run episode
        state, _ = env.reset()

        # Get frame size after reset
        first_frame = env.render()
        height, width = first_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        total_reward = 0
        frame_count = 0

        for step in range(1000):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            total_reward += reward

            # Get frame
            frame = env.render()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Simple text overlay
            cv2.putText(frame, f"Score: {total_reward:.0f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            # Write frame (skip frames for speed)
            if frame_count % int(speed) == 0:
                out.write(frame)
            frame_count += 1

            state = next_state

            if done:
                # Hold last frame for 1 second
                for _ in range(fps):
                    out.write(frame)
                break

        # Cleanup
        out.release()
        env.close()

        print(f"✓ Saved {video_path} (Score: {total_reward:.0f})")

    print(f"\nDone! Created {len(model_files)} videos in {folder}/")


if __name__ == "__main__":
    create_individual_videos()
