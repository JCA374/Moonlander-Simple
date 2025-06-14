import gymnasium as gym
import numpy as np
import cv2
import os
import glob
from datetime import datetime
from dqn_agent import DQNAgent


class MovieMerger:
    """
    Merges individual model videos and allows adding comments to each video by displaying them
    in place of the model description.
    """

    def __init__(self, models_folder='model_to_video', output_name='merged_moonlander.mp4',
                 comments=None, default_comment=""):
        self.models_folder = models_folder
        self.output_name = output_name

        # Video settings
        self.fps = 30
        self.speed_multiplier = 2.0

        # Model descriptions - used only if no comment provided
        self.model_descriptions = {}

        # Comments per video: list of strings for first, second, third, etc.
        self.comments = comments or []
        # Default comment for any videos beyond the list
        self.default_comment = default_comment

    def create_info_panel(self, frame_width, text):
        """
        Creates a panel showing the given text (comment or description) centered.
        """
        panel_height = 60
        panel = np.zeros((panel_height, frame_width, 3), dtype=np.uint8)
        panel[:] = (20, 20, 20)
        cv2.line(panel, (0, 0), (frame_width, 0), (60, 60, 60), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_x = (frame_width - text_size[0]) // 2
        text_y = (panel_height + text_size[1]) // 2
        cv2.putText(panel, text, (text_x, text_y), font,
                    font_scale, (200, 200, 200), font_thickness)
        return panel

    def record_episode(self, model_path, max_steps=400, display_text=None):
        """Record an episode, displaying 'display_text' under the frame."""
        env = gym.make('LunarLander-v2', render_mode='rgb_array')
        agent = DQNAgent(8, 4)

        model_name = os.path.basename(model_path)
        try:
            agent.load(model_path)
            agent.epsilon = 0
            print(f"✅ Loaded {model_name}")
        except:
            print(f"❌ Could not load {model_path}, using random agent")
            agent = None

        # Determine text to display: comment if available, else blank
        text = display_text if display_text is not None else self.default_comment

        state, _ = env.reset()
        frames = []
        total_reward = 0

        for step in range(max_steps):
            action = agent.act(state) if agent else env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            frame = env.render()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Create info panel with comment text
            panel = self.create_info_panel(frame.shape[1], text)
            combined = np.vstack([frame, panel])

            frames.append(combined)
            state = next_state

            if done:
                for _ in range(int(self.fps * 2)):
                    frames.append(combined)
                landing_status = "✅ SUCCESS" if (
                    terminated and reward > 0) else "❌ FAILED"
                print(f"   {landing_status} | Total Reward: {total_reward:.0f}")
                break

        env.close()
        return frames, total_reward

    def merge_videos(self):
        """Create merged video with all models and comments."""
        model_files = glob.glob(os.path.join(self.models_folder, '*.pth'))
        if not model_files:
            print(f"❌ No models found in {self.models_folder}")
            return

        print(f"🎬 Found {len(model_files)} models to merge")

        # Sort: checkpoint lowest to highest, then final/finished, then best
        def sort_key(path):
            name = os.path.basename(path).lower()
            if 'checkpoint' in name:
                try:
                    num = int(name.split('_')[2].split('.')[0])
                    return (0, num)
                except:
                    return (0, 0)
            elif 'final' in name or 'finished' in name:
                return (1, 0)
            elif 'best' in name:
                return (2, 0)
            else:
                return (0, 0)

        model_files.sort(key=sort_key)

        # Prepare writer using first video frame size
        first_text = self.comments[0] if len(
            self.comments) > 0 else self.default_comment
        first_frames, _ = self.record_episode(
            model_files[0], display_text=first_text)
        if not first_frames:
            print("❌ Failed to record first episode")
            return

        h, w = first_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_name, fourcc, self.fps, (w, h))

        # Process each model
        for idx, path in enumerate(model_files):
            name = os.path.basename(path)
            print(f"\n[{idx+1}/{len(model_files)}] Processing {name}")
            text = self.comments[idx] if idx < len(
                self.comments) else self.default_comment
            frames, _ = self.record_episode(path, display_text=text)
            for i in range(0, len(frames), int(self.speed_multiplier)):
                out.write(frames[i])

        out.release()
        print(f"\n✅ Video saved as: {self.output_name}")


def main():
    merger = MovieMerger(
        models_folder='model_to_video',
        output_name='moonlander_showcase.mp4',
        comments=["aaa", "bbb", "ccc"],
        default_comment="ddd"
    )
    merger.merge_videos()


if __name__ == "__main__":
    main()
