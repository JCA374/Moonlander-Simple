import os
import numpy as np
import gymnasium as gym
import cv2
import glob
import re
from datetime import datetime

# Import your trained agent and reward shaper
from dqn_agent import DQNAgent
from reward_shaper import RewardShaper


class VideoCreator:
    """Create a comprehensive LinkedIn video showing AI learning progression using all models in folder."""

    def __init__(
        self,
        video_dir: str = "linkedin_video",
        models_dir: str = None,
        fps: int = 30,
        speed_multiplier: float = 1.0,
        duration_seconds: int = 60,
        max_episode_steps: int = 300,
        overlay_height: int = 100,
        use_reward_shaping: bool = True,
    ):
        # Directories
        self.video_dir = video_dir
        self.models_dir = models_dir or video_dir
        os.makedirs(self.video_dir, exist_ok=True)

        # Video settings (easy to tweak)
        self.fps = fps
        self.speed_multiplier = speed_multiplier
        self.duration_seconds = duration_seconds
        self.max_episode_steps = max_episode_steps
        self.overlay_height = overlay_height
        self.use_reward_shaping = use_reward_shaping

        # Derived values
        self.effective_fps = int(self.fps * self.speed_multiplier)
        self.total_frames = self.fps * self.duration_seconds

        # Initialize reward shaper if requested
        self.reward_shaper = RewardShaper() if use_reward_shaping else None

        # Discover all models in the folder
        self.checkpoints = self.discover_models()

        print(f"Found {len(self.checkpoints)} models to showcase:")
        for cp in self.checkpoints:
            print(f"  - {cp['desc']} ({cp['model'] or 'Random'})")

    def discover_models(self):
        """Automatically discover and organize all models from lowest to highest episode."""
        checkpoints = []

        # Always start with random (no model)
        checkpoints.append({
            "episode": 0,
            "desc": "Episode 0",
            "model": None,
            "priority": 0
        })

        # Find all .pth files
        model_files = glob.glob(os.path.join(self.models_dir, "*.pth"))

        if not model_files:
            print(f"⚠️  No model files found in {self.models_dir}")
            return checkpoints

        model_info = []

        for model_path in model_files:
            filename = os.path.basename(model_path)
            info = self.parse_model_filename(filename)
            if info:
                info['path'] = model_path
                model_info.append(info)

        # Sort by episode number only (lowest to highest)
        model_info.sort(key=lambda x: x['episode'])

        # Add models to checkpoints
        for info in model_info:
            checkpoints.append({
                "episode": info['episode'],
                "desc": info['description'],
                "model": info['filename'],
                "priority": info['priority']
            })

        return checkpoints

    def parse_model_filename(self, filename):
        """Extract episode number for chronological ordering."""
        base_name = filename.replace('.pth', '')

        info = {
            'filename': filename,
            'episode': 0,
            'priority': 1,
            'description': 'Episode 0'
        }

        # Parse different model types - focus on episode numbers
        if filename == 'moonlander_best.pth':
            info.update({
                'priority': 1,
                'episode': 999999,  # Ensure it comes last
                'description': 'Best Model'
            })
        elif filename == 'moonlander_final.pth':
            info.update({
                'priority': 1,
                'episode': 999998,
                'description': 'Final Model'
            })
        elif 'improving' in filename:
            match = re.search(r'improving_(\d+)', filename)
            episode = int(match.group(1)) if match else 0
            info.update({
                'priority': 1,
                'episode': episode,
                'description': f'Episode {episode}'
            })
        elif 'checkpoint' in filename:
            match = re.search(r'checkpoint_(\d+)', filename)
            episode = int(match.group(1)) if match else 0
            info.update({
                'priority': 1,
                'episode': episode,
                'description': f'Episode {episode}'
            })
        elif 'backup' in filename:
            match = re.search(r'backup_(\d+)', filename)
            episode = int(match.group(1)) if match else 0
            info.update({
                'priority': 1,
                'episode': episode,
                'description': f'Episode {episode}'
            })
        else:
            # Try to extract any number from filename
            match = re.search(r'(\d+)', filename)
            if match:
                episode = int(match.group(1))
                info.update({
                    'priority': 1,
                    'episode': episode,
                    'description': f'Episode {episode}'
                })

        return info

    def load_agent(self, model_name: str):
        """Load an agent from model file, with proper error handling."""
        if not model_name:
            return None  # Return None for random actions

        path = os.path.join(self.models_dir, model_name)
        if not os.path.exists(path):
            print(f"⚠️  Model not found: {path}")
            return None

        try:
            agent = DQNAgent(state_size=8, action_size=4)
            agent.load(path)
            agent.epsilon = 0.0  # No exploration for video
            return agent
        except Exception as e:
            print(f"❌ Error loading {model_name}: {e}")
            return None

    def record_episode(self, agent, seed: int):
        """Record a single episode with the given agent."""
        env = gym.make('LunarLander-v2', render_mode='rgb_array')
        state, _ = env.reset(seed=seed)

        if self.reward_shaper:
            self.reward_shaper.reset()

        frames = []
        rewards = []
        original_rewards = []
        actions = []
        total_reward = 0
        total_original = 0

        for step in range(self.max_episode_steps):
            # Capture frame
            frame = env.render()
            frames.append(frame)

            # Get action (random if no agent)
            if agent is None:
                action = env.action_space.sample()
            else:
                action = agent.act(state)

            actions.append(action)

            # Step environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Track rewards
            total_original += reward

            if self.reward_shaper:
                shaped_reward = self.reward_shaper.shape_reward(
                    state, action, reward, done, step, terminated, truncated
                )
                total_reward += shaped_reward
            else:
                total_reward += reward

            rewards.append(total_reward)
            original_rewards.append(total_original)

            state = next_state

            if done:
                break

        env.close()

        # Return comprehensive episode data
        return {
            'frames': frames,
            'rewards': rewards,
            'original_rewards': original_rewards,
            'actions': actions,
            'final_reward': total_reward,
            'final_original': total_original,
            'episode_length': len(frames),
            'success': terminated and reward > 0 if 'reward' in locals() else False
        }

    def create_overlay(self, canvas, vid_w, game_h, checkpoint, episode_data, frame_idx):
        """Create minimal overlay with just episode number in larger font."""
        y_start = game_h
        overlay_bg_color = (0, 0, 0)  # Pure black background

        # Fill overlay background
        cv2.rectangle(canvas, (0, y_start), (vid_w, game_h + self.overlay_height),
                      overlay_bg_color, -1)

        # Large episode text - centered
        episode_text = checkpoint['desc']

        # Use larger font size
        font_scale = 2.5
        thickness = 4
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Get text size for centering
        (text_w, text_h), _ = cv2.getTextSize(
            episode_text, font, font_scale, thickness)

        # Center the text
        text_x = (vid_w - text_w) // 2
        text_y = y_start + (self.overlay_height + text_h) // 2

        # White text with larger size
        cv2.putText(canvas, episode_text, (text_x, text_y),
                    font, font_scale, (255, 255, 255), thickness)

    def make_video(self, seed_offset: int = 42, max_models: int = None):
        """Create the complete video showcasing model progression."""
        # Limit number of models if specified
        checkpoints = self.checkpoints[:max_models] if max_models else self.checkpoints

        if len(checkpoints) == 0:
            print("❌ No models to showcase!")
            return None

        # Video dimensions - increased resolution
        game_w, game_h = 1200, 800  # Doubled resolution
        vid_w = game_w
        vid_h = game_h + self.overlay_height

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = os.path.join(
            self.video_dir, f'moonlander_progression_{timestamp}.mp4')
        writer = cv2.VideoWriter(
            out_path, fourcc, self.effective_fps, (vid_w, vid_h))

        print(f"🎬 Creating video with {len(checkpoints)} models...")
        print(
            f"📊 Video settings: {self.duration_seconds}s @ {self.effective_fps}fps")

        # Calculate frames per model
        frames_per_model = max(1, self.total_frames // len(checkpoints))
        print(f"⏱️  {frames_per_model} frames per model")

        for i, checkpoint in enumerate(checkpoints):
            print(f"🎥 Recording {checkpoint['desc']}...")

            # Load agent
            agent = self.load_agent(checkpoint['model'])

            # Record episode
            episode_data = self.record_episode(agent, seed_offset + i * 100)

            print(f"   📈 Final score: {episode_data['final_reward']:.1f} "
                  f"({'SUCCESS' if episode_data['success'] else 'FAILED'})")

            # Sample frames to fit duration
            num_episode_frames = len(episode_data['frames'])
            if num_episode_frames == 0:
                print(f"   ⚠️  No frames recorded, skipping...")
                continue

            frame_indices = np.linspace(0, num_episode_frames - 1,
                                        frames_per_model, dtype=int)

            # Add frames to video
            for j, frame_idx in enumerate(frame_indices):
                # Get and resize game frame
                game_frame = episode_data['frames'][frame_idx]
                game_frame = cv2.resize(game_frame, (game_w, game_h))

                # Create canvas
                canvas = np.zeros((vid_h, vid_w, 3), dtype=np.uint8)
                canvas[0:game_h] = game_frame

                # Add overlay
                self.create_overlay(canvas, vid_w, game_h, checkpoint,
                                    episode_data, frame_idx)

                # Write frame
                writer.write(canvas)

        writer.release()
        print(f"✅ Video saved: {out_path}")

        # Print summary
        print(f"\n📋 Video Summary:")
        print(f"   Duration: {self.duration_seconds} seconds")
        print(f"   Models showcased: {len(checkpoints)}")
        print(f"   FPS: {self.effective_fps}")
        print(f"   Resolution: {vid_w}x{vid_h}")
        print(
            f"   Reward shaping: {'Enabled' if self.use_reward_shaping else 'Disabled'}")

        return out_path

    def create_comparison_video(self, seed: int = 42):
        """Create a side-by-side comparison of first vs best model."""
        if len(self.checkpoints) < 2:
            print("❌ Need at least 2 models for comparison")
            return None

        first_model = self.checkpoints[0]  # Random
        best_model = next((cp for cp in self.checkpoints
                          if 'best' in cp['desc'].lower()), self.checkpoints[-1])

        print(
            f"🆚 Creating comparison: {first_model['desc']} vs {best_model['desc']}")

        # Record both episodes
        first_agent = self.load_agent(first_model['model'])
        best_agent = self.load_agent(best_model['model'])

        first_data = self.record_episode(first_agent, seed)
        best_data = self.record_episode(best_agent, seed)

        # Create side-by-side video
        game_w, game_h = 1200, 800  # Doubled resolution
        vid_w = game_w * 2  # Side by side
        vid_h = game_h + self.overlay_height

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = os.path.join(
            self.video_dir, f'moonlander_comparison_{timestamp}.mp4')
        writer = cv2.VideoWriter(out_path, fourcc, self.fps, (vid_w, vid_h))

        # Sync frame counts
        max_frames = max(len(first_data['frames']), len(best_data['frames']))

        for i in range(max_frames):
            canvas = np.zeros((vid_h, vid_w, 3), dtype=np.uint8)

            # Left side (first model)
            if i < len(first_data['frames']):
                frame = cv2.resize(first_data['frames'][i], (game_w, game_h))
                canvas[0:game_h, 0:game_w] = frame

            # Right side (best model)
            if i < len(best_data['frames']):
                frame = cv2.resize(best_data['frames'][i], (game_w, game_h))
                canvas[0:game_h, game_w:] = frame

            # Add labels
            cv2.putText(canvas, first_model['desc'], (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(canvas, best_model['desc'], (game_w + 10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            writer.write(canvas)

        writer.release()
        print(f"✅ Comparison video saved: {out_path}")
        return out_path


def main():
    """Main function with customizable settings."""
    # Easy configuration
    creator = VideoCreator(
        video_dir='linkedin_video',
        models_dir='linkedin_video',  # Look for models in linkedin_video folder
        fps=30,                       # Base frame rate
        speed_multiplier=3.0,         # 3x speed - skip every 3rd frame to make episodes faster
        duration_seconds=60,          # This is now ignored - video length depends on episodes
        max_episode_steps=200,        # Limit episode length to 200 steps
        overlay_height=100,           # Smaller space for minimal text
        use_reward_shaping=True,      # Show both shaped and original rewards
    )

    # Create main progression video
    video_path = creator.make_video(seed_offset=42, max_models=8)

    # Optionally create comparison video
    # comparison_path = creator.create_comparison_video(seed=42)

    print(f"\n🎉 Video creation complete!")
    if video_path:
        print(f"📱 Ready for LinkedIn: {video_path}")


if __name__ == '__main__':
    main()
