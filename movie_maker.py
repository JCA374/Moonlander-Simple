import gymnasium as gym
import numpy as np
import cv2
import os
import glob
import re
from datetime import datetime
from dqn_agent import DQNAgent
from reward_shaper import RewardShaper


class ModelVideoMerger:
    """
    Enhanced video creator that merges all models with custom explanations
    and clear Game Score vs AI Score display
    """

    def __init__(self, models_folder='model_to_video', output_folder='merged_videos'):
        self.models_folder = models_folder
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)

        # Video settings
        self.fps = 30
        self.speed_multiplier = 2.0  # 2x speed
        self.resolution = (1200, 900)  # Width x Height for game area
        self.overlay_height = 200  # Space for text at bottom

        # Initialize reward shaper for AI scores
        self.reward_shaper = RewardShaper()

        # Model explanations - CUSTOMIZE THESE!
        self.model_explanations = {
            'moonlander_best.pth': 'The best performing model - masters precision landing',
            'moonlander_final.pth': 'Final model after complete training',
            'moonlander_checkpoint_0.pth': 'Untrained AI - completely random actions',
            'moonlander_checkpoint_2500.pth': 'Early learning - basic movement control',
            'moonlander_checkpoint_5000.pth': 'Developing skills - learning to approach pad',
            'moonlander_checkpoint_7500.pth': 'Intermediate - better fuel management',
            'moonlander_checkpoint_10000.pth': 'Advanced - consistent landing attempts',
            'moonlander_checkpoint_12500.pth': 'Expert level - refined technique',
            'moonlander_checkpoint_15000.pth': 'Near mastery - reliable landings',
            'moonlander_checkpoint_17500.pth': 'Mastery - optimal performance',
            'moonlander_checkpoint_20000.pth': 'Peak performance - precision control',
            'moonlander_improving_4600.pth': 'Breakthrough moment - first consistent landings',
            'moonlander_improving_6400.pth': 'Major improvement - reliable success',
        }

    def discover_and_sort_models(self):
        """Find all models and sort them chronologically"""
        model_files = glob.glob(os.path.join(self.models_folder, '*.pth'))

        if not model_files:
            print(f"❌ No models found in {self.models_folder}/")
            return []

        models = []
        for file_path in model_files:
            filename = os.path.basename(file_path)
            episode_num = self.extract_episode_number(filename)

            models.append({
                'filename': filename,
                'path': file_path,
                'episode': episode_num,
                'explanation': self.model_explanations.get(
                    filename,
                    f'Training model from episode {episode_num}'
                )
            })

        # Sort by episode number (lowest to highest, then best)
        models.sort(key=lambda x: x['episode'])

        print(f"📋 Found {len(models)} models in chronological order:")
        for model in models:
            if model['episode'] < 999990:  # Regular episode numbers
                print(f"  Episode {model['episode']:>6}: {model['filename']}")
            else:  # Special models (final, best)
                print(f"  {'Final/Best':>6}: {model['filename']}")

        return models

    def extract_episode_number(self, filename):
        """Extract episode number for chronological sorting"""
        # Special handling for best/final models to ensure they come last
        if 'best' in filename:
            return 999999  # Best goes very last
        elif 'final' in filename:
            return 999998  # Final goes second to last

        # Extract episode numbers from different filename patterns
        if 'checkpoint' in filename:
            match = re.search(r'checkpoint_(\d+)', filename)
            return int(match.group(1)) if match else 0
        elif 'improving' in filename:
            match = re.search(r'improving_(\d+)', filename)
            return int(match.group(1)) if match else 0
        elif 'backup' in filename:
            match = re.search(r'backup_(\d+)', filename)
            return int(match.group(1)) if match else 0
        else:
            # Try to find any number in filename
            match = re.search(r'(\d+)', filename)
            return int(match.group(1)) if match else 0

    def record_model_episode(self, model_info, seed=42):
        """Record one episode for a model"""
        env = gym.make('LunarLander-v2', render_mode='rgb_array')

        # Load agent
        agent = DQNAgent(8, 4)
        try:
            agent.load(model_info['path'])
            agent.epsilon = 0  # No exploration
        except Exception as e:
            print(f"❌ Error loading {model_info['filename']}: {e}")
            env.close()
            return None

        # Reset environment and reward shaper
        state, _ = env.reset(seed=seed)
        self.reward_shaper.reset()

        frames = []
        game_score = 0  # Original game score
        ai_score = 0    # AI training score

        print(f"🎬 Recording {model_info['filename']}")

        for step in range(800):  # Max steps
            # Capture frame
            frame = env.render()
            frames.append(frame)

            # Get action and step
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Update scores
            game_score += reward

            # Get AI training score using reward shaper
            shaped_reward = self.reward_shaper.shape_reward(
                state, action, reward, done, step, terminated, truncated
            )
            ai_score += shaped_reward

            state = next_state

            if done:
                # Hold final frame for 1 second
                for _ in range(self.fps):
                    frames.append(frame)
                break

        env.close()

        return {
            'frames': frames,
            'game_score': game_score,
            'ai_score': ai_score,
            'success': terminated and reward > 0 if 'reward' in locals() else False,
            'episode_length': len(frames)
        }

    def create_overlay(self, frame, model_info, episode_data, current_game_score, current_ai_score):
        """Create enhanced overlay with scores and explanation"""
        height, width = frame.shape[:2]

        # Create overlay area (bottom section)
        overlay = frame.copy()
        overlay_start_y = height - self.overlay_height

        # Semi-transparent dark background
        cv2.rectangle(overlay, (0, overlay_start_y),
                      (width, height), (0, 0, 0), -1)
        frame_with_overlay = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # Title (model name)
        model_display_name = self.get_display_name(model_info['filename'])
        cv2.putText(frame_with_overlay, model_display_name, (20, overlay_start_y + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        # Explanation text (smaller, wrapped if needed)
        explanation = model_info['explanation']
        self.draw_wrapped_text(frame_with_overlay, explanation,
                               (20, overlay_start_y + 70), width - 40, 0.8, (200, 200, 200))

        # Score displays in top-left corner with final scores
        self.draw_score_display(frame_with_overlay, current_game_score, current_ai_score,
                                episode_data['game_score'], episode_data['ai_score'])

        return frame_with_overlay

    def get_display_name(self, filename):
        """Convert filename to readable display name"""
        if 'best' in filename:
            return '🏆 Best Model'
        elif 'final' in filename:
            return '🎓 Final Model'
        elif 'checkpoint_0' in filename:
            return '🎲 Untrained AI'
        elif 'checkpoint' in filename:
            match = re.search(r'checkpoint_(\d+)', filename)
            if match:
                episode = int(match.group(1))
                return f'📈 Episode {episode:,}'
        elif 'improving' in filename:
            match = re.search(r'improving_(\d+)', filename)
            if match:
                episode = int(match.group(1))
                return f'⭐ Breakthrough: Episode {episode:,}'

        return filename.replace('.pth', '').replace('_', ' ').title()

    def draw_wrapped_text(self, frame, text, start_pos, max_width, font_scale, color):
        """Draw text with word wrapping"""
        x, y = start_pos
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2

        words = text.split(' ')
        lines = []
        current_line = []

        for word in words:
            test_line = ' '.join(current_line + [word])
            (text_width, _), _ = cv2.getTextSize(
                test_line, font, font_scale, thickness)

            if text_width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]

        if current_line:
            lines.append(' '.join(current_line))

        # Draw lines
        line_height = int(font_scale * 30)
        for i, line in enumerate(lines[:2]):  # Max 2 lines
            cv2.putText(frame, line, (x, y + i * line_height),
                        font, font_scale, color, thickness)

    def draw_score_display(self, frame, game_score, ai_score, final_game_score=None, final_ai_score=None):
        """Draw larger score display in top-left with final scores"""
        height, width = frame.shape[:2]

        # Larger background box in top-left
        box_width = 400
        box_height = 180
        box_x = 20  # Left side
        box_y = 20  # Top

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_width, box_y + box_height),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        # White border
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + box_height),
                      (255, 255, 255), 3)

        # Center the content within the box
        content_start_x = box_x + 20

        # Title - larger and centered
        title_text = "SCORES"
        title_size = cv2.getTextSize(
            title_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
        title_x = box_x + (box_width - title_size[0]) // 2
        cv2.putText(frame, title_text, (title_x, box_y + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

        # Current scores - larger font
        y_offset = 70

        # Game Score (current)
        game_text = f"Game: {game_score:.0f}"
        cv2.putText(frame, game_text, (content_start_x, box_y + y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 255, 100), 2)

        # AI Score (current)
        ai_text = f"AI: {ai_score:.0f}"
        cv2.putText(frame, ai_text, (content_start_x, box_y + y_offset + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 150, 255), 2)

        # Final scores (if provided)
        if final_game_score is not None and final_ai_score is not None:
            # Separator line
            cv2.line(frame, (content_start_x, box_y + y_offset + 50),
                     (box_x + box_width - 20, box_y + y_offset + 50), (100, 100, 100), 1)

            # Final scores
            final_text = "FINAL:"
            cv2.putText(frame, final_text, (content_start_x, box_y + y_offset + 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            final_game_text = f"Game: {final_game_score:.0f}"
            cv2.putText(frame, final_game_text, (content_start_x, box_y + y_offset + 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 255, 150), 2)

            final_ai_text = f"AI: {final_ai_score:.0f}"
            cv2.putText(frame, final_ai_text, (content_start_x, box_y + y_offset + 125),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 200, 255), 2)

        # Explanation - centered at bottom
        explanation = "Game=Real | AI=Training"
        exp_size = cv2.getTextSize(
            explanation, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        exp_x = box_x + (box_width - exp_size[0]) // 2
        cv2.putText(frame, explanation, (exp_x, box_y + box_height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    def create_merged_video(self, output_filename='moonlander_progression.mp4', max_models=None):
        """Create the merged video with all models"""
        models = self.discover_and_sort_models()

        if not models:
            print("❌ No models found!")
            return None

        if max_models:
            models = models[:max_models]

        print(f"🎬 Creating merged video with {len(models)} models")

        # Setup video writer
        game_width, game_height = self.resolution
        total_height = game_height + self.overlay_height

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = os.path.join(self.output_folder, output_filename)
        out = cv2.VideoWriter(output_path, fourcc, self.fps,
                              (game_width, total_height))

        for i, model_info in enumerate(models):
            print(
                f"📹 Processing {i+1}/{len(models)}: {model_info['filename']}")

            # Record episode
            episode_data = self.record_model_episode(model_info, seed=42+i)
            if not episode_data:
                continue

            # Process frames
            for frame_idx, frame in enumerate(episode_data['frames']):
                # Skip frames for speed
                if frame_idx % int(self.speed_multiplier) != 0:
                    continue

                # Resize frame
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame = cv2.resize(frame, self.resolution)

                # Calculate current scores (proportional to frame progress)
                progress = frame_idx / len(episode_data['frames'])
                current_game_score = episode_data['game_score'] * progress
                current_ai_score = episode_data['ai_score'] * progress

                # Create frame with overlay
                final_frame = np.zeros(
                    (total_height, game_width, 3), dtype=np.uint8)
                final_frame[:game_height] = frame

                final_frame = self.create_overlay(
                    final_frame, model_info, episode_data,
                    current_game_score, current_ai_score
                )

                out.write(final_frame)

            # Add transition pause
            if i < len(models) - 1:  # Not the last model
                for _ in range(self.fps // 2):  # 0.5 second pause
                    out.write(final_frame)

        out.release()

        print(f"✅ Merged video created: {output_path}")
        self.print_video_summary(models, output_path)

        return output_path

    def print_video_summary(self, models, output_path):
        """Print comprehensive video summary"""
        print(f"\n" + "="*60)
        print("📊 VIDEO CREATION SUMMARY")
        print("="*60)
        print(f"📁 Output: {output_path}")
        print(f"🎬 Models included: {len(models)}")
        print(f"⚡ Speed: {self.speed_multiplier}x")
        print(
            f"📐 Resolution: {self.resolution[0]}x{self.resolution[1] + self.overlay_height}")
        print(f"🎯 Frame rate: {self.fps} FPS")

        print(f"\n📋 Models showcased:")
        for i, model in enumerate(models, 1):
            print(f"  {i:2d}. {self.get_display_name(model['filename'])}")

        print(f"\n💡 Score Legend:")
        print(f"   🎮 Game Score: Original lunar lander scoring (-100 to +300)")
        print(f"   🤖 AI Score: Enhanced training score (helps learning)")
        print("="*60)

    def create_side_by_side_comparison(self, model1_name, model2_name,
                                       output_filename='comparison.mp4'):
        """Create side-by-side comparison of two specific models"""
        models = self.discover_and_sort_models()

        model1 = next(
            (m for m in models if model1_name in m['filename']), None)
        model2 = next(
            (m for m in models if model2_name in m['filename']), None)

        if not model1 or not model2:
            print("❌ One or both models not found!")
            return None

        print(
            f"🆚 Creating comparison: {model1['filename']} vs {model2['filename']}")

        # Record both episodes
        episode1 = self.record_model_episode(model1)
        episode2 = self.record_model_episode(model2)

        if not episode1 or not episode2:
            print("❌ Failed to record episodes")
            return None

        # Setup side-by-side video
        game_width, game_height = self.resolution
        total_width = game_width * 2
        total_height = game_height + self.overlay_height

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = os.path.join(self.output_folder, output_filename)
        out = cv2.VideoWriter(output_path, fourcc, self.fps,
                              (total_width, total_height))

        max_frames = max(len(episode1['frames']), len(episode2['frames']))

        for i in range(max_frames):
            canvas = np.zeros((total_height, total_width, 3), dtype=np.uint8)

            # Left side (model1)
            if i < len(episode1['frames']):
                frame1 = cv2.cvtColor(episode1['frames'][i], cv2.COLOR_RGB2BGR)
                frame1 = cv2.resize(frame1, self.resolution)
                canvas[:game_height, :game_width] = frame1

            # Right side (model2)
            if i < len(episode2['frames']):
                frame2 = cv2.cvtColor(episode2['frames'][i], cv2.COLOR_RGB2BGR)
                frame2 = cv2.resize(frame2, self.resolution)
                canvas[:game_height, game_width:] = frame2

            # Add labels
            cv2.putText(canvas, self.get_display_name(model1['filename']),
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(canvas, self.get_display_name(model2['filename']),
                        (game_width + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

            out.write(canvas)

        out.release()
        print(f"✅ Comparison video created: {output_path}")
        return output_path


def main():
    """Main function with easy customization"""

    # Create the merger
    merger = ModelVideoMerger(
        models_folder='model_to_video',  # Folder containing your models
        output_folder='merged_videos'    # Where to save the final video
    )

    # Customize settings
    merger.fps = 30
    merger.speed_multiplier = 2.0  # 2x speed
    merger.resolution = (1200, 900)  # Higher resolution

    # CUSTOMIZE EXPLANATIONS HERE!
    merger.model_explanations.update({
        'your_model_name.pth': 'Your custom explanation here',
        # Add more as needed...
    })

    print("🎬 MoonLander Model Video Merger")
    print("="*50)

    # Create the main progression video
    video_path = merger.create_merged_video(
        output_filename='moonlander_complete_progression.mp4',
        max_models=10  # Limit to first 10 models, or None for all
    )

    # Optionally create a comparison video
    # comparison_path = merger.create_side_by_side_comparison(
    #     'checkpoint_0', 'best', 'before_vs_after.mp4'
    # )

    print(f"\n🎉 Video creation complete!")
    if video_path:
        print(f"📱 Ready to share: {video_path}")


if __name__ == "__main__":
    main()
