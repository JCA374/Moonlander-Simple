import cv2
import os
import glob
import re
import numpy as np


class MovieMerger:
    """
    Merges existing video files by concatenating them in order of their numeric prefix,
    with a 1-second pause at the end of each clip and high-quality comments with text wrapping.
    """

    def __init__(self, videos_folder='model_to_video', output_name='merged_video.mp4',
                 comments=None, default_comment=""):
        self.videos_folder = videos_folder
        self.output_name = output_name
        # Comments per video: list of strings for first, second, third, etc.
        self.comments = comments or []
        # Default comment for any videos beyond the list
        self.default_comment = default_comment

        # Enhanced panel settings for better quality
        self.panel_height = 120  # Increased height for potential multi-line text
        self.panel_bg_color = (20, 20, 20)
        # Slightly brighter for better visibility
        self.separator_color = (80, 80, 80)
        self.text_color = (255, 255, 255)  # Pure white for better contrast

        # Enhanced font settings for better quality
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.9  # Slightly larger
        self.font_thickness = 2
        self.line_spacing = 35  # Spacing between lines
        self.max_text_width = 0.9  # Use 90% of panel width for text

    def wrap_text(self, text, width):
        """
        Wraps text to fit within the specified width.
        Returns a list of lines.
        """
        words = text.split(' ')
        lines = []
        current_line = []

        max_width = int(width * self.max_text_width)

        for word in words:
            # Test if adding this word exceeds width
            test_line = ' '.join(current_line + [word])
            text_size = cv2.getTextSize(
                test_line, self.font, self.font_scale, self.font_thickness)[0]

            if text_size[0] <= max_width:
                current_line.append(word)
            else:
                if current_line:  # Save current line and start new one
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:  # Single word too long, add it anyway
                    lines.append(word)
                    current_line = []

        if current_line:
            lines.append(' '.join(current_line))

        return lines

    def create_comment_panel(self, width, text):
        """
        Creates a high-quality panel with the comment text, centered and wrapped if needed.
        """
        # Create panel with anti-aliased background
        panel = np.zeros((self.panel_height, width, 3), dtype=np.uint8)
        panel[:] = self.panel_bg_color

        # Draw separator line with anti-aliasing
        cv2.line(panel, (0, 1), (width, 1),
                 self.separator_color, 2, cv2.LINE_AA)

        # Wrap text if needed
        lines = self.wrap_text(text, width)

        # Calculate total height needed for all lines
        total_text_height = len(lines) * self.line_spacing

        # Start Y position to center the text block vertically
        start_y = (self.panel_height - total_text_height) // 2 + \
            self.line_spacing // 2

        # Draw each line centered horizontally
        for i, line in enumerate(lines):
            text_size = cv2.getTextSize(
                line, self.font, self.font_scale, self.font_thickness)[0]
            text_x = (width - text_size[0]) // 2
            text_y = start_y + i * self.line_spacing

            # Draw text with anti-aliasing for better quality
            cv2.putText(panel, line, (text_x, text_y), self.font,
                        self.font_scale, self.text_color, self.font_thickness, cv2.LINE_AA)

        return panel

    def merge_videos(self):
        """Concatenate all video files in ascending numeric order into a single output with high-quality comments."""
        video_files = glob.glob(os.path.join(self.videos_folder, '*.mp4'))
        if not video_files:
            print(f"❌ No video files found in {self.videos_folder}")
            return

        # Sort by numeric prefix (lowest-first)
        def sort_key(path):
            name = os.path.basename(path)
            match = re.search(r"(\d+)", name)
            return int(match.group(1)) if match else float('inf')

        video_files.sort(key=sort_key)
        print(
            f"🎬 Merging {len(video_files)} videos with high-quality text overlays...")

        # Open first video to get properties
        cap0 = cv2.VideoCapture(video_files[0])
        if not cap0.isOpened():
            print(f"❌ Cannot open {video_files[0]}")
            return
        fps = cap0.get(cv2.CAP_PROP_FPS)
        width = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap0.release()

        # Setup writer with extra panel height
        # Use higher quality codec if available
        total_height = height + self.panel_height
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # Create VideoWriter with higher quality settings
        writer = cv2.VideoWriter(
            self.output_name, fourcc, fps, (width, total_height))

        pause_frames = int(fps * 1)  # 1-second pause

        for idx, path in enumerate(video_files):
            name = os.path.basename(path)
            print(f"   [{idx+1}/{len(video_files)}] Adding {name}")
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                print(f"    ❌ Skipping {path}, cannot open file.")
                continue

            # Determine comment text
            text = self.comments[idx] if idx < len(
                self.comments) else self.default_comment

            last_combined = None
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Resize if needed (using INTER_CUBIC for better quality)
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height),
                                       interpolation=cv2.INTER_CUBIC)

                # Create high-quality panel and combine
                panel = self.create_comment_panel(width, text)
                combined = np.vstack([frame, panel])
                writer.write(combined)
                last_combined = combined
                frame_count += 1

            cap.release()
            print(f"      ✓ Added {frame_count} frames")

            # Add 1-second pause of last frame+panel
            if last_combined is not None:
                for _ in range(pause_frames):
                    writer.write(last_combined)

        writer.release()
        print(f"✅ High-quality merged video saved as: {self.output_name}")
        print(f"   Resolution: {width}x{total_height}")
        print(f"   FPS: {fps}")


def main():
    # Example with both short and long comments to demonstrate wrapping
    merger = MovieMerger(
        videos_folder='model_to_video',
        output_name='merged_video_hq.mp4',
        comments=["Episode 400: Basic Control",
                  "Episode 800: Improved control",
                  "Episode 10000: Future improvements",
                  "Episode 10800: Successful landings",
                  "Episode 24000: Learning to land quickly and successfully"],
        default_comment="Training in progress..."
    )
    merger.merge_videos()


if __name__ == "__main__":
    main()
