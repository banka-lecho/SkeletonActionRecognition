import os
import cv2
from tqdm import tqdm
from pathlib import Path


class VideoFrameExtractor:
    def __init__(self, videos_path: str, frames_path: str):
        """
        Initializes the frame extractor from the video.

        :param videos_path: Path to the directory with video files
        :param frames_path: Path for saving extracted frames
        """
        self.videos_path = Path(videos_path)
        self.frames_path = Path(frames_path)
        self.frames_path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def extract_frames_from_video(video_path: Path, output_path: Path) -> None:
        """
        Extracts frames from a single video file and saves them to the specified directory.

        :param video_path: Path to the video file
        :param output_path: Directory for saving frames
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"Error: Couldn't open the video {video_path}")
                return

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_filename = output_path / f"frame_{frame_count:04d}.jpg"
                if not cv2.imwrite(str(frame_filename), frame):
                    raise Exception("Could not write image")
                frame_count += 1

            cap.release()
        except Exception as e:
            print(f"Error in video processing {video_path}: {e}")

    def process_action(self, action_name: str) -> None:
        """
        Processes all videos for the specified action.

        :param action_name: Action name (directory name)
        """
        action_path = self.videos_path / action_name
        if not action_path.is_dir():
            return

        action_output_path = self.frames_path / action_name
        action_output_path.mkdir(exist_ok=True)

        video_files = [f for f in action_path.iterdir() if f.is_file()]
        for video_file in tqdm(video_files, desc=f"Processing {action_name}"):
            video_stem = video_file.stem
            frames_output_path = action_output_path / video_stem
            frames_output_path.mkdir(exist_ok=True)

            self.extract_frames_from_video(video_file, frames_output_path)

    def run(self) -> None:
        """
        Starts the process of extracting frames for the all actions.
        """
        try:
            action_dirs = [d for d in self.videos_path.iterdir() if d.is_dir()]
            for action_dir in action_dirs:
                action_name = action_dir.name
                videos_dirs = [d for d in action_dir.iterdir()]
                for video_path in tqdm(videos_dirs, desc=f"Get frames action's videos -- {action_name}"):
                    frames_output_path = self.frames_path / action_name / video_path.stem
                    os.makedirs(frames_output_path, exist_ok=True)
                    if not any(frames_output_path.iterdir()):
                        self.extract_frames_from_video(video_path, frames_output_path)
        except Exception as e:
            print(f"Error processing videos: {e}")
