import os
import logging
import functools
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from typing import Optional
from multiprocessing import Pool

from pose_estimation.pose_estimator import PoseEstimator
from segmentation.segmenter_clip_seg import ClipSegmentation


class DatasetPreprocessor:
    """
    Class for creating dataset for AdaptedTwoStreamGCN model
        data_dir: directory with files in format:
            data_dir/
                masks/         # .npy files of masks
                    video1
                    ...
                    video_n
                skeletons/     # .npz files of skeletons
                    video1
                    ...
                    video_n
                labels.npy     # labels
    """

    def __init__(
            self,
            videos_path: str,
            frames_path: str,
            output_base_path: str,
            config_path_seg: str,
            model_path_yolo: str,
            logger: Optional[logging.Logger] = None
    ):
        self.logger = logger if logger else logging.getLogger(__name__)
        self.videos_path = Path(videos_path)
        self.frames_path = Path(frames_path)
        self.output_base_path = Path(output_base_path)

        self.output_paths = {
            'masks': self.output_base_path / "action_dataset" / "masks",
            'keypoints': self.output_base_path / "action_dataset" / "keypoints"
        }

        self.action_prompts = {
            "eats": ["bottle", "cup", "food", "plate"],
            "smoke": ["cigarette"],
            "talks_on_phone": ["phone"],
            "interacts_with_laptop": ["laptop"]
        }

        self.yolo_model = PoseEstimator(model_path_yolo)
        self.seg_model = ClipSegmentation(config_path_seg)

    def get_keypoints(self,
                      frame_path: Path,
                      output_path: Path):

        """Get skeleton points from frame and save it in output_dirs"""
        try:
            image = Image.open(frame_path)
            frame_id = frame_path.stem
        except Exception as e:
            self.logger.error(f"Error opening image {frame_path}: {str(e)}", exc_info=True)
            raise

        try:
            keypoints, _ = self.yolo_model.predict(image, verbose=False)
            if keypoints:
                pose_result = keypoints[0]
                keypoints = pose_result.keypoints.xy.cpu().numpy()[0]
                if pose_result.keypoints.conf is not None:
                    conf = pose_result.keypoints.conf.cpu().numpy()[0].squeeze()
                else:
                    conf = 0
                points = {}
                for point_id, (x, y) in enumerate(keypoints, start=0):
                    points[point_id] = (float(x), float(y), float(conf[point_id]))
                edges = self._get_skeleton_edges(len(points))

                np.savez(
                    output_path / f"{frame_id}.npz",
                    points=points,
                    edges=edges
                )
        except Exception as e:
            self.logger.error(f"Error processing keypoints for {frame_path}: {str(e)}", exc_info=True)
            raise

    def get_object_mask(
            self,
            frame_path: Path,
            output_path: Path,
            action_name: Optional[str] = None
    ) -> None:
        """Get object mask from frame and save it in output_dirs"""
        try:
            image = Image.open(frame_path)
            frame_id = frame_path.stem
        except Exception:
            self.logger.exception(f"Error opening image {frame_path}:")
            raise

        try:
            if action_name and action_name in self.action_prompts:
                mask, _ = self.seg_model.predict(image, self.action_prompts[action_name])
            else:
                mask = np.zeros(image.size)
            np.save(output_path / f"{frame_id}.npy", mask)
        except Exception:
            self.logger.exception(f"Error processing masks for {frame_path}:")
            raise

    @staticmethod
    def _get_skeleton_edges(num_points: int) -> np.ndarray:
        """Define skeleton connections between keypoints"""
        edges = [
            [0, 1], [0, 2], [1, 3], [2, 4],  # Head
            [5, 7], [7, 9], [6, 8], [8, 10],  # Arms
            [11, 13], [13, 15], [12, 14], [14, 16],  # Legs
            [5, 6], [11, 12], [5, 11], [6, 12]  # Body
        ]
        return np.array(edges).T if num_points == 17 else np.zeros((2, 0))

    def process_frames(
            self,
            sequence_path: Path,
            action_name: str,
            type_extraction: str
    ) -> None:
        """Process all frames in one frames sequence"""
        output_path = Path(self.output_paths[type_extraction]) / action_name / sequence_path.name
        output_path.mkdir(parents=True, exist_ok=True)
        frame_paths = sorted(sequence_path.glob("*.jpg"))
        if not frame_paths:
            return

        for frame_path in frame_paths:
            if type_extraction == "keypoints":
                self.get_keypoints(frame_path=frame_path, output_path=output_path)
            else:
                self.get_object_mask(frame_path, output_path, action_name)

    def run(self, feature_type: str) -> None:
        try:
            action_dirs = [d for d in self.frames_path.iterdir() if d.is_dir()]
            for action_dir in action_dirs:
                action_name = action_dir.name
                videos_dirs = [d for d in action_dir.iterdir()]
                for frames_path in tqdm(videos_dirs, desc=f"Get frames action's videos -- {action_name}"):
                    try:
                        feature_output_path = self.output_paths[feature_type] / action_name / frames_path.stem
                        os.makedirs(feature_output_path, exist_ok=True)
                        if not any(feature_output_path.iterdir()):
                            self.process_frames(sequence_path=frames_path,
                                                action_name=action_name,
                                                type_extraction=feature_type)
                    except Exception:
                        self.logger.exception(f"Error processing {frames_path}:")
                        continue
        except Exception as e:
            self.logger.critical(f"Critical error in run({feature_type}): {str(e)}", exc_info=True)
            raise


def run_parallel(self) -> None:
    """Main processing pipeline in parallel"""
    action_dirs = [d for d in self.videos_path.iterdir() if d.is_dir()]

    with Pool(processes=4) as pool:
        for action_dir in tqdm(action_dirs, desc="Processing actions"):
            action_name = action_dir.name
            sequence_dirs = [d for d in action_dir.iterdir() if d.is_dir()]

            process_func = functools.partial(
                self.process_video_sequence,
                action_name=action_name
            )

            pool.map(process_func, sequence_dirs)
