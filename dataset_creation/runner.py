import yaml
from pathlib import Path
from logging_config import setup_logging

from dataset_creation.extract_features import DatasetPreprocessor
from dataset_creation.extract_frames import VideoFrameExtractor
from dataset_creation.extract_targets import create_targets_csv

logger = setup_logging(module_name=__name__)


def main():
    try:
        with open("../configs/models.yaml", "r") as f:
            config = yaml.safe_load(f)

        PROJECT_ROOT = Path(__file__).resolve().parent.parent
        videos_input_path = PROJECT_ROOT / config["dataset"]["videos_path"]
        frames_input_path = PROJECT_ROOT / config["dataset"]["frames_path"]
        output_base_path = PROJECT_ROOT / config["dataset"]["action_dataset"]
        labels_path = PROJECT_ROOT / config["dataset"]["labels_path"]
        model_path_yolo = PROJECT_ROOT / config["models"]["model_path"]

        # извлекаем кадры из видео
        frame_extractor = VideoFrameExtractor(videos_input_path, frames_input_path)
        frame_extractor.run()

        # извлекаем признаки для модели распознавания
        data_extractor = DatasetPreprocessor(
            videos_path=videos_input_path,
            frames_path=frames_input_path,
            output_base_path=output_base_path,
            config_path_seg=config["segmentation"]["config_path"],
            model_path_yolo=model_path_yolo)
        data_extractor.run(feature_type='keypoints')
        data_extractor.run(feature_type='masks')

        # извлекаем метки классов и сохраняем в csv
        create_targets_csv(logger, frames_input_path, labels_path)
    except Exception:
        logger.exception("Error in creating dataset")


if __name__ == '__main__':
    main()
