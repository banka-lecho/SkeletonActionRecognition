import logging
import pandas as pd
from typing import Dict
from pathlib import Path
from typing import Optional
from sklearn.model_selection import train_test_split


def split_to_train_test(df: pd.DataFrame) -> pd.DataFrame:
    train_val_df, val_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df['target'],
        random_state=42
    )

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.25,
        stratify=train_val_df['target'],
        random_state=42
    )

    for split_df, split_name in zip([train_df, val_df], ['train', 'val', 'test']):
        split_df['split'] = split_name

    final_df = pd.concat([train_df, val_df], ignore_index=True)
    return final_df


def add_targets(df: pd.DataFrame, frames_dir: Path, encoder: Dict[str, int]) -> pd.DataFrame:
    new_rows = []
    for action_dir in frames_dir.iterdir():
        if not action_dir.is_dir():
            continue

        action_name = action_dir.name
        if action_name not in encoder:
            continue

        for video_dir in action_dir.iterdir():
            video_dir = video_dir.parent / video_dir.stem
            video_name = video_dir.name
            keypoints_path = str(Path('dataset/action_dataset/keypoints') / action_name / video_name)
            if keypoints_path not in df['keypoints_path'].values and not keypoints_path.endswith('.DS_Store'):
                new_row = {
                    'video_path': str(Path('dataset/videos') / action_name / f"{video_name}.mp4"),
                    'class': action_name,
                    'target': encoder[action_name],
                    'split': '',
                    'masks_path': str(Path('dataset/action_dataset/masks') / action_name / video_name),
                    'keypoints_path': keypoints_path,
                    'flow_path': str(Path('dataset/action_dataset/keypoints') / action_name / video_name)
                }
                new_rows.append(new_row)

    return pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)


def create_targets_csv(logger: Optional[logging.Logger], path_to_frames: Path, path_to_labels: Path) -> None:
    try:
        encoder = {
            'hand_interacts_with_person_shakehands': 0,
            'person_eats': 1,
            'person_interacts_with_laptop': 2,
            'person_steals_object': 3,
            'person_talks_on_phone': 4
        }

        if path_to_labels.exists():
            df = pd.read_csv(path_to_labels)
        else:
            df = pd.DataFrame(columns=[
                'video_path', 'class', 'target', 'split',
                'masks_path', 'keypoints_path', 'flow_path'
            ])

        df = add_targets(df, path_to_frames, encoder)
        df = split_to_train_test(df)
        df.to_csv(path_to_labels, index=False)
    except Exception as e:
        logger.critical(f"Critical error in getting targets: {str(e)}", exc_info=True)
        raise
