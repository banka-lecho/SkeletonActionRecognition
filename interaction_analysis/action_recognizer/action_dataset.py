import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from typing import Tuple, Dict, List


class SkeletonDataset(Dataset):
    """Dataset for multimodal action recognition with keypoints sequences.

    Args:
        labels_path (str): Path to CSV file with dataset labels and paths.
        split (str): 'train' or 'val' to select data split.
        frame_step (int): Step between frames when loading sequences. Default: 1.
        sequence_length (int): Number of frames in each sequence. Default: 30.
    """

    def __init__(self,
                 labels_path: str,
                 split: str = 'train',
                 frame_step: int = 1,
                 sequence_length: int = 30):
        self.root_dir = Path(labels_path).resolve().parent.parent.parent
        self.split = split
        self.frame_step = frame_step
        self.sequence_length = sequence_length

        # Load labels
        self.labels_df = pd.read_csv(os.path.join(self.root_dir, labels_path))
        self.labels_df = self.labels_df[self.labels_df['split'] == split]

        # Prepare samples
        self.samples = self._prepare_samples()

        # Build action to index mapping
        self.action_to_idx = {action: idx for idx, action in
                              enumerate(sorted(self.labels_df['target'].unique()))}

    def _prepare_samples(self) -> List[Dict]:
        """Prepare list of samples by scanning the dataset directories."""
        samples = []

        for _, row in self.labels_df.iterrows():
            action = row['class']
            target = row['target']
            video_path = row['video_path']
            keypoints_path = row['keypoints_path']

            # Get all available frames
            keys_dir = os.path.join(self.root_dir, keypoints_path)
            frame_files = sorted([f for f in os.listdir(keys_dir) if f.endswith('.npz')])

            # Create sequences with frame_step
            for i in range(0, len(frame_files) - self.sequence_length * self.frame_step + 1, self.frame_step):
                sequence = []
                for j in range(self.sequence_length):
                    frame_idx = i + j * self.frame_step
                    frame_id = frame_files[frame_idx][:-4]
                    sequence.append({
                        'frame_id': frame_id,
                        'keypoints_path': os.path.join(keypoints_path, frame_id + '.npz')
                    })

                sample = {
                    'action': action,
                    'video': video_path,
                    'target': target,
                    'sequence': sequence
                }
                samples.append(sample)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get sample by index.
        Возвращает Batch data размером [B, T, V, C]
            B — batch size
            T — последовательность кадров
            V — количество вершин
            С — количество признаков для одной точки
        """
        sample = self.samples[idx]
        sequence_data = []

        # Load all frames in the sequence
        for frame in sample['sequence']:
            kp_path = os.path.join(self.root_dir, frame['keypoints_path'])
            kp_data = np.load(kp_path, allow_pickle=True)
            points_dict = kp_data['points'].item()

            # Extract x, y coordinates and confidence (assuming format is [x, y, conf])
            points = np.zeros((17, 3), dtype=np.float32)
            for i in range(17):
                points[i] = points_dict[i]

            sequence_data.append(points)

        sequence_tensor = torch.from_numpy(np.array(sequence_data)).float().permute(2, 0, 1)
        label = self.action_to_idx[sample['target']]

        return sequence_tensor, label

    def get_num_classes(self) -> int:
        return len(self.action_to_idx)

    def get_action_names(self) -> List[str]:
        return list(self.action_to_idx.keys())

    @staticmethod
    def get_adj_matrix():
        num_nodes = 17  # COCO Keypoints
        adj_matrix = torch.zeros(num_nodes, num_nodes)

        edges = [
            (0, 1), (0, 2), (1, 3), (2, 4),
            (0, 5), (0, 6), (5, 7), (7, 9),
            (6, 8), (8, 10), (5, 6), (5, 11),
            (6, 12), (11, 13), (13, 15), (12, 14),
            (14, 16), (11, 12)
        ]

        for i, j in edges:
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1

        # Добавляем self-connections (полезно для GCN)
        adj_matrix += torch.eye(num_nodes)

        # Нормализация (например, симметричная нормализация D^{-1/2} A D^{-1/2})
        # TODO:: вот тут подумать, а нужна ли нам эта нормализация
        degree = adj_matrix.sum(dim=1)
        degree_sqrt_inv = torch.diag(1.0 / torch.sqrt(degree))
        adj_matrix_normalized = degree_sqrt_inv @ adj_matrix @ degree_sqrt_inv
        adj_matrix_normalized = adj_matrix_normalized.unsqueeze(0)
        return adj_matrix_normalized


if __name__ == '__main__':

    dataset = SkeletonDataset(
        labels_path='/labels.csv',
        split='train',
        frame_step=2,
        sequence_length=30
    )
    adj_matrix = dataset.get_adj_matrix()

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )

    for batch_data, batch_labels in dataloader:
        print(f"Input shape: {batch_data.shape}")
