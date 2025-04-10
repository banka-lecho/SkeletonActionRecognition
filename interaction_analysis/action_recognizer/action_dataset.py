import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from typing import Tuple, Dict, List, Optional
from torchvision.transforms.functional import resize


class ActionDataset(Dataset):
    """Dataset for multimodal action recognition with masks, optical flow and keypoints.

    Args:
        split (str): 'train' or 'val' to select data split.
        transform (callable, optional): Optional transform to be applied on samples.
        target_size (Tuple[int, int], optional): Target size for resizing masks and flow.
            If None, keeps original size. Default: (128, 128).
        frame_step (int): Step between frames when loading sequences. Default: 1.
    """

    def __init__(self,
                 labels_path: str,
                 split: str = 'train',
                 transform: Optional[callable] = None,
                 target_size: Tuple[int, int] = (128, 128),
                 frame_step: int = 1):
        self.root_dir = Path(__file__).resolve().parent.parent.parent
        self.split = split
        self.transform = transform
        self.target_size = target_size
        self.frame_step = frame_step

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
            masks_path = row['masks_path']
            flow_path = row['flow_path']
            keypoints_path = row['keypoints_path']

            # Get all available frames
            mask_dir = os.path.join(self.root_dir, masks_path)
            frame_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.npy')])

            # Create samples with frame_step
            for i in range(0, len(frame_files), self.frame_step):
                frame_id = frame_files[i][:-4]  # remove '.npy'
                sample = {
                    'action': action,
                    'video': video_path,
                    'target': target,
                    'frame_id': frame_id,
                    'mask_path': os.path.join(masks_path, frame_id + '.npy'),
                    'flow_path': os.path.join(flow_path, frame_id + '.npy'),
                    'keypoints_path': os.path.join(keypoints_path, frame_id + '.npz')
                }
                samples.append(sample)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], int]:
        sample = self.samples[idx]

        # Load mask
        mask_path = os.path.join(self.root_dir, sample['mask_path'])
        mask = torch.from_numpy(np.load(mask_path)).float().unsqueeze(0)  # [1, H, W]

        # Load optical flow
        flow_path = os.path.join(self.root_dir, sample['flow_path'])
        flow = torch.from_numpy(np.load(flow_path)).float()  # [2, H, W]
        # Проверка размерности
        if flow.dim() == 3 and flow.shape[2] == 2:
            flow = flow.permute(2, 0, 1)  # [2, H, W]

        assert flow.shape[0] == 2, f"Flow must have shape [2, H, W], got {flow.shape}"

        # Load keypoints
        kp_path = os.path.join(self.root_dir, sample['keypoints_path'])
        kp_data = np.load(kp_path, allow_pickle=True)
        points_dict = kp_data['points'].item()
        points = np.zeros((17, 3), dtype=np.float32)

        for i in range(17):
            points[i] = points_dict[i]

        points = torch.from_numpy(points).float()
        edges = torch.from_numpy(kp_data['edges']).long()

        # Resize mask and flow to target size if needed
        if self.target_size is not None:
            mask = resize(mask, self.target_size)
            flow = resize(flow, self.target_size)

        # Apply transforms if any
        if self.transform:
            mask = self.transform(mask)
            flow = self.transform(flow)

        # Prepare output
        data = {
            'mask': mask,
            'optical_flow': flow,
            'skeleton_points': points,
            'edge_index': edges
        }

        label = self.action_to_idx[sample['target']]

        return data, label

    def get_num_classes(self) -> int:
        return len(self.action_to_idx)

    def get_action_names(self) -> List[str]:
        return list(self.action_to_idx.keys())


if __name__ == '__main__':
    dataset = ActionDataset(
        labels_path='dataset/action_dataset/labels.csv',
        split='train',
        target_size=(128, 128),
        frame_step=2
    )

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )

    for batch_data, batch_labels in dataloader:
        print(f"mask batch shape: {batch_data['mask'].shape}")
        print(f"optical flow batch shape: {batch_data['optical_flow'].shape}")
        print(f"skeleton points batch shape: {batch_data['skeleton_points'].shape}")
        break
