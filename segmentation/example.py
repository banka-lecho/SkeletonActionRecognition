import torch
import numpy as np
from PIL import Image
from segmentation.utils import show_masks
from segmentation import SamSegmenter

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

image = Image.open(
    '/Users/anastasiaspileva/PycharmProjects/ActionRecognition/dataset/frames/person_eats/0A2B7027-E7B2-47E2-B384-14E054749B70_0/frame_0000.jpg')
image = np.array(image.convert("RGB"))

sam2_checkpoint = "/Users/anastasiaspileva/PycharmProjects/ActionRecognition/segmentation/checkpoints/sam2.1_hiera_s.pt"
model_cfg = "/Users/anastasiaspileva/PycharmProjects/ActionRecognition/segmentation/configs/sam2.1_hiera_s.yaml"

input_points = np.array([[270, 160]])
input_labels = np.array([1])

seg = SamSegmenter(model_path=sam2_checkpoint, config_path=model_cfg, device=device)
seg.configure_params(image=image, points=input_points, labels=input_labels)

image, masks, scores = seg.predict(image, points=input_points, labels=input_labels)
show_masks(image, masks, scores, point_coords=input_points, input_labels=input_labels)
