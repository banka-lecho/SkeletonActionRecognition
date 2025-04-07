import os
import cv2
import yaml
import logging
import numpy as np
from typing import Dict
from statistics import mode
from prefect import task, flow
from prefect.logging import get_run_logger
from pose_estimation.pose_estimator import PoseEstimator
from interaction_analysis.baseline import PoseMaskAnalyzer
from depth_estimation.depth_estimator import DepthEstimator
from segmentation.segmenter_clip_seg import ClipSegmentation

import warnings

warnings.filterwarnings("ignore")

# Устанавливаем уровень логирования для Prefect
logging.getLogger("prefect").setLevel(logging.WARNING)

with open("configs/models.yaml", "r") as f:
    config = yaml.safe_load(f)

IMAGE_PATH = config["settings"]["rtsp"]
OUTPUT_PATH = config["settings"]["frames_output_dir"]

CONFIG_SEG = config["segmentation"]["config_path"]
CONFIG_DEPTH = config["depth_estimation"]["config_path"]
MODEL_PATH_YOLO = config["pose_estimation"]["model_path"]
MODEL_PATH_DEPTH = config["depth_estimation"]["model_path"]

THRESHOLD_DISTANCE = config["action_recognizer"]["threshold_distance"]
CALL_OPENVINO_MODEL = config["pose_estimation"]["call_openvino"]

DEVICE = config["settings"]["device"]

actions_with_objects = {'food': 'eating',
                        'bottle': 'drinking',
                        'laptop': 'interacts with laptop',
                        'phone': 'interacts with phone',
                        'cigarette': 'smoking',
                        'handrail': 'holding onto the ladder'}

# Initialize models
segmenter = ClipSegmentation(CONFIG_SEG)
yolo_model = PoseEstimator(MODEL_PATH_YOLO)
depth_model = DepthEstimator(model_config=CONFIG_DEPTH, model_path=MODEL_PATH_DEPTH, device=DEVICE)
action_recognizer = PoseMaskAnalyzer(THRESHOLD_DISTANCE)


@task(retries=3, retry_delay_seconds=2, log_prints=True)
def read_image(image_name: str) -> tuple:
    """
    Read an image from the specified path with error handling.

    Args:
        image_name (str): Name of the image file to be read.

    Returns:
        tuple: A tuple containing:
            - image (numpy.ndarray): The loaded image as a NumPy array.
            - image_path (str): Full path to the input image.
            - new_path (str): Full path where the output will be saved.

    Raises:
        Exception: If the image cannot be loaded.
    """
    logger = get_run_logger()
    try:
        image_path = os.path.join(IMAGE_PATH, image_name)
        new_path = os.path.join(OUTPUT_PATH, image_name)
        image = cv2.imread(image_path)
        if image is None:
            raise Exception("Failed to load image")
        return image, image_path, new_path
    except Exception as e:
        logger.error(f"Error in read_image: {e}")
        raise


@task
def run_segmentation_model(image: np.ndarray, prompts: list) -> tuple:
    """
    Run segmentation model on the input image using specified prompts.

    Args:
        image (numpy.ndarray): Input image for segmentation.
        prompts (list): List of text prompts for segmentation.

    Returns:
        tuple: A tuple containing:
            - masks: Segmentation masks generated by the model.
            - label: Predicted label based on the segmentation.

    Raises:
        Exception: If segmentation fails.
    """
    logger = get_run_logger()
    try:
        masks, label = segmenter.predict(image, prompts)
        return masks, label
    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        raise


@task
def run_depth_model(image: np.ndarray) -> np.ndarray:
    """
    Generate depth map for the input image.

    Args:
        image (numpy.ndarray): Input image for depth estimation.

    Returns:
        numpy.ndarray: Depth map of the input image.

    Raises:
        Exception: If depth estimation fails.
    """
    logger = get_run_logger()
    try:
        depth_map = depth_model.predict(image)
        return depth_map
    except Exception as e:
        logger.error(f"Depth estimation failed: {e}")
        raise


@task
def run_keypoints_model(image: np.ndarray) -> tuple:
    """
    Detect 3D keypoints in the input image using either YOLO or OpenVINO model.

    Args:
        image (numpy.ndarray): Input image for keypoint detection.

    Returns:
        tuple: A tuple containing:
            - keypoints: Detected 3D keypoints.
            - annotated_image: Image with keypoints visualized.

    Raises:
        Exception: If no keypoints are detected or if the process fails.
    """
    logger = get_run_logger()
    try:
        if CALL_OPENVINO_MODEL:
            yolo_model.convert_to_openvino()
            keypoints, annotated_image = yolo_model.predict_openvino(image, device="GPU")
        else:
            keypoints, annotated_image = yolo_model.predict(image, verbose=False)

        if not keypoints:
            raise Exception("No skeletal points detected")
        return keypoints, annotated_image
    except Exception as e:
        logger.error(f"Error in run_keypoints_model: {e}")
        raise


@task
def classify_action(image: np.ndarray, mask: np.ndarray,
                    keypoints: list, depth_map: np.ndarray) -> bool:
    """
    Classify the action based on input features.

    Args:
        image (numpy.ndarray): Original input image.
        mask (numpy.ndarray): Segmentation mask.
        keypoints (list): Detected 3D keypoints.
        depth_map (numpy.ndarray): Estimated depth map.

    Returns:
        bool: Classification result (True if action is detected, False otherwise).
    """
    logger = get_run_logger()
    if mask is None or keypoints is None or depth_map is None:
        logger.error("Skipping due to missing data")
    answer = action_recognizer.predict(image, mask, keypoints, depth_map)
    return answer


@task
def annotate_image(answer: bool, annotated_image: np.ndarray,
                   new_path: str, label: str) -> None:
    """
    Annotate the image with the classification result and save it.
    """
    text = f"not {label}" if not answer else f"{label}"

    cv2.putText(annotated_image,
                text,
                org=(10, 50),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.0,
                color=(255, 255, 255),
                thickness=2,
                lineType=cv2.LINE_AA)
    cv2.imwrite(new_path, annotated_image)


@flow(name="action_recognition_flow")
def action_recognition_flow(actions_with_objects: Dict) -> str:
    """
    Main workflow for action recognition pipeline.

    Args:
        actions_with_objects (Dict): Dictionary mapping object names to action labels.
    """
    logger = get_run_logger()
    labels_arr = []
    for image_name in os.listdir(IMAGE_PATH):
        try:
            image, image_path, new_path = read_image(image_name)
            print(f"Картинка {image_path}")
            mask, thing = run_segmentation_model(image, list(actions_with_objects.keys()))
            depth_map = run_depth_model(image)
            keypoints, annotated_image = run_keypoints_model(image)

            action = 'do nothing'
            if thing == 'nothing':
                label = 'do nothing'
            else:
                action = classify_action(image, mask, keypoints, depth_map)
                label = actions_with_objects[thing]
            annotate_image(action, annotated_image, new_path, label)
            labels_arr.append(label)
            print('\n')
        except Exception as e:
            logger.error(f"Error processing {image_name}: {e}")
            continue
    return mode(labels_arr)


def get_action() -> None:
    """Entry point for the action recognition system."""
    action = action_recognition_flow(actions_with_objects)
    return action


get_action()
