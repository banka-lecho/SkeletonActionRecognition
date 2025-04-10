import yaml
import pandas as pd
from sklearn.metrics import precision_score, recall_score
from interaction_analysis.baseline import PoseMaskAnalyzer
from main import get_action


def get_predictions(df: pd.DataFrame, model) -> pd.DataFrame:
    """Get predictions for set"""
    predictions = pd.DataFrame(columns=['video_path', 'target', 'prediction'])
    for row in df.iterrows():
        video_path, target = row['video_path'], row['target']
        pred = get_action(video_path, model)
        predictions.append({'video_path': video_path, 'target': target, 'prediction': pred})
    return predictions


def calc_metrics(predictions: pd.DataFrame, desc='') -> float:
    """Calculate metrics for set"""
    print(f"Calculating map for {desc} videos: {precision_score(predictions['target'], predictions['prediction'])}")
    print(f"Calculating map for {desc} videos: {recall_score(predictions['target'], predictions['prediction'])}")


if __name__ == '__main__':
    with open("configs/models.yaml", "r") as f:
        config = yaml.safe_load(f)

    THRESHOLD_DISTANCE = config["action_recognizer"]["threshold_distance"]

    TRAIN_PATH = config["dataset"]["train_labels_path"]
    VALID_PATH = config["dataset"]["valid_labels_path"]

    model = PoseMaskAnalyzer(base_threshold=THRESHOLD_DISTANCE, use_depth=False)

    train_df, valid_df = pd.read_csv(TRAIN_PATH), pd.read_csv(VALID_PATH)
    predictions_train = get_predictions(train_df, model)
    predictions_valid = get_predictions(valid_df, model)

    calc_metrics(predictions_train, desc='train')
    calc_metrics(predictions_valid, desc='valid')
