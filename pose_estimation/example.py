import cv2

from pose_estimation.pose_estimator import PoseEstimator

# Пути к папкам
input_path = '/Users/anastasiaspileva/Desktop/eating.jpg'
output_path = '/Users/anastasiaspileva/Desktop/eating1.jpg'

image = cv2.imread(input_path)
yolo_model = PoseEstimator("yolo11m-pose.pt")
result, annotated_image = yolo_model.predict(image, verbose=False)
cv2.imwrite(output_path, annotated_image)
