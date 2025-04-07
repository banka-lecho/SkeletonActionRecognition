import cv2
import os

# Папка с кадрами
frames_folder = '/Users/anastasiaspileva/PycharmProjects/ActionRecognition/data/annotated_frames_eating'
frames = sorted([os.path.join(frames_folder, img) for img in os.listdir(frames_folder) if
                 img.endswith(".png") or img.endswith(".jpg")])

if not frames:
    print("Нет кадров в указанной папке.")
    exit()

# Чтение первого кадра для получения размеров
first_frame = cv2.imread(frames[0])
height, width, layers = first_frame.shape

# Используем кодек 'mp4v' для MP4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter('/Users/anastasiaspileva/Desktop/eating.mp4',
                               fourcc, 30, (width, height))

# Запись кадров в видео
for frame in frames:
    img = cv2.imread(frame)
    output_video.write(img)

output_video.release()
print("Видео успешно создано!")
