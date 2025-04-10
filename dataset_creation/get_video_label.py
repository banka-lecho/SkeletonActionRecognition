import pandas as pd
from sklearn.model_selection import train_test_split

# Путь к корневой директории с видео
root_dir = '../dataset/videos'

# Имя выходного CSV файла
output_csv = '/Users/anastasiaspileva/PycharmProjects/ActionRecognition/dataset/action_dataset/labels.csv'
df = pd.read_csv(output_csv)

# Сначала разделим на train+val и test
train_val_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df['target'],
    random_state=42
)

# Затем разделим train+val на train и val
train_df, val_df = train_test_split(
    train_val_df,
    test_size=0.25,  # 0.25 от 0.8 = 20% от общего количества
    stratify=train_val_df['target'],
    random_state=42
)

# Добавляем колонку split
train_df['split'] = 'train'
val_df['split'] = 'val'
test_df['split'] = 'test'

# Объединяем обратно
final_df = pd.concat([train_df, val_df, test_df])

# Сохраняем в CSV
final_df.to_csv('/Users/anastasiaspileva/PycharmProjects/ActionRecognition/dataset/action_dataset/labels.csv',
                index=False)
