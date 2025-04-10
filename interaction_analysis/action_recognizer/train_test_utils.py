import os
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter

from interaction_analysis.action_recognizer.recognizer import SuperFormer
from interaction_analysis.action_recognizer.action_dataset import ActionDataset


# 1. Инициализация датасетов и даталоадеров
def get_dataloaders(labels_path, batch_size=32, target_size=(128, 128), frame_step=1):
    """Создает даталоадеры для обучения и валидации"""
    train_dataset = ActionDataset(
        labels_path=labels_path,
        split='train',
        target_size=target_size,
        frame_step=frame_step
    )

    val_dataset = ActionDataset(
        labels_path=labels_path,
        split='val',
        target_size=target_size,
        frame_step=frame_step
    )

    # Для скелетонных данных нужен специальный коллайт
    def collate_fn(batch):
        masks = torch.stack([item[0]['mask'] for item in batch])
        flows = torch.stack([item[0]['optical_flow'] for item in batch])
        points = torch.stack([item[0]['skeleton_points'] for item in batch])
        edges = batch[0][0]['edge_index']  # предполагаем одинаковую структуру скелета
        labels = torch.tensor([item[1] for item in batch])

        return {
            'mask': masks,
            'optical_flow': flows,
            'skeleton_points': points,
            'edge_index': edges
        }, labels

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return train_loader, val_loader, train_dataset.get_action_names()


# 2. Функции обучения и валидации
def train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, targets) in enumerate(train_loader):
        # Перемещаем данные на устройство
        masks = data['mask'].to(device)
        flows = data['optical_flow'].to(device)
        points = data['skeleton_points'].to(device)
        edges = data['edge_index'].to(device)
        targets = targets.to(device)

        # Обнуляем градиенты
        optimizer.zero_grad()

        # Forward pass
        outputs = model(masks, flows, points, edges)
        loss = criterion(outputs, targets)

        # Backward pass и оптимизация
        loss.backward()
        optimizer.step()

        # Считаем метрики
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Логируем каждые N батчей
        if batch_idx % 50 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(masks)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    # Считаем средние метрики за эпоху
    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total

    # Логируем в TensorBoard
    if writer:
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)

    return train_loss, train_acc


def validate(model, val_loader, criterion, device, epoch, writer, class_names):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_targets = []
    all_predicted = []

    with torch.no_grad():
        for data, targets in val_loader:
            masks = data['mask'].to(device)
            flows = data['optical_flow'].to(device)
            points = data['skeleton_points'].to(device)
            edges = data['edge_index'].to(device)
            targets = targets.to(device)

            outputs = model(masks, flows, points, edges)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            all_targets.extend(targets.cpu().numpy())
            all_predicted.extend(predicted.cpu().numpy())

    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total

    # Логируем в TensorBoard
    if writer:
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        # Confusion matrix
        cm = confusion_matrix(all_targets, all_predicted)
        fig = plot_confusion_matrix(cm, class_names)
        writer.add_figure('Confusion Matrix', fig, epoch)

    print(f'\nValidation set: Average loss: {val_loss:.4f}, Accuracy: {correct}/{total} ({val_acc:.2f}%)\n')

    return val_loss, val_acc


def plot_confusion_matrix(cm, classes):
    """Создает визуализацию матрицы ошибок"""
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='Confusion matrix',
           ylabel='True label',
           xlabel='Predicted label')

    # Поворачиваем подписи и выравниваем
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Добавляем числовые значения
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    return fig


# 3. Основной тренировочный цикл
def train_model(config):
    # Устройство
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Инициализация даталоадеров
    train_loader, val_loader, class_names = get_dataloaders(
        labels_path=config['labels_path'],
        batch_size=config['batch_size'],
        target_size=config['target_size'],
        frame_step=config['frame_step']
    )

    # Модель
    model = SuperFormer(num_classes=len(class_names)).to(device)

    # Критерий и оптимизатор
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)

    # TensorBoard writer
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(config['log_dir'], current_time)
    writer = SummaryWriter(log_dir)

    # Сохранение лучшей модели
    best_acc = 0.0
    best_model_path = os.path.join(config['save_dir'], 'best_model.pth')

    # Тренировочный цикл
    for epoch in range(1, config['epochs'] + 1):
        start_time = time.time()

        # Обучение и валидация
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer)

        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch, writer, class_names)

        # Обновляем learning rate
        scheduler.step(val_acc)

        # Сохраняем лучшую модель
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with accuracy: {best_acc:.2f}%")

        # Логируем время эпохи
        epoch_time = time.time() - start_time
        print(f'Epoch {epoch} completed in {epoch_time:.2f}s')

    # Закрываем writer
    writer.close()

    # Загружаем лучшую модель для тестирования
    model.load_state_dict(torch.load(best_model_path))
    return model, best_acc


# 4. Конфигурация и запуск обучения
if __name__ == "__main__":
    config = {
        'labels_path': 'dataset/action_dataset/labels.csv',
        'batch_size': 32,
        'target_size': (128, 128),
        'frame_step': 1,
        'epochs': 50,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'log_dir': 'runs',
        'save_dir': 'saved_models'
    }

    # Создаем директории если их нет
    os.makedirs(config['log_dir'], exist_ok=True)
    os.makedirs(config['save_dir'], exist_ok=True)

    # Запускаем обучение
    trained_model, best_acc = train_model(config)
    print(f"Training completed. Best validation accuracy: {best_acc:.2f}%")
