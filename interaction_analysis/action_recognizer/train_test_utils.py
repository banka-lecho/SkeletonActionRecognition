import torch
import time
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import precision_score, recall_score
from interaction_analysis.action_recognizer.action_former import ST_GCN_Net
from interaction_analysis.action_recognizer.action_dataset import SkeletonDataset


def compute_metrics(outputs, labels):
    """Вычисляет precision и recall"""
    _, predicted = torch.max(outputs, 1)
    predicted = predicted.cpu().numpy()
    labels = labels.cpu().numpy()

    precision = precision_score(labels, predicted, average='macro', zero_division=0)
    recall = recall_score(labels, predicted, average='macro', zero_division=0)
    return precision, recall


def log_gradients(model, writer, step):
    """Логирует гистограммы градиентов"""
    for name, param in model.named_parameters():
        if param.grad is not None:
            writer.add_histogram(f'gradients/{name}', param.grad, step)


def log_weights(model, writer, step):
    """Логирует гистограммы весов"""
    for name, param in model.named_parameters():
        writer.add_histogram(f'weights/{name}', param, step)


def train_epoch(model, dataloader, optimizer, writer, epoch, criterion, device):
    total_loss = 0
    all_labels = []
    all_outputs = []

    loss = 0
    for batch_idx, (points, labels) in enumerate(dataloader):
        # Forward
        outputs = model(points)
        loss = criterion(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Собираем данные для метрик
        all_labels.extend(labels.to(device).numpy())
        all_outputs.extend(outputs.detach().to(device).numpy())

        # Логируем градиенты и веса для каждого батча
        if epoch % 5 == 0:
            log_gradients(model, writer, epoch * len(dataloader) + batch_idx)
            log_weights(model, writer, epoch * len(dataloader) + batch_idx)
            writer.add_scalar('train/batch_loss', loss.item(), epoch * len(dataloader) + batch_idx)

    # Вычисляем метрики для эпохи
    all_outputs = torch.tensor(np.array(all_outputs))
    all_labels = torch.tensor(np.array(all_labels))
    precision, recall = compute_metrics(all_outputs, all_labels)
    avg_loss = total_loss / len(dataloader)

    # Логируем метрики для эпохи
    writer.add_scalar('train/loss', avg_loss, epoch)
    writer.add_scalar('train/precision', precision, epoch)
    writer.add_scalar('train/recall', recall, epoch)

    print(f"Epoch {epoch}, Loss: {loss.item()}")
    return avg_loss, precision


def validate(model, loader, device, writer, epoch):
    model.eval()
    total_loss = 0
    correct = 0
    all_labels = []
    all_outputs = []

    with torch.no_grad():
        for masks, labels in loader:
            masks = masks.to(device)
            labels = labels.to(device)

            outputs = model(masks)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())

    # Вычисляем метрики
    all_outputs = torch.tensor(np.array(all_outputs))
    all_labels = torch.tensor(np.array(all_labels))
    precision, recall = compute_metrics(all_outputs, all_labels)
    avg_loss = total_loss / len(loader)

    # Логируем метрики валидации
    writer.add_scalar('val/loss', avg_loss, epoch)
    writer.add_scalar('val/precision', precision, epoch)
    writer.add_scalar('val/recall', recall, epoch)

    return avg_loss, precision


if __name__ == '__main__':
    # def augment_keypoints(keypoints):
    #     # Добавление шума
    #     noise = torch.randn_like(keypoints) * 0.01
    #     keypoints += noise
    #
    #     # Случайный сдвиг
    #     shift = torch.rand(2) * 0.1 - 0.05
    #     keypoints += shift
    #
    #     return keypoints

    train_dataset = SkeletonDataset(
        labels_path='/labels.csv',
        split='train',
        frame_step=2,
        sequence_length=30
    )

    val_dataset = SkeletonDataset(
        labels_path='/labels.csv',
        split='test',
        frame_step=2,
        sequence_length=30
    )

    adj_matrix = SkeletonDataset.get_adj_matrix()

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )

    model = ST_GCN_Net(num_classes=5, adj_matrix=adj_matrix)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    time_experiment = time.time()
    writer = SummaryWriter(
        f'/Users/anastasiaspileva/PycharmProjects/ActionRecognition/interaction_analysis/action_recognizer/runs/{time_experiment}')
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Цикл обучения
    best_val_loss = float('inf')
    for epoch in range(60):
        train_loss, train_prec = train_epoch(model=model, dataloader=train_loader,
                                             optimizer=optimizer, writer=writer,
                                             epoch=epoch, criterion=criterion, device=device)

        val_loss, val_prec = validate(model, val_loader, device, writer, epoch)

        scheduler.step(val_loss)

        print(f'Epoch {epoch + 1}:')
        print(f'Train Loss: {train_loss:.4f} | Precision: {train_prec:.2%}')
        print(f'Val Loss: {val_loss:.4f} | Precision: {val_prec:.2%}')

        # Сохраняем лучшую модель
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

    # Закрываем writer
    writer.close()
