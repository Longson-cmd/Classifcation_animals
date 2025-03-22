from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision import models
import torch


# Define Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to a fixed size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Pre-trained model normalization
])
# /kaggle/input/animal-dataset/animal/train.py
# /kaggle/input/animal-dataset/animal/model.py

# Define paths to your dataset
train_dir = '/kaggle/input/animal-dataset/animal/my_dataset/train'  # Path to the 'train' folder
test_dir = '/kaggle/input/animal-dataset/animal/my_dataset/test'    # Path to the 'test' folder

# Load Data Using ImageFolder
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

# Create DataLoader Instances
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

# Example: Print the number of samples in each dataset
print(f"Number of training images: {len(train_dataset)}")
print(f"Number of val images: {len(val_dataset)}")

#################################################

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

# Model Definition
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

model = resnet18(weights = ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(in_features=model.fc.in_features, out_features=10)


# Utility Functions
def log_confusion_matrix(y_true, y_pred, writer, epoch, class_names):
    cm = confusion_matrix(y_true, y_pred)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100  

    fig, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(cm_percentage, annot=True, cmap='Reds', fmt='.1f', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted classes')
    plt.ylabel('Actual classes')
    plt.title('Confusion Matrix')

    writer.add_figure('Confusion Matrix', fig, epoch)
    plt.close(fig)

def save_checkpoint(model, optimizer, best_val_acc, epoch, checkpoint_path='checkpoint.pth'):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'best_val_acc': best_val_acc
    }
    torch.save(checkpoint, checkpoint_path)

def load_checkpoint(model, optimizer, checkpoint_path='checkpoint.pth'):
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['best_val_acc']
        print(f'Checkpoint loaded from epoch {start_epoch}')
        return start_epoch, best_val_acc
    else:
        print('No checkpoint found, starting fresh')
        return 0, 0.0

# Training Function
def train(model, train_loader, val_loader, criterion, optimizer, number_epochs, device):
    start_epoch, best_val_acc = load_checkpoint(model, optimizer)

    for epoch in range(start_epoch, number_epochs + 1):
        model.train()
        total_train, correct_train, running_loss_train = 0, 0, 0.0

        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch} / {number_epochs}', leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            running_loss_train += loss.item()

        train_acc = 100 * correct_train / total_train
        train_loss = running_loss_train / len(train_loader)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Loss/train", train_loss, epoch)

        # Evaluation
        model.eval()
        total_val, correct_val, running_loss_val = 0, 0, 0.0
        y_true, y_pred = [], []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Evaluating', leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                running_loss_val += loss.item()
                y_true.extend(labels.cpu().numpy()) 
                y_pred.extend(predicted.cpu().numpy())

        val_acc = 100 * correct_val / total_val
        val_loss = running_loss_val / len(val_loader)
        writer.add_scalar("Accuracy/validation", val_acc, epoch)
        writer.add_scalar("Loss/validation", val_loss, epoch)

        log_confusion_matrix(y_true, y_pred, writer, epoch, class_names)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_accuracy_model.pth')
            print(f'New best accuracy : {best_val_acc:.2f}% on epoch {epoch} ')
        save_checkpoint(model, optimizer, best_val_acc, epoch)

if __name__ == '__main__':
    writer = SummaryWriter()
    class_names = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    
    # Loss function & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train
    number_epochs = 3
    train(model, train_loader, val_loader, criterion, optimizer, number_epochs, device)

    print("Training completed.")



