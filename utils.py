import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

def log_confusion_matrix(y_true, y_pred, writer, epoch, class_names):
    cm = confusion_matrix(y_true, y_pred)
    cm_percentage = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis] * 100  ###

    fig, ax = plt.subplots(figsize = (12, 12))
    sns.heatmap(cm_percentage, annot=True, cmap='Reds', fmt = '.1f', xticklabels=class_names, yticklabels=class_names)

    plt.xlabel('Predicted classes') ###
    plt.ylabel('Actual classes') ###
    plt.title('Confusion Matrix')

    writer.add_figure('Confusion Matrix', fig, epoch)
    plt.close(fig)

def save_checkpoint(model, optimizer, best_val_acc, epoch, checkpoint_path = 'checkpoint.pth'):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'best_val_acc': best_val_acc
    }

    torch.save(checkpoint, checkpoint_path) ####


def load_checkpoint(model, optimizer, checkpoint_path = 'checkpoint.pth'):
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict']) ###
        optimizer.load_state_dict(checkpoint['optimizer_state_dict']) ####

        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['best_val_acc']
        print(f'Checkpoint loaded from epoch {start_epoch}')
        return start_epoch, best_val_acc
    
    else:
        print('No checkpoint found, start fresh')
        return 0, 0.0

