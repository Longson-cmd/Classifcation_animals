import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import get_dataloader
from model import Simple_model
from utils import log_confusion_matrix, save_checkpoint, load_checkpoint

writer = SummaryWriter()
class_names = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']
device = torch.device('cpu')


def train(model, train_loader, val_loader, criterion, optimizer, number_epochs):
    start_epoch, best_val_acc = load_checkpoint(model, optimizer, checkpoint_path='checkpoint.pth')

    for epoch in range(start_epoch, number_epochs + 1):

        # training mode
        model.train()
        total_train, correct_train, running_loss_train = 0, 0, 0.0

        for images, labels in tqdm(train_loader, desc= f'Epoch {epoch} / {number_epochs}', leave=False):
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
        train_loss = running_loss_train/ len(train_loader)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Loss/train", train_loss, epoch)

        model.eval()
        total_val, correct_val, running_loss_val = 0, 0, 0.0
        y_true, y_pred = [], []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc = 'Evaluating', leave=False):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                running_loss_val += loss.item()
                y_true.extend(labels.cpu().numpy()) ###
                y_pred.extend(predicted.cpu().numpy()) ###

        val_acc = 100 * correct_val/ total_val
        val_loss = running_loss_val / len(val_loader)
        writer.add_scalar("Accuracy/validation", val_acc, epoch)
        writer.add_scalar("Loss/validation", val_loss, epoch)

        log_confusion_matrix(y_true, y_pred, writer, epoch, class_names)
        

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_accuracy_model.pth')
            print(f'New best accuracy : {best_val_acc:.2f} %')
            save_checkpoint(model, optimizer, best_val_acc, epoch, checkpoint_path = 'checkpoint.pth')

        
if __name__ == '__main__':
    model = Simple_model()
    model = model.to(device)
    train_loader, val_loader = get_dataloader()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    number_epochs = 3
    train(model, train_loader, val_loader, criterion, optimizer, number_epochs)


        






            
        
