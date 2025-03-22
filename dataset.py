import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


import matplotlib.pyplot as plt
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])



def get_dataloader(batch_size = 16):
    train_dataset = ImageFolder(root = 'my_dataset/train', transform=transform)
    val_dataset = ImageFolder(root = 'my_dataset/test', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=False)

    # class_names = train_dataset.classes
    # print(f'Class_names :', class_names)
    return train_loader, val_loader


get_dataloader(batch_size = 16)
    




# Show some sample
# print(len(train_dataset))
# print(type(train_dataset))
# image, label = train_dataset[7500]
# transform = transforms.ToTensor()
# image_tensor = transform(image)


# plt.imshow(image_tensor.permute(1,2, 0))
# plt.title(f'Class: {train_dataset.classes[label]}')
# plt.axis('off')
# plt.show()
