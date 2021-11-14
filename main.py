from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import os
import torch
import torchvision
from torch.utils.data import random_split
import torchvision.models as models

res = ''
data_dir = './Garbage classification/Garbage classification'

classes = os.listdir(data_dir)

transformations = transforms.Compose(
    [transforms.Resize((256, 256)), transforms.ToTensor()])

dataset = ImageFolder(data_dir, transform=transformations)

# 'exec(%matplotlib inline)'
# # %matplotlib inline

random_seed = 42
torch.manual_seed(random_seed)

train_ds, val_ds, test_ds = random_split(dataset, [1593, 176, 758])
len(train_ds), len(val_ds), len(test_ds)

batch_size = 32

train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size*2, shuffle=True, num_workers=4, pin_memory=True)
from torchvision.utils import make_grid

def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
        break

if __name__ == '__main__':
    show_batch(train_dl)

from resnet import ResNet
model = ResNet(dataset)

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

device = get_default_device()

from device_data_loader import DeviceDataLoader

train_dl = DeviceDataLoader(train_dl, device)
print('before')
print(val_dl)
# val_dl = DeviceDataLoader(val_dl, device)
print('after')
print(val_dl)
to_device(model, device)

@torch.no_grad()
def evaluate(model, val_loader):
    print(val_loader)#[]
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    print(outputs)#[]
    if len(outputs) == 0:
        return
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []

        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print("train_losses:")
        print(train_losses)

                    # Validation phase
        result = evaluate(model, val_loader)
        return []
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history

model = to_device(ResNet(dataset), device)
if __name__ == '__main__':
    evaluate(model, val_dl)

# Let's start training the model:
num_epochs = 1
opt_func = torch.optim.Adam
lr = 5.5e-5
if __name__ == '__main__':
    history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)

    def plot_accuracies(history):
        if(history):
            accuracies = [x['val_acc'] for x in history]
            plt.plot(accuracies, '-x')
            plt.xlabel('epoch')
            plt.ylabel('accuracy')
            plt.title('Accuracy vs. No. of epochs')

    plot_accuracies(history)

    def plot_losses(history):
        train_losses = [x.get('train_loss') for x in history]
        val_losses = [x['val_loss'] for x in history]
        plt.plot(train_losses, '-bx')
        plt.plot(val_losses, '-rx')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['Training', 'Validation'])
        plt.title('Loss vs. No. of epochs')

    plot_losses(history)

filename = "model.pth"
torch.save(model, filename)
# img, label = test_ds[17]
# plt.imshow(img.permute(1, 2, 0))
# print('Label:', dataset.classes[label], ', Predicted:', predict_image(img, model))
# img, label = test_ds[23]
# plt.imshow(img.permute(1, 2, 0))
# print('Label:', dataset.classes[label], ', Predicted:', predict_image(img, model))
