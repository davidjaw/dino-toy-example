import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils import DINOLoss
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchvision.utils import make_grid


def load_mnist_data(batch_size=256):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize(28, antialias=None),
                                    transforms.Normalize((0.1307,), (0.3081,))])
    train_transform = transforms.Compose([transforms.RandomCrop(23, padding=7),
                                          transforms.RandomAffine(degrees=20, translate=(0.1, 0.1)),
                                          transforms.Resize(28, antialias=None),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))])
    # Load the training and test datasets
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transform)
    train_t_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    # Data loaders for the training and test datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=3)
    train_t_loader = DataLoader(train_t_dataset, batch_size=batch_size, shuffle=False, num_workers=3)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=3, pin_memory=True)
    return train_loader, train_t_loader, test_loader


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.SiLU(),
        )
        self.fc = nn.Sequential(
            nn.LazyLinear(256),
            nn.LazyLinear(64),
            nn.LazyLinear(128),
        )
        self.flat = nn.Flatten()
        self.cls_head = nn.Sequential(
            nn.LazyLinear(256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.LazyLinear(10),
        )

    def forward(self, x):
        x = self.convs(x)
        x = self.flat(x)
        cls_head = self.cls_head(x)
        x = F.silu(self.fc(x))
        return x, cls_head


def ema_update(net, ema_net, decay):
    """
    Perform Exponential Moving Average (EMA) update on the network.

    :param net: The original neural network (PyTorch model).
    :param ema_net: The shadow model whose parameters are the EMA of the original model's parameters.
    :param decay: The decay rate for EMA.
    """
    with torch.no_grad():
        # Iterate over parameters of both networks
        for param, ema_param in zip(net.parameters(), ema_net.parameters()):
            # Update the shadow parameter based on the EMA formula
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)


def get_next_batch(data_iter, data_loader):
    try:
        data, target = next(data_iter)
    except StopIteration:
        data_iter = iter(data_loader)
        data, target = next(data_iter)
    return data, target, data_iter


def train_model_classification(model, train_loader, optimizer, device, epoch, writer):
    model.train()
    train_loss = 0
    correct = 0
    loader = tqdm(train_loader, desc=f'Epoch {epoch}')
    accuracy = 0
    for b_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        _, output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        accuracy = correct / len(train_loader.dataset)
        loader.set_postfix(loss=loss.item(), accuracy=accuracy)
    writer.add_scalar('train/loss', train_loss / len(train_loader.dataset), epoch)
    writer.add_scalar('train/accu', accuracy, epoch)


def eval_and_print(net, test_loader, device):
    net.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            _, output = net(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    print(f'Accuracy: {correct / len(test_loader.dataset)}')


def main():
    train_loader, train_t_loader, test_loader = load_mnist_data()
    denorm = lambda x: x * 0.3081 + 0.1307
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)

    writer = SummaryWriter('runs/wo_pretrain')
    opti = torch.optim.SGD(model.parameters(), lr=1e-3)
    # Train a model that without pretraining nor EMA
    for epoch in range(20):
        train_model_classification(model, train_loader, opti, device, epoch, writer)
    writer.close()
    # evaluate accuracy
    eval_and_print(model, test_loader, device)

    writer = SummaryWriter('runs/pretrain')
    model = Net().to(device)
    weight_path = 'mnist.pt'
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load('mnist.pt'))
    ema_model = deepcopy(model).to(device)

    t_iter = iter(train_t_loader)

    opti = torch.optim.SGD(model.parameters(), lr=5e-5)
    dino_loss = DINOLoss(128)
    for epoch in range(100):
        loader = tqdm(train_loader, desc=f'Epoch {epoch}')
        embedding, ema_embedding, data, t_data = None, None, None, None
        loss = None
        for batch_idx, (data, target) in enumerate(loader):
            embedding, _ = model(data.to(device))
            with torch.no_grad():
                t_data, _, t_iter = get_next_batch(t_iter, train_t_loader)
                ema_embedding, _ = ema_model(t_data.to(device))
                ema_embedding = dino_loss.sinkhorn_knopp_teacher(ema_embedding, 0.04)

            loss = dino_loss([embedding], [ema_embedding])
            loss.backward()

            # clip gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), 15)
            opti.step()

            ema_update(model, ema_model, 0.994)
            loader.set_postfix(loss=loss.item())

        data_grid = make_grid(denorm(data))
        t_data_grid = make_grid(denorm(t_data))
        # Add the grid to TensorBoard
        writer.add_image('Original Images', data_grid, global_step=epoch)
        writer.add_image('Transformed Images', t_data_grid, global_step=epoch)
        writer.add_histogram('embedding', embedding, epoch)
        writer.add_histogram('ema_embedding', ema_embedding, epoch)
        writer.add_scalar('loss', loss.item(), epoch)

        # save model
        torch.save(ema_model.state_dict(), 'mnist.pt')

    # add embedding of dataset
    emb = []
    img = []
    meta = []
    for batch_idx, (data, target) in enumerate(train_loader):
        embedding = ema_model(data.to(device))
        emb.append(embedding)
        img.append(denorm(data))
        meta.append(target)
        if batch_idx > 100:
            break
    writer.add_embedding(torch.cat(emb), metadata=torch.cat(meta), label_img=torch.cat(img))
    writer.close()

    writer = SummaryWriter('runs/w_pretrain')
    # Train a model that without pretraining nor EMA
    for epoch in range(20):
        train_model_classification(model, train_loader, opti, device, epoch, writer)
    # evaluate accuracy
    eval_and_print(model, test_loader, device)


if __name__ == '__main__':
    main()


