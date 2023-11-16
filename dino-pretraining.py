import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils import DINOLoss
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def load_mnist_data(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])
    # Load the training and test datasets
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    # Data loaders for the training and test datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    return train_loader, test_loader


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 128)

    def forward(self, x):
        x = F.silu(F.max_pool2d(self.conv1(x), 2))
        x = F.silu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.silu(self.fc1(x))
        return x


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


def main():
    _, train_loader = load_mnist_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    # model.load_state_dict(torch.load('mnist.pt'))
    ema_model = deepcopy(model).to(device)

    writer = SummaryWriter('visualize500')
    opti = torch.optim.SGD(model.parameters(), lr=1e-4)
    dino_loss = DINOLoss(128)
    for epoch in range(500):
        loader = tqdm(train_loader, desc=f'Epoch {epoch}')
        embedding, ema_embedding = None, None
        loss = None
        for batch_idx, (data, target) in enumerate(loader):
            embedding = model(data.to(device))
            with torch.no_grad():
                ema_embedding = ema_model(data.to(device))
                ema_embedding = dino_loss.sinkhorn_knopp_teacher(ema_embedding, 0.04)
            loss = dino_loss([embedding], [ema_embedding])
            loss.backward()
            opti.step()

            ema_update(model, ema_model, 0.994)
            loader.set_postfix(loss=loss.item())

        writer.add_histogram('embedding', embedding, epoch)
        writer.add_histogram('ema_embedding', ema_embedding, epoch)
        writer.add_scalar('loss', loss.item(), epoch)

    # # save model
    # torch.save(ema_model.state_dict(), 'mnist.pt')

    # add embedding of dataset
    denorm = lambda x: x * 0.3081 + 0.1307
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


if __name__ == '__main__':
    main()


