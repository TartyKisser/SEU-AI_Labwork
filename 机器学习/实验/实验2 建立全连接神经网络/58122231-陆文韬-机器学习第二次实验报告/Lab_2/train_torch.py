import torch
from torch import nn
from torchvision import datasets
from config import args as arg
from utils import *
device = torch.device(get_default_device())

batch_size = arg.batch_size
my_lr = arg.lr
my_max_lr = 10 * my_lr
grad_clip = 0.1
my_epochs = arg.epochs
my_momentum = arg.momentum
my_weight_decay = arg.weight_decay
seed = arg.seed

torch.manual_seed(seed)

train_data = datasets.FashionMNIST(root="data",
                                   train=True,
                                   download=True,
                                   transform=train_tfms)

test_data = datasets.FashionMNIST(root="data",
                                  train=False,
                                  download=True,
                                  transform=test_tfms)

train_dl = MpsDataLoader(train_data,
                         batch_size=batch_size,
                         shuffle=True,
                         pin_memory=True)
test_dl = MpsDataLoader(test_data,
                        batch_size=batch_size,
                        pin_memory=True)


class SingleLayerNN(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hidden, initializer=None):
        super().__init__()
        self.num_inputs, self.num_outputs, self.num_hidden = num_inputs, num_outputs, num_hidden
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        if initializer:
            initializer(self.fc1.weight)
            initializer(self.fc2.weight)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class TrainStep:
    def __init__(self,
                 model,
                 lr=my_lr,
                 max_lr=my_max_lr,
                 epochs=my_epochs,
                 momentum=my_momentum,
                 weight_decay=my_weight_decay):
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(model.parameters(),
                                         lr=lr,
                                         momentum=momentum,
                                         weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr,
            epochs=epochs,
            steps_per_epoch=len(train_dl))
    
    def __call__(self, model, images, labels):
        self.optimizer.zero_grad()
        loss = self.loss_fn(model(images), labels)
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        return loss


if __name__ == "__main__":
    my_model = to_device(SingleLayerNN(784, 10, 256), get_default_device())

    history = fit_and_test(my_model, my_epochs, train_dl, test_dl, TrainStep(my_model))
    torch.save(my_model.state_dict(), 'single_layer_nn.pth')
