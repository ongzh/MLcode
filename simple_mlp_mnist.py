import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm.notebook import tqdm

mnist_train = datasets.MNIST(root="./datasets", train=True, transform=transforms.ToTensor(), download=True)
mnist_test = datasets.MNIST(root="./datasets", train=False, transform=transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=64, shuffle=False)


class MNIST_MLP_MODEL(nn.Module):
    def __init__(self):
        
        super().__init__()
        self.flatten = nn.Flatten()
        self.lin = nn.Sequential(
        
            nn.Linear(784,500),
            nn.ReLU(),
            nn.Linear(500,10)
        )
    
    def forward(self,x):
        x = self.flatten(x)
        logits = self.lin(x)
        return logits
    
model = MNIST_MLP_MODEL()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

batches = iter(train_loader)
test_batches = iter(test_loader)

def train(train_loader,model,loss_fn,optimizer):
    size = len(train_loader.dataset)
    for num, batch in enumerate(tqdm(batches)):
        img, label = batch
        #forward
        pred = model(img)
        loss = loss_fn(pred,label)
        #backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if num % 64 == 0:
            loss, current = loss.item(), num * len(pred)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        

def test(test_loader,model):
    size = len(test_loader.dataset)
    test_loss, correct = 0,0
    with torch.no_grad():
        for num, batch in enumerate(tqdm(test_batches)):
            model.eval()
            img,label = batch
            pred = model(img)
            loss = loss_fn(pred,label)
            test_loss += loss.item()
            #softmax = nn.Softmax(dim=1)
            correct += (pred.argmax(1) == label).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_loader, model, loss_fn, optimizer)
    test(test_loader, model)
print("Done!")
