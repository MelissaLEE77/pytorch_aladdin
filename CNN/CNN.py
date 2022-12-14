# Imports
from select import KQ_NOTE_RENAME
from this import d
from turtle import Turtle, forward
import torch
import torchvision # torch package for vision related things
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
import torchvision.datasets as datasets  # Standard datasets
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules
from torch.utils.data import DataLoader  # Gives easier dataset managment by creating mini batches etc.
from tqdm import tqdm  # For nice progress bar!


# Create simple CNN
# convoulution layer -> image 정보 보존, image data(채널*세로*가로)로 구성, 3차원 data input -> output도 3차원
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.fc1 = nn.Linear(16*7*7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)
        return x

def save_checkpoint(state, filename = "my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint):
    print("=> loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

## Check model
#model = CNN()
#x = torch.randn(64,1,28,28)
#print(model(x).shape)


#Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyperparmeter
in_channels = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 5
load_model = True #False

# Load data
train_dataset = datasets.MNIST(root='./PYTORCH_CHEETSHEET/dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size = batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='./PYTORCH_CHEETSHEET/dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=train_dataset, batch_size = batch_size, shuffle=True)

#Initialize network
model = CNN().to(device)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"))

#Train Network
for epoch in range(num_epochs):
    losses = []

    # checkpoint로 model, optimizer, acuuracy 등을 저장
    if epoch % 3 == 0:
        checkpoint = {'state_dict':model.state_dict(),'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint)

    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)
        #print(data.shape) -> data 확인용

        #forward
        scores = model(data)
        loss = criterion(scores, targets)
        losses.append(loss.item())
        #backward
        optimizer.zero_grad()
        loss.backward()
        #gradient descent or adam step
        optimizer.step()

# Check accuracy on training & test to see how good our model

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    return num_correct/num_samples


print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:.2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")



