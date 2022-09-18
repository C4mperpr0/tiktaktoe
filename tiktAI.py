import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

if torch.cuda.is_available():
    print("Cuda is available!")
else:
    while True:
        print("Cude is not available!")
        input()


class TiktaktoeAI(nn.Module):
    def __init__(self):
        super(TiktaktoeAI, self).__init__()
        self.lin1 = nn.Linear(10, 10)
        self.lin2 = nn.Linear(10, 10)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num = 1
        for i in size:
            num *= i
        return num


tiktaktoeAI = TiktaktoeAI()
print(tiktaktoeAI)

for i in range(100):
    data_input = [0, 1, 0, 0, 0, 1, 0, 1, 0, 1]
    out = tiktaktoeAI(torch.Tensor([data_input for _ in range(10)]))

    data_expected = [1, 0, 1, 1, 1, 0, 1, 0, 1, 0]
    target = Variable(torch.Tensor([data_expected for _ in range(10)]))

    criterion = nn.MSELoss()
    loss = criterion(out, target)
    print(loss)

    tiktaktoeAI.zero_grad()
    loss.backward()
    optimizer = optim.SGD(tiktaktoeAI.parameters(), lr=0.11)
    optimizer.step()

















"""
# https://www.youtube.com/watch?v=8gZR4Q3262k&list=PLNmsVeXQZj7rx55Mai21reZtd_8m-qe27&index=10
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
# PROBLEME MIT CUDA (auskommentierter Code)
print(f"Cuda available: {torch.cuda.is_available()}\nMain-Thread: {__name__ == '__main__'}\n\n\n")

kwargs = {} #{'num_workers': 1, 'pin_memory': True}
train_data = torch.utils.data.DataLoader(datasets.MNIST('data', train=True, download=True,
                                                    transform=transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize((0.1307,),(0.3081,))])),
                                     batch_size=64, shuffle=True, **kwargs)
test_data = torch.utils.data.DataLoader(datasets.MNIST('data', train=False,
                                                    transform=transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize((0.1307,),(0.3081,))])),
                                     batch_size=64, shuffle=True, **kwargs)



class Netz(nn.Module):
    def __init__(self):
        super(Netz, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv_dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 60)
        self.fc2 = nn.Linear(60, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.conv_dropout(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)

model = Netz()
model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.8)

def train(epoch):
    model.train()
    for batch_id, (data, target) in enumerate(train_data):
        #data = data.cuda()
        #target = taget.cuda()
        data = Variable(data)
        target = Variable(target)

        optimizer.zero_grad()
        out = model(data)
        criterion = F.nll_loss
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        print(f'Epoche {epoch}')

"""
def test():
    model.eval()
    loss = 0
    correct = 0
"""

for epoch in range(1, 10):
    train(epoch)

"""








