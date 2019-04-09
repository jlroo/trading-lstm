#imports
import h5py
import pandas as pd
import torch
import torchvision
import time
import random
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
batch_size = 100
output_dim = 1
input_dim = ...
num_layers=2
hidden_dim = ...
dropoutRate = 0.5
Learning_rate = 0.01
Num_Epochs = 100
# Numpy data array

h5_file = h5py.File(file_path)
self.data = h5_file.get('data')
self.target = h5_file.get('label')


transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_set = torchvision.datasets.(root='./dta', train=True, download=True, transform=transform_train)
test_set = torchvision.datasets.(root='./', train=False, download=True, transform=transform_test)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=1)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=1)

class LSTM(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1, num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # Define the LSTM layer
        self.LSTM = torch.nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, dropout = dropoutRate)

        # Define the output layer
        self.linear = torch.nn.Linear(self.hidden_dim, output_dim)

    def forward(self, input):
        # Pass forward thw LSTM model
        output,_ = self.LSTM(input.view(len(input), self.batch_size, -1))

        # Only take the output from the final timetep
        output = self.linear(output[-1].view(self.batch_size, -1))
        return output.view(-1)


model = LSTM(input_dim, hidden_dim, batch_size=batch_size, output_dim=output_dim, num_layers=num_layers)
# Put on the GPU
model.cuda()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr = Learning_rate)


for epoch in range(Num_Epochs):
    # Modify the learning rate at 50th and 75th epochs
    if epoch == 50:
        for param_group in optimizer.param_groups:
            param_group['lr'] = Learning_rate / 10.0
    if epoch == 75:
        for param_group in optimizer.param_groups:
            param_group['lr'] = Learning_rate / 100.0

    time1 = time.time()
    model.train()
    for i, (data, classes) in enumerate(train_loader):
        data, classes = Variable(data.cuda()), Variable(classes.cuda())
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, classes)
        loss.backward()
        optimizer.step()
    model.eval()
    # Test Loss
    counter = 0
    test_accuracy_sum = 0.0
    for i, (data, classes) in enumerate(test_loader):
        data, classes = Variable(data.cuda()), Variable(classes.cuda())
        output = model(data)
        prediction = output.data.max(1)[1]
        accuracy = (float(prediction.eq(classes.data).sum())/float(batch_size))*100.0
        counter += 1
        test_accuracy_sum = test_accuracy_sum + accuracy
    test_accuracy_ave = test_accuracy_sum/float(counter)
    time2 = time.time()
    time_elapsed = time2 - time1
    print(epoch, test_accuracy_ave, time_elapsed)