# TASK - To create a sequence like a list of odd numbers and then build a model to train it to predict the next digit in the sequence.
# A simple neural networks with 2 layers and 1 output layer is used to train the model.

# Importing the libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class config:
    num_epochs = 500
    train = False

cfg = config()

# for reproducibility
torch.manual_seed(1)
np.random.seed(1)

# Creating the dataset
# Creating a sequence of odd numbers
seq = [i for i in range(1, 100, 2)]
print(seq)

# Creating the input and output data
# Input data
X = []
# Output data
y = []

# Creating the input and output data
for i in range(len(seq) - 1):
    X.append(seq[i])
    y.append(seq[i + 1])

# Converting the input and output data to numpy arrays
X = np.array(X)
y = np.array(y)

# Converting the input and output data to tensors
X = torch.from_numpy(X).float()
Y = torch.from_numpy(y).float()

# Creating the model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 1)

    def forward(self, x):
        x = self.fc1(x)
        return x
    
# Creating the model object
model = Net()

# Creating the optimizer
optimizer = optim.Adam(model.parameters(), lr = 0.001)

# Creating the loss function
loss_function = nn.MSELoss()

# Creating the training loop
def train():
    for i in range(cfg.num_epochs):
        # input 
        for x, y in zip(X, Y):
            x = x.view(1, -1)
            y = y.view(1, -1)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            output = model(x)
            loss = loss_function(output, y)
            loss.backward()
            optimizer.step()
        # print statistics
        if i % 100 == 0:
            print("Epoch: ", i, " Loss: ", loss.item())
    # saving the model
    torch.save(model, "model.pt")

if config.train:
    train()

# loading the model
model = torch.load("model.pt")

# Testing the model
# Predicting the next number in the sequence
print("Predicted number: ", model(torch.tensor([99.0])).item())

# Testing the model
# Predicting the next number in the sequence
seq = [i for i in range(111, 200, 2)]
# converting the sequence to a tensor
seq = torch.from_numpy(np.array(seq)).float()

output = []
for i in range(len(seq) - 1):
    output.append(model(torch.tensor([seq[i]])).item())

print(output)
