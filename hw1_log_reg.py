# Don't change batch size
batch_size = 64

from torch.utils.data.sampler import SubsetRandomSampler
import torch
import torch.nn as nn
from torchvision import datasets, transforms

## USE THIS SNIPPET TO GET BINARY TRAIN/TEST DATA

train_data = datasets.MNIST('./data', train=True, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ]))
test_data = datasets.MNIST('./data', train=False, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))

subset_indices = ((train_data.targets == 0) + (train_data.targets == 1)).nonzero()
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                           shuffle=False, sampler=SubsetRandomSampler(subset_indices.view(-1)))

subset_indices = ((test_data.targets == 0) + (test_data.targets == 1)).nonzero()
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                          shuffle=False, sampler=SubsetRandomSampler(subset_indices.view(-1)))

# The number of epochs is at least 10, you can increase it to achieve better performance
num_epochs = 10
# Additional hyperparams
input_dim = 784  # image is 28^2
num_features = 1  # image is 1 or -1
learn_rate = 0.001  # eta for SGD
momentum = 0.9  # for momentum method of optimization

#Define the loss function
class Reg_Loss(nn.modules.Module):
    def __init__(self):
        super(Reg_Loss,self).__init__()
    def forward(self,outputs,labels):
        batch_size = outputs.size()[0]
        return torch.sum(torch.log(1 + torch.exp(-(outputs.t() * labels)))) / batch_size

#Create the the logistic regression model
logist_model = 0
logist_model = nn.Linear(input_dim,num_features)
crit = Reg_Loss()
optimizer = torch.optim.SGD(logist_model.parameters(),lr=learn_rate,momentum=momentum)
step = len(train_loader)

# Training the Model
for epoch in range(num_epochs):
    total_loss = 0
    avg_loss = 0
    num_batches = 0
    for i, (images, labels) in enumerate(train_loader):
        images = images.view(-1, 28 * 28)
        # Convert labels from 0,1 to -1,1
        labels = 2 * (labels.float() - 0.5)
        outputs = logist_model(images)
        loss = crit(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        num_batches += 1
        total_loss += loss.item()

    ## Print your results every epoch
    avg_loss = total_loss/num_batches
    print ('Epoch [{}/{}], Average loss for epoch {} = {:.4f}'
                   .format(epoch+1, num_epochs, epoch+1, avg_loss ))

# Test the Model
correct = 0.
total = 0.
for images, labels in test_loader:
    images = images.view(-1, 28 * 28)

    ## Put your prediction code here, currently use a random prediction
    #prediction = torch.randint(0, 2, labels.shape)
    test_outs = torch.sigmoid(logist_model(images))
    prediction = test_outs.data >= 0.5

    correct += (prediction.view(-1).long() == labels).sum()
    total += images.shape[0]
print('Accuracy of the model on the test images: %f %%' % (100 * (correct.float() / total)))
