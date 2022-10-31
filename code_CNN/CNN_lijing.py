import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets
import numpy as np
import pandas as pd
num_epochs = 1
num_classes = 10
batch_size = 100
learning_rate = 0.001


DATA_PATH = "/data/users/lijing/PycharmProjects/AD_DL_YQS/data/"
MODEL_STORE_PATH = "/data/users/lijing/PycharmProjects/AD_DL_YQS/data/"

rd = np.random.RandomState(888)
matrix_1 = rd.randint(0, 1, (999, 10))
matrix_2 = rd.randint(1, 2, (999, 10))
matrix_3 = rd.randint(2, 3, (999, 10))
matrix_4 = rd.randint(3, 4, (999, 10))
matrix_5 = rd.randint(4, 5, (999, 10))
matrix_6 = rd.randint(5, 6, (999, 10))
matrix_7 = rd.randint(6, 7, (999, 10))

label = np.array([0]*333+[1]*333+[2]*333)
samples = [matrix_1, matrix_2, matrix_3, matrix_4, matrix_5, matrix_6, matrix_7]

# sample1 = np.dstack(((matrix_1,matrix_2,matrix_3,matrix_4)))
# sample2 = sample1
# sample3 = sample1
# sample4 = sample1
# ##idx        0       1      2       3  - 19
#             # 10x4    1X10x4
# samples = [sample1,sample2,sample3,sample4]



trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])



train_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=True, transform=trans, download=True)
test_dataset = torchvision.datasets.MNIST(root=DATA_PATH, train=False, transform=trans)


train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data,label):
        self.data = data
        self.label = label
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
       # print("getitem")
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        X = self.data[idx]
        y = self.label[idx]

        return X, y
####调用 20 10 4
from torch.utils.data import DataLoader
Dataset = CustomDataset(samples,label)

train_dataloader = DataLoader(Dataset,batch_size=2,shuffle=True)

Xs,ys = next(iter(train_dataloader))
print(Xs,ys)





class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet,self).__init__()
         #  3   64  64    -> 32    48   48    64 x 7 x7

        self.layer1 = nn.Sequential(
            
            nn.Conv2d(1,32,kernel_size=5,stride=1,padding=2),
            
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=2,stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=5,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2))


        self.drop_out = nn.Dropout()
        
        self.fc1 = nn.Linear(7*7*64,1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000,10)

    
    
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        
        out = out.reshape(out.size(0),-1)
        
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


model = ConvNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr = learning_rate)


total_step = len(train_loader) 
loss_list = []
acc_list = []



for epoch in range(10):

    for i,(images,labels) in enumerate(train_loader):
        outputs = model(images)

        loss = criterion(outputs,labels)
        loss_list.append(loss.item())


        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        total = labels.size(0)

        _,predicted = torch.max(outputs.data,1)
        correct = (predicted == labels).sum().item()
        acc_list.append(correct/total)

        if (i+1) % 100 == 0:
            print('Epoch[{}/{}],Step[{},{}],Loss:{:.4f},Accuracy:{:.2f}%'
            .format(epoch+1,num_epochs,i+1,total_step,loss.item(),(correct/total)*100))



model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images,labels in test_loader:
        outputs = model(images)
        _,predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Test Accuracy of the model on the 1w test images:{} %'.format((correct / total) * 100))

# save the model
torch.save(model.state_dict(),MODEL_STORE_PATH + 'conv_net_model.ckpt')



