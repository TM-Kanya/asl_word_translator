
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Adjust paths to match dataset structure
def dataLoader(batch_size, root_dir="C:\\Users\\leeho\\OneDrive - University of Toronto\\EngSci - Year 3 (Robo)\\Winter_2025\\APS360_Applied Fundamentals of Deep Learning\\Project\\x_MHI_FINAL"):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize if needed
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dir = "C:\\Users\\leeho\\OneDrive - University of Toronto\\EngSci - Year 3 (Robo)\\Winter_2025\\APS360_Applied Fundamentals of Deep Learning\\Project\\x_train_sample_TRIAL"
    val_dir = "C:\\Users\\leeho\\OneDrive - University of Toronto\\EngSci - Year 3 (Robo)\\Winter_2025\\APS360_Applied Fundamentals of Deep Learning\\Project\\x_val_sample_TRIAL"


    train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = torchvision.datasets.ImageFolder(val_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


## similar to lab 3 

class Cnn(nn.Module):
    
    # count = 0
    
    def __init__(self, num_classes=10):  # adjust num_classes based on dataset
        super(Cnn, self).__init__()
        self.name = "cnn"

        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)

        # fully connected layer to match classes 
        self.fc1 = nn.Linear(64 * 8 * 8, 256)  
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)  

    def forward(self, x):
        x = self.maxpool(self.relu(self.conv1(x)))  
        x = self.maxpool(self.relu(self.conv2(x)))  
        x = self.maxpool(self.relu(self.conv3(x)))  

        # Print the shape before flattening (debugging step)
        # count+=1
        # print(count, " Shape before flattening:", x.shape)

        x = x.view(-1, 64 * 8 * 8)  # Flatten for FC layer
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x

def eval(net, loader):
    correct = 0
    total = 0
    device = next(net.parameters()).device

    net.eval()  # Set model to eval mode
    with torch.no_grad():
        for inputs, labels in loader:
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)  # Ensures batch shape

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

    return correct / total



from tqdm import tqdm

def train(net, batch_size, learning_rate, num_epochs=30):
    torch.manual_seed(1000)

    train_loader, val_loader = dataLoader(batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    train_acc = np.zeros(num_epochs)
    val_acc = np.zeros(num_epochs)

    for epoch in range(num_epochs):
        net.train()
        correct, total = 0, 0
        
        # wrap the training loop with tqdm to display the progress bar
        loop = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch', ncols=100)
        
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

            # progress bar
            loop.set_postfix(loss=loss.item(), accuracy=correct / total)

        train_acc[epoch] = correct / total
        val_acc[epoch] = eval(net, val_loader)

        print(f"Epoch {epoch + 1}: Train acc: {train_acc[epoch]:.4f} | Validation acc: {val_acc[epoch]:.4f}")

        # model path for each epoch
        model_path = f"model_{net.name}_bs{batch_size}_lr{learning_rate}_epoch{epoch}.pth"
        torch.save(net.state_dict(), model_path)

    print(train_acc)
    print(val_acc)
    np.savetxt("{}_train_acc.csv".format(model_path), train_acc)
    np.savetxt("{}_val_acc.csv".format(model_path), val_acc)

    print("Training complete!")

# Training Example:
net = Cnn(num_classes=51)  # Adjust num_classes to match your dataset


print("_________________________________ 1. Sz = 256, LR = 0.005, Ep = 50 (NEW BEST ~40% ACCURACY) _________________________________")
train(net, batch_size = 256, learning_rate=0.005, num_epochs=20)



# train(net, batch_size = 128, learning_rate=0.05, num_epochs=20)
# print("_________________________________ 1. Sz = 128, LR = 0.05, Ep = 20 _________________________________")
# print("/n")

# train(net, batch_size = 128, learning_rate=0.005, num_epochs=20)
# print("_________________________________ 2. Sz = 128, LR = 0.05, Ep = 20 _________________________________")
# print("/n")


# train(net, batch_size = 256, learning_rate=0.05, num_epochs=20)
# print("_________________________________ 3. Sz = 256, LR = 0.05, Ep = 20 _________________________________")
# print("/n")



# print("_________________________________ 4. Sz = 256, LR = 0.005, Ep = 25 (NEW BEST ~40% ACCURACY) _________________________________")
print("/n")


# train(net, batch_size = 32, learning_rate=0.01, num_epochs=15)
# print("_________________________________ 5. Sz = 32, LR = 0.01, Ep = 15  (this is the best one for now)_________________________________")
# print("/n")


# train(net, batch_size = 64, learning_rate=0.05, num_epochs=15)
# print("_________________________________ 6. Sz = 64, LR = 0.05, Ep = 15  (this is the best one for now)__________________________________")
# print("/n")


# train(net, batch_size = 64, learning_rate=0.005, num_epochs=20)
# print("_________________________________ 7. Sz = 64, LR = 0.05, Ep = 20  (this is the best one for now)__________________________________")
# print("/n")


