'''import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

from PIL import Image


class SafeImageFolder(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except (OSError, IOError, Image.DecompressionBombError) as e:
            print(f"Skipping corrupted image at index {index}: {e}")
            return None  # Return None for bad images

class Cnn(nn.Module):
    def __init__(self, num_classes=10):  # adjust num_classes based on dataset
        super(Cnn, self).__init__()
        self.name = "cnn"

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)

        # Fully connected layers for classification
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.maxpool(self.relu(self.conv1(x)))
        x = self.maxpool(self.relu(self.conv2(x)))
        x = self.maxpool(self.relu(self.conv3(x)))

        x = x.view(-1, 64 * 8 * 8)  # Flatten for FC layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Cnn(num_classes=1710)  # Ensure num_classes matches the training config
# model_path = "model_cnn_bs64_lr0.01_epoch79.pth"  # Adjust to your latest trained model file
model_path = "C:\\Users\\leeho\\OneDrive - University of Toronto\\EngSci - Year 3 (Robo)\\Winter_2025\\APS360_Applied Fundamentals of Deep Learning\\Project\\model_cnn_bs64_lr0.01_epoch79.pth"
net.load_state_dict(torch.load(model_path, map_location=device))
net.to(device)
net.eval()
def testLoader(batch_size, root_dir="path_to_test_frames"):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_dir = "C:\\Users\\leeho\\Downloads\\APS(PALS)x_test_new_1\\testing_frames"
    
    # test_dir = "C:\\Users\\leeho\\OneDrive - University of Toronto\\EngSci - Year 3 (Robo)\\Winter_2025\\APS360_Applied Fundamentals of Deep Learning\\Project\\x_test_new_frames - Copy"
    
    # test_dir = "C:\\Users\\leeho\\OneDrive - University of Toronto\\EngSci - Year 3 (Robo)\\Winter_2025\\APS360_Applied Fundamentals of Deep Learning\\Project\\x_test_new_frames3"

    test_dataset = SafeImageFolder(test_dir, transform=transform)

    # Remove any bad images
    test_dataset.samples = [s for s in test_dataset.samples if s is not None]

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

# Load test dataset
test_loader = testLoader(batch_size=1710)

correct = 0
total = 0
predictions = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        pred = outputs.argmax(dim=1)
        
        predictions.extend(pred.cpu().numpy())  # Store predictions
        correct += (pred == labels).sum().item()
        total += labels.size(0)

test_accuracy = correct / total
print(f"Test Accuracy: {test_accuracy:.4f}")

# Save predictions to a file
np.savetxt("test_predictions2.csv", predictions, delimiter=",")
'''




import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt  
from PIL import Image

class SafeImageFolder(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except (OSError, IOError, Image.DecompressionBombError) as e:
            print(f"Skipping corrupted image at index {index}: {e}")
            return None  # Return None for bad images

class Cnn(nn.Module):
    def __init__(self, num_classes=10):  # adjust num_classes based on dataset
        super(Cnn, self).__init__()
        self.name = "cnn"

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)

        # Fully connected layers for classification
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.maxpool(self.relu(self.conv1(x)))
        x = self.maxpool(self.relu(self.conv2(x)))
        x = self.maxpool(self.relu(self.conv3(x)))

        x = x.view(-1, 64 * 8 * 8)  # Flatten for FC layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Cnn(num_classes=1710)  # Ensure num_classes matches the training config
# model_path = "model_cnn_bs64_lr0.01_epoch79.pth"  # Adjust to your latest trained model file
model_path = "C:\\Users\\leeho\\OneDrive - University of Toronto\\EngSci - Year 3 (Robo)\\Winter_2025\\APS360_Applied Fundamentals of Deep Learning\\Project\\model_cnn_bs64_lr0.01_epoch79.pth"
net.load_state_dict(torch.load(model_path, map_location=device))
net.to(device)
net.eval()

def testLoader(batch_size, root_dir="path_to_test_frames"):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # test_dir = "C:\\Users\\leeho\\Downloads\\APS(PALS)x_test_new_1\\testing_frames"
    
    test_dir = "C:\\Users\\leeho\\OneDrive - University of Toronto\\EngSci - Year 3 (Robo)\\Winter_2025\\APS360_Applied Fundamentals of Deep Learning\\Project\\x_test_new_frames - Copy"
    
    # test_dir = "C:\\Users\\leeho\\OneDrive - University of Toronto\\EngSci - Year 3 (Robo)\\Winter_2025\\APS360_Applied Fundamentals of Deep Learning\\Project\\x_test_new_frames3"
    
    test_dir = "C:\\Users\\leeho\\Downloads\\WASL_testing\\Frames"


    test_dataset = SafeImageFolder(test_dir, transform=transform)

    # Remove any bad images
    test_dataset.samples = [s for s in test_dataset.samples if s is not None]

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

# Load test dataset
test_loader = testLoader(batch_size=1710)
# Extract gloss names from the dataset
gloss_mapping = test_loader.dataset.classes  # list: classification to label classes

correct = 0
total = 0
predictions = []

'''with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        pred = outputs.argmax(dim=1)
        
        predictions.extend(pred.cpu().numpy())  # Store predictions
        correct += (pred == labels).sum().item()
        total += labels.size(0)

        # Visualize some images with their predictions and actual labels
        for i in range(min(5, inputs.size(0))):  # Show at most 5 images
            img = inputs[i].cpu().numpy().transpose((1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
            img = (img * 0.5) + 0.5  # Unnormalize the image

            # Plot the image
            plt.imshow(img)
            plt.axis('off')  # Hide axes
            plt.title(f"Pred: {pred[i].item()}, Actual: {labels[i].item()}")  # Show prediction and true label
            plt.show()

test_accuracy = correct / total
print(f"Test Accuracy: {test_accuracy:.4f}")'''


with torch.no_grad():

    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        pred = outputs.argmax(dim=1)

        unique_preds = set(pred.cpu().numpy())
        print(f"Unique predicted indices: {unique_preds}")

        # Ensure predictions are within valid range
        pred_glosses = [
            gloss_mapping[idx.item()] if idx.item() < len(gloss_mapping) else "UNKNOWN"
            for idx in pred
        ]


        # Store predictions
        predictions.extend(pred_glosses)

        correct += (pred == labels).sum().item()
        total += labels.size(0)
        
            
        # Print predictions with actual labels
        # for i in range((inputs.size(0))):  # Show at most 5 images
        
        for i in range(inputs.size(0)):  # Show at most 5 images
            img = inputs[i].cpu().numpy().transpose((1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
            img = (img * 0.5) + 0.5  # Unnormalize the image

            # Plot the image
            plt.imshow(img)
            plt.axis('off')  # Hide axes
            # if pred_glosses[i] == "UNKNOWN": 
            #     pred_glosses[i] = "bed"
            #     correct+=1
                
            plt.title(f"Pred:{pred_glosses[i]}, Actual: {gloss_mapping[labels[i].item()]}")  # Show prediction and true label
            plt.show()
            
            if correct ==1: 
                break


            print(f"Predicted Gloss: {pred_glosses[i]}, Actual: {gloss_mapping[labels[i].item()]}")




'''with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        pred = outputs.argmax(dim=1)

        # Convert predictions to gloss labels
        pred_glosses = [gloss_mapping[idx.item()] for idx in pred]

        # Store predictions
        predictions.extend(pred_glosses)

        correct_predictions = (pred == labels)  # Boolean mask for correct predictions
        correct += correct_predictions.sum().item()
        total += labels.size(0)

        # Print and visualize all correct predictions
        for i in range(inputs.size(0)):  
            if correct_predictions[i]:  # Check if the prediction was correct
                img = inputs[i].cpu().numpy().transpose((1, 2, 0))  # Convert (C, H, W) -> (H, W, C)
                img = (img * 0.5) + 0.5  # Unnormalize

                # Plot the correctly classified image
                plt.imshow(img)
                plt.axis('off')  # Hide axes
                plt.title(f"Pred: {pred_glosses[i]}, Actual: {gloss_mapping[labels[i].item()]}")
                plt.show()

                print(f"Correct Prediction: Predicted Gloss: {pred_glosses[i]}, Actual: {gloss_mapping[labels[i].item()]}")
'''

test_accuracy = correct / total

print(f"Test Accuracy: {test_accuracy:.4f}")

# Save predictions to a file
np.savetxt("test_predictions_gloss.csv", predictions, fmt="%s", delimiter=",")


# Save predictions to a file
# np.savetxt("test_predictions2.csv", predictions, delimiter=",")
