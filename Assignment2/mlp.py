import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt



transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Define MLP_1, MLP_2, and MLP_3 models
class MLP_1(nn.Module):
    def __init__(self, input_size, hidden_dims):
        super(MLP_1, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_dims[0]))
        layers.append(nn.ReLU())
        
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dims[-1], 10))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input tensor
        return self.model(x)

class MLP_2(nn.Module):
    def __init__(self, input_size, hidden_dims):
        super(MLP_2, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.2))
        
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
        
        layers.append(nn.Linear(hidden_dims[-1], 10))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input tensor
        return self.model(x)

class MLP_3(nn.Module):
    def __init__(self, input_size, hidden_dims):
        super(MLP_3, self).__init__()
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.2))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], 10))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input tensor
        return self.model(x)

def mlp_train(model_num, input_size, learning_rate, batch_size, hidden_dims, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Training on: {device}")
    
    if model_num == 1:
        model = MLP_1(input_size, hidden_dims).to(device)
    elif model_num == 2:
        model = MLP_2(input_size, hidden_dims).to(device)
    elif model_num == 3:
        model = MLP_3(input_size, hidden_dims).to(device)
    else:
        print("Pass Either 1, 2, or 3 for the model to train")
        return None
    

    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(model)
    
    loss_f = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            images, labels = images.to(device).float(), labels.to(device)
            output = model(images)
            loss = loss_f(output, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_accuracy = 100 * correct / total
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}, Accuracy: {train_accuracy}%")
    
    # Test the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device).float(), labels.to(device)
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_accuracy = 100 * correct / total
    print(f"Test accuracy: {test_accuracy}%")
    return model
    
def mlp_apply(model, test_indexes):
    # Set model to evaluation mode
    model.eval()
    
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    # Get the selected test images and labels
    images = []
    labels_true = []
    for index in test_indexes:
        image, label = test_dataset[index]
        images.append(image)
        labels_true.append(label)
    
    # Convert images and labels to tensors
    images = torch.stack(images).to(device)
    labels_true = torch.tensor(labels_true).to(device)
    
    correct = 0
    # Perform inference
    with torch.no_grad():
        # Forward pass
        logits = model(images.view(-1, 28 * 28))
        _, predicted = torch.max(logits, dim=1)

        correct += (predicted == labels_true).sum().item()
    
    print('Accuracy of the network on the test images: %d %%' % (100 * correct / len(test_indexes)))
    print()
    
    # Plot the images with true and predicted labels
    fig, axs = plt.subplots(2, 5, figsize=(12, 6))
    fig.suptitle('MLP Classification Results')
    for i, ax in enumerate(axs.flat):
        ax.imshow(images[i].cpu().numpy().squeeze(), cmap='gray')
        ax.axis('off')
        ax.set_title(f"True: {labels_true[i].item()}\nPred: {predicted[i].item()}")
    plt.show()
