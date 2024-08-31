import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Define the CNN model
class FashionMNISTCNN(nn.Module):
    def __init__(self, num_filters=8):
        super(FashionMNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, num_filters, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(num_filters, num_filters*2, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(num_filters*2 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Set up data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the Fashion MNIST dataset
def load_data(batch_size):
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# Training function
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:
            print(f'[{i + 1}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

# Evaluation function
def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy on test images: {accuracy:.2f}%')
    return accuracy

# Function to run experiments with different hyperparameters
def run_experiments(filter_values, batch_sizes, num_epochs):
    results = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for num_filters in filter_values:
        for batch_size in batch_sizes:
            print(f"Running experiment with {num_filters} filters and batch size {batch_size}")
            model = FashionMNISTCNN(num_filters=num_filters).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            train_loader, test_loader = load_data(batch_size)

            for epoch in range(num_epochs):
                print(f'Epoch {epoch+1}/{num_epochs}')
                train(model, train_loader, criterion, optimizer, device)
            
            accuracy = evaluate(model, test_loader, device)
            results.append((num_filters, batch_size, accuracy))
            
            # Save the model
            torch.save(model.state_dict(), f'fashion_mnist_cnn_{num_filters}_filters_{batch_size}_batch.pth')

    return results

# Function to plot results (accuracy vs number of filters)
def plot_accuracy_vs_filters(results):
    filter_values = sorted(set(result[0] for result in results))
    batch_sizes = sorted(set(result[1] for result in results))

    plt.figure(figsize=(10, 6))
    for batch_size in batch_sizes:
        accuracies = [result[2] for result in results if result[1] == batch_size]
        plt.plot(filter_values, accuracies, marker='o', label=f'Batch Size {batch_size}')

    plt.title('Model Accuracy vs Number of Filters')
    plt.xlabel('Number of Filters')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy_vs_filters.png')
    plt.close()

# Function to plot results (accuracy vs batch size)
def plot_accuracy_vs_batch_size(results):
    batch_sizes = sorted(set(result[1] for result in results))
    filter_values = sorted(set(result[0] for result in results))

    plt.figure(figsize=(10, 6))
    for num_filters in filter_values:
        accuracies = [result[2] for result in results if result[0] == num_filters]
        plt.plot(batch_sizes, accuracies, marker='o', label=f'Filters {num_filters}')

    plt.title('Model Accuracy vs Batch Size')
    plt.xlabel('Batch Size')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy_vs_batch_size.png')
    plt.close()

# Main execution
if __name__ == "__main__":
    filter_values = [8, 16, 32]
    batch_sizes = [32, 64]
    num_epochs = 5

    results = run_experiments(filter_values, batch_sizes, num_epochs)

    # Print results
    for num_filters, batch_size, accuracy in results:
        print(f'Num filters: {num_filters}, Batch size: {batch_size}, Accuracy: {accuracy:.2f}%')

    # Plot results
    plot_accuracy_vs_filters(results)
    plot_accuracy_vs_batch_size(results)

    print("Experiments completed. Results saved and plotted.")
