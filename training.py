import torch
import torch.nn as nn
from torch.optim import AdamW
from torchvision.models import resnet18 
from model_architecture import LightweightNetwork
from data_preprocessing import get_data_loaders
from torchvision.models import ResNet18_Weights

def train_model(train_loader, model, feature_extractor, optimizer, criterion, num_epochs=10):
    model.train()
    feature_extractor.eval()  # Freeze the feature extractor
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()
            # Extract features using the pre-trained model
            with torch.no_grad():
                features = feature_extractor(images)
            # Forward pass through the lightweight network
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

if __name__ == '__main__':
    input_channels = 512  # The output size of ResNet18's last convolutional layer
    hidden_dim = 256
    num_layers = 6
    # Initialize feature extractor
    feature_extractor = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    feature_extractor = nn.Sequential(*list(feature_extractor.children())[:-2])  # Remove the classifier layers
    # Initialize the lightweight network
    side_network = LightweightNetwork(input_channels, hidden_dim, num_layers)
    # Set up optimizer and loss function
    optimizer = AdamW(side_network.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    # Get data loaders
    data_dir = 'data/'
    train_loader, val_loader, test_loader = get_data_loaders(data_dir)
    # Train the model
    train_model(train_loader, side_network, feature_extractor, optimizer, criterion)

    torch.save(side_network.state_dict(), 'models/fpt_plus_trained_model.pth')
    print("Model has been successfully saved as 'fpt_plus_trained_model.pth'.")
