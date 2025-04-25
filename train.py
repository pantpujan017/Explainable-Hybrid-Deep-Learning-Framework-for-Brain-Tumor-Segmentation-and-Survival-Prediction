import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from model import AXUNet, BCEDiceLoss
from dataset import BrainTumorDataset

def train_model(model, train_loader, optimizer, criterion, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for inputs, masks in pbar:
            inputs, masks = inputs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())

if __name__ == "__main__":
    # Paths to preprocessed data
    train_dir = "D:\\brain\\data\\train"
    mask_dir = "D:\\brain\\data\\mask"

    # Hyperparameters
    batch_size = 64  # Use a small batch size for CPU
    num_epochs = 40
    learning_rate = 1e-4

    # Initialize dataset and DataLoader
    dataset = BrainTumorDataset(train_dir, mask_dir)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss, and optimizer
    device = torch.device("cpu")
    model = AXUNet().to(device)
    criterion = BCEDiceLoss().to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, train_loader, optimizer, criterion, num_epochs, device)

    # Save the trained model
    torch.save(model.state_dict(), "brain_tumor_model.pth")
    print("Model saved as brain_tumor_model.pth")