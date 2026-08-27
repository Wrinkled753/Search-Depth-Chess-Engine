import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from training.model import ChessEvalNet

# Hyperparameters
BATCH_SIZE = 1024
EPOCHS = 10
LEARNING_RATE = 0.001
HIDDEN_SIZE = 256
NUM_LAYERS = 3

DATA_DIR = "data/processed"
MODELS_DIR = "models"
PLOT_FILE = "loss_curve.png"

def load_data():
    print("Loading datasets...")
    train_ds = torch.load(os.path.join(DATA_DIR, "train.pt"), weights_only=False)
    val_ds = torch.load(os.path.join(DATA_DIR, "val.pt"), weights_only=False)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, val_loader

def train_model():
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_loader, val_loader = load_data()
    
    model = ChessEvalNet(hidden_size=HIDDEN_SIZE, num_hidden_layers=NUM_LAYERS).to(device)
    
    # Smooth L1 Loss is often preferred over MSE for regression to avoid huge gradients from outliers
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print("Starting training...")
    for epoch in range(EPOCHS):
        # Training Phase
        model.train()
        running_train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * inputs.size(0)
            
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item() * inputs.size(0)
                
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
        
        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            model_path = os.path.join(MODELS_DIR, "eval_net.pt")
            torch.save(model.state_dict(), model_path)
            print(f" -> Saved new best model to {model_path}")
            
    print("Training complete!")
    
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, EPOCHS+1), train_losses, label='Train Loss')
    plt.plot(range(1, EPOCHS+1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Smooth L1 Loss')
    plt.title('Training & Validation Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOT_FILE)
    print(f"Loss curve saved to {PLOT_FILE}")

if __name__ == "__main__":
    train_model()
