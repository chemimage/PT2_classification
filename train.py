# Required Libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import numpy as np
from torchvision import models
import os
from dataset import *
from model import *
from IPython.core.debugger import set_trace
import argparse

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs of training')
    parser.add_argument('--exp', type=int, default=0, help='experiment id')
    parser.add_argument('--batch_size', type=int, default=6, help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')



    opt = parser.parse_args()
    print(opt)
    # U-Net Model Initialization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=7,
        n_classes=4,
        depth=4,
        wf=6,
        padding=True,
        batch_norm=True,
        up_mode='upsample')

    model.to(device)

    os.makedirs('models', exist_ok=True)

    # Data Paths
    train_ir_dir = "../../data_classification/train/ir"
    train_labels_dir = "../../data_classification/train/target"
    test_ir_dir = "../../data_classification/test/ir"
    test_labels_dir = "../../data_classification/test/target"

    # Data Loaders
    train_data = IRDataset(train_ir_dir, train_labels_dir)
    val_data = IRDataset(test_ir_dir, test_labels_dir,mode = 'val')
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=True)

    # Loss function, Optimizer and Scheduler

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()


    # Training Loop with Early Stopping
    num_epochs = 200
    patience = 5
    best_val_loss = None

    for epoch in range(opt.n_epochs):
        model.train()
        epoch_val_loss = 0.0
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        scheduler.step()
        epoch_loss = running_loss / len(train_data)
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss}')

        # Validation loop
        model.eval()
        running_loss = 0.0

        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(val_data)
        print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {epoch_loss}')

        if best_val_loss is None or epoch_loss < best_val_loss:
            best_val_loss = epoch_loss
            torch.save(model.state_dict(), 'model_best'+'_'+ str(opt.exp) +'.pth')
            num_epochs_no_improvement = 0
        else:
            num_epochs_no_improvement += 1

        if num_epochs_no_improvement >= patience:
            print("Early stopping")
            break

if __name__ == '__main__':
    
    main()
