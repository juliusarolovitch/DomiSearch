import time
from torch.optim import Adam, lr_scheduler
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import random
from queue import PriorityQueue
import matplotlib.pyplot as plt
import os
import astar_module
from tqdm import tqdm  # Import tqdm for the loading bar
import multiprocessing
from tqdm import tqdm


num_maps = 3000
width, height, density = 20, 20, 0.3

# Check for GPU availability and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class MapDataset(Dataset):
    def __init__(self, map_directories, path_directories, transform=None, max_files_per_directory=1000000, sets_per_map=500):
        self.map_directories = map_directories
        self.path_directories = path_directories
        self.transform = transform
        self.sets_per_map = sets_per_map
        self.max_files = max_files_per_directory
        self.items = []

        for map_directory, path_directory in zip(map_directories, path_directories):
            base_filenames = [f for f in os.listdir(map_directory) if f.endswith(
                '.txt') and not '_set_' in f][:self.max_files]
            for base_filename in base_filenames:
                map_index = base_filename.split('_')[1].split('.')[0]
                before_map_filepath = os.path.join(
                    map_directory, base_filename)
                for set_num in range(sets_per_map):
                    path_filename = f"path_{map_index}_set_{set_num+1}.txt"
                    path_filepath = os.path.join(path_directory, path_filename)
                    after_map_filename = f"map_{map_index}_set_{set_num+1}_after.txt"
                    after_map_filepath = os.path.join(
                        map_directory, after_map_filename)
                    if os.path.exists(path_filepath) and os.path.exists(after_map_filepath):
                        self.items.append(
                            (before_map_filepath, path_filepath, after_map_filepath))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        before_map_filepath, path_filepath, after_map_filepath = self.items[idx]
        before_map = np.loadtxt(before_map_filepath, dtype=np.float32)
        after_map = np.loadtxt(after_map_filepath, dtype=np.float32)
        path_data = np.loadtxt(path_filepath, dtype=np.float32)

        if len(path_data.shape) == 1:  # Ensure path_data is two-dimensional
            path_data = np.expand_dims(path_data, axis=0)

        # Create channels for start, end, and path
        start_channel = np.zeros_like(before_map)
        end_channel = np.zeros_like(before_map)

        # if 0 <= int(path_data[0, 0]) < height and 0 <= int(path_data[0, 1]) < width:
        #     start_channel[int(path_data[0, 0]), int(path_data[0, 1])] = 1
        # else:
        #     raise ValueError(
        #         f"Invalid path coordinates: {path_data[0, 0]}, {path_data[0, 1]}")
        start_channel[int(path_data[0, 0]), int(path_data[0, 1])] = 1
        end_channel[int(path_data[-1, 0]), int(path_data[-1, 1])] = 1

        input_tensor = np.stack(
            [before_map, start_channel, end_channel], axis=0)
        if self.transform:
            input_tensor = self.transform(input_tensor)
            after_map = self.transform(after_map)

        return torch.tensor(input_tensor, dtype=torch.float32), torch.tensor(after_map, dtype=torch.float32)


class MapNet(nn.Module):
    def __init__(self):
        super(MapNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv_final = nn.Conv2d(512, 1, kernel_size=1)  # Output 1 channel
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout(F.relu(self.bn5(self.conv5(x))))
        x = torch.sigmoid(self.conv_final(x))
        return x


def train_model(model, train_loader, val_loader, epochs=20):
    model.train()
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    best_val_loss = float('inf')
    early_stopping_counter = 0

    for epoch in tqdm(range(epochs), desc='Epochs', leave=True):
        running_loss = 0.0
        total_loss = 0.0
        # Using tqdm to wrap the training loader to display progress
        train_iterator = tqdm(
            train_loader, desc=f'Training Epoch {epoch+1}', leave=False)
        for i, (input_tensors, targets) in enumerate(train_iterator):
            input_tensors, targets = input_tensors.to(
                device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(input_tensors)

            # Ensure targets tensor has an additional dimension to match output
            targets = targets.unsqueeze(1)

            loss = criterion(outputs, targets.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_loss += loss.item()

            # Update tqdm bar with the latest loss information
            train_iterator.set_postfix(loss=(running_loss / (i + 1)))

        average_loss = total_loss / len(train_loader)
        print(
            f'Epoch [{epoch+1}/{epochs}] Complete. Average Loss: {average_loss:.4f}')

        scheduler.step()

        # Validate after each epoch and update progress bar in the outer loop
        val_loss = validate_model(model, val_loader, criterion)
        tqdm.write(
            f'Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), "best_mapnet_model_20.pth")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= 3:
                tqdm.write(f"Early stopping triggered at epoch {epoch + 1}")
                break


def validate_model(model, val_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    with torch.no_grad():
        for input_tensors, targets in val_loader:
            input_tensors, targets = input_tensors.to(
                device), targets.to(device)
            outputs = model(input_tensors)
            targets = targets.unsqueeze(1)  # Adjusting dimensions for BCELoss
            loss = criterion(outputs, targets.float())
            total_loss += loss.item()

    average_loss = total_loss / len(val_loader)
    return average_loss


def custom_collate_fn(batch):
    if not batch:
        return torch.empty(0, 3, 20, 20), torch.empty(0, 1, 20, 20)

    # Unpack the batch.
    maps, targets = zip(*batch)

    # Determine the maximum size in the batch.
    max_height = max(map.size(1) for map in maps)
    max_width = max(map.size(2) for map in maps)

    padded_maps = []
    padded_targets = []

    # Pad each map and corresponding target to the maximum size.
    for map, target in zip(maps, targets):
        # Calculate padding. We add padding to the bottom and right edges.
        padding_height = max_height - map.size(1)
        padding_width = max_width - map.size(2)
        # Padding: left, right, top, bottom
        pad_dims = (0, padding_width, 0, padding_height)

        # Apply padding.
        padded_map = F.pad(map, pad_dims, mode='constant', value=0)
        padded_target = F.pad(target, pad_dims, mode='constant', value=0)

        padded_maps.append(padded_map)
        padded_targets.append(padded_target)

    # Stack all the padded maps and targets into two tensors.
    return torch.stack(padded_maps), torch.stack(padded_targets)


if __name__ == "__main__":
    map_suffixes = [20]
    map_directories = [f"maps_{suffix}" for suffix in map_suffixes]
    path_directories = [f"paths_{suffix}" for suffix in map_suffixes]

    full_dataset = MapDataset(
        map_directories=map_directories, path_directories=path_directories)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,
                              drop_last=True, collate_fn=custom_collate_fn, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False,
                            drop_last=True, collate_fn=custom_collate_fn, num_workers=8)

    model = MapNet().to(device)

    train_model(model, train_loader, val_loader, epochs=50)

    torch.save(model.state_dict(), "mapnet_model_20.pth")

    def __getitem__(self, idx):
        before_map_filepath, path_filepath, after_map_filepath = self.items[idx]
        before_map = np.loadtxt(before_map_filepath, dtype=np.float32)
        after_map = np.loadtxt(after_map_filepath, dtype=np.float32)
        path_data = np.loadtxt(path_filepath, dtype=np.float32)

        if len(path_data.shape) == 1:  # Ensure path_data is two-dimensional
            path_data = np.expand_dims(path_data, axis=0)

        # Create channels for start, end, and path
        start_channel = np.zeros_like(before_map)
        end_channel = np.zeros_like(before_map)

        if 0 <= int(path_data[0, 0]) < height and 0 <= int(path_data[0, 1]) < width:
            start_channel[int(path_data[0, 0]), int(path_data[0, 1])] = 1
        else:
            raise ValueError(
                f"Invalid path coordinates: {path_data[0, 0]}, {path_data[0, 1]}")

        end_channel[int(path_data[-1, 0]), int(path_data[-1, 1])] = 1

        input_tensor = np.stack(
            [before_map, start_channel, end_channel], axis=0)
        if self.transform:
            input_tensor = self.transform(input_tensor)
            after_map = self.transform(after_map)

        return torch.tensor(input_tensor, dtype=torch.float32), torch.tensor(after_map, dtype=torch.float32)
