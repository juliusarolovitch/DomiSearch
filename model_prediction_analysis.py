import time
from torch.optim import Adam, lr_scheduler
from torch.utils.data import random_split, Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import random
from queue import PriorityQueue
import matplotlib.pyplot as plt
import os
import astar_module
from tqdm import tqdm

import multiprocessing
from multiprocessing import Pool, set_start_method


class MapDataset(Dataset):
    def __init__(self, map_directory, path_directory, transform=None, max_files=1000000, sets_per_map=500):
        self.map_directory = map_directory
        self.path_directory = path_directory
        self.transform = transform
        self.sets_per_map = sets_per_map
        self.max_files = max_files
        self.base_filenames = [f for f in os.listdir(
            map_directory) if f.endswith('.txt') and not '_set_' in f][:max_files]
        self.items = []
        for base_filename in self.base_filenames:
            map_index = base_filename.split('_')[1].split('.')[0]
            before_map_filepath = os.path.join(map_directory, base_filename)
            for set_num in range(sets_per_map):
                path_filepath = os.path.join(
                    path_directory, f"path_{map_index}_set_{set_num+1}.txt")
                after_map_filepath = os.path.join(
                    map_directory, f"map_{map_index}_set_{set_num+1}_after.txt")
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
        if len(path_data.shape) == 1:
            path_data = np.expand_dims(path_data, axis=0)
        start_channel = np.zeros_like(before_map)
        end_channel = np.zeros_like(before_map)
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
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout = nn.Dropout(0.25)
        self.conv_final = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.sigmoid(self.conv_final(x))
        return x


def plot_prediction_distribution(predictions, actuals):
    bins = np.linspace(0, 1, 100)
    digitized = np.digitize(predictions, bins) - 1
    positives = np.zeros(len(bins)-1, dtype=np.float32)
    totals = np.zeros(len(bins)-1, dtype=np.float32)
    for i in range(len(bins)-1):
        mask = digitized == i
        totals[i] = np.sum(mask)
        positives[i] = np.sum(actuals[mask])
    proportions = np.where(totals > 0, positives / totals, 0)
    plt.bar(bins[:-1], proportions, width=bins[1]-bins[0], align='edge')
    plt.xlabel('Predicted Value Bins')
    plt.ylabel('Proportion of True Positives')
    plt.title('True Positive Rate by Prediction Confidence')
    plt.savefig('distributions.png')


def adjusted_sigmoid(x, k=10):
    x_centered = 2 * (x - 0.5)
    return 1 / (1 + torch.exp(-k * x_centered))


def evaluate_accuracy_at_threshold(args):
    model, device, loader, threshold = args
    print(f"Now testing accuracy at {threshold}")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            outputs_transformed = adjusted_sigmoid(outputs)
            predicted_classes_transformed = (
                outputs_transformed > threshold).float()
            correct += (predicted_classes_transformed == labels).sum().item()
            total += labels.numel()
    accuracy = correct / total
    print(f"Accuracy at {threshold} was {accuracy}")
    return threshold, accuracy


def analyze_predictions_and_metrics_parallel(model, loader, device, num_processes=4):
    thresholds = np.arange(0.45, .55, 0.001)
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    pool = Pool(processes=num_processes)
    results = pool.map(evaluate_accuracy_at_threshold, [
                       (model, device, loader, threshold) for threshold in thresholds])
    pool.close()
    pool.join()

    best_threshold, best_accuracy = max(results, key=lambda x: x[1])
    return best_threshold, best_accuracy


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "best_mapnet_model.pth"
    dataset = MapDataset(map_directory="maps", path_directory="paths")
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = MapNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    best_threshold, best_accuracy = analyze_predictions_and_metrics_parallel(
        model, loader, device)
    print(f"Best Threshold: {best_threshold:.1f}")
    print(f"Best Accuracy: {best_accuracy:.2f}")
