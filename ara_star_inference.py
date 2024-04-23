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
from tqdm import tqdm  # Import tqdm for the loading bar
import heapq


def heuristic(a, b):
    return abs(b[0] - a[0]) + abs(b[1] - a[1])


def a_star(grid, start, goal, heuristic_grid, num_iterations):
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up

    expansions = 0

    for i in range(num_iterations):
        start_time = time.time()  # Record start time for the iteration
        # Initialize the open set and other data structures for each iteration
        open_set = []
        heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start))
        came_from = {}
        cost_so_far = {}
        came_from[start] = None
        cost_so_far[start] = 0

        adjusted_heuristic_grid = heuristic_grid - \
            i * (heuristic_grid - 1) / num_iterations
        if i == num_iterations-1:
            adjusted_heuristic_grid = np.ones_like(adjusted_heuristic_grid)

        while open_set:
            _, current_cost, current = heapq.heappop(open_set)

            if current == goal:
                path = []
                while current:
                    path.append([current[0], current[1]])
                    current = came_from[current]
                path.reverse()
                print(f"Path cost after iteration {i + 1}: {len(path)}")
                break

            expansions += 1
            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                if 0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0]):
                    if grid[neighbor[0]][neighbor[1]] == 2:
                        continue

                    new_cost = current_cost + 1
                    if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                        cost_so_far[neighbor] = new_cost
                        priority = new_cost + \
                            heuristic(neighbor, goal) * \
                            adjusted_heuristic_grid[neighbor[0]][neighbor[1]]

                        heapq.heappush(
                            open_set, (priority, new_cost, neighbor))
                        came_from[neighbor] = current

        # Calculate time taken for the iteration
        iteration_time = time.time() - start_time
        print(f"Time for iteration {i + 1}: {iteration_time:.4f} seconds")

    return path, expansions  # Return an empty path if no path is found


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


def display_map_and_model_output(model_path, map_directory, path_directory, map_index, set_num):
    before_map_filepath = os.path.join(map_directory, f"map_{map_index}.txt")
    path_filepath = os.path.join(
        path_directory, f"path_{map_index}_set_{set_num}.txt")
    after_map_filepath = os.path.join(
        map_directory, f"map_{map_index}_set_{set_num}_after.txt")

    if not os.path.exists(before_map_filepath):
        print("before map path doesn't exist")
        return
    if not os.path.exists(after_map_filepath):
        print("after map path doesn't exist")
        return
    if not os.path.exists(path_filepath):
        print("path filepath doesn't exist")
        return

    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'cpu')
    model = MapNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    before_map = np.loadtxt(before_map_filepath, dtype=np.float32)
    after_map = np.loadtxt(after_map_filepath, dtype=np.float32)
    path_data = np.loadtxt(path_filepath, dtype=np.float32)

    num_iterations = 4  # Define the number of iterations
    print("ARA* on original graph:")
    path, expansions = a_star(before_map, (int(path_data[0, 0]), int(
        path_data[0, 1])), (int(path_data[-1, 0]), int(path_data[-1, 1])), np.ones_like(before_map)*10, num_iterations)
    print(f"Expansions on regular maze: {expansions}, cost: {len(path)}")

    if len(path_data.shape) == 1:
        path_data = np.expand_dims(path_data, axis=0)

    start_channel = np.zeros_like(before_map)
    end_channel = np.zeros_like(before_map)
    start_channel[int(path_data[0, 0]), int(path_data[0, 1])] = 1
    end_channel[int(path_data[-1, 0]), int(path_data[-1, 1])] = 1

    input_tensor = torch.tensor(np.stack([before_map, start_channel, end_channel], axis=0),
                                dtype=torch.float32).unsqueeze(0)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        model_output = model(input_tensor).squeeze(0).squeeze(0).cpu().numpy()

    # model_output_transformed = model_output*2
    model_output_transformed = np.exp(5 * (model_output - 0.5))
    print("\n\n\nARA* mutant:")
    adjusted_path, expansions = a_star(before_map, (int(path_data[0, 0]), int(
        path_data[0, 1])), (int(path_data[-1, 0]), int(path_data[-1, 1])), model_output_transformed, num_iterations)
    print(
        f"Expansions on adjusted maze: {expansions}, cost: {len(adjusted_path)}")

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    titles = ['Before Map', 'After Map',
              'Model Output', 'Model Output Transformed']
    path = np.array(path)
    adjusted_path = np.array(adjusted_path)
    data = [before_map, after_map, model_output, model_output_transformed]
    for ax, datum, title in zip(axs, data, titles):
        im = ax.imshow(datum, cmap='viridis', interpolation='nearest')
        ax.set_title(title)
        ax.axis('on')
        plt.colorbar(im, ax=ax)

        if title == 'Model Output Transformed':
            for i in range(datum.shape[0]):
                for j in range(datum.shape[1]):
                    if datum[i, j] <= 1:
                        ax.plot(j, i, 'go', markersize=4)
            ax.plot(adjusted_path[:, 1], adjusted_path[:, 0], 'r-',
                    marker='o', markersize=2, linestyle='-', linewidth=1)
        elif title == 'Model Output':
            for i in range(datum.shape[0]):
                for j in range(datum.shape[1]):
                    if datum[i, j] <= .5:
                        ax.plot(j, i, 'go', markersize=4)
        else:
            ax.plot(path[:, 1], path[:, 0], 'r-',
                    marker='o', markersize=2, linestyle='-', linewidth=1)

    plt.tight_layout()
    fig.savefig('output_figure.png')


model_path = 'best_mapnet_model.pth'
map_directory = 'maps_20'
path_directory = 'paths_20'
map_index = 1
set_num = 26

display_map_and_model_output(
    model_path, map_directory, path_directory, map_index, set_num)
