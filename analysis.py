import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import heapq


def heuristic(a, b):
    return abs(b[0] - a[0]) + abs(b[1] - a[1])


def a_star(grid, start, goal, heuristic_grid):
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start))
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0
    expansions = 0

    while open_set:
        _, current_cost, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current:
                path.append([current[0], current[1]])
                current = came_from[current]
            path.reverse()
            return path, expansions, current_cost
        expansions += 1
        for dx, dy in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0]) and grid[neighbor[0]][neighbor[1]] != 2:
                new_cost = current_cost + 1
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = (new_cost + heuristic(neighbor, goal)) * \
                        heuristic_grid[neighbor[0]][neighbor[1]]
                    heapq.heappush(open_set, (priority, new_cost, neighbor))
                    came_from[neighbor] = current
    return None, expansions, float('inf')


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
        self.dropout = nn.Dropout(0.5)
        self.conv_final = nn.Conv2d(512, 1, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout(F.relu(self.bn5(self.conv5(x))))
        x = torch.sigmoid(self.conv_final(x))
        return x


def display_map_and_model_output(model_path, map_directory, path_directory):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MapNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    total_expansions_original = 0
    total_cost_original = 0
    total_expansions_adjusted = 0
    total_cost_adjusted = 0
    num_optimal_paths = 0
    num_expansions_more_adjusted = 0
    num_trials = 0

    for map_index in range(1, 51):
        for set_num in range(1, 51):
            before_map_filepath = os.path.join(
                map_directory, f"map_{map_index}.txt")
            path_filepath = os.path.join(
                path_directory, f"path_{map_index}_set_{set_num}.txt")

            if not os.path.exists(before_map_filepath) or not os.path.exists(path_filepath):
                continue

            before_map = np.loadtxt(before_map_filepath, dtype=np.float32)
            path_data = np.loadtxt(path_filepath, dtype=np.float32)
            if len(path_data.shape) == 1:
                path_data = np.expand_dims(path_data, axis=0)

            path, expansions_original, cost_original = a_star(before_map, (int(path_data[0, 0]), int(
                path_data[0, 1])), (int(path_data[-1, 0]), int(path_data[-1, 1])), np.ones_like(before_map))
            total_expansions_original += expansions_original
            total_cost_original += cost_original

            start_channel = np.zeros_like(before_map)
            end_channel = np.zeros_like(before_map)
            start_channel[int(path_data[0, 0]), int(path_data[0, 1])] = 1
            end_channel[int(path_data[-1, 0]), int(path_data[-1, 1])] = 1

            input_tensor = torch.tensor(np.stack(
                [before_map, start_channel, end_channel], axis=0), dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                model_output = model(input_tensor).squeeze(
                    0).squeeze(0).cpu().numpy()

            model_output_transformed = model_output * 2
            path_adjusted, expansions_adjusted, cost_adjusted = a_star(before_map, (int(path_data[0, 0]), int(
                path_data[0, 1])), (int(path_data[-1, 0]), int(path_data[-1, 1])), model_output_transformed)
            total_expansions_adjusted += expansions_adjusted
            total_cost_adjusted += cost_adjusted
            num_trials += 1

            if len(path) == len(path_adjusted):
                num_optimal_paths += 1
            if expansions_adjusted > expansions_original:
                num_expansions_more_adjusted += 1

    if num_trials > 0:
        average_expansions_drop_percent = (
            (total_expansions_original - total_expansions_adjusted) / total_expansions_original) * 100
        average_cost_difference_percent = (
            (total_cost_adjusted - total_cost_original) / total_cost_original) * 100
        percent_optimal_paths = (num_optimal_paths / num_trials) * 100
        percent_expansions_more_adjusted = (
            num_expansions_more_adjusted / num_trials) * 100

        print(
            f"Average drop in expansions (%): {average_expansions_drop_percent}, Average difference in cost (%): {average_cost_difference_percent}")
        print(f"Percentage of optimal paths: {percent_optimal_paths}%")
        print(
            f"Percentage of paths with more expansions in adjusted path than original path: {percent_expansions_more_adjusted}%")
    else:
        print("No valid data to process.")


model_path = 'best_mapnet_model.pth'
map_directory = 'maps_100'
path_directory = 'paths_100'

display_map_and_model_output(model_path, map_directory, path_directory)
