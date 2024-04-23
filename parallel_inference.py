import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from queue import PriorityQueue
import threading
import time


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
        self.conv_final = nn.Conv2d(512, 1, kernel_size=1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout(F.relu(self.bn5(self.conv5(x))))
        x = torch.sigmoid(self.conv_final(x))
        return x


def heuristic(a, b):
    return abs(b[0] - a[0]) + abs(b[1] - a[1])


def reconstruct_path(came_from, current):
    path = []
    while current:
        path.append([current[0], current[1]])
        current = came_from[current]
    path.reverse()
    return path


def a_star(grid, start, goal, heuristic_grid, ready_event):
    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    open_set = PriorityQueue()
    open_set.put((0 + heuristic(start, goal), 0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}
    expansions = 0

    search_start_time = time.time()

    while not open_set.empty():
        time.sleep(.005)
        _, current_cost, current = open_set.get()

        if current == goal:
            search_end_time = time.time()
            search_time = search_end_time - search_start_time
            print(f"A* search completed in {search_time:.4f} seconds")
            path = reconstruct_path(came_from, current)
            print(f"Path cost: {len(path)} with {expansions} expansions")
            return path, expansions

        expansions += 1
        for dx, dy in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0]):
                if grid[neighbor[0]][neighbor[1]] == 2:
                    continue  # Skip obstacles

                new_cost = current_cost + 1
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    if ready_event.is_set():  # Check if the heuristic grid is ready
                        heuristic_multiplier = heuristic_grid[neighbor[0]
                                                              ][neighbor[1]]
                    else:
                        heuristic_multiplier = 1  # Default heuristic before model output
                    priority = new_cost + \
                        heuristic(neighbor, goal) * heuristic_multiplier
                    open_set.put((priority, new_cost, neighbor))
                    came_from[neighbor] = current

    print("No path found.")
    return None, expansions


def model_inference_thread(model, input_tensor, heuristic_grid, ready_event):
    with torch.no_grad():
        output = model(input_tensor).squeeze(0).squeeze(0).cpu().numpy()
    output_transformed = np.exp(5 * (output - 0.5))
    np.copyto(heuristic_grid, output_transformed)
    ready_event.set()


def display_map_and_model_output(model_path, map_directory, path_directory, map_index, set_num):
    before_map_filepath = os.path.join(map_directory, f"map_{map_index}.txt")
    path_filepath = os.path.join(
        path_directory, f"path_{map_index}_set_{set_num}.txt")

    before_map = np.loadtxt(before_map_filepath, dtype=np.float32)
    path_data = np.loadtxt(path_filepath, dtype=np.float32)

    start = (1, 0)
    goal = (1999, 1999)

    start_channel = np.zeros_like(before_map)
    end_channel = np.zeros_like(before_map)
    start_channel[int(path_data[0, 0]), int(path_data[0, 1])] = 1
    end_channel[int(path_data[-1, 0]), int(path_data[-1, 1])] = 1

    input_tensor = torch.tensor(np.stack(
        [before_map, start_channel, end_channel], axis=0), dtype=torch.float32).unsqueeze(0).to(device)
    heuristic_grid = np.ones_like(before_map, dtype=np.float32)
    ready_event = threading.Event()

    # Conduct A* search without model inference
    search_start_time = time.time()
    path_no_inference, expansions_no_inference = a_star(
        before_map, (int(path_data[0, 0]), int(
            path_data[0, 1])), (int(path_data[-1, 0]), int(path_data[-1, 1])), heuristic_grid, ready_event)
    search_end_time = time.time()
    search_time_no_inference = search_end_time - search_start_time

    print("\nResults without model inference:")
    print(f"A* search completed in {search_time_no_inference:.4f} seconds")
    if path_no_inference is not None:
        print(
            f"Path cost: {len(path_no_inference)} with {expansions_no_inference} expansions")

    # Conduct A* search with model inference
    model = MapNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    inference_start_time = time.time()
    inference_thread = threading.Thread(target=model_inference_thread, args=(
        model, input_tensor, heuristic_grid, ready_event))
    search_thread = threading.Thread(target=a_star, args=(
        before_map, (int(path_data[0, 0]), int(
            path_data[0, 1])), (int(path_data[-1, 0]), int(path_data[-1, 1])), heuristic_grid, ready_event))

    inference_thread.start()
    search_thread.start()

    inference_thread.join()
    inference_end_time = time.time()
    search_thread.join()
    search_end_time = time.time()

    inference_time = inference_end_time - inference_start_time
    search_time = search_end_time - inference_start_time

    print("\nResults with model inference:")
    if inference_time < search_time:
        print("Model inference completed before search.")
    else:
        print("Search completed before model inference.")

    print(f"Time taken for model inference: {inference_time:.4f} seconds")
    print(f"A* search completed in {search_time:.4f} seconds")
    if path_no_inference is not None:
        print(
            f"Path cost: {len(path_no_inference)} with {expansions_no_inference} expansions")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'best_mapnet_model.pth'
map_directory = 'maps_100'
path_directory = 'paths_100'
map_index = 0
set_num = 6

display_map_and_model_output(
    model_path, map_directory, path_directory, map_index, set_num)
