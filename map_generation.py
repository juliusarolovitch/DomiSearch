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
from tqdm import tqdm
import multiprocessing
from tqdm import tqdm


def generate_map(width, height, density):
    total_cells = width * height
    obstacle_count = int(total_cells * density)

    grid = np.zeros((height, width), dtype=int)

    obstacles = random.sample(range(total_cells), obstacle_count)
    for obstacle in obstacles:
        x = obstacle % width
        y = obstacle // width
        grid[y][x] = 2

    return grid


def path_cost(path):
    """Calculate the total cost of a given path based on the length."""
    return len(path) - 1  # Subtract 1 because we include the start point in the path length


def is_dead_end(map, start, goal, state):
    direct_path = astar_module.astar(map, start, goal)
    if direct_path is None:
        return True  # Cannot determine a dead end if no direct path exists from start to goal

    path_to_state = astar_module.astar(map, start, state)
    path_from_state = astar_module.astar(map, state, goal)

    if not path_to_state or not path_from_state:
        return True  # No valid path via the state means it's effectively a dead end or isolated

    # Calculate the cost of going through the state
    through_state_cost = path_cost(path_to_state) + path_cost(path_from_state)
    direct_cost = path_cost(direct_path)

    return through_state_cost > direct_cost


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def have_overlap(path1, path2):
    return not set(path1).isdisjoint(path2)

# Assuming generate_map, astar, and heuristic functions are defined as before


def generate_valid_start_goal(map):
    height, width = map.shape
    valid_points = [(y, x) for y in range(height)
                    for x in range(width) if map[y][x] == 0]

    if len(valid_points) < 2:
        raise ValueError(
            "Not enough open spaces to select distinct start and goal points.")

    start, goal = random.sample(valid_points, 2)
    path = astar_module.astar(map, start, goal)
    while path is None:  # Make sure there's a valid path
        start, goal = random.sample(valid_points, 2)
        path = astar_module.astar(map, start, goal)

    return start, goal


def mark_unreachable_as_blocked(map, start, goal):
    height, width = map.shape
    for y in range(height):
        for x in range(width):
            if map[y, x] == 0:  # Only check unblocked cells
                if astar_module.astar(map, (y, x), start) is None or astar_module.astar(map, (y, x), goal) is None:
                    # Mark as blocked if no path to start or goal
                    map[y, x] = float('inf')


def save_map_to_file(map, filename):
    with open(filename, 'w') as f:
        for row in map:
            f.write(' '.join(str(cell) for cell in row) + '\n')


def save_path_to_file(path, filename):
    with open(filename, 'w') as f:
        for state in path:
            f.write(f"{state[0]} {state[1]}\n")


def worker_task(args):
    i, width, height, density, directory, sets_per_map = args
    attempts = 0
    paths_generated = 0
    original_map = generate_map(width, height, density)
    original_map_filename = os.path.join(directory, f"map_{i}.txt")
    save_map_to_file(original_map, original_map_filename)

    while paths_generated < sets_per_map:
        if attempts >= 100:
            print(
                f"Regenerating map {i+1} after 100 unsuccessful attempts to find valid paths.")
            attempts = 0
            original_map = generate_map(width, height, density)
            original_map_filename = os.path.join(directory, f"map_{i}.txt")
            save_map_to_file(original_map, original_map_filename)
            paths_generated = 0

        map_copy = np.copy(original_map)
        start, goal = generate_valid_start_goal(map_copy)
        path = astar_module.astar(map_copy, start, goal)

        if path:
            paths_generated += 1
            attempts = 0

            path_filename = os.path.join(
                "paths_test", f"path_{i}_set_{paths_generated}.txt")
            save_path_to_file(path, path_filename)

            mark_unreachable_as_blocked(map_copy, start, goal)
            height, width = map_copy.shape
            dead_ends = {}

            for y in range(height):
                for x in range(width):
                    if map_copy[y][x] != 0:  # Ignore obstacles
                        continue
                    state = (y, x)
                    if is_dead_end(map_copy, start, goal, state) is True:
                        cost_diff = 1
                    else:
                        cost_diff = 0
                    map_copy[y][x] = cost_diff

            # Change all '2's to '1's
            map_copy[map_copy == 2] = 1

            after_filename = os.path.join(
                directory, f"map_{i}_set_{paths_generated}_after.txt")
            save_map_to_file(map_copy, after_filename)
        else:
            attempts += 1


def generate_and_save_maps_parallel(num_maps, width, height, density, directory="maps_test", sets_per_map=400):
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists("paths"):
        os.makedirs("paths")

    args = [(i, width, height, density, directory, sets_per_map)
            for i in range(num_maps)]
    with multiprocessing.Pool(processes=40) as pool:
        # Generate progress bar for map generation
        for _ in tqdm(pool.imap_unordered(worker_task, args), total=num_maps, desc='Generating maps'):
            pass


# # Example usage
num_maps = 1
width, height, density = 200, 200, 0.3
generate_and_save_maps_parallel(num_maps, width, height, density)
