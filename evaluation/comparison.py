import numpy as np
import pickle
import os
import sys
sys.path.append('.')

from environment.map_generator import generate_2d_map, get_valid_start_goal
from environment.terrain_generator import generate_3d_terrain
from planning.astar import astar_path
from navigation.run_navigation import run_ann_navigation
from evaluation.metrics import compute_metrics
from model.train_ann import load_model
from config import *

def compare_algorithms(num_scenarios=10, map_size=DEFAULT_MAP_SIZE):
    """Compare A* and ANN performance on energy efficiency"""

    # Load the trained ANN model
    try:
        model = load_model(phase=2)
        print("ANN model loaded successfully")
    except Exception as e:
        print(f"Failed to load ANN model: {e}")
        return None

    astar_results = []
    ann_results = []

    print(f"Running comparison on {num_scenarios} scenarios...")

    for i in range(num_scenarios):
        print(f"Scenario {i+1}/{num_scenarios}")

        # Generate environment
        grid = generate_2d_map(map_size, density=DEFAULT_OBSTACLE_DENSITY)
        terrain = generate_3d_terrain(map_size, terrain_type="mixed")
        start, goal = get_valid_start_goal(grid)

        # Test A*
        astar_path_result = astar_path(grid, terrain, start, goal, phase=2)
        astar_metrics = compute_metrics(astar_path_result, "success" if astar_path_result else "failed", terrain, phase=2)
        astar_results.append(astar_metrics)

        # Test ANN
        try:
            ann_result = run_ann_navigation(grid, terrain, start, goal, model, phase=2)
            ann_path_result, ann_status = ann_result
            ann_metrics = compute_metrics(ann_path_result, "success" if ann_status == "success" else "failed", terrain, phase=2)
        except Exception as e:
            print(f"ANN failed on scenario {i+1}: {e}")
            ann_metrics = compute_metrics(None, "failed", terrain, phase=2)
        ann_results.append(ann_metrics)

    # Calculate averages
    def avg_metric(results, key):
        values = [r[key] for r in results if r['success']]
        return np.mean(values) if values else float('inf')

    astar_avg_energy = avg_metric(astar_results, 'total_energy')
    ann_avg_energy = avg_metric(ann_results, 'total_energy')

    astar_success_rate = sum(1 for r in astar_results if r['success']) / len(astar_results)
    ann_success_rate = sum(1 for r in ann_results if r['success']) / len(ann_results)

    print("\n" + "="*50)
    print("COMPARISON RESULTS")
    print("="*50)
    print(f"A* Average Energy:    {astar_avg_energy:.2f}")
    print(f"ANN Average Energy:   {ann_avg_energy:.2f}")
    print(f"A* Success Rate:      {astar_success_rate:.1%}")
    print(f"ANN Success Rate:     {ann_success_rate:.1%}")

    if ann_avg_energy < astar_avg_energy:
        energy_improvement = ((astar_avg_energy - ann_avg_energy) / astar_avg_energy) * 100
        print(f"ANN Energy Improvement: {energy_improvement:.1f}%")
    else:
        energy_degradation = ((ann_avg_energy - astar_avg_energy) / astar_avg_energy) * 100
        print(f"ANN Energy Degradation: {energy_degradation:.1f}%")

    return {
        'astar_avg_energy': astar_avg_energy,
        'ann_avg_energy': ann_avg_energy,
        'astar_success_rate': astar_success_rate,
        'ann_success_rate': ann_success_rate
    }

if __name__ == "__main__":
    results = compare_algorithms(num_scenarios=20)
    print("\nComparison completed!")
