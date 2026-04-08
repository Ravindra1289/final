from config import *
import math

def calculate_energy(path, terrain=None, phase=2):
    energy_dist = 0.0
    energy_turn = 0.0
    energy_slope = 0.0
    
    if not path or len(path) == 0:
        return 0.0, 0.0, 0.0, 0.0
        
    for i in range(len(path) - 1):
        cy, cx, hd1 = path[i]
        ny, nx, hd2 = path[i+1]
        
        if hd1 != hd2:
            energy_turn += TURN_COST
        elif cy != ny or cx != nx:
            energy_dist += DISTANCE_COST
            if phase == 2 and terrain is not None:
                slope = max(0, terrain[ny, nx] - terrain[cy, cx])
                energy_slope += slope * SLOPE_COST
                
    total_energy = energy_dist + energy_turn + energy_slope
    return total_energy, energy_dist, energy_turn, energy_slope

FAIL_PENALTY = 1000.0
SUCCESS_STATUSES = {"success", "success_with_rescue"}

def compute_metrics(path, status, terrain=None, phase=2):
    if not path:
        return {
            "status": status,
            "path_length": 0,
            "turns": 0,
            "total_energy": FAIL_PENALTY,
            "success": False
        }
    
    total, d_e, t_e, s_e = calculate_energy(path, terrain, phase)
    if status not in SUCCESS_STATUSES:
        total += FAIL_PENALTY
        
    turns = int(t_e / TURN_COST)
    path_len = len([p for i, p in enumerate(path) if i==0 or (path[i][:2] != path[i-1][:2])]) - 1
    
    return {
        "status": status,
        "path_length": max(0, path_len),
        "turns": turns,
        "total_energy": total,
        "success": (status in SUCCESS_STATUSES)
    }
