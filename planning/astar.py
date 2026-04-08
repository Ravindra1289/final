import heapq
import math
from config import *

def astar_path(grid, terrain, start, goal, phase=2):
    """
    A* returning a list of tuples (y, x, heading).
    terrain: matching shape grid of heights. For Phase 1, pass an array of zeros or None.
    """
    sy, sx = start
    gy, gx = goal
    
    # Priority Queue: (f_score, g_score, (y, x, heading), path_so_far)
    open_set = []
    
    # We want to allow the robot to start facing towards the goal initially to avoid random starting penalties
    dy_goal = gy - sy
    dx_goal = gx - sx
    if abs(dy_goal) > abs(dx_goal):
        initial_hd = SOUTH if dy_goal > 0 else NORTH
    else:
        initial_hd = EAST if dx_goal > 0 else WEST
        
    start_state = (sy, sx, initial_hd)
    
    def heuristic(state):
        (cy, cx, _) = state
        return math.sqrt((cy - gy)**2 + (cx - gx)**2) * DISTANCE_COST
        
    heapq.heappush(open_set, (heuristic(start_state), 0, start_state, [start_state]))
    g_scores = {start_state: 0}
    
    while open_set:
        _, current_g, current_state, path = heapq.heappop(open_set)
        cy, cx, hd = current_state
        
        if (cy, cx) == goal:
            return path
            
        for action in [FORWARD, LEFT, RIGHT]:
            if action == FORWARD:
                new_hd = hd
                dy, dx = MOVES[new_hd]
                ny, nx = cy + dy, cx + dx
                cost_dist = DISTANCE_COST
                cost_turn = 0
            elif action == LEFT:
                new_hd = (hd - 1) % 4
                ny, nx = cy, cx
                cost_dist = 0
                cost_turn = TURN_COST
            elif action == RIGHT:
                new_hd = (hd + 1) % 4
                ny, nx = cy, cx
                cost_dist = 0
                cost_turn = TURN_COST
                
            slope_c = 0
            # Check bounds and obstacles only if moving
            if action == FORWARD:
                if not (0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]):
                    continue
                if grid[ny, nx] == 1:
                    continue
                
                if phase == 2 and terrain is not None:
                    h_curr = terrain[cy, cx]
                    h_next = terrain[ny, nx]
                    slope = max(0, h_next - h_curr)
                    if slope > MAX_SLOPE:
                        continue # Tool steep -> Obstacle
                    slope_c = slope * SLOPE_COST
            
            new_g = current_g + cost_dist + cost_turn + slope_c
            new_state = (ny, nx, new_hd)
            
            if new_state not in g_scores or new_g < g_scores[new_state]:
                g_scores[new_state] = new_g
                f_score = new_g + heuristic(new_state)
                heapq.heappush(open_set, (f_score, new_g, new_state, path + [new_state]))
                
    return None # No path found
