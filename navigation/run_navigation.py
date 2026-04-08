import numpy as np # type: ignore
import math # type: ignore
from config import NORTH, SOUTH, EAST, WEST, FORWARD, LEFT, RIGHT, MOVES, MAX_SLOPE, DISTANCE_COST, TURN_COST, SLOPE_COST # type: ignore
from dataset.dataset_generator import extract_features # type: ignore
from planning.astar import astar_path # type: ignore


def _try_teacher_rescue(path, grid, terrain, current_position, goal, phase):
    rescue_path = astar_path(grid, terrain, current_position, goal, phase=phase)
    if rescue_path and len(rescue_path) > 1:
        path.extend(rescue_path[1:])
        return True
    return False

def run_ann_navigation(grid, terrain, start, goal, model, phase=2, max_steps=500):
    cy, cx = start
    gy, gx = goal
    
    dy_goal = gy - cy
    dx_goal = gx - cx
    if abs(dy_goal) > abs(dx_goal):
        hd = SOUTH if dy_goal > 0 else NORTH
    else:
        hd = EAST if dx_goal > 0 else WEST
        
    max_steps = max(max_steps, grid.shape[0] * grid.shape[1] * 4)
    path = [(cy, cx, hd)]
    status = "running"
    consecutive_spins = 0
    consecutive_blocked = 0  # Track how many times forward is blocked in a row
    state_visits = {(cy, cx, hd): 1}
    position_visits = {(cy, cx): 1}
    best_goal_distance = math.sqrt((goal[0] - cy) ** 2 + (goal[1] - cx) ** 2)
    steps_since_progress = 0
    used_rescue = False
    
    for _ in range(max_steps):
        if (cy, cx) == goal:
            status = "success"
            break
            
        features = extract_features(grid, terrain, (cy, cx, hd), goal, phase=phase)
        X = np.array(features).reshape(1, -1)
        probs = model.predict_proba(X)[0]

        # Enhanced ANN Logic: Smart decision making with goal awareness
        # Calculate direct distance to goal and current heading efficiency
        dy_goal = goal[0] - cy
        dx_goal = goal[1] - cx
        dist_to_goal = math.sqrt(dy_goal**2 + dx_goal**2)
        if dist_to_goal + 1e-6 < best_goal_distance:
            best_goal_distance = dist_to_goal
            steps_since_progress = 0
        else:
            steps_since_progress += 1

        # Determine optimal heading toward goal
        if abs(dy_goal) > abs(dx_goal):
            optimal_hd = SOUTH if dy_goal > 0 else NORTH
        else:
            optimal_hd = EAST if dx_goal > 0 else WEST

        # Calculate heading alignment bonus
        heading_alignment = 1.0 - abs((hd - optimal_hd) % 4) / 2.0  # 1.0 for perfect alignment, 0.5 for 90° off, 0.0 for 180° off

        # Adaptive hysteresis based on situation
        if dist_to_goal < 5:  # Close to goal - prioritize direct movement
            HYSTERESIS_BIAS = 0.3
        elif consecutive_blocked > 5:  # Stuck - encourage exploration
            HYSTERESIS_BIAS = -0.1  # Actually discourage forward when stuck
        else:  # Normal navigation
            HYSTERESIS_BIAS = 0.15

        # Apply goal-directed bias when heading is misaligned and we're not stuck
        if consecutive_blocked <= 3 and heading_alignment < 0.7:
            # Boost turning actions that move toward optimal heading
            left_turn_hd = (hd - 1) % 4
            right_turn_hd = (hd + 1) % 4
            left_alignment = 1.0 - abs((left_turn_hd - optimal_hd) % 4) / 2.0
            right_alignment = 1.0 - abs((right_turn_hd - optimal_hd) % 4) / 2.0

            if left_alignment > right_alignment:
                probs[LEFT] += 0.2  # Boost left turn
            else:
                probs[RIGHT] += 0.2  # Boost right turn

        # Enhanced decision making: ANN predictions with energy-aware tie-breaking
        probs = np.array(probs)

        # Calculate energy costs for all actions (always needed)
        energy_costs = [1000, 1000, 1000]  # [FORWARD, LEFT, RIGHT]

        # Calculate actual energy costs for valid actions
        for action_idx in [FORWARD, LEFT, RIGHT]:
            if action_idx == FORWARD:
                dy, dx = MOVES[hd]
                ny, nx = cy + dy, cx + dx
                if (0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1] and
                    grid[ny, nx] == 0 and
                    (phase == 1 or terrain is None or max(0, terrain[ny, nx] - terrain[cy, cx]) <= MAX_SLOPE)):
                    energy_costs[action_idx] = DISTANCE_COST
                    if phase == 2 and terrain is not None:
                        slope = max(0, terrain[ny, nx] - terrain[cy, cx])
                        energy_costs[action_idx] += slope * SLOPE_COST
            else:  # LEFT or RIGHT turn
                energy_costs[action_idx] = TURN_COST

        # Blend the ANN probabilities with progress, loop-avoidance, and energy cost.
        action_scores = np.full(3, -1e9, dtype=float)
        candidate_states = {}
        current_dist = dist_to_goal

        for action_idx in [FORWARD, LEFT, RIGHT]:
            if action_idx == FORWARD:
                dy, dx = MOVES[hd]
                ny, nx = cy + dy, cx + dx
                new_hd = hd
                if energy_costs[action_idx] >= 1000:
                    continue
            elif action_idx == LEFT:
                ny, nx = cy, cx
                new_hd = (hd - 1) % 4
            else:
                ny, nx = cy, cx
                new_hd = (hd + 1) % 4

            new_state = (ny, nx, new_hd)
            candidate_states[action_idx] = new_state
            next_dist = math.sqrt((goal[0] - ny) ** 2 + (goal[1] - nx) ** 2)
            progress_bonus = current_dist - next_dist
            revisit_penalty = 0.22 * state_visits.get(new_state, 0) + 0.08 * position_visits.get((ny, nx), 0)
            energy_penalty = 0.035 * min(energy_costs[action_idx], 20)

            action_scores[action_idx] = (
                probs[action_idx]
                + 0.35 * progress_bonus
                - revisit_penalty
                - energy_penalty
            )

        action = int(np.argmax(action_scores))

        # If we are revisiting the same state too often, favor the least-visited valid action.
        current_state = (cy, cx, hd)
        if state_visits.get(current_state, 0) >= 3:
            valid_actions = [idx for idx, score in enumerate(action_scores) if score > -1e8]
            if valid_actions:
                action = min(
                    valid_actions,
                    key=lambda idx: (
                        state_visits.get(candidate_states[idx], 0),
                        position_visits.get(candidate_states[idx][:2], 0),
                        energy_costs[idx],
                        -action_scores[idx],
                    ),
                )

        should_rescue = (
            state_visits.get(current_state, 0) >= 5
            or steps_since_progress >= 18
            or consecutive_blocked >= 10
        )
        if should_rescue and _try_teacher_rescue(path, grid, terrain, (cy, cx), goal, phase):
            used_rescue = True
            status = "success_with_rescue"
            break

        # Apply hysteresis bias to encourage forward movement
        if action == FORWARD:
            pass  # Keep forward
        elif consecutive_blocked == 0:  # Only apply bias when not blocked
            # Small bias toward forward when possible
            forward_cost = energy_costs[FORWARD] if energy_costs[FORWARD] < 1000 else 1000
            turn_cost = energy_costs[action]
            if forward_cost < turn_cost * 1.5:  # Forward is reasonably better
                action = FORWARD
        
        if action == FORWARD:
            dy, dx = MOVES[hd]
            ny, nx = cy + dy, cx + dx  # type: ignore
            
            # Check if forward move is valid
            blocked = False
            if not (0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]):
                blocked = True
            elif grid[ny, nx] == 1:
                blocked = True
            elif phase == 2 and terrain is not None:
                slope = max(0, terrain[ny, nx] - terrain[cy, cx])
                if slope > MAX_SLOPE:
                    blocked = True
            
            if blocked:
                # Enhanced Recovery: Smart obstacle avoidance with goal awareness
                consecutive_blocked += 1
                if consecutive_blocked > 25:
                    if _try_teacher_rescue(path, grid, terrain, (cy, cx), goal, phase):
                        used_rescue = True
                        status = "success_with_rescue"
                    else:
                        status = "stuck"
                    break

                # Calculate which direction would best avoid obstacles and move toward goal
                best_turn = None
                best_score = -1

                for turn_action, turn_name in [(LEFT, "left"), (RIGHT, "right")]:
                    test_hd = (hd - 1) % 4 if turn_action == LEFT else (hd + 1) % 4
                    dy_test, dx_test = MOVES[test_hd]
                    ny_test, nx_test = cy + dy_test, cx + dx_test

                    # Check if turn leads to valid position
                    if not (0 <= ny_test < grid.shape[0] and 0 <= nx_test < grid.shape[1]):
                        continue
                    if grid[ny_test, nx_test] == 1:
                        continue
                    if phase == 2 and terrain is not None:
                        slope = max(0, terrain[ny_test, nx_test] - terrain[cy, cx])
                        if slope > MAX_SLOPE:
                            continue

                    # Score this turn based on goal direction and obstacle avoidance
                    dy_to_goal = gy - ny_test
                    dx_to_goal = gx - nx_test
                    if abs(dy_to_goal) > abs(dx_to_goal):
                        goal_hd = SOUTH if dy_to_goal > 0 else NORTH
                    else:
                        goal_hd = EAST if dx_to_goal > 0 else WEST

                    alignment_score = 1.0 - abs((test_hd - goal_hd) % 4) / 2.0

                    # Check for obstacles in the forward direction after this turn
                    obstacle_penalty = 0
                    for check_dist in range(1, 4):  # Look ahead 3 steps
                        check_y = ny_test + dy_test * check_dist
                        check_x = nx_test + dx_test * check_dist
                        if not (0 <= check_y < grid.shape[0] and 0 <= check_x < grid.shape[1]):
                            obstacle_penalty += 0.3
                            break
                        if grid[check_y, check_x] == 1:
                            obstacle_penalty += 0.5 / check_dist  # Closer obstacles penalize more
                            break

                    turn_score = alignment_score - obstacle_penalty

                    if turn_score > best_score:
                        best_score = turn_score
                        best_turn = turn_action

                # Execute best turn, or fallback to simple turn if no good option
                if best_turn is not None:
                    action = best_turn
                    hd = (hd - 1) % 4 if action == LEFT else (hd + 1) % 4
                else:
                    # Fallback: simple turn toward goal
                    dy_g = gy - cy
                    dx_g = gx - cx
                    if abs(dx_g) >= abs(dy_g):
                        desired_hd = EAST if dx_g > 0 else WEST
                    else:
                        desired_hd = SOUTH if dy_g > 0 else NORTH
                    left_dist = (hd - desired_hd) % 4
                    right_dist = (desired_hd - hd) % 4
                    if left_dist <= right_dist:
                        hd = (hd - 1) % 4
                    else:
                        hd = (hd + 1) % 4

                path.append((cy, cx, hd))
            else:
                # Move forward successfully
                consecutive_spins = 0
                consecutive_blocked = 0
                cy, cx = ny, nx
                path.append((cy, cx, hd))
        elif action == LEFT:
            hd = (hd - 1) % 4
            consecutive_spins += 1
            path.append((cy, cx, hd))
        elif action == RIGHT:
            hd = (hd + 1) % 4
            consecutive_spins += 1
            path.append((cy, cx, hd))
            
        if consecutive_spins >= 16:
            if _try_teacher_rescue(path, grid, terrain, (cy, cx), goal, phase):
                used_rescue = True
                status = "success_with_rescue"
            else:
                status = "spinning_failure"
            break

        latest_state = path[-1]
        state_visits[latest_state] = state_visits.get(latest_state, 0) + 1
        latest_pos = latest_state[:2]
        position_visits[latest_pos] = position_visits.get(latest_pos, 0) + 1
        
    if status == "running":
        if _try_teacher_rescue(path, grid, terrain, (cy, cx), goal, phase):
            used_rescue = True
            status = "success_with_rescue"
        else:
            status = "max_steps"

    if status == "success" and used_rescue:
        status = "success_with_rescue"
        
    return path, status
