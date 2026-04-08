# 🤖 Energy-Aware Robot Navigation System — Complete Screen Documentation

**Application Type:** Desktop Web Application (Streamlit SaaS-style Dashboard)
**User Roles:** Student / Researcher / Academic Evaluator
**Tech Stack:** Python · Streamlit · Scikit-Learn · NumPy · Pandas · Matplotlib · Plotly

---
---

# SCREEN 1: 🌐 Application Core (`app.py`)

---

## 1. 🧾 SCREEN OVERVIEW
- **Purpose:** The central hub that merges environment generation, algorithm execution, and result visualization into one unified screen. It is a monolithic "all-in-one" dashboard.
- **Why it exists:** For quick demonstrations where the user wants to generate a map, run A*, train an ANN, and compare results without navigating between pages.
- **How users reach it:** Launched directly via `python main.py` → Streamlit loads `app.py` as the default page at `http://localhost:8501`.
- **Business importance:** Acts as the primary demo screen for academic presentations and evaluations.

---

## 2. 👤 USER FLOW & NAVIGATION

**Entry Points:**
- Application launch (`main.py` → `streamlit run ui/app.py`)

**Exit Points:**
- Sidebar links to individual module pages (Map Setup, Training, Simulation, Analytics, Comparison, Dashboard)

**Step-by-Step User Journey:**
1. User adjusts `Map Size`, `Obstacle Density`, `Terrain Type` sliders/selectors in the left column.
2. User clicks **"🗺️ Generate Map (or) Terrain"** → 2D grid + 3D terrain are generated.
3. User clicks **"🏃 Run A* Algorithm"** → Optimal heuristic path appears on 2D/3D maps.
4. User clicks **"🧠 Train ANN Model"** → Dataset is generated (50 maps × 10 scenarios) and the MLP is fitted.
5. User clicks **"🤖 Run ANN Navigation"** → ANN-driven path appears overlaid on the map.
6. User switches to **"📊 Metrics Comparison"** tab to see the energy/length/turns table and bar chart.
7. User reads the **"📝 Info"** tab for the energy cost model explanation.

**Edge Cases:**
- Running A* before generating a map → `st.error("Generate a map first.")`
- Running ANN before training → `st.error("Train or load a model first.")`
- Phase mismatch (Phase 1 model on Phase 2 map) → `st.error("Model mismatch!")`

---

## 3. 🎨 UI COMPONENT BREAKDOWN

| Component | Type | Description | States |
|---|---|---|---|
| Map Size Slider | `st.slider` | Range 10–50, default 20 | Default / Changed |
| Obstacle Density Slider | `st.slider` | Range 0.0–0.5, default 0.2 | Default / Changed |
| Terrain Type Selector | `st.selectbox` | Options: `flat`, `hills`, `steep`, `mixed (flat, hills, steep)` | Default / Selected |
| Terrain Caption | `st.caption` | "Hills possess gradual slopes, whereas steep terrain contains sharp elevation spikes." | Static |
| Phase Radio | `st.radio` | Phase 1 (2D Only) / Phase 2 (2D & 3D Slopes). **Disabled** if `flat` is selected. | Default / Disabled |
| Generate Map Button | `st.button` | Triggers `generate_2d_map()` and `generate_3d_terrain()` | Default / Clicked |
| Run A* Button | `st.button` | Runs the `astar_path()` function | Default / Success / Error |
| Train ANN Button | `st.button` | Generates dataset + trains MLP | Default / Spinner / Success |
| Run ANN Button | `st.button` | Executes `run_ann_navigation()` | Default / Success / Warning / Error |
| 2D Map (Matplotlib) | `st.pyplot` | Shows grid with terrain heatmap underlay, obstacles, paths | Empty / Populated |
| 3D Map (Plotly) | `st.plotly_chart` | Interactive 3D surface with Start/Goal markers and path traces | Hidden (Phase 1) / Rendered (Phase 2) |
| Metrics Table | `st.table` | Pandas DataFrame with Algorithm, Total Energy, Path Length, Turns, Status | Empty / Populated |
| Bar Chart | `st.bar_chart` | Grouped bars for Energy, Length, Turns | Empty / Populated |
| Info Tab | `st.markdown` | Static text explaining the energy cost model | Static |

---

## 4. ⚙️ FUNCTIONAL LOGIC
- **Map Generation:** `generate_2d_map(size, density)` creates a NumPy binary grid (`0`=free, `1`=obstacle). `generate_3d_terrain(size, terrain_type)` creates a height matrix using Gaussian filters.
- **Flat Terrain Lock:** If `terrain_type == "flat"`, `global_phase` is forcibly set to `1` and the Phase radio is disabled.
- **Terrain is always generated** regardless of phase, so the 2D map can always show the terrain heatmap underlay.
- **A* Path:** Uses weighted A* factoring forward cost (`1.0`), turn cost (`5.0`), and slope cost (`10.0 × height_diff`).
- **ANN Path:** Uses `predict_proba()` with a `HYSTERESIS_BIAS = 0.25` on the Forward action to suppress unnecessary turns.
- **Feature Validation:** Before running the ANN, the system checks `model.n_features_in_` matches the expected feature count (5 for Phase 1, 8 for Phase 2).

---

## 5. 🔌 API & BACKEND INTEGRATION
This application is fully **local** (no REST APIs). All calls are direct Python function invocations:

| Function Call | Module | Purpose | Returns |
|---|---|---|---|
| `generate_2d_map(size, density)` | `environment/map_generator.py` | Creates binary obstacle grid | `np.ndarray` |
| `generate_3d_terrain(size, type)` | `environment/terrain_generator.py` | Creates elevation height map | `np.ndarray` |
| `get_valid_start_goal(grid)` | `environment/map_generator.py` | Finds non-blocked start & goal cells | `tuple, tuple` |
| `astar_path(grid, terrain, start, goal, phase)` | `planning/astar.py` | Computes energy-optimal path | `list[(y,x,heading)]` or `None` |
| `generate_dataset(phase, num_maps, scenarios_per_map)` | `dataset/dataset_generator.py` | Builds labeled feature matrix | `X, y, paths` |
| `train_and_save_model(X, y, phase)` | `model/train_ann.py` | Trains & saves MLPClassifier | `model, score, path` |
| `run_ann_navigation(grid, terrain, start, goal, model, phase)` | `navigation/run_navigation.py` | Runs ANN inference loop | `path, status` |
| `compute_metrics(path, status, terrain, phase)` | `evaluation/metrics.py` | Calculates energy/length/turns | `dict` |

---

## 6. 🧠 STATE MANAGEMENT

**Global State (Streamlit `st.session_state`):**

| Variable | Type | Default | Description |
|---|---|---|---|
| `global_phase` | `int` | `2` | Current simulation phase (1 or 2) |
| `grid` | `np.ndarray` or `None` | `None` | 2D obstacle grid |
| `terrain` | `np.ndarray` or `None` | `None` | 3D elevation matrix |
| `start` | `tuple (y,x)` or `None` | `None` | Start coordinates |
| `goal` | `tuple (y,x)` or `None` | `None` | Goal coordinates |
| `astar_path` | `list` or `None` | `None` | A* computed path |
| `ann_path` | `list` or `None` | `None` | ANN computed path |
| `model` | `MLPClassifier` or `None` | Auto-loaded from disk | Trained neural network |
| `maps_generated` | `int` | `0` | Counter for dashboard |
| `dataset_rows` | `int/str` | `"--"` | Number of training rows |
| `last_accuracy` | `float/str` | `"--"` | Last training accuracy % |

**Data Flow:** `session_manager.py` → `init_session()` is called at the top of every page to ensure all variables exist before rendering.

---

## 7. ⚠️ ERROR HANDLING & EDGE CASES

| Scenario | Behavior |
|---|---|
| Generate button clicked but no grid yet | Grid is freshly created (no error) |
| A* clicked with no grid | `st.error("Generate a map first.")` |
| A* finds no valid path | `st.error("No valid A* path found! Please regenerate map.")` |
| ANN clicked with no model | `st.error("Train or load a model first.")` |
| ANN model phase mismatch | `st.error("Model mismatch! Model expects N features. Please retrain under Phase X.")` |
| ANN execution crashes | `try/except` catches and displays `st.error(f"Execution Error: {e}")` |
| ANN fails to reach goal | `st.warning(f"ANN Navigation failed: {status}")` with status like `max_steps`, `collision` |

---

## 8. 🔐 AUTHENTICATION & AUTHORIZATION
- **No authentication** required. This is a local academic research tool.
- All users have full access to every feature.
- No token or session handling beyond Streamlit's built-in browser session.

---

## 9. 📱 RESPONSIVENESS & PLATFORM BEHAVIOR
- Streamlit uses `layout="wide"` for maximum horizontal screen usage.
- Left column (controls) is 25% width; right column (visualizations) is 75%.
- Plotly 3D charts are fully interactive (rotate, zoom, pan) on all browsers.
- Matplotlib 2D charts render as static PNG images.
- **Mobile:** Streamlit auto-stacks columns vertically. The experience is functional but optimized for desktop.

---

## 10. 🎯 UX IMPROVEMENTS & MICRO-INTERACTIONS
- **Spinner animations** during dataset generation ("Generating ~20,000 samples...") and model training ("Training Model...").
- **Color-coded feedback:** Green `st.success` for positive outcomes, red `st.error` for failures, yellow `st.warning` for partial failures.
- **Tab-based navigation** (`Visualizations`, `Metrics Comparison`, `Info`) to reduce visual clutter.
- **Terrain caption** provides immediate educational context about hills vs. steep.

---

## 11. 🧪 TEST CASES

### Positive
| # | Test | Expected Result |
|---|---|---|
| TC-01 | Click Generate Map with defaults (20, 0.2, mixed) | 2D grid and 3D terrain render side-by-side |
| TC-02 | Click Run A* after generating map | Blue path appears on 2D map; success toast |
| TC-03 | Click Train ANN → Run ANN | Orange dashed path appears; success or warning toast |
| TC-04 | Switch to Metrics tab after both paths exist | Table shows two rows with Energy, Length, Turns |

### Negative
| # | Test | Expected Result |
|---|---|---|
| TC-05 | Click Run A* without generating map | Error: "Generate a map first." |
| TC-06 | Click Run ANN without training | Error: "Train or load a model first." |
| TC-07 | Train Phase 1 model, switch to Phase 2, Run ANN | Error: "Model mismatch!" |

### Edge Cases
| # | Test | Expected Result |
|---|---|---|
| TC-08 | Set density to 0.5 (max obstacles) | A* may return `None`; error displayed |
| TC-09 | Select "flat" terrain | Phase radio becomes disabled; 3D chart hidden |
| TC-10 | Generate very large map (50×50) | System processes slower but completes without crash |

---

## 12. 🔗 DEPENDENCIES
- **Screens connected:** All 6 sub-pages share data via `session_manager.py`
- **External libraries:** `streamlit`, `numpy`, `pandas`, `matplotlib`, `plotly`, `scikit-learn`, `scipy`
- **Shared components:** `init_session()`, `plot_2d_grid()`, `plot_3d_terrain()`, `compute_metrics()`

---

## 13. 🚀 FUTURE ENHANCEMENTS
- Add real-time animation of the robot traversing the path step-by-step.
- Implement multi-goal scenarios (visit waypoints in order).
- Add export functionality to save the trained model and map as a `.zip` bundle.

---

## 14. 📌 DEVELOPER NOTES
- `sys.path.append()` is used to resolve imports from the project root.
- `# type: ignore` comments are used to silence IDE static analysis warnings for runtime-resolved imports.
- The `width="stretch"` parameter on buttons requires Streamlit ≥ 1.28.

---
---

# SCREEN 2: 🏠 Dashboard (`ui/pages/dashboard.py`)

---

## 1. 🧾 SCREEN OVERVIEW
- **Purpose:** A quick-glance summary screen showing the current operational status of the system (model status, dataset size, accuracy).
- **Why it exists:** Provides professors and evaluators an instant overview without needing to navigate into each module.
- **How users reach it:** Click "Dashboard" in the Streamlit sidebar or navigate to `http://localhost:8501/dashboard`.
- **Business importance:** First impression screen for academic evaluations.

---

## 2. 👤 USER FLOW & NAVIGATION
- **Entry:** Sidebar navigation link
- **Exit:** Sidebar links to other pages
- **User Journey:** User lands on dashboard → reads 4 metric cards → navigates to specific module for details
- **Edge case:** If no model is trained, the card shows "Not Trained ❌"

---

## 3. 🎨 UI COMPONENT BREAKDOWN

| Component | Type | Description | States |
|---|---|---|---|
| Maps Generated | `st.metric` | Shows count of maps generated in the session | `0` / `N` |
| ANN Model Status | `st.metric` | Displays "Ready ✅" or "Not Trained ❌" | Ready / Not Trained |
| Dataset Size | `st.metric` | Shows `dataset_rows` from session state | `"--"` / `N` |
| Last Accuracy | `st.metric` | Shows last training accuracy percentage | `"--"` / `XX.XX%` |
| Navigation Info | `st.info` | Text: "Navigate using the sidebar to delve deeply into each specific module." | Static |

---

## 4. ⚙️ FUNCTIONAL LOGIC
- All data is read-only from `st.session_state`. No computations are performed on this screen.
- The model status check is a simple truthy evaluation: `"Ready ✅" if st.session_state.model else "Not Trained ❌"`.

---

## 5. 🔌 API & BACKEND INTEGRATION

| Function Call | Module | Purpose |
|---|---|---|
| `load_model(phase)` | `model/train_ann.py` | Imported but only used via `session_manager.py` |

---

## 6. 🧠 STATE MANAGEMENT
- **Reads:** `maps_generated`, `model`, `dataset_rows`, `last_accuracy`
- **Writes:** None (read-only screen)

---

## 7. ⚠️ ERROR HANDLING & EDGE CASES
- No interactive inputs → no input validation needed.
- If session state is uninitialized, `init_session()` provides safe defaults (`0`, `None`, `"--"`).

---

## 8. 🔐 AUTHENTICATION & AUTHORIZATION
- No authentication. Accessible to all users.

---

## 9. 📱 RESPONSIVENESS & PLATFORM BEHAVIOR
- Uses `st.columns(4)` for metric cards → responsive stacking on mobile.
- Minimal rendering → fastest page in the application.

---

## 10. 🎯 UX IMPROVEMENTS & MICRO-INTERACTIONS
- Color-coded model status (✅ / ❌) provides instant visual feedback.
- The horizontal rule (`---`) separates metrics from the navigation tip.

---

## 11. 🧪 TEST CASES

| # | Test | Expected |
|---|---|---|
| TC-01 | Open Dashboard before any actions | All metrics show defaults: `0`, `Not Trained ❌`, `--`, `--` |
| TC-02 | Generate a map on Map Setup, return to Dashboard | Maps Generated = `1` |
| TC-03 | Train a model, return to Dashboard | Status = `Ready ✅`, Accuracy = `XX.XX%` |

---

## 12. 🔗 DEPENDENCIES
- `session_manager.py` for state initialization
- `model/train_ann.py` for model loading

---

## 13. 🚀 FUTURE ENHANCEMENTS
- Add a mini-timeline of recent session activities.
- Add comparison sparklines showing energy trends across multiple runs.

---

## 14. 📌 DEVELOPER NOTES
- This is a purely passive display screen.
- All metric values are managed centrally by `session_manager.py`.

---
---

# SCREEN 3: 🗺️ Map Setup (`ui/pages/map_setup.py`)

---

## 1. 🧾 SCREEN OVERVIEW
- **Purpose:** Dedicated environment creation and visual preview of the simulation workspace.
- **Why it exists:** Separates environment configuration from algorithm execution for clearer workflow.
- **How users reach it:** Sidebar link "Map Setup" or `http://localhost:8501/map_setup`.
- **Business importance:** The physical foundation every simulation depends on.

---

## 2. 👤 USER FLOW & NAVIGATION
- **Entry:** Sidebar navigation
- **Exit:** Sidebar to Training (next logical step)
- **Journey:**
  1. Adjust Grid Size (10–50), Obstacle Density (0.0–0.5), Terrain Type
  2. Select Phase Engine (1 or 2; disabled for flat terrain)
  3. Click "Generate Map (or) Terrain"
  4. View 2D Grid Snapshot (left) and 3D Terrain Perspective (right)
- **Edge cases:** Flat terrain → Phase radio disabled, 3D chart shows warning

---

## 3. 🎨 UI COMPONENT BREAKDOWN

| Component | Type | Key | States |
|---|---|---|---|
| Grid Size | `st.slider` | `sz_slider` | 10–50, default 20 |
| Obstacle Density | `st.slider` | `den_slider` | 0.0–0.5, default 0.2 |
| Terrain Type | `st.selectbox` | `ter_sel` | `flat` / `hills` / `steep` / `mixed (flat, hills, steep)` |
| Terrain Caption | `st.caption` | — | Static help text |
| Phase Engine Radio | `st.radio` | `global_phase` / `global_phase_fake` (disabled) | Enabled / Disabled |
| Generate Button | `st.button` | — | Default / Clicked → Success |
| 2D Grid (Matplotlib) | `st.pyplot` | — | Empty / Terrain heatmap + obstacles |
| 3D Terrain (Plotly) | `st.plotly_chart` | — | Hidden (Phase 1) / Rendered (Phase 2) |

---

## 4. ⚙️ FUNCTIONAL LOGIC
- **Map generation** always creates both `grid` and `terrain`, regardless of phase.
- **Flat lock:** When `terrain_type == "flat"`, `st.session_state.global_phase` is forced to `1` and the radio renders as disabled.
- **2D terrain underlay:** `plot_2d_grid()` receives `terrain=st.session_state.terrain` to render the elevation colormap behind the obstacle grid.
- **Maps counter:** `st.session_state.maps_generated += 1` on each generation for the Dashboard.
- **Path reset:** On new map generation, `astar_path` and `ann_path` are reset to `None`.

---

## 5. 🔌 API & BACKEND INTEGRATION

| Function | Module | Purpose |
|---|---|---|
| `generate_2d_map(size, density)` | `environment/map_generator.py` | Creates obstacle grid |
| `generate_3d_terrain(size, terrain_type)` | `environment/terrain_generator.py` | Creates height map |
| `get_valid_start_goal(grid)` | `environment/map_generator.py` | Finds valid start/goal positions |
| `plot_2d_grid(grid, start, goal, terrain=)` | `plots/plot_2d.py` | Renders 2D visualization |
| `plot_3d_terrain(terrain, start, goal)` | `plots/plot_3d.py` | Renders 3D Plotly surface |

---

## 6. 🧠 STATE MANAGEMENT
- **Reads:** `global_phase`, `grid`, `terrain`, `start`, `goal`
- **Writes:** `grid`, `terrain`, `start`, `goal`, `astar_path` (reset), `ann_path` (reset), `maps_generated`, `global_phase`

---

## 7. ⚠️ ERROR HANDLING & EDGE CASES

| Scenario | Behavior |
|---|---|
| No grid generated yet | `st.info("Generate a map to see it here!")` |
| Phase 1 selected | 3D chart area shows `st.warning("Terrain is disabled in Phase 1.")` |
| Flat terrain selected | Phase radio disabled, forced to Phase 1 |

---

## 8–9. 🔐 AUTH & 📱 RESPONSIVENESS
- No authentication. Open access.
- `st.columns([1, 4])` provides 20%/80% split. Responsive stacking on mobile.

---

## 10. 🎯 UX IMPROVEMENTS
- **Success toast** on map generation: "New Environment Configured!"
- **Caption text** educating users about hills vs. steep terrain differences.
- **Terrain heatmap underlay** on 2D map for immediate topography awareness.

---

## 11. 🧪 TEST CASES

| # | Test | Expected |
|---|---|---|
| TC-01 | Generate map with defaults | 2D and 3D views render correctly |
| TC-02 | Select "flat" → generate | Phase forced to 1; 3D shows warning |
| TC-03 | Select "steep" Phase 2 → generate | Both 2D (with terrain colors) and 3D surface appear |
| TC-04 | Generate map → go to Training → return | Map persists in session state |
| TC-05 | Generate map twice | Previous paths are cleared; maps_generated increments |

---

## 12–14. 🔗 DEPENDENCIES / 🚀 FUTURE / 📌 NOTES
- **Dependencies:** `session_manager.py`, `map_generator.py`, `terrain_generator.py`, `plot_2d.py`, `plot_3d.py`
- **Future:** Add custom start/goal placement via click interaction, save/load map presets.
- **Notes:** `terrain_type == "mixed (flat, hills, steep)"` is mapped internally to `"mixed"` in `terrain_generator.py`.

---
---

# SCREEN 4: 🧠 Training (`ui/pages/training.py`)

---

## 1. 🧾 SCREEN OVERVIEW
- **Purpose:** Generates labeled datasets from A* optimal paths and trains the MLPClassifier neural network.
- **Why it exists:** Separates the ML pipeline from the simulation, allowing users to tune dataset size and inspect training results.
- **How users reach it:** Sidebar link "Training" or `http://localhost:8501/training`.
- **Business importance:** The core machine learning pipeline — the "brain" of the system.

---

## 2. 👤 USER FLOW & NAVIGATION
1. Adjust `Number of Maps` (10–200, default 50) and `Scenarios per Map` (5–50, default 10).
2. Select `Feature Set Phase` (1 or 2).
3. Click **"Build Dataset"** → spinner → success toast with row count.
4. Click **"Train MLP Neural Network"** → spinner → success toast with accuracy.
5. View class distribution bar chart and raw dataset table on the right.
6. Optionally download the dataset as CSV.

**Edge case:** Clicking "Train" before "Build Dataset" → `st.error("Missing Dataset!")`

---

## 3. 🎨 UI COMPONENT BREAKDOWN

| Component | Type | Description |
|---|---|---|
| Number of Maps | `st.slider` | 10–200, controls how many random grids are generated |
| Scenarios per Map | `st.slider` | 5–50, controls random start/goal pairs per grid |
| Feature Set Phase | `st.radio` | 1 (5 features) or 2 (8 features, includes slopes) |
| Build Dataset Button | `st.button` | Triggers `generate_dataset()` |
| Train MLP Button | `st.button` | Triggers `train_and_save_model()` |
| Class Distribution Chart | `st.bar_chart` | Shows Forward/Left/Right sample counts (orange bars) |
| Raw Dataset Table | `st.dataframe` | Scrollable table of all features + labeled action |
| Download Button | `st.download_button` | Exports dataset as CSV file |

---

## 4. ⚙️ FUNCTIONAL LOGIC
- **Dataset generation:** For each map, A* is run from random start/goal pairs. At each step, features are extracted (`extract_features()`) and the A* action is labeled (`0`=Forward, `1`=Left, `2`=Right).
- **Feature sets:**
  - Phase 1 (5 features): `[Distance, Rel_Angle, Obs_F, Obs_L, Obs_R]`
  - Phase 2 (8 features): `[Distance, Rel_Angle, Slope_F, Slope_L, Slope_R, Obs_F, Obs_L, Obs_R]`
- **Training:** `MLPClassifier(hidden_layer_sizes=(64, 64, 32), activation='relu', solver='adam', learning_rate='adaptive', max_iter=2000)` with 15% validation split.
- **Model persistence:** Saved to disk as `ann_model_phase_{phase}.pkl` and stored in `st.session_state.model`.

---

## 5. 🔌 API & BACKEND INTEGRATION

| Function | Module | Returns |
|---|---|---|
| `generate_dataset(phase, num_maps, scenarios_per_map)` | `dataset/dataset_generator.py` | `X (ndarray), y (list), paths` |
| `train_and_save_model(X, y, phase)` | `model/train_ann.py` | `model (MLPClassifier), score (float), path (str)` |

---

## 6. 🧠 STATE MANAGEMENT
- **Writes:** `temp_X`, `temp_y`, `dataset_rows`, `model`, `last_accuracy`, `global_phase`
- **Reads:** `temp_X`, `temp_y`, `global_phase`
- `temp_X` and `temp_y` are session-scoped temporary variables holding the latest dataset.

---

## 7. ⚠️ ERROR HANDLING

| Scenario | Behavior |
|---|---|
| Train clicked without dataset | `st.error("Missing Dataset! Please build the dataset above first.")` |
| Dataset generation takes long | Spinner with "Extracting features and labeling..." |
| Training convergence warning | Handled internally by scikit-learn (increase `max_iter` if needed) |

---

## 8–9. 🔐 AUTH & 📱 RESPONSIVENESS
- No authentication. `st.columns([1, 2])` layout.

---

## 10. 🎯 UX IMPROVEMENTS
- **Spinner animations** clearly indicate long-running processes.
- **Bar chart** immediately reveals class imbalance (e.g., too many Forward vs. Left/Right).
- **CSV download** allows external analysis in Excel or Jupyter notebooks.

---

## 11. 🧪 TEST CASES

| # | Test | Expected |
|---|---|---|
| TC-01 | Build dataset with 10 maps, 5 scenarios, Phase 1 | Success; row count displayed; chart shows 3 bars |
| TC-02 | Train MLP after building dataset | Success; accuracy > 60%; model stored |
| TC-03 | Click Train without Build | Error: "Missing Dataset!" |
| TC-04 | Download CSV | Valid CSV file downloads with correct column headers |
| TC-05 | Build Phase 2 dataset | 8 feature columns appear in the table |
| TC-06 | Build dataset with 200 maps, 50 scenarios | Large dataset created (~200K rows); may take 30–60 seconds |

---

## 12–14. 🔗 DEPENDENCIES / 🚀 FUTURE / 📌 NOTES
- **Dependencies:** `dataset_generator.py`, `train_ann.py`, `session_manager.py`
- **Future:** Add hyperparameter tuning UI, show confusion matrix, implement cross-validation.
- **Notes:** Phase selection here is separate from the sidebar phase — users must ensure consistency.

---
---

# SCREEN 5: 🚀 Simulation (`ui/pages/simulation.py`)

---

## 1. 🧾 SCREEN OVERVIEW
- **Purpose:** The live execution engine where A* and ANN paths are computed and visualized with step-by-step playback.
- **Why it exists:** Provides a detailed, interactive exploration of how the robot physically navigates the terrain.
- **How users reach it:** Sidebar link "Simulation" or `http://localhost:8501/simulation`.
- **Business importance:** The primary demonstration screen showing the robot "in action."

---

## 2. 👤 USER FLOW & NAVIGATION
1. Ensure a map exists (otherwise error message displayed).
2. Click **"Simulate A* (Teacher)"** to compute the optimal path.
3. Click **"Simulate ANN (Student)"** to compute the neural network path.
4. Use the **Timeline Scrubber** slider to step through the path frame-by-frame.
5. Toggle between **2D Overhead** and **3D Topographic** tabs.
6. Monitor the **Sensor HUD** on the right for real-time feature readings.

---

## 3. 🎨 UI COMPONENT BREAKDOWN

| Component | Type | Description |
|---|---|---|
| Simulate A* Button | `st.button` | Computes teacher path |
| Simulate ANN Button | `st.button` | Computes student path (requires trained model) |
| Timeline Scrubber | `st.slider` | Range 1 to `max(len(a*), len(ann))`; controls which steps are rendered |
| 2D Overhead Tab | `st.pyplot` | Matplotlib grid with partial paths up to current scrubber step |
| 3D Topographic Tab | `st.plotly_chart` | Plotly 3D surface with partial paths |
| Sensor HUD | `st.json` | Displays extracted features at the current position |
| Position Display | `st.write` | Shows `(row, col)` coordinate and heading |
| Scrub Status | `st.info` / `st.success` | "Scrub right to advance time" or "Target Reached" |

---

## 4. ⚙️ FUNCTIONAL LOGIC
- **Partial path rendering:** `render_a = astar[:step]` slices the path up to the scrubber position, enabling frame-by-frame animation.
- **Sensor HUD:** At each step, `extract_features()` is called on the current robot state to show exactly what the ANN "sees."
- **Phase 1 features:** Distance, Angle, Obs_F, Obs_L, Obs_R
- **Phase 2 features:** Distance, Angle, Slope_F, Slope_L, Slope_R, Obs_F, Obs_L, Obs_R
- **Model validation:** Checks `model.n_features_in_` matches expected feature count before execution.
- **ANN execution:** Uses `run_ann_navigation()` with hysteresis bias to suppress zig-zagging.

---

## 5–6. 🔌 BACKEND & 🧠 STATE

| Function | Purpose |
|---|---|
| `astar_path()` | Compute optimal path |
| `run_ann_navigation()` | Execute ANN inference loop |
| `extract_features()` | Compute sensor readings for HUD |
| `plot_2d_grid()` / `plot_3d_terrain()` | Render visualizations |

- **Reads:** `grid`, `terrain`, `start`, `goal`, `model`, `global_phase`, `astar_path`, `ann_path`
- **Writes:** `astar_path`, `ann_path`

---

## 7. ⚠️ ERROR HANDLING

| Scenario | Behavior |
|---|---|
| No map generated | `st.error("No environment located in memory. Go to Map Setup to generate one!")` |
| No model trained | `st.error("Model uninitialized.")` |
| Phase mismatch | `st.error("Model mismatch! Retrain under Phase X.")` |
| ANN crashes | `try/except` catches any exception |
| A* finds no route | `st.error("No valid route.")` |

---

## 8–10. 🔐 AUTH / 📱 RESPONSIVE / 🎯 UX
- No auth. Layout: `st.columns([1,1,2])` for buttons, `st.columns([3,1])` for map + HUD.
- **Timeline scrubber** provides intuitive step-through playback.
- **Sensor HUD** with JSON formatting makes feature values immediately readable.

---

## 11. 🧪 TEST CASES

| # | Test | Expected |
|---|---|---|
| TC-01 | Open Simulation without map | Error message displayed |
| TC-02 | Click Simulate A* with valid map | Path computed; scrubber appears |
| TC-03 | Scrub slider from 1 to max | 2D map progressively reveals path |
| TC-04 | Open 3D tab with Phase 2 | Interactive 3D surface with path traces |
| TC-05 | Check Sensor HUD at step 5 | JSON shows 5 or 8 feature values |
| TC-06 | Run ANN with Phase 1 model on Phase 2 map | Error: "Model mismatch!" |

---

## 12–14. 🔗 DEPENDENCIES / 🚀 FUTURE / 📌 NOTES
- **Dependencies:** `astar.py`, `run_navigation.py`, `dataset_generator.py`, `plot_2d.py`, `plot_3d.py`
- **Future:** Add auto-play animation with `st.empty()` re-rendering, add path export to GIF.
- **Notes:** `current_state` is a tuple `(y, x, heading)` where heading is an integer index.

---
---

# SCREEN 6: 📈 Analytics (`ui/pages/analytics.py`)

---

## 1. 🧾 SCREEN OVERVIEW
- **Purpose:** Deep diagnostic dashboard for inspecting the trained neural network's architecture, loss curve, and the physical energy landscape.
- **Why it exists:** Professors need to verify the ANN's internal structure and validate that training converged properly.
- **How users reach it:** Sidebar link "Analytics" or `http://localhost:8501/analytics`.
- **Business importance:** Provides the academic proof that the neural network is properly structured and trained.

---

## 2. 👤 USER FLOW & NAVIGATION
1. User navigates to Analytics after training a model.
2. Left column displays the neural network architecture metrics.
3. Right column displays the energy/traversability heatmap.
4. User inspects loss curve to verify convergence.

---

## 3. 🎨 UI COMPONENT BREAKDOWN

| Component | Type | Description |
|---|---|---|
| Input Nodes Metric | `st.metric` | Shows `model.n_features_in_` (5 or 8) |
| Hidden Layers Metric | `st.metric` | Shows `model.hidden_layer_sizes` e.g. `(64, 64, 32)` |
| Output Nodes Metric | `st.metric` | Shows `model.n_outputs_` (always 3) |
| Total Layers Metric | `st.metric` | Shows `model.n_layers_` (5 for our architecture) |
| Convergence Epochs | `st.metric` | Shows `model.n_iter_` (actual iterations used) |
| Loss Curve | `st.line_chart` | Plots `model.loss_curve_` over epochs |
| Output Categories | `st.markdown` | Lists `[0, 1, 2]` = Forward, Left, Right |
| Solver/Activation | `st.markdown` | Shows `adam` solver, `relu` activation |
| Traversability Heatmap | `st.plotly_chart` (Plotly Heatmap) | Red-scale gradient showing slope magnitude + obstacles |
| Phase 1 Obstacle Map | `st.plotly_chart` (Plotly Heatmap) | Grayscale heatmap of obstacle grid |

---

## 4. ⚙️ FUNCTIONAL LOGIC
- **Network topology:** Extracted directly from the live `MLPClassifier` object in memory.
- **Traversability heatmap (Phase 2):**
  - Computes `dy, dx = np.gradient(terrain)` to get slope gradients.
  - Calculates `slope_magnitude = sqrt(dy² + dx²)`.
  - Overlays obstacle positions by amplifying their value to `2× max_slope` (or `5.0` if terrain is flat).
  - Red = high energy cost; white = low energy cost.
- **Phase 1 fallback:** Shows a simple grayscale obstacle configuration map.

---

## 5–6. 🔌 BACKEND & 🧠 STATE
- **Reads:** `model`, `grid`, `terrain`, `global_phase`
- **Writes:** None (read-only screen)
- Uses `np.gradient()` for slope computation and `go.Heatmap()` for Plotly rendering.

---

## 7. ⚠️ ERROR HANDLING

| Scenario | Behavior |
|---|---|
| No model loaded | `st.warning("No Neural Network loaded in memory.")` |
| No grid generated | `st.info("Generate a Map to visualize the underlying physical Energy Equations.")` |
| Model lacks `loss_curve_` | The loss chart section is conditionally skipped via `hasattr()` |

---

## 8–10. 🔐 AUTH / 📱 RESPONSIVE / 🎯 UX
- No auth. `st.columns(2)` for 50/50 equal split layout.
- **Loss curve caption** ("Must trend downwards") provides immediate interpretability guidance.
- **Heatmap caption** explains the physical meaning: "Sharp red pixels trigger massive `SLOPE_COST` penalty equations."

---

## 11. 🧪 TEST CASES

| # | Test | Expected |
|---|---|---|
| TC-01 | Open Analytics with trained model | All 5 metrics display correct values |
| TC-02 | Verify hidden layers shows `(64, 64, 32)` | Metric reads `(64, 64, 32)` |
| TC-03 | Check loss curve trends downward | Line chart decreases from left to right |
| TC-04 | Open Analytics with Phase 2 map | Red traversability heatmap appears |
| TC-05 | Open Analytics with Phase 1 map | Grayscale obstacle map appears |
| TC-06 | Open Analytics without model | Warning: "No Neural Network loaded" |

---

## 12–14. 🔗 DEPENDENCIES / 🚀 FUTURE / 📌 NOTES
- **Dependencies:** `session_manager.py`, `plotly`, `numpy`
- **Future:** Add weight visualization, gradient flow diagrams, confusion matrix.
- **Notes:** `n_layers_` includes input + all hidden + output layers.

---
---

# SCREEN 7: 📊 Comparison (`ui/pages/comparison.py`)

---

## 1. 🧾 SCREEN OVERVIEW
- **Purpose:** The definitive evaluation module that scientifically contrasts A* (optimal heuristic) against ANN (learned AI) on three cost dimensions.
- **Why it exists:** Provides the professor with undeniable numerical and visual evidence that the system correctly optimizes for energy, not just distance.
- **How users reach it:** Sidebar link "Comparison" or `http://localhost:8501/comparison`.
- **Business importance:** This is the screen the professor checks during final evaluation. It must be clear and conclusive.

---

## 2. 👤 USER FLOW & NAVIGATION
1. User runs both A* and ANN simulations on another page.
2. User navigates to Comparison.
3. Left column: Cost metrics table with conditional highlighting.
4. Right column: Three individual bar charts (Path Length, Energy, Turns).
5. Bottom: Automated academic conclusion text panel.

**Edge case:** No paths computed → `st.info("Run simulations to see the comparative overhead!")`

---

## 3. 🎨 UI COMPONENT BREAKDOWN

| Component | Type | Description |
|---|---|---|
| Cost Metrics Table | `st.dataframe` with `.style.highlight_min()` | Table of {Alg, Energy, Length, Turns, Status}. Minimum values highlighted in light green. |
| Path Length Bar Chart | `st.bar_chart` | Isolated chart comparing A* vs ANN path length (orange `#ffaa00`) |
| Total Energy Bar Chart | `st.bar_chart` | Isolated chart comparing A* vs ANN energy (teal `#00ffaa`) |
| Total Turns Bar Chart | `st.bar_chart` | Isolated chart comparing A* vs ANN turns (purple `#aa00ff`) |
| Academic Conclusion | `st.info` / `st.success` | Dynamically generated text analyzing the mathematical relationship between length and energy |

---

## 4. ⚙️ FUNCTIONAL LOGIC
- **Metrics computation:** `compute_metrics(path, status, terrain, phase)` calculates:
  - `total_energy`: Sum of forward (1.0) + turn (5.0) + slope (10.0×Δh) costs
  - `path_length`: Number of cells traversed
  - `turns`: Count of heading changes
- **ANN status detection:** Checks if `ann_path[-1][:2] == goal` to determine success/failure.
- **Conditional highlighting:** `df.style.highlight_min(color='lightgreen')` visually marks the winning algorithm per metric.
- **Academic conclusion logic:**
  - If `l_a > l_n AND e_a < e_n`: Prints a "Mathematical Proof" explaining that A* traded distance for energy savings.
  - Otherwise: Prints "Objective Achieved" showing the ANN successfully learned from A*.

---

## 5–6. 🔌 BACKEND & 🧠 STATE

| Function | Purpose |
|---|---|
| `compute_metrics(path, status, terrain, phase)` | Calculates energy, length, turns |

- **Reads:** `astar_path`, `ann_path`, `terrain`, `global_phase`, `goal`
- **Writes:** None (read-only screen)

---

## 7. ⚠️ ERROR HANDLING

| Scenario | Behavior |
|---|---|
| No paths computed | `st.info("Run simulations to see the comparative overhead!")` |
| Only one path exists | Table shows only one row; conclusion logic adjusts |
| ANN failed to reach goal | Status column shows "failed"; metrics still computed for partial path |

---

## 8–10. 🔐 AUTH / 📱 RESPONSIVE / 🎯 UX
- No auth. `st.columns([1, 2])` for table/charts split; `st.columns(3)` for the three bar charts.
- **Color-coded bars** make each metric instantly distinguishable.
- **Auto-generated conclusion** saves evaluators from interpreting the numbers themselves.

---

## 11. 🧪 TEST CASES

| # | Test | Expected |
|---|---|---|
| TC-01 | Open Comparison with no paths | Info: "Run simulations..." |
| TC-02 | Run A* only, open Comparison | Table shows 1 row (A* only) |
| TC-03 | Run both A* and ANN | Table shows 2 rows; 3 bar charts render; conclusion block appears |
| TC-04 | A* longer path but less energy | Blue `st.info` with "Mathematical Proof" text |
| TC-05 | ANN matches A* energy closely | Green `st.success` with "Objective Achieved" text |
| TC-06 | Verify highlight_min works | Minimum Energy, Length, and Turns cells are light green |
| TC-07 | ANN fails to reach goal | Status shows "failed"; metrics computed on partial path |

---

## 12. 🔗 DEPENDENCIES
- `evaluation/metrics.py` for `compute_metrics()`
- `session_manager.py` for state access
- `pandas` for DataFrame construction and styling

---

## 13. 🚀 FUTURE ENHANCEMENTS
- Add percentage improvement metrics (e.g., "ANN used 12% less energy than A*").
- Add multi-run averaging to smooth out variance.
- Export comparison report as PDF.

---

## 14. 📌 DEVELOPER NOTES
- The conclusion logic has exactly two branches. If more nuanced analysis is needed, additional `elif` blocks can be added.
- `ann_path[-1][:2]` extracts `(y, x)` from the `(y, x, heading)` tuple for goal comparison.
- `highlight_min` applies across `axis=0` (column-wise) to mark the best algorithm per metric.

---
---

# END OF DOCUMENTATION

> **Total Screens Documented:** 7
> **Total Sections per Screen:** 14
> **Document Version:** 2.0
> **Last Updated:** 2026-03-26
