# Distance_Traveled_Algorithm
# TGSIM Trajectory-Based Distance Estimation

This repository implements a custom algorithm to compute lane-aligned traveled distances for all vehicles in the TGSIM Stationary Trajectory Dataset. The approach constructs a lane-specific reference trajectory by chaining vehicle segments and then projects other vehicles onto the reference to estimate their traveled distances. This method overcomes the limitations of standard Euclidean accumulation, particularly on curved roads.

---

## 📁 Input

- `TGSIM_Full.csv`: Main vehicle trajectory dataset with the following required columns:
  - `id`: Vehicle ID
  - `time`: Time stamp
  - `xloc_kf`: X-coordinate in local map frame
  - `yloc_kf`: Y-coordinate in local map frame
  - `run_index`: Experiment run ID
  - `lane_kf`: Lane index of the vehicle

---

## ⚙️ How It Works

### 1. `compute_travelled_distance(df)`
Computes per-vehicle cumulative Euclidean distance using `(xloc_kf, yloc_kf)`, grouped by `id`.

### 2. `find_reference_vehicle(df)`
For each `run_index` and `lane_kf`, this function constructs a **reference trajectory** by:
- Selecting the vehicle point closest to origin `(0,0)` as the seed.
- Iteratively appending the most "forward" vehicle segment with the **maximum turning angle** to ensure coherent chaining.
- Saving the synthetic reference vehicle (with ID `-run_index * 10 + lane_kf`) to CSV.

### 3. `find_closest_reference_point(vehicle_df, reference_trajectory)`
For every non-reference vehicle, finds the point closest to the reference trajectory (based on Euclidean distance). If the minimum distance > 1.5 m, logs the result in a CSV.

### 4. `assign_travelled_distance(df, reference_vehicles)`
For every vehicle that is not the reference:
- Finds its closest point on the reference trajectory.
- Computes directional angle to determine whether to add or subtract the distance.
- Propagates distance forward and backward in time using Euclidean distances between points.

---

## ✅ Output

- `processed_data_full.csv`: A fully updated dataset with a new column:
  - `travelled_distance`: Aligned distance along the lane-specific reference trajectory
- `reference_trajectory_run_{run}_lane_{lane}.csv`: Each lane’s reference trajectory
- `distance_exceed_1.5_run_{run}.csv`: (Optional) Logs of vehicle-reference distances exceeding 1.5m

---

## 🚀 Run Instructions

```bash
# Step 1: Prepare your CSV as TGSIM_Full.csv
# Step 2: Run the main script
python your_script_name.py
