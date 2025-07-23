import numpy as np
import pandas as pd
import time
import os

def compute_travelled_distance(df):
    """Compute the travelled distance for each vehicle by summing Euclidean distances."""
    df = df.sort_values(by=["id", "time"])  # Ensure sorted by id and time
    df["dx"] = df.groupby("id")["xloc_kf"].diff()
    df["dy"] = df.groupby("id")["yloc_kf"].diff()
    df["segment_distance"] = np.sqrt(df["dx"]**2 + df["dy"]**2)
    df["segment_distance"].fillna(0, inplace=True)  # Fill NaN for first point of each vehicle
    df["travelled_distance"] = df.groupby("id")["segment_distance"].cumsum()
    df.drop(columns=["dx", "dy", "segment_distance"], inplace=True)  # Cleanup
    return df

def find_best_candidate(last_point, second_last_point, remaining_df):
    """Find the best candidate vehicle point based on the largest angle."""
    threshold = 1.0  # Initial distance threshold in meters

    # 🔹 Keep increasing threshold until candidates are found
    while True:
        # Find candidate points within threshold
        remaining_df["dist_to_last"] = np.sqrt(
            (remaining_df["xloc_kf"] - last_point["xloc_kf"])**2 +
            (remaining_df["yloc_kf"] - last_point["yloc_kf"])**2
        )
        candidates = remaining_df[remaining_df["dist_to_last"] <= threshold]

        if not candidates.empty:
            break  # Exit threshold-increasing loop once we have candidates

        # 🔹 Increase threshold if no points found and retry
        threshold += 0.5
        if threshold > 10:  # Prevent infinite loop
            print("OVER threshold")
            break
            return None  # No valid candidate found

    # Compute previous vector
    vector_prev = np.array([
        last_point["xloc_kf"] - second_last_point["xloc_kf"],
        last_point["yloc_kf"] - second_last_point["yloc_kf"]
    ])
    norm_prev = np.linalg.norm(vector_prev)

    best_theta = -1
    best_candidate = None

    # 🔹 Iterate through candidates and find the one with the maximum angle
    for _, cand in candidates.iterrows():
        vector_next = np.array([
            cand["xloc_kf"] - last_point["xloc_kf"],
            cand["yloc_kf"] - last_point["yloc_kf"]
        ])
        norm_next = np.linalg.norm(vector_next)

        if norm_prev * norm_next == 0:
            continue  # Avoid division by zero

        # Compute dot product and angle
        cos_theta = np.dot(vector_prev, vector_next) / (norm_prev * norm_next)
        theta = np.arccos(np.clip(cos_theta, -1, 1)) * 180 / np.pi

        # Select the candidate with the largest angle
        if theta > best_theta:
            best_theta = theta
            best_candidate = cand

    return best_candidate


def find_reference_vehicle(df):
    """Create a reference vehicle trajectory for each lane based on the new method."""
    reference_vehicles = {}  # Dictionary to store reference vehicle for each lane

    for run_index, run_df in df.groupby("run_index"):
        for lane in run_df["lane_kf"].unique():
            lane_df = run_df[run_df["lane_kf"] == lane].copy()
            if lane_df.empty:
                continue

            # Preserve original row indices and IDs
            lane_df["original_id"] = lane_df["id"]  # Store original vehicle IDs
            lane_df["row_index"] = lane_df.index  # Store original row indices

            # Step 1: Find starting point r_0 closest to (0,0)
            lane_df["dist_to_origin"] = np.sqrt(lane_df["xloc_kf"]**2 + lane_df["yloc_kf"]**2)
            r_0_idx = lane_df["dist_to_origin"].idxmin()
            r_0 = lane_df.loc[r_0_idx]
            vehicle_id = r_0["id"]
            ref_trajectory = lane_df[lane_df["id"] == vehicle_id].sort_values("time").copy()

            # Step 3: Iteratively find next vehicle points
            while True:
                last_point = ref_trajectory.iloc[-1]
                second_last_point = ref_trajectory.iloc[-2] if len(ref_trajectory) >= 2 else ref_trajectory.iloc[-1]

                # Remove vehicles already used in trajectory
                remaining_df = lane_df[~lane_df.index.isin(ref_trajectory.index)].copy()

                # 🔹 Ensure next vehicle is in front based on xloc_kf and yloc_kf
                remaining_df = remaining_df[
                    (remaining_df["xloc_kf"] > last_point["xloc_kf"]) &
                    (remaining_df["yloc_kf"] > last_point["yloc_kf"])
                ].copy()

                best_candidate = find_best_candidate(last_point, second_last_point, remaining_df)

                # 🔹 If no valid candidate is found, stop trajectory
                if best_candidate is None:
                    print("No valid next point found, stopping trajectory.")
                    break

                # 🔹 Append the full trajectory of the new vehicle
                new_vehicle_id = best_candidate["id"]
                new_vehicle_trajectory = lane_df[(lane_df["id"] == new_vehicle_id)].sort_values("time")
                new_vehicle_trajectory = new_vehicle_trajectory[new_vehicle_trajectory["time"] >= best_candidate["time"]]

                ref_trajectory = pd.concat([ref_trajectory, new_vehicle_trajectory], ignore_index=True)
                print(ref_trajectory)

                # Remove the processed vehicle from remaining_df
                remaining_df = remaining_df[remaining_df["id"] != new_vehicle_id]

            # Compute travelled_distance for the entire trajectory
            ref_vehicle_id = -int(f"{run_index}{int(lane)}")
            ref_trajectory["id"] = ref_vehicle_id

            # Compute cumulative Euclidean distance for the entire trajectory
            ref_trajectory["dx"] = ref_trajectory["xloc_kf"].diff()
            ref_trajectory["dy"] = ref_trajectory["yloc_kf"].diff()
            ref_trajectory["segment_distance"] = np.sqrt(ref_trajectory["dx"]**2 + ref_trajectory["dy"]**2)
            ref_trajectory["segment_distance"].fillna(0, inplace=True)  # First point is 0
            ref_trajectory["travelled_distance"] = ref_trajectory["segment_distance"].cumsum()
            ref_trajectory.drop(columns=["dx", "dy", "segment_distance"], inplace=True)
            ref_trajectory.reset_index(drop=True, inplace=True)
            ref_trajectory["row_index"] = ref_trajectory.index  # Assign row number

            # Store in reference_vehicles dictionary
            reference_vehicles[lane] = (ref_vehicle_id, ref_trajectory)
            ref_trajectory.drop(columns=["dist_to_origin"], inplace=True)  # Cleanup

            # Save to CSV with original_id and row_index
            ref_trajectory.to_csv(f"reference_trajectory_run_{run_index}_lane_{lane}.csv", index=False)

    print("DONE")
    return reference_vehicles  # Dictionary {lane_kf: (reference_vehicle_id, reference_trajectory)}



    
def find_closest_reference_point(vehicle_df, reference_trajectory):
    """Find the closest reference trajectory point for ALL points of a surrounding vehicle."""
    min_distance = float("inf")
    best_vehicle_point = None
    best_ref_point = None

    # Empty list to record points > 1.5m
    record_data = []

    run_index = vehicle_df["run_index"].iloc[0]  
    surrounding_vehicle_id = vehicle_df["id"].iloc[0]  

    # Find reference vehicle id
    reference_vehicle_id = reference_trajectory["id"].iloc[0]  

    for _, vehicle_point in vehicle_df.iterrows():
        v_x, v_y = vehicle_point["xloc_kf"], vehicle_point["yloc_kf"]

        for _, ref_point in reference_trajectory.iterrows():
            r_x, r_y = ref_point["xloc_kf"], ref_point["yloc_kf"]

            distance = np.sqrt((v_x - r_x) ** 2 + (v_y - r_y) ** 2)

            if distance < min_distance:
                min_distance = distance
                best_vehicle_point = vehicle_point  
                best_ref_point = ref_point  

    # Record rows have min_distance > 1.5 
    if min_distance > 1.5:
        record_data.append({
            "run_index": run_index,  # Record run_index
            "surrounding_vehicle_id": surrounding_vehicle_id,  # Record surrounding vehicle id
            "vehicle_xloc_kf": best_vehicle_point["xloc_kf"],
            "vehicle_yloc_kf": best_vehicle_point["yloc_kf"],
            "reference_vehicle_id": reference_vehicle_id,  # Record reference vehicle id
            "ref_xloc_kf": best_ref_point["xloc_kf"],
            "ref_yloc_kf": best_ref_point["yloc_kf"],
            "lane_kf": best_vehicle_point["lane_kf"],  # Record lane_kf
            "min_distance": min_distance
        })

    # Generate CSV 
    if record_data:
        df_record = pd.DataFrame(record_data)
        file_name = f"distance_exceed_1.5_run_{run_index}.csv"  

        # If there is no such file, create one; otherwise write on the existing ones
        if not os.path.exists(file_name):
            df_record.to_csv(file_name, mode='w', header=True, index=False)  # Headings
        else:
            df_record.to_csv(file_name, mode='a', header=False, index=False)  # Add data

    return best_vehicle_point, best_ref_point, min_distance

def assign_travelled_distance(df, reference_vehicles):
    """Assign travelled distance for all surrounding vehicles based on their lane reference vehicle."""
    # Group by id and lane
    for (vehicle_id, lane), vehicle_df in df.groupby(["id", "lane_kf"]):
        if lane not in reference_vehicles:
            continue  # Meaning no reference vehicle
        
        reference_vehicle_id, reference_trajectory = reference_vehicles[lane]
        
        if vehicle_id == reference_vehicle_id:
            continue  # Keep travelled distance of the ref vehicle unchanged
        
        if vehicle_df.empty:
            continue

        # Calculate travelled distance for every lane
        s_p, r_p, d_p = find_closest_reference_point(vehicle_df, reference_trajectory)

        # Find next reference point
        next_ref_points = reference_trajectory[reference_trajectory["row_index"] > r_p["row_index"]]

        if next_ref_points.empty:
            prev_ref_points = reference_trajectory[reference_trajectory["row_index"] < r_p["row_index"]]
            if prev_ref_points.empty:
                print("AAAAAA")
                continue  

            r_p_prev = prev_ref_points.iloc[-1]

            # Compute angle using r_p_prev
            vector_rp_prev_rp = np.array([r_p["xloc_kf"] - r_p_prev["xloc_kf"], 
                                          r_p["yloc_kf"] - r_p_prev["yloc_kf"]])
            vector_rp_sp = np.array([s_p["xloc_kf"] - r_p["xloc_kf"], 
                                     s_p["yloc_kf"] - r_p["yloc_kf"]])

            dot_product = np.dot(vector_rp_prev_rp, vector_rp_sp)
            norm_rp_prev_rp = np.linalg.norm(vector_rp_prev_rp)
            norm_rp_sp = np.linalg.norm(vector_rp_sp)

            cos_theta = dot_product / (norm_rp_prev_rp * norm_rp_sp) if norm_rp_prev_rp * norm_rp_sp != 0 else 0
            angle = np.arccos(np.clip(cos_theta, -1, 1)) * 180 / np.pi  

            if 0 <= angle < 90:
                df.loc[s_p.name, "travelled_distance"] = r_p["travelled_distance"] - d_p  
            else:
                df.loc[s_p.name, "travelled_distance"] = r_p["travelled_distance"] + d_p  
                
        else:
            r_p_next = next_ref_points.iloc[0]

            # Compute angle to determine front/back relation
            vector_rp_rp1 = np.array([r_p_next["xloc_kf"] - r_p["xloc_kf"], 
                                      r_p_next["yloc_kf"] - r_p["yloc_kf"]])
            vector_rp_sp = np.array([s_p["xloc_kf"] - r_p["xloc_kf"], 
                                     s_p["yloc_kf"] - r_p["yloc_kf"]])

            dot_product = np.dot(vector_rp_rp1, vector_rp_sp)
            norm_rp_rp1 = np.linalg.norm(vector_rp_rp1)
            norm_rp_sp = np.linalg.norm(vector_rp_sp)

            cos_theta = dot_product / (norm_rp_rp1 * norm_rp_sp) if norm_rp_rp1 * norm_rp_sp != 0 else 0
            angle = np.arccos(np.clip(cos_theta, -1, 1)) * 180 / np.pi  

            if 0 <= angle < 90:
                df.loc[s_p.name, "travelled_distance"] = r_p["travelled_distance"] + d_p 
            else:
                df.loc[s_p.name, "travelled_distance"] = r_p["travelled_distance"] - d_p  

        # Propagate travelled distance
        prev_points = vehicle_df[vehicle_df["time"] <= s_p["time"]].sort_values("time", ascending=False)
        prev_travelled_distance = df.loc[s_p.name, "travelled_distance"]  

        prev_points_indices = prev_points.index.to_list()
        for i in range(len(prev_points_indices) - 1):
            idx = prev_points_indices[i]
            idx_next = prev_points_indices[i + 1]

            p1 = df.loc[idx]
            p2 = df.loc[idx_next]  

            distance = np.sqrt((p1["xloc_kf"] - p2["xloc_kf"])**2 +
                               (p1["yloc_kf"] - p2["yloc_kf"])**2) 

            prev_travelled_distance -= distance  
            df.loc[idx_next, "travelled_distance"] = prev_travelled_distance 

        next_points = vehicle_df[vehicle_df["time"] >= s_p["time"]].sort_values("time")
        next_travelled_distance = df.loc[s_p.name, "travelled_distance"]  

        next_points_indices = next_points.index.to_list()
        for i in range(len(next_points_indices) - 1):
            idx = next_points_indices[i]
            idx_next = next_points_indices[i + 1]

            p1 = df.loc[idx]
            p2 = df.loc[idx_next]  

            distance = np.sqrt((p1["xloc_kf"] - p2["xloc_kf"])**2 +
                               (p1["yloc_kf"] - p2["yloc_kf"])**2)  

            next_travelled_distance += distance  
            df.loc[idx_next, "travelled_distance"] = next_travelled_distance  

    return df

# Main Execution
time_in = time.time()

# Load Data
df = pd.read_csv("TGSIM_Full.csv")
#df = df[df['lane_kf']==1]
#df = pd.read_csv("test_224.csv")
# Group by run_index
processed_dfs = []  # to store processed data for every run_index

for run, sub_df in df.groupby("run_index"):  # run_index group
    print(f"Processing run_index: {run}")

    # Compute travelled distances for each vehicle
    sub_df = compute_travelled_distance(sub_df)

    # Find reference vehicles for each lane
    reference_vehicles = find_reference_vehicle(sub_df)

    # Assign travelled distances for surrounding vehicles
    sub_df = assign_travelled_distance(sub_df, reference_vehicles)

    processed_dfs.append(sub_df)  # save sub_df

# Concat all sub_df
df = pd.concat(processed_dfs, ignore_index=True)

# Save processed data
df.to_csv("processed_data_full.csv", index=False)
