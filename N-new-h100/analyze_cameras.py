import json
import numpy as np
import os

# --- IMPORTANT: Set this to the correct path for your dataset ---
# Make sure this points to the JSON file you are using for training.
path_to_json = './lego/transforms_train.json'
# -------------------------------------------------------------

def analyze_transforms(json_path):
    """Loads a NeRF JSON file and prints a summary of camera positions."""
    
    if not os.path.exists(json_path):
        # Fallback for datasets that use different naming
        fallback_path = os.path.join(os.path.dirname(json_path), 'transforms.json')
        if os.path.exists(fallback_path):
            print(f"'{json_path}' not found. Using '{fallback_path}' instead.")
            json_path = fallback_path
        else:
            print(f"FATAL: Could not find JSON file at '{json_path}' or '{fallback_path}'.")
            return

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return

    if 'frames' not in data:
        print("Error: 'frames' key not in JSON file.")
        return

    origins = []
    print("--- Analyzing Camera Origins ---")
    for frame in data['frames']:
        if 'transform_matrix' in frame:
            c2w = np.array(frame['transform_matrix'])
            # The origin is the first 3 elements of the last column
            origin = c2w[:3, 3]
            origins.append(origin)
    
    if not origins:
        print("No camera poses found in the JSON file.")
        return

    origins = np.array(origins)
    min_bounds = np.min(origins, axis=0)
    max_bounds = np.max(origins, axis=0)

    print(f"\n--- Analysis Summary ---")
    print(f"Found {len(origins)} camera poses.")
    print(f"Min camera coordinates (X,Y,Z): ({min_bounds[0]:.2f}, {min_bounds[1]:.2f}, {min_bounds[2]:.2f})")
    print(f"Max camera coordinates (X,Y,Z): ({max_bounds[0]:.2f}, {max_bounds[1]:.2f}, {max_bounds[2]:.2f})")
    print("--------------------------\n")


if __name__ == '__main__':
    analyze_transforms(path_to_json)