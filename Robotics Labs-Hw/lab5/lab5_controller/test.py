import numpy as np
import matplotlib.pyplot as plt

def world_to_map(world_coords):
    """
    Convert world coordinates (in meters) to map indices (0 to 359).
    Assumes:
      - World x is in [-12, 0] where 0 maps to 359 and -12 maps to 0.
      - World y is in [-12, 0] where 0 maps to 0 and -12 maps to 359.
    """
    wx, wy = world_coords
    map_x = int((wx + 12) / 12 * 359)
    map_y = int((0 - wy) / 12 * 359)
    return (map_x, map_y)

# Load configuration space (which is in map coordinates) and the path (in world coordinates)
config_space = np.load("config_space.npy")
world_path = np.load("path.npy", allow_pickle=True)

# Display the configuration space
plt.figure(figsize=(6, 6))
# Use origin='upper' because our map array is indexed with (0,0) at the top left.
plt.imshow(config_space, cmap='gray', origin='upper')

# If there is a planned path, convert each waypoint to map coordinates for display.
if len(world_path) > 0:
    map_path = [world_to_map(pt) for pt in world_path]
    map_path_x, map_path_y = zip(*map_path)
    plt.plot(map_path_x, map_path_y, color='red', linewidth=2,
             marker='o', markersize=2, label="Planned Path")

plt.legend()
plt.title("Path Over Configuration Space (Converted from World Coordinates)")
plt.colorbar(label="Occupancy (0 = Free, 1 = Obstacle)")
plt.show()
