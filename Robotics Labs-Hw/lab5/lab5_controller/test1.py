import numpy as np
import matplotlib.pyplot as plt

def main():
    # Load the map data from map.npy.
    try:
        map_data = np.load("map.npy")
    except FileNotFoundError:
        print("Error: 'map.npy' not found. Make sure it exists in the current directory.")
        return

    # Display the map. You can change 'origin' to 'lower' if needed.
    plt.imshow(map_data, cmap='gray', origin='upper')
    plt.title("Map Visualization")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.colorbar(label="Occupancy")
    plt.show()

if __name__ == "__main__":
    main()
