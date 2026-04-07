import pandas as pd
import matplotlib.pyplot as plt
import os

def load_entropy_data(file_dict):
    """
    Load all CSVs into a dictionary of DataFrames.
    Assumes each file has columns: Step, entropy, entropy__MIN, entropy__MAX
    """
    dataframes = {}
    for label, filepath in file_dict.items():
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            dataframes[label] = df
        else:
            print(f"Warning: File not found - {filepath}")
    return dataframes

def plot_entropy_curves(dataframes, name, color_dict=None, title="Entropy over Training Steps", save_path=None):
    """
    dataframes: dict with {label: DataFrame}
    color_dict: optional dict with {label: color}
    save_path: optional path to save the figure (e.g., "entropy_plot.png")
    """
    plt.figure(figsize=(10, 6))

    for label, df in dataframes.items():
        steps = df['Step']
        mean = df[df.columns[1]]
        min_ = df[df.columns[2]]
        max_ = df[df.columns[3]]

        color = color_dict[label] if color_dict and label in color_dict else None

        plt.plot(steps, mean, label=label, color=color, linewidth=1.5)
        plt.fill_between(steps, min_, max_, color=color, alpha=0.2)

    plt.xlabel("Epochs")
    plt.ylabel("Entropy")
    # plt.title(title)
    # plt.legend()
    # plt.grid(True)
    plt.tight_layout()

    plt.savefig(f'{name}_entropy_plot.png', dpi=600, bbox_inches='tight', pad_inches=0.05) #, transparent=True)

    print(f"Saved plot to {save_path}")


# ==== Example usage ====
env_name = 'Cartpole'

file_dict = {
    "Box": "/Users/abhilashchenreddy/PycharmProjects/CQL_AC/Robust SAC-N/visualiaztion_data/Cartpole/box.csv",
    "Convex Hull": "/Users/abhilashchenreddy/PycharmProjects/CQL_AC/Robust SAC-N/visualiaztion_data/Cartpole/hull.csv",
    "Ellipsoid": "/Users/abhilashchenreddy/PycharmProjects/CQL_AC/Robust SAC-N/visualiaztion_data/Cartpole/ellipsoid.csv",
    "Epinet": "/Users/abhilashchenreddy/PycharmProjects/CQL_AC/Robust SAC-N/visualiaztion_data/Cartpole/epinet.csv"
}

color_dict = {
    "Box": (0.16, 0.67, 0.53),
    "Convex Hull": (0.19, 0.55, 0.91),
    "Ellipsoid": (0.8, 0.25, 0.33),
    "Epinet": "pink"
}

save_path = '/Users/abhilashchenreddy/PycharmProjects/CQL_AC/Robust SAC-N/visualiaztion_data/entropy_plot.png'

dataframes = load_entropy_data(file_dict)
plot_entropy_curves(dataframes,env_name, color_dict, save_path)  # Set save_path="entropy_plot.png" to save
