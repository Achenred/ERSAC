import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import re


def plot_saved_images_in_grid(plots_dir, taus, betas, output_dir):
    """
    Load and display the saved plots in a grid, with tau on the y-axis and beta on the x-axis.

    Args:
        plots_dir (str): Directory containing the saved plot images.
        taus (list): List of tau values.
        betas (list): List of beta values.
        output_dir (str): Directory to save the combined grid plot.
    """
    # Get all saved plot filenames
    plot_filenames = os.listdir(plots_dir)

    # Create a dictionary to map (tau, beta) to plot filenames
    plot_map = {}

    # Regular expression to extract tau and beta values from the filename
    filename_pattern = r'tau([0-9\.]+)_beta([0-9\.]+)\.png'

    for filename in plot_filenames:
        # Match the filename with the regex pattern
        match = re.search(filename_pattern, filename)
        if match:
            tau_val = float(match.group(1))  # Extract tau value
            beta_val = float(match.group(2))  # Extract beta value

            # Add the plot to the dictionary if the tau and beta values are in the given lists
            if tau_val in taus and beta_val in betas:
                plot_map[(tau_val, beta_val)] = filename
        else:
            print(f"Skipping invalid filename format: {filename}")

    # Create the subplots grid
    fig, axes = plt.subplots(len(taus), len(betas), figsize=(18, 12), sharex=True, sharey=True)

    # Iterate over the grid of taus and betas and plot the corresponding images
    for idx_tau, tau in enumerate(taus):
        for idx_beta, beta in enumerate(betas):
            ax = axes[idx_tau, idx_beta]

            # Check if a plot exists for this (tau, beta) pair
            if (tau, beta) in plot_map:
                plot_path = os.path.join(plots_dir, plot_map[(tau, beta)])
                img = mpimg.imread(plot_path)
                ax.imshow(img)
                ax.axis('off')  # Turn off the axis for cleaner presentation
                ax.set_title(f"Tau: {tau}, Beta: {beta}", fontsize=8)
            else:
                ax.axis('off')  # If no plot is available for this (tau, beta), leave the axis off

    # Adjust layout to prevent overlap
    plt.tight_layout(pad=0.5)  # Increase padding between plots
    plt.subplots_adjust(hspace=0.1, wspace=0.1)  # Increase horizontal and vertical spacing

    # Save the figure with high resolution as a PDF
    filename = os.path.join(output_dir, "combined_action_probabilities_grid.pdf")
    plt.savefig(filename, format='pdf', dpi=300)  # Set dpi for higher resolution
    plt.close()  # Close the plot to free up memory


# Example tau and beta values
taus =  [0.1,0.3,0.5,0.7,0.9] #[0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
betas = [0, 0.001, 0.01, 0.1]

# Specify the directory where the plots are saved and the output directory for the combined plot
plots_dir = "/Users/abhilashchenreddy/PycharmProjects/CQL_AC/tabular/plots/riverswim"
output_dir = plots_dir

# Call the function to load and display the plots in a grid
# plot_saved_images_in_grid(plots_dir, taus, betas, output_dir)


import numpy as np

states = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
samples = [np.random.choice(states) for _ in range(1000)]
print("Sample distribution:", np.unique(samples, return_counts=True))