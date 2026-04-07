# import matplotlib.pyplot as plt
# import numpy as np
#
# # Function to create the individual boxplot
# def create_boxplot(ax, ticks, group_1_data, group_2_data, group_3_data, group_4_data):
#     y_positions = np.arange(len(ticks)) * 6.0  # Increased space between categories
#     group_offsets = [-0.75, 0.75]  # Offsets for the two groups
#
#     for i, y in enumerate(y_positions):
#         # First group of boxplots
#         ax.boxplot(group_1_data[i],
#                    vert=False,
#                    positions=[y + group_offsets[0] - 0.3],
#                    widths=0.2, patch_artist=True,
#                    boxprops=dict(facecolor='#D7191C', alpha=0.8))
#         ax.boxplot(group_2_data[i],
#                    vert=False,
#                    positions=[y + group_offsets[0] + 0.3],
#                    widths=0.2, patch_artist=True,
#                    boxprops=dict(facecolor='#D7191C', alpha=0.5))
#
#         # Second group of boxplots
#         ax.boxplot(group_3_data[i],
#                    vert=False,
#                    positions=[y + group_offsets[1] - 0.3],
#                    widths=0.2, patch_artist=True,
#                    boxprops=dict(facecolor='#2C7BB6', alpha=0.8))
#         ax.boxplot(group_4_data[i],
#                    vert=False,
#                    positions=[y + group_offsets[1] + 0.3],
#                    widths=0.2, patch_artist=True,
#                    boxprops=dict(facecolor='#2C7BB6', alpha=0.5))
#
#         # Calculate the maximum value across all groups in the category
#         max_y = max(
#             max(group_1_data[i]), max(group_2_data[i]),
#             max(group_3_data[i]), max(group_4_data[i])
#         )
#
#         # Add group labels at the same level for the category
#         ax.text(max_y + 2, y + group_offsets[0], 'a', ha='center', fontsize=10)
#         ax.text(max_y + 2, y + group_offsets[1], 'b', ha='center', fontsize=10)
#
#     # Add y-ticks and labels
#     ax.set_yticks(y_positions)
#     ax.set_yticklabels(ticks, rotation=0, fontsize=8)
#     ax.set_ylim(-2, len(ticks) * 6)
#     ax.set_xlim(0, 70)
#     # ax.set_title('Grouped Boxplots', fontsize=10)
#     ax.set_ylabel('Categories', fontsize=9)
#     ax.set_xlabel('Values', fontsize=9)
#
#
# # Prepare the data and ticks
# ticks = ['1', '2', '3', '4', '5','6', '7', '8', '9', '10']
#
# group_1_data = [
#     [10, 12, 11], [20, 22, 19], [15, 18, 16], [25, 28, 27], [30, 32, 29],
#     [35, 38, 36], [40, 42, 41], [45, 48, 46], [50, 52, 49], [55, 58, 56]
# ]
# group_2_data = [
#     [8, 9, 7], [18, 19, 17], [12, 13, 11], [22, 23, 21], [28, 29, 26],
#     [32, 33, 31], [38, 39, 37], [42, 43, 40], [48, 49, 47], [54, 55, 53]
# ]
# group_3_data = [
#     [14, 15, 13], [24, 25, 23], [20, 21, 19], [30, 31, 29], [34, 35, 32],
#     [40, 41, 38], [44, 45, 43], [50, 51, 48], [56, 57, 54], [60, 61, 58]
# ]
# group_4_data = [
#     [16, 17, 15], [26, 27, 24], [22, 23, 20], [32, 33, 30], [36, 37, 34],
#     [42, 43, 40], [46, 47, 44], [52, 53, 50], [58, 59, 56], [62, 63, 60]
# ]
#
# # Create a 3x3 grid of subplots with increased vertical size
# fig, axes = plt.subplots(3, 3, figsize=(20, 20))  # Increased height
#
# # Populate each subplot with the boxplot
# for ax in axes.flat:
#     create_boxplot(ax, ticks, group_1_data, group_2_data, group_3_data, group_4_data)
#
# plt.tight_layout()
# plt.show()


import pickle
import numpy as np
from itertools import combinations
from collections import defaultdict
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

policy_path = "/Users/abhilashchenreddy/PycharmProjects/CQL_AC/results/tabular_results_policies_20250128_061939.pkl"


    # "/Users/abhilashchenreddy/PycharmProjects/CQL_AC/results/tabular_results_policies_20250128_070725.pkl"


    # "/Users/abhilashchenreddy/PycharmProjects/CQL_AC/results/tabular_results_policies_20250128_061939.pkl"


    # "/Users/abhilashchenreddy/PycharmProjects/CQL_AC/results/tabular_results_policies_20250121_181751.pkl"

# Open and load the pickle file
with open(policy_path, 'rb') as file:
    policies = pickle.load(file)


import numpy as np
from itertools import combinations
from collections import defaultdict

import numpy as np
from itertools import combinations,product
from collections import defaultdict
import csv


# Define the target methods
target_methods = {'tabular_EDAC', 'tabular_robustEDAC', 'true_model', 'VI_model'}

# Group policies by (tau, beta, dsf, seed)
grouped_policies = defaultdict(list)
for policy in policies:
    key = (policy['risk_tau'], policy['beta'], policy['dsf'], policy['seed'])
    grouped_policies[key].append(policy)

# Placeholder for pairwise distances
distances = defaultdict(lambda: defaultdict(list))

# Compute pairwise distances for each (tau, beta, dsf, seed)
for (tau, beta, dsf, seed), entries in grouped_policies.items():
    # Extract policies for target methods
    methods = {entry['method_name']: entry['policy'] for entry in entries if entry['method_name'] in target_methods}

    # Ensure we have all 4 methods
    if set(methods.keys()) == target_methods:
        # Determine random policy size from one of the policies
        example_policy = next(iter(methods.values()))
        num_states, num_actions = example_policy.shape
        random_policy = np.full((num_states, num_actions), 1 / num_actions)

        # Generate all combinations of methods (including repeated distances)
        all_combinations = list(product(target_methods, repeat=2))

        # Compute pairwise distances for all combinations
        for method1, method2 in all_combinations:
            policy1 = methods[method1]
            policy2 = methods[method2]
            distance = np.linalg.norm(policy1 - policy2)  # Euclidean distance
            distances[(tau, beta, dsf)][(method1, method2)].append(distance)

        # Compute distances to the random policy
        for method, policy in methods.items():
            random_distance = np.linalg.norm(policy - random_policy)
            distances[(tau, beta, dsf)][(method, 'random_policy')].append(random_distance)

# Aggregate distances at (tau, beta, dsf) level
summary = []
for (tau, beta, dsf), method_data in distances.items():
    for (method1, method2), dist_list in method_data.items():
        summary.append({
            'tau': tau,
            'data_size': dsf,
            'beta': beta,
            'method1': method1,
            'method2': method2,
            'distances': dist_list,
        })


# Save the summary to a CSV file
output_csv_path = "aggregated_policy_distances.csv"
# df = pd.DataFrame(summary)
# df.to_csv(output_csv_path, index=False)

print(f"Results saved to {output_csv_path}")


df = pd.read_csv(output_csv_path)
df = df.sort_values(['tau', 'data_size', 'method1', 'method2']).reset_index(drop=True)
df['distances'] = df['distances'].apply(lambda x: sorted(x) if isinstance(x, list) else x)


# print(df[(df.tau == 0.1) & (df.data_size == 25)][['method1', 'method2']])
# import sys
# sys.exit()

# Adjust the data to simulate box plots using means and standard deviations for each combination
# Extract unique values for tau and data_size
taus = sorted(df['tau'].unique())
data_sizes = sorted(df['data_size'].unique())

# Create a grid of subplots
fig, axes = plt.subplots(len(data_sizes), len(taus), figsize=(12, 8), sharex=False, sharey=False)
fig.subplots_adjust(hspace=0.8, wspace=0.5)

# Define custom palette and x-tick labels
custom_palette = {
    'tabular_EDAC': 'cornflowerblue',
    'tabular_robustEDAC': 'indianred',
    'VI_model': 'black',
    'true_model': 'forestgreen',
    'random_policy': 'darkorange'
}
custom_xtick_labels = ['EDAC', 'Robust-EDAC', 'VI Model', 'Beh. Model', 'Random']
hue_order = ['tabular_EDAC', 'tabular_robustEDAC', 'VI_model', 'true_model', 'random_policy']


# Helper function to convert string lists to numeric lists
def to_numeric_list(cell):
    try:
        return [float(x) for x in cell.strip('[]').split(',')]
    except:
        return []


df['distances'] = df['distances'].apply(to_numeric_list)
global_min = 0
global_max = np.round(1 + max([max(d) for d in df['distances'] if d]), 2)  # Maximum y-axis value

custom_xtick_labels = ['EDAC', 'Robust-EDAC', 'VI Model', 'True Model']

# Create subplots
fig, axes = plt.subplots(len(data_sizes), len(taus), figsize=(12, 8), sharex=False, sharey=False)

for i, data_size in enumerate(data_sizes):
    for j, tau in enumerate(taus):
        # Filter data for the current combination of tau and data_size
        subset = df[(df['tau'] == tau) & (df['data_size'] == data_size)]
        subset = subset[['method1', 'method2', 'distances']]

        # Explode the 'distances' column
        subset['distances'] = subset['distances'].apply(lambda x: [float(d) for d in x] if isinstance(x, list) else [])
        data_exploded = subset.explode('distances')

        # Skip if no data
        if data_exploded.empty:
            continue

        data_exploded['distances'] = data_exploded['distances'].astype(float)

        # Filter and set categorical order for method1
        xtick_order = ['tabular_EDAC', 'tabular_robustEDAC', 'VI_model', 'true_model']
        hue_order = ['tabular_EDAC', 'tabular_robustEDAC', 'VI_model', 'true_model', 'random_policy']
        data_exploded = data_exploded[data_exploded['method1'].isin(xtick_order)]
        data_exploded['method1'] = pd.Categorical(data_exploded['method1'], categories=xtick_order, ordered=True)

        # Define the custom color palette
        custom_palette = {
            'tabular_EDAC': 'cornflowerblue',
            'tabular_robustEDAC': 'indianred',
            'VI_model': 'black',
            'true_model': 'forestgreen',
            'random_policy': 'darkorange'
        }

        # Create the box plot
        ax = axes[i, j]
        sns.boxplot(
            data=data_exploded,
            x='method1',
            y='distances',
            hue='method2',
            ax=ax,
            palette=custom_palette,
            hue_order=hue_order,
            showfliers=False,
            width=0.5,
            dodge=True,
            linewidth=0.2,
            medianprops={'color': 'none'}
        )

        # Set subplot title and labels
        ax.set_title(f'Tau: {tau}, Data Size: {data_size}')
        ax.set_xlabel('Model Groups' if i == len(data_sizes) - 1 else '')
        ax.set_ylabel('Distances' if j == 0 else '')
        ax.set_xticklabels(custom_xtick_labels, rotation=0, ha='center')

        # Set uniform y-axis limits and ticks
        ax.set_ylim(global_min, global_max)
        rounded_ticks = np.round(np.arange(global_min, global_max + 0.75, 0.75), 2)
        ax.set_yticks(rounded_ticks)

        # Add thin dotted horizontal lines
        for y in rounded_ticks:
            ax.axhline(y=y, color='gray', linestyle=':', linewidth=0.5)

        # Remove individual legends
        ax.get_legend().remove()

# Add a single legend at the bottom
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, custom_xtick_labels, loc='lower center', ncol=5)

# Final layout adjustments
plt.tight_layout(rect=[0, 0.1, 1, 0.95])
plt.show()


import sys
sys.exit()

# Create a folder for the plots
output_folder = "policy_distance_plots"
os.makedirs(output_folder, exist_ok=True)

# Filter the data to keep only rows where 'method2' is 'model2'
base_model = "tabular_robustEDAC"
df_filtered = df[df['method2'] == base_model]

models = df.method1.unique()
print(models)
import sys
sys.exit()

print(df_filtered.shape)

# Iterate through unique data sizes to create plots
for data_size in df_filtered['data_size'].unique():
    # Filter the data for the current data size
    data_subset = df_filtered[df_filtered['data_size'] == data_size]

    print(data_subset)

    # Create a new plot
    plt.figure(figsize=(10, 6))
    plt.title(f"Policy Distance for Data Size {data_size} (Base: {base_model})", fontsize=14)
    plt.xlabel("Tau", fontsize=12)
    plt.ylabel("Mean Distance", fontsize=12)

    # Plot lines for each method1
    for method1 in data_subset['method1'].unique():
        method_data = data_subset[data_subset['method1'] == method1]
        plt.plot(
            method_data['tau'],
            method_data['mean_distance'],
            label=f"{method1} vs {base_model}",
            marker='o'
        )

    # Add legend and grid
    plt.legend()
    plt.grid(alpha=0.4)

    # Save the plot
    plot_path = os.path.join(output_folder, f"policy_distance_data_size_{data_size}.png")
    plt.savefig(plot_path)
    plt.show()
    plt.close()

print(f"Plots saved in folder: {output_folder}")


import sys
sys.exit()


df = pd.read_csv("combined_data.csv")
df.columns = ["seed", "model", "tau", "df", "freq", "action"]

# Pivot the dataset as per the user's instructions
pivoted_data = df.pivot_table(
    index=["seed", "tau", "df", "freq"],
    columns="model",
    values="action"
).reset_index()


# Create the requested difference columns
pivoted_data["EDAC_behv"] = (
    pivoted_data["tabular_EDAC"] - pivoted_data["true_model"]
)
pivoted_data["EDAC_VI"] = (
    pivoted_data["tabular_EDAC"] - pivoted_data["VI_model"]
)
pivoted_data["RobustEDAC_behv"] = (
    pivoted_data["tabular_robustEDAC"] - pivoted_data["true_model"]
)
pivoted_data["RobustEDAC_VI"] = (
    pivoted_data["tabular_robustEDAC"] - pivoted_data["VI_model"]
)

# Display the resulting dataframe
print(pivoted_data.head(3).T)

import os
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd

def analyze_data_with_plots(df, group_cols, target_vars, independent_var):
    results = []

    # Create a folder for regression plots if it doesn't exist
    plots_folder = "regression_plots"
    os.makedirs(plots_folder, exist_ok=True)

    # Group the data by specified columns (e.g., tau and df, ignoring seed)
    grouped = df.groupby(group_cols)

    for group_name, group_data in grouped:
        # Initialize a plot for this group
        plt.figure(figsize=(10, 6))
        plt.title(f"Group: {group_name}", fontsize=14)
        plt.xlabel(independent_var, fontsize=12)
        plt.ylabel("Target Variables", fontsize=12)

        # For each group, fit regressions and compute sums for each target variable
        result = {
            "tau": group_name[0],
            "df": group_name[1]
        }

        for target_var in target_vars:
            # Fit regression model
            X = sm.add_constant(group_data[independent_var])  # Add constant for intercept
            y = group_data[target_var]
            model = sm.OLS(y, X).fit()

            # Extract p-value for the independent variable
            p_value = model.pvalues[independent_var]

            # Compute the sum of the target variable
            target_sum = y.sum()

            # Append p-value and sum for each target variable in the result dictionary
            result[f"{target_var}_p_value"] = p_value
            result[f"{target_var}_sum"] = target_sum

            # Plot data points and regression line
            plt.scatter(group_data[independent_var], y, label=f"{target_var} data points", alpha=0.6)
            plt.plot(group_data[independent_var], model.predict(X), label=f"{target_var} regression line", linewidth=2)

        # Add the result to the list
        results.append(result)

        # Add a legend and save the plot
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_folder, f"group_{group_name}.png"))
        plt.close()

    return pd.DataFrame(results)



# Define a function to fit regressions and calculate p-values and sums for target variables
def analyze_data(df, group_cols, target_vars, independent_var):
    results = []

    # Group the data by specified columns (e.g., tau and df, ignoring seed)
    grouped = df.groupby(group_cols)

    for group_name, group_data in grouped:
        # For each group, fit regressions and compute sums for each target variable
        result = {
            "tau": group_name[0],
            "df": group_name[1]
        }

        for target_var in target_vars:
            # Fit regression model
            X = sm.add_constant(group_data[independent_var])  # Add constant for intercept
            y = group_data[target_var]
            model = sm.OLS(y, X).fit()

            # Extract p-value for the independent variable
            p_value = model.pvalues[independent_var]

            # Compute the sum of the target variable
            target_sum = y.sum()

            # Append p-value and sum for each target variable in the result dictionary
            result[f"{target_var}_p_value"] = p_value
            result[f"{target_var}_sum"] = target_sum

        # Add the result to the list
        results.append(result)

    return pd.DataFrame(results)


# Define columns and variables for analysis
group_columns = ["tau", "df"]  # Group by tau and df only
target_variables = ["EDAC_behv", "EDAC_VI", "RobustEDAC_behv", "RobustEDAC_VI"]
independent_variable = "freq"

# Perform the analysis
analysis_results = analyze_data_with_plots(pivoted_data, group_columns, target_variables, independent_variable)

# Save the results to a CSV file
output_analysis_path = "regression_analysis_results_pivoted.csv"
analysis_results.to_csv(output_analysis_path, index=False)

print(f"Analysis results saved to {output_analysis_path}")
import sys
sys.exit()


# Read the CSV file
data = pd.read_csv("model.csv")

# Assuming the columns are named as follows:
# state_freq (independent variable), action_diff_model_1_true_model, action_diff_model_1_VI_model,
# action_diff_model_2_true_model, action_diff_model_2_VI_model

independent_variable = data["state"]  # First column
results = {}

# Loop through each of the last 4 columns and build regression
for column in data.columns[1:]:
    dependent_variable = data[column]

    # Add a constant for the regression model
    X = sm.add_constant(independent_variable)  # Add constant term (intercept)
    model = sm.OLS(dependent_variable, X)  # Ordinary Least Squares regression
    result = model.fit()  # Fit the model

    # Store the summary and p-value
    results[column] = {
        "coef": result.params,
        "p_values": result.pvalues,
        "summary": result.summary().as_text()
    }

# Print p-values for each regression
for column, output in results.items():
    print(f"Regression: {column} ~ state_freq")
    print("P-values:")
    print(output["p_values"])
    print("\n")

