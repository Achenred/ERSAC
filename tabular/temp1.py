import numpy as np
import os
from collections import defaultdict
import pandas as pd

def collect_raw_state_action_frequencies(base_path, taus, seeds, num_samples):
    """
    Collects individual (state, action, tau, frequency) entries across seeds,
    to allow boxplot rendering with natural whiskers.
    """
    rows = []

    for tau in taus:
        for seed in seeds:
            file_path = os.path.join(
                base_path, f"{seed}", f"{tau}", f"{num_samples}", "samples_train.npz"
            )

            if not os.path.exists(file_path):
                print(f"Missing file: {file_path}")
                continue

            data = np.load(file_path)
            states = data["idstatefrom"]
            actions = data["idaction"]

            sa_counts = defaultdict(int)
            total = len(states)

            for s, a in zip(states, actions):
                sa_counts[(s, a)] += 1

            # Convert to frequency and store each entry
            for (s, a), count in sa_counts.items():
                freq = count / total
                rows.append({
                    'tau': tau,
                    'state': s,
                    'action': a,
                    'frequency': freq,
                    'seed': seed
                })

    return pd.DataFrame(rows)


def to_dataframe(freq_dict):
    rows = []
    for tau, dist in freq_dict.items():
        for (s, a), prob in dist.items():
            rows.append({'tau': tau, 'state': s, 'action': a, 'frequency': prob})
    return pd.DataFrame(rows)


base_path = "/Users/abhilashchenreddy/PycharmProjects/CQL_AC/offline_data/machine-v2/"
taus = [0.1, 0.5, 0.9]
seeds = [0,1,2,3]
num_samples = 125  # or the subfolder name if variable

freqs = collect_raw_state_action_frequencies(base_path, taus, seeds, num_samples)

print(freqs.head(5))


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_state_frequencies_grouped_by_action(df, save_path=None):
    """
    Plots a grouped boxplot:
    - X-axis: (s, a) tuple, grouped by state
    - Hue: tau with custom colors
    - Dotted line between state groups
    - No background grid
    - No legend
    - Optional: Save to file with high DPI
    """
    # Format x-axis as tuple (s, a)
    df['x'] = df.apply(lambda row: f"({int(row['state'])}, {int(row['action'])})", axis=1)
    x_order = sorted(df['x'].unique(), key=lambda x: (int(x.split(',')[0][1:]), int(x.split(',')[1][:-1])))
    df['x'] = pd.Categorical(df['x'], categories=x_order, ordered=True)

    # Custom tau colors
    tau_palette = {
        0.1: "orange",
        0.5: "plum",
        0.9: "cyan"
    }

    plt.figure(figsize=(16, 6))
    sns.set(style="white")  # no grid

    sns.boxplot(
        data=df,
        x='x',
        y='frequency',
        hue='tau',
        width=0.5,
        palette=tau_palette,
        showfliers=True
    )

    # Axis labels and title
    plt.xticks(rotation=0)
    plt.xlabel("State-Action (s, a)")
    plt.ylabel("Frequency")
    # plt.title("State-Action Frequency Distribution Grouped by Action")

    # Add dashed lines between states
    xtick_labels = [tick.get_text() for tick in plt.gca().get_xticklabels()]
    state_transitions = []
    last_state = None
    for idx, label in enumerate(xtick_labels):
        state = label.split(',')[0][1:]
        if last_state is not None and state != last_state:
            state_transitions.append(idx - 0.5)
        last_state = state

    for x in state_transitions:
        plt.axvline(x=x, linestyle='--', color='gray', linewidth=1)

    # Remove legend
    plt.gca().legend_.remove()

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=1200, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


plot_state_frequencies_grouped_by_action(freqs, save_path= '/Users/abhilashchenreddy/PycharmProjects/CQL_AC/state_action_freq.png')
