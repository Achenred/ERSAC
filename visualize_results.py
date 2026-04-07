# visualize_results.py
"""
Visualize and compare results from run_all_experiments.py
Creates plots and comparison tables
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List


def load_results(results_dir: Path) -> Dict:
    """Load experiment results from directory"""
    summary_path = results_dir / "summary.json"
    detailed_path = results_dir / "detailed_results.json"
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    with open(detailed_path, 'r') as f:
        detailed = json.load(f)
    
    return {'summary': summary, 'detailed': detailed}


def plot_learning_curves(detailed_results: List[Dict], save_path: Path):
    """Plot learning curves for all methods"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Learning Curves Comparison', fontsize=16, fontweight='bold')
    
    # Extract data
    methods_data = {}
    for result in detailed_results:
        method = result['method']
        eval_history = result['eval_history']
        
        steps = [e['step'] for e in eval_history]
        q_values = [e['mean_q_value'] for e in eval_history]
        action_agree = [e['action_agreement'] for e in eval_history]
        rewards = [e['mean_reward'] for e in eval_history]
        
        methods_data[method] = {
            'steps': steps,
            'q_values': q_values,
            'action_agree': action_agree,
            'rewards': rewards
        }
    
    # Plot 1: Q-values
    ax = axes[0, 0]
    for method, data in methods_data.items():
        ax.plot(data['steps'], data['q_values'], marker='o', label=method.upper(), linewidth=2)
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Mean Q-Value', fontsize=12)
    ax.set_title('Q-Value Estimates', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Action Agreement
    ax = axes[0, 1]
    for method, data in methods_data.items():
        ax.plot(data['steps'], data['action_agree'], marker='s', label=method.upper(), linewidth=2)
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Action Agreement', fontsize=12)
    ax.set_title('Action Agreement with Greedy Policy', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Mean Reward
    ax = axes[1, 0]
    for method, data in methods_data.items():
        ax.plot(data['steps'], data['rewards'], marker='^', label=method.upper(), linewidth=2)
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Mean Reward', fontsize=12)
    ax.set_title('Mean Reward in Test Set', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Training Time
    ax = axes[1, 1]
    methods = list(methods_data.keys())
    times = [detailed_results[i]['train_time'] for i in range(len(detailed_results))]
    colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))
    bars = ax.bar(range(len(methods)), times, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.upper() for m in methods], rotation=45, ha='right')
    ax.set_ylabel('Training Time (seconds)', fontsize=12)
    ax.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.1f}s',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Learning curves saved to: {save_path}")
    plt.close()


def plot_final_comparison(summary: Dict, save_path: Path):
    """Create bar chart comparing final performance"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Final Performance Comparison', fontsize=16, fontweight='bold')
    
    results = summary['results']
    methods = [r['method'] for r in results]
    test_q = [r['final_test_q'] for r in results]
    action_agree = [r['action_agreement'] for r in results]
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
    
    # Plot 1: Test Q-value
    ax = axes[0]
    bars = ax.bar(range(len(methods)), test_q, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.upper() for m in methods], rotation=45, ha='right')
    ax.set_ylabel('Mean Q-Value', fontsize=12)
    ax.set_title('Final Test Q-Value', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, test_q):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 2: Action Agreement
    ax = axes[1]
    bars = ax.bar(range(len(methods)), action_agree, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.upper() for m in methods], rotation=45, ha='right')
    ax.set_ylabel('Action Agreement', fontsize=12)
    ax.set_title('Final Action Agreement', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])
    
    # Add value labels
    for bar, val in zip(bars, action_agree):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Final comparison saved to: {save_path}")
    plt.close()


def create_results_table(summary: Dict, save_path: Path):
    """Create formatted results table"""
    results = summary['results']
    
    # Sort by test Q-value
    sorted_results = sorted(results, key=lambda x: x['final_test_q'], reverse=True)
    
    # Create table
    lines = []
    lines.append("="*90)
    lines.append("OFFLINE RL METHODS COMPARISON - FINAL RESULTS")
    lines.append("="*90)
    lines.append(f"Dataset: {summary['dataset']}")
    lines.append(f"Training Steps: {summary['num_steps']:,}")
    lines.append(f"Train Size: {summary['train_size']:,} transitions")
    lines.append(f"Test Size: {summary['test_size']:,} transitions")
    lines.append(f"Random Seed: {summary['seed']}")
    lines.append("="*90)
    lines.append("")
    lines.append(f"{'Rank':<6} {'Method':<15} {'Test Q-Value':<15} {'Action Agree':<15} {'Time (s)':<12}")
    lines.append("-"*90)
    
    for rank, result in enumerate(sorted_results, 1):
        method = result['method'].upper()
        test_q = result['final_test_q']
        action_agree = result['action_agreement']
        train_time = result['train_time']
        
        # Add medal emoji for top 3
        if rank == 1:
            rank_str = "🥇 1"
        elif rank == 2:
            rank_str = "🥈 2"
        elif rank == 3:
            rank_str = "🥉 3"
        else:
            rank_str = f"   {rank}"
        
        lines.append(f"{rank_str:<6} {method:<15} {test_q:<15.4f} {action_agree:<15.3f} {train_time:<12.1f}")
    
    lines.append("="*90)
    lines.append("")
    lines.append("PERFORMANCE INSIGHTS:")
    lines.append("-"*90)
    
    # Best method
    best_method = sorted_results[0]
    lines.append(f"Best Test Q-Value: {best_method['method'].upper()} ({best_method['final_test_q']:.4f})")
    
    # Best action agreement
    best_agreement = max(sorted_results, key=lambda x: x['action_agreement'])
    lines.append(f"Best Action Agreement: {best_agreement['method'].upper()} ({best_agreement['action_agreement']:.3f})")
    
    # Fastest
    fastest = min(sorted_results, key=lambda x: x['train_time'])
    lines.append(f"Fastest Training: {fastest['method'].upper()} ({fastest['train_time']:.1f}s)")
    
    # Q-value statistics
    q_values = [r['final_test_q'] for r in sorted_results]
    lines.append(f"Q-Value Range: [{min(q_values):.4f}, {max(q_values):.4f}]")
    lines.append(f"Q-Value Std Dev: {np.std(q_values):.4f}")
    
    lines.append("="*90)
    
    # Save to file
    table_text = "\n".join(lines)
    with open(save_path, 'w') as f:
        f.write(table_text)
    
    # Print to console
    print("\n" + table_text)
    print(f"\nResults table saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize experiment results")
    parser.add_argument("results_dir", type=str,
                       help="Directory containing experiment results")
    parser.add_argument("--no_plots", action="store_true",
                       help="Skip generating plots")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return
    
    print(f"Loading results from: {results_dir}")
    
    # Load results
    data = load_results(results_dir)
    summary = data['summary']
    detailed = data['detailed']
    
    # Create visualizations
    if not args.no_plots:
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            
            print("\nGenerating visualizations...")
            
            # Learning curves
            plot_learning_curves(detailed, results_dir / "learning_curves.png")
            
            # Final comparison
            plot_final_comparison(summary, results_dir / "final_comparison.png")
            
        except ImportError:
            print("Warning: matplotlib not found, skipping plots")
    
    # Create results table
    create_results_table(summary, results_dir / "results_table.txt")
    
    print(f"\n✅ Visualization complete!")
    print(f"All outputs saved to: {results_dir}")


if __name__ == "__main__":
    main()
