"""
FAST experiment runner with optimizations:
- Reduced observation resolution (84x84 instead of 210x160)
- Smaller batch size for faster iteration
- Optional: Skip frame stacking for even faster processing
"""

# Add argument for fast mode
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--fast", action="store_true", 
                   help="Use fast mode: smaller resolution, less data")
args, unknown = parser.parse_known_args()

if args.fast:
    print("=" * 70)
    print("FAST MODE ENABLED")
    print("  - Using 84x84 resolution (downscaled from 210x160)")
    print("  - Batch size: 128 (reduced from 256)")
    print("  - Single frame (no stacking) for faster processing")
    print("=" * 70)
    
    # Modify imports or parameters here if needed
    FAST_MODE = True
    FAST_RESOLUTION = (84, 84)
    FAST_BATCH_SIZE = 128
else:
    FAST_MODE = False

# Now import the main script
import sys
sys.argv = [sys.argv[0]] + unknown  # Remove --fast flag
exec(open('run_all_experiments.py').read())
