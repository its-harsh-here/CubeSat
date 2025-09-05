
"""
Main script using Ultra-Flexible Dataset Processor
Handles ANY dataset structure automatically
"""

import sys
from pathlib import Path
import argparse
from ultra_flexible_processor import FlexibleDatasetProcessor
from ber_calculator import BERCalculator

def main():
    """Main function using flexible processing"""
    parser = argparse.ArgumentParser(description='Project ICARUS - Flexible Dataset Processor')
    parser.add_argument('--dataset-path', default='../cubesat_dataset', 
                       help='Path to cubesat_dataset directory')
    parser.add_argument('--results-path', default='../results',
                       help='Path to save results and plots')
    parser.add_argument('--evaluate-ber', action='store_true',
                       help='Run BER evaluation after processing')
    parser.add_argument('--generate-plots', action='store_true', 
                       help='Generate performance plots')

    args = parser.parse_args()
    # Initialize flexible processor
    processor = FlexibleDatasetProcessor(args.dataset_path)

    # Process entire dataset
    success = processor.process_entire_dataset()

    if not success:
        print("\nDataset processing failed!")
        return 1

    # Generate processing plots
    if args.generate_plots:
        processor.generate_performance_plots()

if __name__ == "__main__":
    sys.exit(main())