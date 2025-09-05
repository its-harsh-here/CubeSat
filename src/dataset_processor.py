
"""
Dataset Processing Module for Project ICARUS
Handles traversal and processing of the entire dataset
"""

import numpy as np
import json
from pathlib import Path
from bpsk_receiver import BPSKReceiver
import matplotlib.pyplot as plt

class DatasetProcessor:
    def __init__(self, dataset_path="../cubesat_dataset"):
        """
        Initialize dataset processor

        Args:
            dataset_path: Path to cubesat_dataset directory relative to src/
        """
        self.dataset_path = Path(dataset_path)
        self.receiver = BPSKReceiver()
        self.results = {}

    def load_sample_data(self, sample_path):
        """Load rx.npy and meta.json from sample directory"""
        try:
            rx_data = np.load(sample_path / 'rx.npy')

            with open(sample_path / 'meta.json', 'r') as f:
                meta_data = json.load(f)

            return rx_data, meta_data
        except Exception as e:
            print(f"Error loading {sample_path}: {e}")
            return None, None

    def save_decoded_bits(self, sample_path, decoded_bits):
        """Save decoded bits to sample directory"""
        try:
            # Ensure decoded_bits is integer array of 0s and 1s
            decoded_bits = np.array(decoded_bits, dtype=np.int32)
            np.save(sample_path / 'decoded_bits.npy', decoded_bits)
            return True
        except Exception as e:
            print(f"Error saving to {sample_path}: {e}")
            return False

    def process_sample_directory(self, sample_path, phase_name):
        """Process a single sample directory"""
        print(f"  Processing {sample_path.name}...")

        # Load data
        rx_data, meta_data = self.load_sample_data(sample_path)
        if rx_data is None or meta_data is None:
            return None

        try:
            # Process with receiver
            decoded_bits, symbols = self.receiver.process_sample(rx_data, meta_data, phase_name)

            # Save results
            success = self.save_decoded_bits(sample_path, decoded_bits)

            if success:
                # Calculate metrics
                result = {
                    'sample_path': str(sample_path),
                    'n_bits': len(decoded_bits),
                    'snr_metadata': meta_data.get('snr_db', 'unknown'),
                    'processing_success': True
                }

                # Estimate SNR if possible
                if len(symbols) > 0:
                    decisions = self.receiver.symbol_decision(symbols)
                    estimated_snr = self.receiver.estimate_snr(symbols, decisions)
                    result['estimated_snr'] = estimated_snr

                return result
            else:
                return {'processing_success': False, 'sample_path': str(sample_path)}

        except Exception as e:
            print(f"    Error processing sample: {e}")
            return {'processing_success': False, 'error': str(e), 'sample_path': str(sample_path)}

    def process_phase_directory(self, phase_path):
        """Process all samples in a phase directory"""
        phase_name = phase_path.name
        print(f"\nProcessing {phase_name}...")

        # Find all sample directories
        sample_dirs = sorted([d for d in phase_path.iterdir() if d.is_dir() and d.name.startswith('sample')])

        if not sample_dirs:
            print(f"  No sample directories found in {phase_path}")
            return []

        print(f"  Found {len(sample_dirs)} sample directories")

        phase_results = []
        successful_samples = 0

        # Process each sample directory
        for sample_dir in sample_dirs:
            result = self.process_sample_directory(sample_dir, phase_name)
            if result:
                phase_results.append(result)
                if result.get('processing_success', False):
                    successful_samples += 1

        print(f"  Completed: {successful_samples}/{len(sample_dirs)} samples processed successfully")

        # Store results
        self.results[phase_name] = phase_results

        return phase_results

    def process_entire_dataset(self):
        """Process the entire cubesat_dataset in tree order"""
        print("=== Project ICARUS - Dataset Processing ===")
        print(f"Dataset path: {self.dataset_path}")

        if not self.dataset_path.exists():
            print(f"ERROR: Dataset path {self.dataset_path} does not exist!")
            print("Please ensure cubesat_dataset/ is in the correct location relative to src/")
            return False

        # Find all phase directories
        phase_dirs = sorted([d for d in self.dataset_path.iterdir() 
                           if d.is_dir() and d.name.startswith('phase')])

        if not phase_dirs:
            print(f"No phase directories found in {self.dataset_path}")
            return False

        print(f"Found {len(phase_dirs)} phase directories")

        total_samples_processed = 0

        # Process each phase directory
        for phase_dir in phase_dirs:
            phase_results = self.process_phase_directory(phase_dir)
            successful_in_phase = sum(1 for r in phase_results if r.get('processing_success', False))
            total_samples_processed += successful_in_phase

        print(f"\n=== Processing Complete ===")
        print(f"Total samples successfully processed: {total_samples_processed}")

        # Generate summary report
        self.generate_summary_report()

        return True

    def generate_summary_report(self):
        """Generate a summary report of processing results"""
        print("\n=== Processing Summary ===")

        for phase_name, phase_results in self.results.items():
            successful = [r for r in phase_results if r.get('processing_success', False)]
            failed = [r for r in phase_results if not r.get('processing_success', False)]

            print(f"\n{phase_name}:")
            print(f"  Successful: {len(successful)}")
            print(f"  Failed: {len(failed)}")

            if successful:
                # Show SNR statistics if available
                snr_values = [r.get('estimated_snr') for r in successful if r.get('estimated_snr') is not None]
                if snr_values:
                    print(f"  Estimated SNR range: {min(snr_values):.2f} to {max(snr_values):.2f} dB")

                # Show bit count statistics
                bit_counts = [r.get('n_bits', 0) for r in successful]
                if bit_counts:
                    print(f"  Decoded bits per sample: {min(bit_counts)} to {max(bit_counts)}")

            # Show failed samples
            if failed:
                print(f"  Failed samples:")
                for fail in failed[:3]:  # Show first 3 failures
                    sample_name = Path(fail['sample_path']).name
                    error = fail.get('error', 'Unknown error')
                    print(f"    {sample_name}: {error}")
                if len(failed) > 3:
                    print(f"    ... and {len(failed) - 3} more")

    def generate_performance_plots(self):
        """Generate required performance plots"""
        print("\nGenerating performance plots...")

        try:
            # Create results directory if it doesn't exist
            results_dir = Path("../results")
            results_dir.mkdir(exist_ok=True)

            # Plot processing success rates by phase
            fig, ax = plt.subplots(figsize=(10, 6))

            phases = []
            success_rates = []

            for phase_name, phase_results in self.results.items():
                phases.append(phase_name)
                successful = sum(1 for r in phase_results if r.get('processing_success', False))
                total = len(phase_results)
                success_rates.append(successful / total * 100 if total > 0 else 0)

            ax.bar(phases, success_rates)
            ax.set_ylabel('Success Rate (%)')
            ax.set_title('Processing Success Rate by Phase')
            ax.set_ylim(0, 100)

            # Add value labels on bars
            for i, rate in enumerate(success_rates):
                ax.text(i, rate + 1, f'{rate:.1f}%', ha='center')

            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(results_dir / 'processing_success_rates.png', dpi=300, bbox_inches='tight')
            plt.close()

            print(f"  Saved processing_success_rates.png")

            return True

        except Exception as e:
            print(f"Error generating plots: {e}")
            return False
