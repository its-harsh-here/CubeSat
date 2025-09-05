
"""
Ultra-Flexible Dataset Processor for Project ICARUS
Handles ANY folder structure - finds rx.npy and meta.json files recursively
"""

import numpy as np
import json
from pathlib import Path
from bpsk_receiver import BPSKReceiver
import matplotlib.pyplot as plt

class FlexibleDatasetProcessor:
    def __init__(self, dataset_path="../cubesat_dataset"):
        """Initialize with ultra-flexible directory detection"""
        self.dataset_path = Path(dataset_path)
        self.receiver = BPSKReceiver()
        self.results = {}

    def find_data_files_recursive(self, root_path):
        """Recursively find all directories containing rx.npy and meta.json"""
        data_dirs = []

        # Recursively search for directories containing both required files
        for path in root_path.rglob("*"):
            if path.is_dir():
                rx_file = path / "rx.npy"
                meta_file = path / "meta.json"

                if rx_file.exists() and meta_file.exists():
                    data_dirs.append(path)

        return sorted(data_dirs)

    def load_sample_data(self, sample_path):
        """Load rx.npy and meta.json from any directory"""
        try:
            rx_data = np.load(sample_path / 'rx.npy')

            with open(sample_path / 'meta.json', 'r') as f:
                meta_data = json.load(f)

            return rx_data, meta_data
        except Exception as e:
            print(f"Error loading {sample_path}: {e}")
            return None, None

    def save_decoded_bits(self, sample_path, decoded_bits):
        """Save decoded bits to the data directory"""
        try:
            decoded_bits = np.array(decoded_bits, dtype=np.int32)
            np.save(sample_path / 'decoded_bits.npy', decoded_bits)
            return True 
        except Exception as e:
            print(f"Error saving to {sample_path}: {e}")
            return False

    def determine_phase_from_path(self, sample_path):
        """Determine which phase this sample belongs to based on path"""
        path_str = str(sample_path).lower()

        # Check for phase indicators in the path
        if 'phase1' in path_str or 'timing' in path_str:
            return 'phase1_timing'
        elif 'phase2' in path_str or 'snr' in path_str:
            return 'phase2_snr'
        elif 'phase3' in path_str or 'coding' in path_str or 'error' in path_str:
            return 'phase3_coding'
        elif 'phase4' in path_str or 'doppler' in path_str:
            return 'phase4_doppler'
        else:
            # Default to phase1 if we can't determine
            return 'unknown_phase'

    def process_sample_directory(self, sample_path):
        """Process a single sample directory"""
        # Determine phase from path
        phase_name = self.determine_phase_from_path(sample_path)

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
                    'relative_path': str(sample_path.relative_to(self.dataset_path)),
                    'phase_detected': phase_name,
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
            return {'processing_success': False, 'error': str(e), 'sample_path': str(sample_path)}

    def process_entire_dataset(self):
        """Process entire dataset by finding all data files recursively"""
        if not self.dataset_path.exists():
            print(f"ERROR: Dataset path {self.dataset_path} does not exist!")
            return False

        # Find ALL directories containing data files, regardless of structure
        data_directories = self.find_data_files_recursive(self.dataset_path)

        if not data_directories:
            self.analyze_dataset_structure()
            return False

        # Group by detected phase
        phase_groups = {}
        for data_dir in data_directories:
            phase = self.determine_phase_from_path(data_dir)
            if phase not in phase_groups:
                phase_groups[phase] = []
            phase_groups[phase].append(data_dir)

        # Show what we found
        for phase, dirs in phase_groups.items():
            print(f"  {phase}: {len(dirs)} samples")

        total_samples_processed = 0

        # Process each data directory

        for data_dir in data_directories:
            result = self.process_sample_directory(data_dir)
            if result and result.get('processing_success', False):
                total_samples_processed += 1

                # Add to phase results
                phase = result['phase_detected']
                if phase not in self.results:
                    self.results[phase] = []
                self.results[phase].append(result)
        # Generate summary report
        self.generate_summary_report()

        return True

    def analyze_dataset_structure(self):
        print("="*50)
        def show_tree(path, prefix="", max_depth=3, current_depth=0):
            if current_depth >= max_depth:
                return

            items = sorted(path.iterdir()) if path.is_dir() else []
            dirs = [item for item in items if item.is_dir()]
            files = [item for item in items if item.is_file()]
        show_tree(self.dataset_path)

        # Look for any .npy or .json files
        npy_files = list(self.dataset_path.rglob("*.npy"))
        json_files = list(self.dataset_path.rglob("*.json"))

        if npy_files:
            print("\n .npy files found:")
            for f in npy_files[:10]:  # Show first 10
                print(f"  {f.relative_to(self.dataset_path)}")
            if len(npy_files) > 10:
                print(f"  ... and {len(npy_files) - 10} more")

        if json_files:
            print("\n .json files found:")
            for f in json_files[:10]:  # Show first 10
                print(f"  {f.relative_to(self.dataset_path)}")
            if len(json_files) > 10:
                print(f"  ... and {len(json_files) - 10} more")

    def generate_summary_report(self):
        """Generate summary report grouped by phase"""
        total_successful = 0
        total_failed = 0

        for phase_name, phase_results in self.results.items():
            successful = [r for r in phase_results if r.get('processing_success', False)]
            failed = [r for r in phase_results if not r.get('processing_success', False)]

            total_successful += len(successful)
            total_failed += len(failed)

            print(f"\n{phase_name}:")
            print(f"   Successful: {len(successful)}")
            print(f"   Failed: {len(failed)}")

            if successful:
                # Show paths for verification
                snr_values = [r.get('estimated_snr') for r in successful if r.get('estimated_snr') is not None]
                if snr_values:
                    print(f"   SNR range: {min(snr_values):.2f} to {max(snr_values):.2f} dB")

                # Show bit counts
                bit_counts = [r.get('n_bits', 0) for r in successful]
                if bit_counts:
                    print(f"   Bits per sample: {min(bit_counts)} to {max(bit_counts)}")

        print(f"\n" + "="*50)
        print(f"OVERALL SUMMARY:")
        print(f"   Total successful: {total_successful}")
        print(f"   Total failed: {total_failed}")
        print(f"   Success rate: {total_successful/(total_successful+total_failed)*100:.1f}%" if (total_successful+total_failed) > 0 else "N/A")

    def generate_performance_plots(self):
        """Generate performance plots"""
        print("\nGenerating performance plots...")

        try:
            results_dir = Path("../results")
            results_dir.mkdir(exist_ok=True)

            if not self.results:
                print("No results to plot")
                return False

            # Plot processing success rates by phase
            fig, ax = plt.subplots(figsize=(12, 6))

            phases = []
            success_rates = []
            sample_counts = []

            for phase_name, phase_results in self.results.items():
                phases.append(phase_name.replace('_', '\n'))  # Line break for better display
                successful = sum(1 for r in phase_results if r.get('processing_success', False))
                total = len(phase_results)
                success_rates.append(successful / total * 100 if total > 0 else 0)
                sample_counts.append(total)

            bars = ax.bar(phases, success_rates, color=['skyblue', 'lightgreen', 'orange', 'lightcoral'])
            ax.set_ylabel('Success Rate (%)')
            ax.set_title('Processing Success Rate by Phase')
            ax.set_ylim(0, 100)

            # Add value labels on bars
            for i, (rate, count) in enumerate(zip(success_rates, sample_counts)):
                ax.text(i, rate + 2, f'{rate:.1f}%\n({count} samples)', ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig(results_dir / 'processing_success_rates.png', dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Saved processing_success_rates.png")
            return True

        except Exception as e:
            print(f"Error generating plots: {e}")
            return False
