
"""
BER/FER Calculation Module for Project ICARUS
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

class BERCalculator:
    def __init__(self):
        """Initialize BER calculator"""
        self.ber_results = {}

    def calculate_ber(self, transmitted_bits, received_bits):
        """Calculate bit error rate"""
        if len(transmitted_bits) != len(received_bits):
            min_len = min(len(transmitted_bits), len(received_bits))
            transmitted_bits = transmitted_bits[:min_len]
            received_bits = received_bits[:min_len]

        errors = np.sum(transmitted_bits != received_bits)
        total_bits = len(transmitted_bits)
        ber = errors / total_bits if total_bits > 0 else 0

        return ber, errors, total_bits

    def calculate_fer(self, transmitted_frames, received_frames):
        """Calculate frame error rate"""
        frame_errors = 0
        total_frames = len(transmitted_frames)

        for tx_frame, rx_frame in zip(transmitted_frames, received_frames):
            if not np.array_equal(tx_frame, rx_frame):
                frame_errors += 1

        fer = frame_errors / total_frames if total_frames > 0 else 0

        return fer, frame_errors, total_frames

    def theoretical_bpsk_ber(self, snr_db):
        """Calculate theoretical BPSK BER curve"""
        snr_linear = 10**(snr_db / 10)
        ber = 0.5 * np.erfc(np.sqrt(snr_linear))
        return ber

    def evaluate_sample(self, sample_path):
        """Evaluate BER for a single sample"""
        try:
            # Load metadata to get ground truth
            with open(sample_path / 'meta.json', 'r') as f:
                meta_data = json.load(f)

            # Load decoded bits
            decoded_bits = np.load(sample_path / 'decoded_bits.npy')

            # Get ground truth bits from metadata
            if 'ground_truth_bits' in meta_data:
                ground_truth = np.array(meta_data['ground_truth_bits'])

                # Calculate BER
                ber, errors, total_bits = self.calculate_ber(ground_truth, decoded_bits)

                return {
                    'ber': ber,
                    'errors': errors,
                    'total_bits': total_bits,
                    'snr_db': meta_data.get('snr_db', None),
                    'sample_path': str(sample_path)
                }
            else:
                print(f"Warning: No ground truth bits in {sample_path}")
                return None

        except Exception as e:
            print(f"Error evaluating {sample_path}: {e}")
            return None

    def evaluate_phase(self, phase_path):
        """Evaluate all samples in a phase"""
        phase_name = phase_path.name
        sample_dirs = sorted([d for d in phase_path.iterdir() 
                            if d.is_dir() and d.name.startswith('sample')])

        phase_results = []

        for sample_dir in sample_dirs:
            result = self.evaluate_sample(sample_dir)
            if result:
                phase_results.append(result)

        self.ber_results[phase_name] = phase_results
        return phase_results

    def plot_ber_curves(self, output_path="../results"):
        """Plot BER vs SNR curves"""
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True)

        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot theoretical BPSK curve
        snr_range = np.linspace(-5, 20, 100)
        theoretical_ber = self.theoretical_bpsk_ber(snr_range)
        ax.semilogy(snr_range, theoretical_ber, 'k--', label='Theoretical BPSK', linewidth=2)

        # Plot results for each phase
        colors = ['blue', 'green', 'red', 'orange']
        for i, (phase_name, results) in enumerate(self.ber_results.items()):
            if results:
                snr_values = [r['snr_db'] for r in results if r['snr_db'] is not None]
                ber_values = [r['ber'] for r in results if r['snr_db'] is not None]

                if snr_values and ber_values:
                    ax.semilogy(snr_values, ber_values, 'o-', color=colors[i % len(colors)], 
                              label=phase_name, markersize=6)

        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('Bit Error Rate')
        ax.set_title('BER vs SNR Performance')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(1e-6, 1)

        plt.tight_layout()
        plt.savefig(output_path / 'ber_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved BER curves to {output_path / 'ber_curves.png'}")

    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        print("\n=== Performance Evaluation Results ===")

        for phase_name, results in self.ber_results.items():
            if results:
                print(f"\n{phase_name}:")

                ber_values = [r['ber'] for r in results]
                snr_values = [r['snr_db'] for r in results if r['snr_db'] is not None]

                print(f"  Samples evaluated: {len(results)}")
                print(f"  BER range: {min(ber_values):.6f} to {max(ber_values):.6f}")

                if snr_values:
                    print(f"  SNR range: {min(snr_values):.2f} to {max(snr_values):.2f} dB")

                # Calculate average BER at different SNR levels
                snr_bins = [(0, 5), (5, 10), (10, 15), (15, 20)]
                for snr_min, snr_max in snr_bins:
                    bin_results = [r for r in results if r['snr_db'] is not None and 
                                 snr_min <= r['snr_db'] < snr_max]
                    if bin_results:
                        avg_ber = np.mean([r['ber'] for r in bin_results])
                        print(f"  Average BER at {snr_min}-{snr_max} dB: {avg_ber:.6f}")
