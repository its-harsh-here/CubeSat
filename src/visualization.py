
"""
Visualization Module for Project ICARUS
Generate constellation diagrams and other performance plots
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from bpsk_receiver import BPSKReceiver

class Visualizer:
    def __init__(self):
        """Initialize visualizer"""
        self.receiver = BPSKReceiver()

    def plot_constellation(self, symbols, title="Constellation Diagram", 
                          save_path=None, snr_db=None):
        """Plot constellation diagram"""
        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot symbols
        ax.scatter(np.real(symbols), np.imag(symbols), alpha=0.6, s=20)

        # Add ideal constellation points for BPSK
        ax.scatter([-1, 1], [0, 0], c='red', s=100, marker='x', 
                  linewidth=3, label='Ideal BPSK')

        ax.set_xlabel('In-Phase')
        ax.set_ylabel('Quadrature')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.axis('equal')

        if snr_db is not None:
            ax.text(0.05, 0.95, f'SNR: {snr_db:.1f} dB', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_eye_diagram(self, samples, sps=4, title="Eye Diagram", save_path=None):
        """Plot eye diagram for timing analysis"""
        # Reshape samples into eye traces
        n_traces = len(samples) // (2 * sps)
        eye_traces = samples[:n_traces * 2 * sps].reshape(n_traces, 2 * sps)

        fig, ax = plt.subplots(figsize=(10, 6))

        time_axis = np.linspace(-1, 1, 2 * sps)

        for trace in eye_traces[:min(100, n_traces)]:  # Plot max 100 traces
            ax.plot(time_axis, np.real(trace), 'b-', alpha=0.1)

        ax.set_xlabel('Normalized Time')
        ax.set_ylabel('Amplitude')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        # Add sampling points
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Sampling Points')
        ax.legend()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def analyze_sample(self, sample_path, save_plots=True):
        """Analyze a single sample and generate plots"""
        try:
            # Load data
            rx_data = np.load(sample_path / 'rx.npy')
            with open(sample_path / 'meta.json', 'r') as f:
                meta_data = json.load(f)

            phase_name = sample_path.parent.name

            # Process sample to get symbols at different stages
            filtered = self.receiver.matched_filter(rx_data)
            symbols = self.receiver.gardner_timing_recovery(filtered)
            symbols = self.receiver.costas_loop(symbols)

            if save_plots:
                # Create output directory
                output_dir = Path("../results") / "constellations" / phase_name
                output_dir.mkdir(parents=True, exist_ok=True)

                # Generate constellation diagram
                sample_name = sample_path.name
                snr_db = meta_data.get('snr_db')
                title = f"Constellation - {phase_name} - {sample_name}"

                self.plot_constellation(symbols, title=title,
                                      save_path=output_dir / f"{sample_name}_constellation.png",
                                      snr_db=snr_db)

                # Generate eye diagram from filtered samples
                if len(filtered) > 100:  # Ensure enough samples
                    eye_title = f"Eye Diagram - {phase_name} - {sample_name}"
                    self.plot_eye_diagram(filtered, title=eye_title,
                                        save_path=output_dir / f"{sample_name}_eye.png")

                print(f"  Generated plots for {sample_name}")

            return True

        except Exception as e:
            print(f"Error analyzing {sample_path}: {e}")
            return False

    def generate_phase_analysis(self, phase_path, max_samples=5):
        """Generate analysis plots for a phase (limited number of samples)"""
        phase_name = phase_path.name
        print(f"\nGenerating analysis plots for {phase_name}...")

        sample_dirs = sorted([d for d in phase_path.iterdir() 
                            if d.is_dir() and d.name.startswith('sample')])

        # Process limited number of samples to avoid too many plots
        samples_to_process = sample_dirs[:max_samples]

        successful = 0
        for sample_dir in samples_to_process:
            if self.analyze_sample(sample_dir):
                successful += 1

        print(f"  Generated plots for {successful}/{len(samples_to_process)} samples")
        return successful

def main():
    """Main function for visualization"""
    visualizer = Visualizer()

    dataset_path = Path("../cubesat_dataset")

    if not dataset_path.exists():
        print(f"Dataset path {dataset_path} not found!")
        return

    # Generate analysis for each phase
    for phase_dir in sorted(dataset_path.glob('phase*')):
        if phase_dir.is_dir():
            visualizer.generate_phase_analysis(phase_dir)

    print("\nVisualization complete! Check ../results/constellations/ for plots.")

if __name__ == "__main__":
    main()
