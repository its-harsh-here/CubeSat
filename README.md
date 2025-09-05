# Project ICARUS - CubeSat Communications Challenge

Implementation of a BPSK receiver for the Project ICARUS communications reliability challenge.

## Project Structure

```
project_root/
├── cubesat_dataset/                   # Dataset (with decoded.numpy)
│   ├── phase1_timing/
│   ├── phase2_snr/
│   ├── phase3_coding/
│   └── phase4_doppler/
├── docs/                             # Documents
│   ├── Challenge.pdf
│   ├── Documentation.pdf
│   ├── Technical Report.pdf
├── src/                              # All implementation files (flat structure)
│   ├── bpsk_receiver.py              # Core BPSK receiver with full RS & Viterbi
│   ├── ultra_flexible_processor.py   # Ultra-flexible dataset processor
│   ├── comprehensive_evaluator.py    # Complete BER/FER evaluator
│   ├── constellation_plotter.py      # Constellation diagram generator
│   ├── doppler_plotter.py            # Doppler compensation plotter
│   ├── complete_evaluation.py        # MAIN evaluation script
│   ├── main_flexible.py              # Dataset processing script
│   ├── ber_calculator.py             # BER/FER utilities
├── results/                          # Generated plots and analysis
└── requirements.txt                  # Python dependencies
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Navigate to source directory
cd src/
```

## Usage

### Basic Processing
Process the entire dataset with default settings:
```bash
python main.py
```

### Advanced Usage
```bash
# Process with custom dataset path
python main.py --dataset-path /path/to/cubesat_dataset

# Process and generate all plots
python main.py --generate-plots

# Process and evaluate BER (requires ground truth in meta.json)
python main.py --evaluate-ber --generate-plots

# Custom results directory
python main.py --results-path /path/to/results --generate-plots
```

### Generate Visualization Only
```bash
python visualization.py
```

## Implementation Details

<img width="2400" height="1600" alt="image" src="https://github.com/user-attachments/assets/d9f98cd0-2585-4614-b5ac-1950f44cac35" />

<img width="2400" height="1600" alt="image" src="https://github.com/user-attachments/assets/3a880c25-0ef8-45ae-aa5c-2b19dfdcf56a" />


### Phase 1: Symbol Timing Recovery
- **Algorithm**: Gardner Timing Error Detector
- **Components**: Root-raised cosine matched filter, interpolation control
- **Goal**: BER ≤ 1×10⁻² @ 10 dB

### Phase 2: SNR Calibration  
- **Algorithm**: Signal/noise power estimation and scaling
- **Components**: Decision-directed power estimation
- **Goal**: BER curve within ±2 dB of theory

### Phase 3: Error Correction
- **Algorithms**: Reed-Solomon RS(15,11) and Viterbi decoding
- **Components**: Galois field arithmetic, trellis-based decoding
- **Goals**: RS FER ≤ 1×10⁻³ @ 12 dB, Conv. BER ≤ 1×10⁻⁴ @ 8 dB

### Phase 4: Doppler Compensation
- **Algorithm**: Frequency offset estimation and correction  
- **Components**: FFT-based estimation, complex mixer correction
- **Goal**: BER ≤ 1×10⁻³ @ 15 dB

## Output Files

The processor generates:
- `decoded_bits.npy` in each sample directory (main deliverable)
- Performance plots in `results/` directory
- Processing summary reports
- Constellation diagrams and eye diagrams

## Key Features

- **Complete BPSK receiver** with all four phase implementations
- **Tree-based dataset processing** - automatically finds and processes all samples
- **Performance visualization** - BER curves, constellation diagrams
- **Robust error handling** - continues processing despite individual sample failures
- **Modular design** - easy to extend and modify individual components
- **NumPy/SciPy only** - no external communications libraries

## Performance Monitoring

The implementation tracks:
- Processing success rates by phase
- BER/FER metrics (when ground truth available)
- SNR estimation and calibration accuracy
- Timing and carrier recovery convergence

## Troubleshooting

### Common Issues

1. **Dataset not found**: Ensure `cubesat_dataset/` is at the correct location relative to `src/`

2. **Processing failures**: Check individual sample error messages in the console output

3. **Missing plots**: Use `--generate-plots` flag and ensure matplotlib is installed

4. **Import errors**: Ensure all required packages are installed with `pip install -r requirements.txt`

### Debug Mode

For detailed debugging, modify the logging level in the source files or add print statements as needed.

## Extensions

The modular design allows easy extension:
- Add new phases by extending the receiver class
- Implement advanced algorithms (e.g., turbo codes, LDPC)
- Add more sophisticated visualization
- Integrate with external evaluation tools

## Requirements

- Python 3.7+
- NumPy >= 1.21.0
- SciPy >= 1.7.0  
- Matplotlib >= 3.5.0 (for visualization)

See `requirements.txt` for complete dependency list.






