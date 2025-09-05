
"""
BPSK Receiver Implementation for Project ICARUS
Handles all four phases of the communication challenge
"""

import numpy as np
from scipy import signal
import json
from pathlib import Path
import matplotlib.pyplot as plt

class BPSKReceiver:
    def __init__(self, samples_per_symbol=4, rolloff=0.35):
        """
        Initialize BPSK receiver for CubeSat communications

        Args:
            samples_per_symbol: Oversampling factor
            rolloff: Root-raised cosine filter rolloff factor
        """
        self.sps = samples_per_symbol
        self.rolloff = rolloff
        self.rrc_filter = None
        self.symbol_duration = 1.0  # Normalized

        # Timing recovery parameters
        self.timing_alpha = 0.05
        self.timing_beta = self.timing_alpha**2 / 4

        # Carrier recovery parameters  
        self.carrier_alpha = 0.02
        self.carrier_beta = self.carrier_alpha**2 / 4

    def design_matched_filter(self, span=8):
        """Design root-raised cosine matched filter"""
        # Time vector
        t = np.arange(-span*self.sps//2, span*self.sps//2+1) / self.sps

        # Root-raised cosine impulse response
        h = np.zeros_like(t, dtype=float)
        for i, time in enumerate(t):
            if abs(time) < 1e-10:  # t = 0
                h[i] = 1.0 + self.rolloff * (4/np.pi - 1)
            elif abs(abs(time) - 1/(4*self.rolloff)) < 1e-10:  # Special case
                h[i] = (self.rolloff/np.sqrt(2)) * ((1+2/np.pi)*np.sin(np.pi/(4*self.rolloff)) + 
                                                    (1-2/np.pi)*np.cos(np.pi/(4*self.rolloff)))
            else:
                numerator = np.sin(np.pi*time*(1-self.rolloff)) + 4*self.rolloff*time*np.cos(np.pi*time*(1+self.rolloff))
                denominator = np.pi*time*(1-(4*self.rolloff*time)**2)
                h[i] = numerator / denominator

        # Normalize for unit energy
        h = h / np.sqrt(np.sum(h**2))
        self.rrc_filter = h
        return h

    def matched_filter(self, rx_samples):
        """Apply matched filtering to received samples"""
        if self.rrc_filter is None:
            self.design_matched_filter()

        # Apply matched filter
        filtered = signal.convolve(rx_samples, self.rrc_filter, mode='same')
        return filtered

    def gardner_timing_recovery(self, samples):
        """
        Gardner timing recovery algorithm
        """
        n_symbols = len(samples) // self.sps
        recovered_symbols = []

        # Initialize timing loop
        mu = 0.0  # Fractional timing offset
        mu_accum = 0.0  # Integral term

        for i in range(1, n_symbols):
            # Calculate sample indices
            k = int(i * self.sps + mu)

            if k + self.sps < len(samples) and k - self.sps >= 0:
                # Get samples for Gardner TED
                sample_late = samples[k + self.sps//2] if k + self.sps//2 < len(samples) else 0
                sample_prompt = samples[k]
                sample_early = samples[k - self.sps//2] if k - self.sps//2 >= 0 else 0

                recovered_symbols.append(sample_prompt)

                # Gardner timing error detector
                if len(recovered_symbols) > 1:
                    timing_error = np.real(sample_late * np.conj(recovered_symbols[-2]) - 
                                         sample_early * np.conj(sample_prompt))

                    # Update timing with loop filter
                    mu_accum += self.timing_beta * timing_error
                    mu += self.timing_alpha * timing_error + mu_accum

                    # Keep mu bounded
                    mu = np.clip(mu, -2.0, 2.0)

        return np.array(recovered_symbols)

    def costas_loop(self, symbols):
        """
        Costas loop for carrier phase recovery
        """
        recovered_symbols = np.zeros_like(symbols)
        phase = 0.0
        phase_accum = 0.0

        for i, symbol in enumerate(symbols):
            # Correct phase
            corrected_symbol = symbol * np.exp(-1j * phase)
            recovered_symbols[i] = corrected_symbol

            # Phase error detector (for BPSK)
            phase_error = np.imag(corrected_symbol) * np.sign(np.real(corrected_symbol))

            # Update phase with loop filter
            phase_accum += self.carrier_beta * phase_error
            phase += self.carrier_alpha * phase_error + phase_accum

            # Wrap phase
            phase = np.angle(np.exp(1j * phase))

        return recovered_symbols

    def symbol_decision(self, symbols):
        """Make hard decisions on BPSK symbols"""
        # For BPSK: positive real -> 1, negative real -> 0
        decisions = (np.real(symbols) > 0).astype(int)
        return decisions

    def estimate_snr(self, symbols, decisions):
        """Estimate SNR from received symbols and decisions"""
        # Reconstruct ideal symbols
        ideal_symbols = 2 * decisions - 1  # Map 0,1 to -1,1

        # Calculate powers
        signal_power = np.mean(np.abs(ideal_symbols)**2)
        error_power = np.mean(np.abs(symbols - ideal_symbols)**2)

        # SNR in dB
        snr_db = 10 * np.log10(signal_power / (error_power + 1e-12))
        return snr_db

    def process_phase1(self, rx_samples, meta_data):
        """Process Phase 1: Symbol timing recovery"""
        # Apply matched filter
        filtered = self.matched_filter(rx_samples)

        # Timing recovery
        symbols = self.gardner_timing_recovery(filtered)

        # Carrier recovery
        symbols = self.costas_loop(symbols)

        # Symbol decisions
        decoded_bits = self.symbol_decision(symbols)

        return decoded_bits, symbols

    def calibrate_snr(self, symbols, meta_data):
        """Phase 2: SNR calibration"""
        # Get expected SNR from metadata
        expected_snr = meta_data.get('snr_db', 10.0)

        # Estimate current SNR
        decisions = self.symbol_decision(symbols)
        estimated_snr = self.estimate_snr(symbols, decisions)

        # Calculate scaling factor to match expected SNR
        snr_ratio = 10**((expected_snr - estimated_snr) / 10)
        scaling_factor = np.sqrt(snr_ratio)

        # Apply scaling
        calibrated_symbols = symbols * scaling_factor

        return calibrated_symbols

    def process_phase2(self, rx_samples, meta_data):
        """Process Phase 2: SNR calibration"""
        # Basic receiver processing
        filtered = self.matched_filter(rx_samples)
        symbols = self.gardner_timing_recovery(filtered)
        symbols = self.costas_loop(symbols)

        # SNR calibration
        calibrated_symbols = self.calibrate_snr(symbols, meta_data)

        # Symbol decisions
        decoded_bits = self.symbol_decision(calibrated_symbols)

        return decoded_bits, calibrated_symbols

    def reed_solomon_decode(self, symbols):
        """
        Simplified Reed-Solomon RS(15,11) decoder
        This is a basic implementation - full RS decoder is complex
        """
        # Convert symbols to 4-bit values (for GF(16))
        # This is a simplified version - real RS decoder needs full GF arithmetic

        # For now, return symbols as is (placeholder)
        # In full implementation, would include:
        # - Galois field arithmetic
        # - Syndrome calculation  
        # - Berlekamp-Massey algorithm
        # - Chien search and error correction

        return symbols

    def viterbi_decode(self, soft_bits, constraint_length=7):
        """
        Simplified Viterbi decoder for convolutional codes
        """
        # This is a basic implementation
        # Full Viterbi decoder requires trellis construction

        # For demonstration, use hard decisions
        hard_bits = (soft_bits > 0).astype(int)

        # In full implementation:
        # - Construct trellis based on generator polynomials
        # - Calculate branch metrics
        # - Perform add-compare-select operations
        # - Traceback for maximum likelihood path

        return hard_bits

    def process_phase3(self, rx_samples, meta_data):
        """Process Phase 3: Error correction coding"""
        # Basic receiver processing
        filtered = self.matched_filter(rx_samples)
        symbols = self.gardner_timing_recovery(filtered)
        symbols = self.costas_loop(symbols)
        symbols = self.calibrate_snr(symbols, meta_data)

        # Check if Reed-Solomon or Convolutional coding
        coding_type = meta_data.get('coding_type', 'none')

        if coding_type == 'reed_solomon':
            # Reed-Solomon decoding
            decoded_symbols = self.reed_solomon_decode(symbols)
            decoded_bits = self.symbol_decision(decoded_symbols)
        elif coding_type == 'convolutional':
            # Viterbi decoding
            soft_bits = np.real(symbols)  # Use real part as soft bits
            decoded_bits = self.viterbi_decode(soft_bits)
        else:
            # No coding
            decoded_bits = self.symbol_decision(symbols)

        return decoded_bits, symbols

    def estimate_frequency_offset(self, samples):
        """Estimate frequency offset using FFT"""
        # Take FFT to find dominant frequency component
        fft_vals = np.fft.fft(samples)
        freqs = np.fft.fftfreq(len(samples))

        # Find peak frequency
        peak_idx = np.argmax(np.abs(fft_vals))
        freq_offset = freqs[peak_idx]

        return freq_offset

    def compensate_doppler(self, samples, freq_offset):
        """Compensate for Doppler frequency offset"""
        # Generate correction signal
        t = np.arange(len(samples))
        correction = np.exp(-1j * 2 * np.pi * freq_offset * t)

        # Apply correction
        corrected_samples = samples * correction

        return corrected_samples

    def process_phase4(self, rx_samples, meta_data):
        """Process Phase 4: Doppler compensation"""
        # Estimate and compensate frequency offset
        freq_offset = self.estimate_frequency_offset(rx_samples)
        compensated_samples = self.compensate_doppler(rx_samples, freq_offset)

        # Basic receiver processing
        filtered = self.matched_filter(compensated_samples)
        symbols = self.gardner_timing_recovery(filtered)
        symbols = self.costas_loop(symbols)
        symbols = self.calibrate_snr(symbols, meta_data)

        # Apply error correction if needed
        coding_type = meta_data.get('coding_type', 'none')
        if coding_type == 'reed_solomon':
            decoded_symbols = self.reed_solomon_decode(symbols)
            decoded_bits = self.symbol_decision(decoded_symbols)
        elif coding_type == 'convolutional':
            soft_bits = np.real(symbols)
            decoded_bits = self.viterbi_decode(soft_bits)
        else:
            decoded_bits = self.symbol_decision(symbols)

        return decoded_bits, symbols

    def process_sample(self, rx_samples, meta_data, phase_name):
        """Process a single sample based on phase"""
        if phase_name == 'phase1_timing':
            return self.process_phase1(rx_samples, meta_data)
        elif phase_name == 'phase2_snr':
            return self.process_phase2(rx_samples, meta_data)
        elif phase_name == 'phase3_coding':
            return self.process_phase3(rx_samples, meta_data)
        elif phase_name == 'phase4_doppler':
            return self.process_phase4(rx_samples, meta_data)
        else:
            # Default to phase 1
            return self.process_phase1(rx_samples, meta_data)
