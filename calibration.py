"""
Sound Calibration
generates one CSV file
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from psychopy import sound, core

class SimpleCalibration:
    def __init__(self, calibration_file="calibration.csv"):
        self.calibration_file = calibration_file
        self.calibration_data = None
        self.base_noise = self.create_base_noise()

    def create_base_noise(self, duration=8.0, sample_rate=44100):
        np.random.seed(42) # ensure same base sound each time
        n_samples = int(duration * sample_rate)
        signal = np.random.normal(0, 1, n_samples)
        rms = np.sqrt(np.mean(signal**2)) # RMS normalization
        if rms > 0:
            signal = signal / rms

        return signal.astype(np.float32)
    
    def generate_white_noise(self, amplitude, duration=8.0, sample_rate=44100):
        n_samples = int(duration * sample_rate)
        signal = self.base_noise[:n_samples] * amplitude
        return sound.Sound(
            value=signal,
            stereo=True,
            sampleRate=sample_rate
        )
    
    def measure_dBA(self, amplitude):
        """Measure dBA for a given amplitude."""
        print(f"\n=== Amplitude = {amplitude:.3f} ===")
        tone = self.generate_white_noise(amplitude)
        print("Playing white noise (8 seconds)...")
        tone.play()
        core.wait(tone.getDuration()) # wait for the tone to finish playing
        dB = float(input(f"Enter dB(A): ")) # ask the user to enter the dB(A)
        return dB # return the dB(A)
    
    def run(self, amplitudes=None, n_repeats=5): # measure 5 times
        """Run repeated calibration sessions."""

        if amplitudes is None:
            amplitudes = np.geomspace(0.01, 1, 20).tolist()

        print(f"\n=== Calibration ===")
        print(f"{n_repeats} repeats")
        print(f"{len(amplitudes)} amplitudes per repeat")

        data = []
        fig, ax = plt.subplots(figsize=(8, 6))
        for repeat in range(1, n_repeats + 1):
            print(f"\n========== Repeat {repeat}/{n_repeats} ==========")
            repeat_data = []
            for i, amp in enumerate(amplitudes, 1):
                print(f"\nPoint {i}/{len(amplitudes)}")
                dB = self.measure_dBA(amp)
                row = {
                    'repeat': repeat,
                    'amplitude': amp,
                    'log_amp': np.log(amp),
                    'dBA': dB
                }
                data.append(row)
                repeat_data.append(row)
            # preview fit for this repeat
            repeat_df = pd.DataFrame(repeat_data)
            x = repeat_df['log_amp'].values
            y = repeat_df['dBA'].values
            a, b = np.polyfit(x, y, 1)
            x_smooth = np.linspace(x.min(), x.max(), 100)
            y_smooth = a * x_smooth + b
            ax.plot(
                x_smooth,
                y_smooth,
                '--',
                color='gray',
                alpha=0.5,
                linewidth=1.5,
                label='Preview fits' if repeat == 1 else None
            )
            input("\nPress Enter to continue to next repeat...")

        self.calibration_data = pd.DataFrame(data)
        self.calibration_data.to_csv(self.calibration_file, index=False)
        print(f"\n=== Done ===")
        print(f"Saved to: {self.calibration_file}")
        self.plot(ax)
        return self.calibration_data
    
    def plot(self, ax=None):
        if self.calibration_data is None:
            return

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        x = self.calibration_data['log_amp'].values
        y = self.calibration_data['dBA'].values

        # Raw points
        ax.scatter(
            x,
            y,
            s=40,
            c='blue',
            alpha=0.6,
            label='Measured'
        )

        # Final fit
        a, b = np.polyfit(x, y, 1)
        self.fit_a = a
        self.fit_b = b
        x_smooth = np.linspace(x.min(), x.max(), 100)
        y_smooth = a * x_smooth + b
        y_pred = a * x + b
        r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2) # calculate R^2
        print(f"\nFinal R² = {r2:.4f}")

        ax.plot(
            x_smooth,
            y_smooth,
            'r-',
            linewidth=3,
            label=f'Final fit: dBA = {a:.4f}*ln(amplitude) + {b:.4f}'
        )

        ax.set_xlabel('ln(Amplitude)')
        ax.set_ylabel('dBA')
        ax.set_title('Sound Calibration')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_file = self.calibration_file.replace('.csv', '_plot.png')
        fig.savefig(plot_file, dpi=300)
        plt.show()
        print(f"Plot saved: {plot_file}")
        print(f"dBA = {a:.4f} * ln(amplitude) + {b:.4f}")
        print(f"Amplitude = exp((dBA - {b:.4f}) / {a:.4f})")

        with open("calibration_fit.txt", "w") as f: # save parameters to txt
            f.write(f"{self.fit_a}\n")
            f.write(f"{self.fit_b}\n")
    
    def dba_to_amplitude(self, target_dba):
        return np.exp((target_dba - self.fit_b) / self.fit_a)

if __name__ == "__main__":
    # Run calibration
    cal = SimpleCalibration("my_calibration.csv")
    cal.run()
    