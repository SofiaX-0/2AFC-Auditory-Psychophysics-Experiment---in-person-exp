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

    def create_base_noise(self, duration=5.0, sample_rate=44100):
        np.random.seed(42)
        n_samples = int(duration * sample_rate)
        signal = np.random.normal(0, 1, n_samples)
        rms = np.sqrt(np.mean(signal**2))
        if rms > 0:
            signal = signal / rms

        return signal.astype(np.float32)
    
    def generate_white_noise(self, amplitude, sample_rate=44100):

        signal = self.base_noise * amplitude

        return sound.Sound(value=signal, stereo=True, sampleRate=sample_rate)
    
    def measure_dBA(self, amplitude):
        """Measure dBA for a given amplitude."""
        print(f"\n=== Amplitude = {amplitude:.3f} ===")
        tone = self.generate_white_noise(amplitude)
        print("Playing white noise (5 seconds)...")
        tone.play()
        core.wait(tone.getDuration()) # wait for the tone to finish playing
        dB = float(input(f"Enter dB(A): ")) # ask the user to enter the dB(A)
        return dB # return the dB(A)
    
    def run(self, amplitudes=None):
        """Run the calibration."""
        if amplitudes is None:
            # Use logarithmically spaced amplitude values
            amplitudes = np.geomspace(0.01, 1, 20).tolist()
        
        print(f"\n=== Calibration ===") # print the calibration title
        print(f"Measure {len(amplitudes)} points") # print the number of points
        
        data = [] # create an empty list to store the data
        for i, amp in enumerate(amplitudes, 1): # loop through the amplitudes
            print(f"\nPoint {i}/{len(amplitudes)}") # print the point number
            dB = self.measure_dBA(amp) # measure the dBA
            data.append({'amplitude': amp, 'log_amp': np.log(amp), 'dBA': dB}) # add the data to the list
        self.calibration_data = pd.DataFrame(data)
        self.calibration_data.to_csv(self.calibration_file, index=False)
        
        print(f"\n=== Done ===")
        print(f"Saved to: {self.calibration_file}")
        
        self.plot() # plot the data
        return self.calibration_data
    
    def plot(self):
        if self.calibration_data is None:
            return # if the data is not available, return
        
        fig, ax = plt.subplots(figsize=(8, 6)) # create a figure and axis
        
        x = self.calibration_data['log_amp'].values
        y = self.calibration_data['dBA'].values # change pandas data to numpy array
        
        ax.scatter(x, y, s=50, c='blue', label='Measured') # plot the data; s is the size of the points, c is the color, label is the label of the points
        
        # Linear fit
        a, b = np.polyfit(x, y, 1) # fit a linear model to the data
        self.fit_a = a
        self.fit_b = b
        x_smooth = np.linspace(x.min(), x.max(), 100) # create a smooth curve, np.linspace(start, end, n)- generate n points between start and end evenly
        y_smooth = a * x_smooth + b # calculate the y values for the smooth curve
        y_pred = a * x + b
        r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)

        print(f"R² = {r2:.4f}")

        ax.plot(x_smooth, y_smooth, 'r-', linewidth=2, label=f'Log fit: dBA = {a:.1f}*ln(amplitude) + {b:.1f}')
        
        ax.set_xlabel('ln(Amplitude)') # set the x label
        ax.set_ylabel('dBA') # set the y label
        ax.set_title('Sound Calibration') # set the title
        ax.legend() # show the legend
        ax.grid(True, alpha=0.3) # show the grid
        
        plot_file = self.calibration_file.replace('.csv', '_plot.png') # create the plot file name
        fig.savefig(plot_file, dpi=300) # save the plot
        plt.show() # show the plot
        print(f"Plot saved: {plot_file}") # print the plot file name
        print(f"dBA = {a:.4f} * ln(amplitude) + {b:.4f}")
        print(f"Amplitude = exp((dBA - {b:.4f}) / {a:.4f})")
    
    def dba_to_amplitude(self, target_dba):
        return np.exp((target_dba - self.fit_b) / self.fit_a)


if __name__ == "__main__":
    # Run calibration
    cal = SimpleCalibration("my_calibration.csv")
    cal.run()
    