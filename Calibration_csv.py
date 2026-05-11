"""
Simple Sound Calibration
Only generates one CSV file, no JSON
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from psychopy import sound, core
import os
import tkinter as tk

class SimpleCalibration:
    def __init__(self, calibration_file="calibration.csv"):
        self.calibration_file = calibration_file
        self.calibration_data = None
    
    def generate_white_noise(self, amplitude, duration=0.5, sample_rate=44100):
        """Build Gaussian white noise, peak-normalize, scale by amplitude, return PsychoPy Sound."""
        n_samples = int(duration * sample_rate)
        signal = np.random.normal(0, 1, n_samples)
        max_val = np.max(np.abs(signal))
        if max_val > 0:
            signal = signal / max_val  # [-1, 1] before scaling so amplitude is interpretable
        signal = signal * amplitude  # user-controlled level for SPL calibration
        signal = signal.astype(np.float32) # convert to float32 for PsychoPy
        return sound.Sound(value=signal, stereo=True, sampleRate=sample_rate)  # create PsychoPy Sound object
    
    def measure_dB_A(self, amplitude):
        """Measure dB SPL for a given amplitude."""
        print(f"\n=== Amplitude = {amplitude:.3f} ===")
        tone = self.generate_white_noise(amplitude, duration=5.0)
        print("Playing white noise (5 seconds)...")
        tone.play()
        core.wait(tone.getDuration()) # wait for the tone to finish playing
        dB = float(input(f"Enter dB(A): ")) # ask the user to enter the dB(A)
        return dB # return the dB(A)
    
    def run(self, amplitudes=None):
        """Run the calibration."""
        if amplitudes is None:
            amplitudes = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        print(f"\n=== Calibration ===") # print the calibration title
        print(f"Measure {len(amplitudes)} points") # print the number of points
        
        data = [] # create an empty list to store the data
        for i, amp in enumerate(amplitudes, 1): # loop through the amplitudes
            print(f"\nPoint {i}/{len(amplitudes)}") # print the point number
            dB = self.measure_dB_A(amp) # measure the dBA
            data.append({'amplitude': amp, 'dB_SPL': dB}) # add the data to the list
        self.calibration_data = pd.DataFrame(data)
        self.calibration_data.to_csv(self.calibration_file, index=False)
        
        print(f"\n=== Done ===")
        print(f"Saved to: {self.calibration_file}")
        
        self.plot() # plot the data
        return self.calibration_data
    37
    def plot(self):
        if self.calibration_data is None:
            return # if the data is not available, return
        
        fig, ax = plt.subplots(figsize=(8, 6)) # create a figure and axis
        
        x = self.calibration_data['amplitude'].values
        y = self.calibration_data['dB_SPL'].values # change pandas data to numpy array
        
        ax.scatter(x, y, s=50, c='blue', label='Measured') # plot the data; s is the size of the points, c is the color, label is the label of the points
        
        # Linear fit
        a, b = np.polyfit(x, y, 1) # fit a linear model to the data
        x_smooth = np.linspace(0, 1, 100) # create a smooth curve, np.linspace(start, end, n)- generate n points between start and end evenly
        y_smooth = a * x_smooth + b # calculate the y values for the smooth curve
        ax.plot(x_smooth, y_smooth, 'r-', linewidth=2, label=f'Linear fit: dB = {a:.1f}*amp + {b:.1f}')
        
        ax.set_xlabel('Amplitude') # set the x label
        ax.set_ylabel('dB SPL') # set the y label
        ax.set_title('Sound Calibration') # set the title
        ax.legend() # show the legend
        ax.grid(True, alpha=0.3) # show the grid
        
        plot_file = self.calibration_file.replace('.csv', '_plot.png') # create the plot file name
        fig.savefig(plot_file, dpi=300) # save the plot
        plt.show() # show the plot
        print(f"Plot saved: {plot_file}") # print the plot file name


# Simple GUI viewer
class SimpleViewer:
    def __init__(self, root):
        # root is the main window of the application
        self.root = root
        self.root.title("Calibration Viewer")
        self.root.geometry("400x300")
        
        self.data = None
        
        # Load button
        tk.Button(root, text="Load CSV", command=self.load_csv, 
                 bg="blue", fg="white", font=("Arial", 12)).pack(pady=10)
        
        self.file_label = tk.Label(root, text="No file loaded", fg="gray")
        self.file_label.pack()
        
        # Converter
        frame = tk.LabelFrame(root, text="Convert Amplitude to dB", padx=10, pady=10)
        frame.pack(pady=20, padx=20, fill="both")
        
        tk.Label(frame, text="Amplitude (0-1):").pack()
        self.amp_entry = tk.Entry(frame, font=("Arial", 14))
        self.amp_entry.pack(pady=5)
        self.amp_entry.bind('<Return>', self.convert)
        
        tk.Button(frame, text="Convert", command=self.convert).pack(pady=5)
        
        self.result_label = tk.Label(frame, text="---", font=("Arial", 18, "bold"), fg="green")
        self.result_label.pack(pady=10)
    
    def load_csv(self):
        # open a file dialog to select a CSV file
        from tkinter import filedialog
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        # if the file is selected, read the file and store the data in self.data
        if path:
            self.data = pd.read_csv(path)
            self.file_label.config(text=os.path.basename(path), fg="green")
            # print the number of points and the range of the data
            print(f"Loaded: {len(self.data)} points, range: {self.data['dB_SPL'].min():.0f}-{self.data['dB_SPL'].max():.0f} dB")
    
    def convert(self, event=None):
        # if the data is not loaded, print an error message
        if self.data is None:
            self.result_label.config(text="Load CSV first", fg="red")
            return
        
        try:
            # get the amplitude from the entry field
            amp = float(self.amp_entry.get())
            # if the amplitude is not between 0 and 1, print an error message
            if amp < 0 or amp > 1:
                self.result_label.config(text="0-1 only", fg="red")
                return
            
            # Linear interpolation
            dB = np.interp(amp, self.data['amplitude'], self.data['dB_SPL'])
            self.result_label.config(text=f"{dB:.1f} dB", fg="green")
        except:
            # if the amplitude is not a number, print an error message
            self.result_label.config(text="Invalid", fg="red")


if __name__ == "__main__":
    # Run calibration
    cal = SimpleCalibration("my_calibration.csv")
    cal.run()
    
    # Or run viewer only
    # root = tk.Tk()
    # viewer = SimpleViewer(root)
    # root.mainloop()