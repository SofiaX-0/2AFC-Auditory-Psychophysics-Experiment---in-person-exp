"""
Calibration Viewer GUI
Load calibration file and convert amplitude to dB SPL
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import json
import numpy as np
import os

class CalibrationViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Sound Calibration Viewer")
        self.root.geometry("500x400")
        
        self.calibration_data = None
        self.amplitude_to_dB = None
        self.mapping_type = None
        self.mapping_params = None
        
        self.create_widgets()
    
    def create_widgets(self):
        # File loading frame
        file_frame = tk.LabelFrame(self.root, text="Load Calibration", padx=10, pady=10)
        file_frame.pack(pady=10, padx=20, fill="x")
        
        self.file_label = tk.Label(file_frame, text="No calibration file loaded", fg="gray")
        self.file_label.pack(side=tk.LEFT, padx=5)
        
        load_btn = tk.Button(file_frame, text="Load CSV", command=self.load_calibration)
        load_btn.pack(side=tk.RIGHT, padx=5)
        
        # Calibration info frame
        info_frame = tk.LabelFrame(self.root, text="Calibration Info", padx=10, pady=10)
        info_frame.pack(pady=10, padx=20, fill="x")
        
        self.info_text = tk.Label(info_frame, text="No calibration loaded", font=("Arial", 10))
        self.info_text.pack()
        
        # Conversion frame
        convert_frame = tk.LabelFrame(self.root, text="Amplitude to dB SPL Converter", padx=10, pady=10)
        convert_frame.pack(pady=10, padx=20, fill="both", expand=True)
        
        # Input
        input_frame = tk.Frame(convert_frame)
        input_frame.pack(pady=10)
        
        tk.Label(input_frame, text="Amplitude (0-1):", font=("Arial", 12)).pack(side=tk.LEFT, padx=5)
        self.amp_entry = tk.Entry(input_frame, width=15, font=("Arial", 12))
        self.amp_entry.pack(side=tk.LEFT, padx=5)
        self.amp_entry.bind('<Return>', self.convert)
        
        convert_btn = tk.Button(input_frame, text="Convert", command=self.convert, 
                                bg="blue", fg="white", font=("Arial", 10))
        convert_btn.pack(side=tk.LEFT, padx=10)
        
        # Output
        output_frame = tk.Frame(convert_frame)
        output_frame.pack(pady=10)
        
        tk.Label(output_frame, text="dB SPL:", font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=5)
        self.dB_label = tk.Label(output_frame, text="---", font=("Arial", 16, "bold"), fg="green")
        self.dB_label.pack(side=tk.LEFT, padx=5)
        
        # Range display
        range_frame = tk.Frame(convert_frame)
        range_frame.pack(pady=10)
        
        self.range_label = tk.Label(range_frame, text="", font=("Arial", 9), fg="gray")
        self.range_label.pack()
        
        # Quick reference frame
        quick_frame = tk.LabelFrame(self.root, text="Quick Reference", padx=10, pady=10)
        quick_frame.pack(pady=10, padx=20, fill="x")
        
        self.quick_text = tk.Text(quick_frame, height=6, width=50)
        self.quick_text.pack()
    
    def load_calibration(self):
        """Load calibration CSV file"""
        file_path = filedialog.askopenfilename(
            title="Select Calibration CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            # Load CSV
            self.calibration_data = pd.read_csv(file_path)
            
            # Try to load corresponding mapping JSON
            json_path = file_path.replace('.csv', '_mapping.json')
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    mapping = json.load(f)
                self.mapping_type = mapping.get('mapping_type')
                self.mapping_params = mapping.get('mapping_params')
                
                # Recreate mapping function
                self._create_mapping_function()
            
            # Update UI
            self.file_label.config(text=os.path.basename(file_path), fg="green")
            
            # Show info
            min_dB = self.calibration_data['dB_SPL'].min()
            max_dB = self.calibration_data['dB_SPL'].max()
            min_amp = self.calibration_data['amplitude'].min()
            max_amp = self.calibration_data['amplitude'].max()
            
            info_str = f"Range: amplitude [{min_amp:.3f}, {max_amp:.3f}] -> dB [{min_dB:.1f}, {max_dB:.1f}]"
            if self.mapping_type:
                info_str += f"\nMapping: {self.mapping_type}"
            self.info_text.config(text=info_str)
            
            # Update quick reference
            self._update_quick_reference()
            
            messagebox.showinfo("Success", f"Loaded calibration from {os.path.basename(file_path)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load calibration: {e}")
    
    def _create_mapping_function(self):
        """Recreate mapping function from saved parameters"""
        if self.mapping_type == "linear":
            a, b = self.mapping_params
            self.amplitude_to_dB = lambda amp: a * amp + b
        elif self.mapping_type == "quadratic":
            a, b, c = self.mapping_params
            self.amplitude_to_dB = lambda amp: a * amp**2 + b * amp + c
        elif self.mapping_type == "exponential":
            a, b, c = self.mapping_params
            self.amplitude_to_dB = lambda amp: a * np.exp(b * amp) + c
        else:
            # Fallback to interpolation
            self.amplitude_to_dB = None
    
    def _update_quick_reference(self):
        """Update quick reference table"""
        if self.calibration_data is None:
            return
        
        self.quick_text.delete(1.0, tk.END)
        
        # Show a few reference points
        ref_amplitudes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        self.quick_text.insert(tk.END, "Amplitude -> dB SPL (interpolated):\n")
        
        for amp in ref_amplitudes:
            if self.amplitude_to_dB:
                dB = self.amplitude_to_dB(amp)
            else:
                # Linear interpolation from data
                dB = np.interp(amp, self.calibration_data['amplitude'], self.calibration_data['dB_SPL'])
            self.quick_text.insert(tk.END, f"  {amp:.1f} -> {dB:.1f} dB\n")
    
    def convert(self, event=None):
        """Convert amplitude to dB SPL"""
        if self.calibration_data is None:
            messagebox.showwarning("Warning", "Please load calibration file first")
            return
        
        try:
            amp = float(self.amp_entry.get())
            
            if amp < 0 or amp > 1:
                self.dB_label.config(text="Invalid (0-1)", fg="red")
                return
            
            # Get dB value
            if self.amplitude_to_dB:
                dB = self.amplitude_to_dB(amp)
            else:
                # Linear interpolation
                dB = np.interp(amp, self.calibration_data['amplitude'], self.calibration_data['dB_SPL'])
            
            # Check range
            min_dB = self.calibration_data['dB_SPL'].min()
            max_dB = self.calibration_data['dB_SPL'].max()
            
            if dB < min_dB or dB > max_dB:
                self.dB_label.config(text=f"{dB:.1f} (extrapolated)", fg="orange")
                self.range_label.config(text=f"Range: {min_dB:.1f}-{max_dB:.1f} dB")
            else:
                self.dB_label.config(text=f"{dB:.1f} dB", fg="green")
                self.range_label.config(text="")
            
        except ValueError:
            self.dB_label.config(text="Invalid number", fg="red")

def main():
    root = tk.Tk()
    app = CalibrationViewer(root)
    root.mainloop()

if __name__ == "__main__":
    main()
        
