"""
Sound Calibration
generates one CSV file
DELETE any calibration.csv file before the first run.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from psychopy import sound, core
from psychopy import prefs
prefs.hardware['audioLib'] = ['ptb']
prefs.hardware['audioLatencyMode'] = 2

# prefs.hardware['audioDevice'] = "扬声器 (Realtek(R) Audio)"
# prefs.hardware['audioDevice'] = "Speakers (FiiO K7)"
prefs.hardware['audioDevice'] = "Speakers (2- FiiO K7)" # ALSO CHANGE IN CALIBRATION


class SimpleCalibration:

    def __init__(
        self,
        calibration_file="calibration.csv",
        stimulus_file="calibration_4s.wav" # 4s: 10 repetitions of a fixed 400ms white-noise segment
    ):

        self.calibration_file = calibration_file
        self.calibration_data = None
        self.stimulus_file = stimulus_file

        print(f"Loaded stimulus: {stimulus_file}")

    def measure_dBA(self, amplitude):
        print(f"\n=== Amplitude = {amplitude:.6f} ===")
        tone = sound.Sound(self.stimulus_file)
        tone.setVolume(amplitude)
        print("Playing calibration stimulus...")
        tone.play()
        core.wait(tone.getDuration())
        dB = float(input("Enter dB(A): "))

        return dB

    def run(
        self,
        amplitudes=None,
        n_repeats=1 
    ):
        """
        Run repeated(Here only 1) calibration sessions.
        """

        if amplitudes is None:
            # x-axis: log10(amplitude), to get equal x intervals
            '''
             == np.logspace(np.log10(0.003), np.log10(0.1),20)
            '''
            amplitudes = np.geomspace(0.004,0.4,20).tolist()

        print("\n=== Calibration ===")
        print(f"{n_repeats} repeats")
        print(f"{len(amplitudes)} amplitudes per repeat")

        if os.path.exists(self.calibration_file):

            old_df = pd.read_csv(self.calibration_file)
            data = old_df.to_dict("records")
            print(f"Loaded {len(old_df)} existing points.")
            print(f"Current total points: {len(data)}")

        else: # no existing file

            data = []

        fig, ax = plt.subplots(figsize=(8, 6))

        ## read from calibration.csv
        if len(data) > 0:
            start_repeat = (max(row["repeat"]for row in data) + 1)
        else:
            start_repeat = 1

        for repeat in range(start_repeat, start_repeat + n_repeats): # eg, the 2nd time to run: range(2,3) => repeat==2
            print(f"\n========== Repeat {repeat} ==========")
            repeat_data = []
            for i, amp in enumerate(amplitudes, 1):
                print(f"\nPoint {i}/{len(amplitudes)}")
                dB = self.measure_dBA(amp)

                row = {
                    "repeat": repeat,
                    "amplitude": amp,
                    "log_amp": np.log10(amp),
                    "dBA": dB
                }

                data.append(row)

                repeat_data.append(row) # record the new round data

            repeat_df = pd.DataFrame(repeat_data)

            x = repeat_df["log_amp"].values
            y = repeat_df["dBA"].values
            ### FIT (for this repeat as a preview)
            a, b = np.polyfit(x,y,1) ## y = a*x + b
            ### PLOT SMOOTH LINE
            x_smooth = np.linspace(x.min(),x.max(),100)
            y_smooth = (a * x_smooth + b)

            ax.plot(
                x_smooth,
                y_smooth,
                "--",
                color="gray", ## gray line preview only for this repeat
                alpha=0.5,
                linewidth=1.5,
                label=(
                    "Preview fits"
                    if repeat == 1
                    else None
                )
            )

        self.calibration_data = pd.DataFrame(data)

        self.calibration_data.to_csv(
            self.calibration_file,
            index=False
        )

        print("\n=== Done ===")
        print(f"Saved to: {self.calibration_file}")
        ### DRAW
        self.plot(ax)

        return self.calibration_data

    def plot(self, ax=None):
        if self.calibration_data is None:
            return
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        x = self.calibration_data["log_amp"].values
        y = self.calibration_data["dBA"].values

        ax.scatter(
            x,
            y,
            s=40,
            c="blue", # blue dots
            alpha=0.6,
            label="Measured"
        )

        a, b = np.polyfit(x,y,1)

        self.fit_a = a
        self.fit_b = b
        x_smooth = np.linspace(x.min(), x.max(), 100)
        y_smooth = (a * x_smooth + b)
        y_pred = (a * x + b)

        r2 = 1 - (np.sum((y - y_pred) ** 2)/np.sum((y - np.mean(y)) ** 2))

        print(f"\nFinal R² = {r2:.4f}")

        ax.plot(
            x_smooth,
            y_smooth,
            "r-", # final red line
            linewidth=3,
            label=(
                f"Final fit: "
                f"dBA = {a:.4f}*log10(amplitude) + {b:.4f}"
            )
        )

        ax.set_xlabel("log10(Amplitude)")
        ax.set_ylabel("dBA")
        ax.set_title("Sound Calibration")

        ax.legend()
        ax.grid(True,alpha=0.3)

        plot_file = (
            self.calibration_file.replace(
                ".csv",
                "_plot.png"
            )
        )

        fig.savefig(plot_file,dpi=300)
        print(f"Plot saved: {plot_file}")
        print(f"dBA = {a:.4f} * log10(amplitude) + {b:.4f}")
        print(f"Amplitude = 10 ** ((dBA - {b:.4f}) / {a:.4f})")

        fit_file = (
            self.calibration_file.replace(
                ".csv",
                "_fit.txt" # save a txt file
            )
        )

        with open(fit_file,"w") as f:
            f.write(f"{self.fit_a}\n")
            f.write(f"{self.fit_b}\n")

        print(f"Calibration parameters saved to {fit_file}")

        plt.show()

    def dba_to_amplitude(self,target_dba):
        return 10 ** ((target_dba - self.fit_b)/self.fit_a)


if __name__ == "__main__":

    cal = SimpleCalibration(
        "calibration.csv",
        "calibration_4s.wav"
    )

    cal.run()