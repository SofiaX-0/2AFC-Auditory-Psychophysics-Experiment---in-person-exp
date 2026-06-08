"""
Sound Calibration
generates one CSV file
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from psychopy import sound, core
from psychopy import prefs
prefs.hardware['audioLib'] = ['ptb']
prefs.hardware['audioLatencyMode'] = 2
# prefs.hardware['audioDevice'] = 'Speakers (8- US-4x4)' --> used in our lab
prefs.hardware['audioDevice'] = "扬声器 (Realtek(R) Audio)"
# prefs.hardware['audioDevice'] = "Speakers (FiiO K7)" --> used in our lab


class SimpleCalibration:

    def __init__(
        self,
        calibration_file="calibration.csv",
        stimulus_file="calibration_4s.wav"
    ):

        self.calibration_file = calibration_file
        self.calibration_data = None

        self.stimulus_file = stimulus_file

        print(
            f"Loaded stimulus: {stimulus_file}"
        )

    def measure_dBA(self, amplitude):

        print(
            f"\n=== Amplitude = {amplitude:.6f} ==="
        )

        tone = sound.Sound(
            self.stimulus_file
        )

        tone.setVolume(
            amplitude
        )

        print(
            "Playing calibration stimulus..."
        )

        tone.play()

        core.wait(
            tone.getDuration()
        )

        dB = float(
            input("Enter dB(A): ")
        )

        return dB

    def run(
        self,
        amplitudes=None,
        n_repeats=5
    ):
        """
        Run repeated calibration sessions.
        """

        if amplitudes is None:

            amplitudes = np.geomspace(
                0.001,
                0.005,
                20
            ).tolist()

        print("\n=== Calibration ===")

        print(
            f"{n_repeats} repeats"
        )

        print(
            f"{len(amplitudes)} amplitudes per repeat"
        )

        data = []

        fig, ax = plt.subplots(
            figsize=(8, 6)
        )

        for repeat in range(
            1,
            n_repeats + 1
        ):

            print(
                f"\n========== Repeat {repeat}/{n_repeats} =========="
            )

            repeat_data = []

            for i, amp in enumerate(
                amplitudes,
                1
            ):

                print(
                    f"\nPoint {i}/{len(amplitudes)}"
                )

                dB = self.measure_dBA(
                    amp
                )

                row = {
                    "repeat": repeat,
                    "amplitude": amp,
                    "log_amp": np.log10(amp),
                    "dBA": dB
                }

                data.append(row)

                repeat_data.append(row)

            repeat_df = pd.DataFrame(
                repeat_data
            )

            x = repeat_df[
                "log_amp"
            ].values

            y = repeat_df[
                "dBA"
            ].values

            a, b = np.polyfit(
                x,
                y,
                1
            )

            x_smooth = np.linspace(
                x.min(),
                x.max(),
                100
            )

            y_smooth = (
                a * x_smooth + b
            )

            ax.plot(
                x_smooth,
                y_smooth,
                "--",
                color="gray",
                alpha=0.5,
                linewidth=1.5,
                label=(
                    "Preview fits"
                    if repeat == 1
                    else None
                )
            )

            input(
                "\nPress Enter to continue to next repeat..."
            )

        self.calibration_data = pd.DataFrame(
            data
        )

        self.calibration_data.to_csv(
            self.calibration_file,
            index=False
        )

        print("\n=== Done ===")

        print(
            f"Saved to: {self.calibration_file}"
        )

        self.plot(ax)

        return self.calibration_data

    def plot(self, ax=None):

        if self.calibration_data is None:
            return

        if ax is None:

            fig, ax = plt.subplots(
                figsize=(8, 6)
            )

        else:

            fig = ax.figure

        x = self.calibration_data[
            "log_amp"
        ].values

        y = self.calibration_data[
            "dBA"
        ].values

        ax.scatter(
            x,
            y,
            s=40,
            c="blue",
            alpha=0.6,
            label="Measured"
        )

        a, b = np.polyfit(
            x,
            y,
            1
        )

        self.fit_a = a
        self.fit_b = b

        x_smooth = np.linspace(
            x.min(),
            x.max(),
            100
        )

        y_smooth = (
            a * x_smooth + b
        )

        y_pred = (
            a * x + b
        )

        r2 = 1 - (
            np.sum(
                (y - y_pred) ** 2
            )
            /
            np.sum(
                (y - np.mean(y)) ** 2
            )
        )

        print(
            f"\nFinal R² = {r2:.4f}"
        )

        ax.plot(
            x_smooth,
            y_smooth,
            "r-",
            linewidth=3,
            label=(
                f"Final fit: "
                f"dBA = {a:.4f}*log10(amplitude) + {b:.4f}"
            )
        )

        ax.set_xlabel(
            "log10(Amplitude)"
        )

        ax.set_ylabel(
            "dBA"
        )

        ax.set_title(
            "Sound Calibration"
        )

        ax.legend()

        ax.grid(
            True,
            alpha=0.3
        )

        plot_file = (
            self.calibration_file.replace(
                ".csv",
                "_plot.png"
            )
        )

        fig.savefig(
            plot_file,
            dpi=300
        )

        print(
            f"Plot saved: {plot_file}"
        )

        print(
            f"dBA = {a:.4f} * log10(amplitude) + {b:.4f}"
        )

        print(
            f"Amplitude = 10 ** ((dBA - {b:.4f}) / {a:.4f})"
        )

        fit_file = (
            self.calibration_file.replace(
                ".csv",
                "_fit.txt"
            )
        )

        with open(
            fit_file,
            "w"
        ) as f:

            f.write(
                f"{self.fit_a}\n"
            )

            f.write(
                f"{self.fit_b}\n"
            )

        print(
            f"Calibration parameters saved to {fit_file}"
        )

        plt.show()

    def dba_to_amplitude(
        self,
        target_dba
    ):

        return 10 ** (
            (
                target_dba
                - self.fit_b
            )
            /
            self.fit_a
        )


if __name__ == "__main__":

    cal = SimpleCalibration(
        "my_calibration.csv",
        "calibration_4s.wav"
    )

    cal.run()