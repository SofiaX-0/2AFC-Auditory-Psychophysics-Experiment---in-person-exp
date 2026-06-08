# - For calibration: 0.4s white noise, sampleRate = 44000 Hz

import numpy as np
from scipy.io.wavfile import write

# ==================================================
# Fixed seed
# ==================================================

np.random.seed(42)

# ==================================================
# Basic signal utilities
# ==================================================

def rms(x):
    return np.sqrt(np.mean(x**2))


def rms_normalise(x):
    return x / (rms(x) + 1e-12)


# ==================================================
# Parameters
# ==================================================

Fs = 44000

STIMULUS_DURATION = 0.4      # 400 ms
N_REPEATS = 10               # calibration playback

A1_sigma = 1.0


# ==================================================
# Generate fixed 400 ms white noise
# ==================================================

n_samples = int(STIMULUS_DURATION * Fs)

stimulus = np.random.randn(n_samples)

# RMS normalisation
stimulus = rms_normalise(stimulus)

# apply amplitude scale
stimulus = stimulus * A1_sigma


# ==================================================
# Generate calibration stimulus
# ==================================================

calibration = np.tile(
    stimulus,
    N_REPEATS
)

# ==================================================
# Save 400 ms stimulus
# ==================================================

stimulus_wav = np.int16(
    stimulus / np.max(np.abs(stimulus)) * 32767
)

write(
    "stimulus_400ms.wav",
    Fs,
    stimulus_wav
)

# ==================================================
# Save calibration WAV
# ==================================================

calibration_wav = np.int16(
    calibration / np.max(np.abs(calibration)) * 32767
)

write(
    "calibration_4s.wav",
    Fs,
    calibration_wav
)

# ==================================================
# Diagnostics
# ==================================================

print("Saved: stimulus_400ms.wav")
print("Saved: calibration_4s.wav")

print()
print(f"Sample rate: {Fs} Hz")
print(f"Stimulus duration: {STIMULUS_DURATION:.1f} s")
print(f"Calibration duration: {STIMULUS_DURATION * N_REPEATS:.1f} s")
print(f"Repeats: {N_REPEATS}")

print()
print(f"Stimulus RMS: {rms(stimulus):.6f}")
print(f"Calibration RMS: {rms(calibration):.6f}")