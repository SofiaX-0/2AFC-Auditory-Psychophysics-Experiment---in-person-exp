"""
final result: 0.004 - 0.4
0.4  around 90dba
0.3 around 87dba
0.2 around 83dba
0.15170760762929 aroun 80dba
TO TEST CALIBRATION around 80dba
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from psychopy import sound, core
from psychopy import prefs

# import sounddevice as sd
# print('sound device:', sd.default.device)
# for i, dev in enumerate(sd.query_devices()):
#     print (i, dev['name'])


prefs.hardware['audioLib'] = ['ptb']
prefs.hardware['audioLatencyMode'] = 3
# prefs.hardware['audioDevice'] = "扬声器 (Realtek(R) Audio)"
prefs.hardware['audioDevice'] = "Speakers (2- FiiO K7)"


def test(file, amplitude):
        print(f"\n=== Amplitude = {amplitude:.6f} ===")
        tone = sound.Sound(file)
        tone.setVolume(amplitude)
        print("Playing calibration stimulus...")
        tone.play()
        core.wait(tone.getDuration())
        dB = 19.3675 * math.log10(amplitude) + 97.0546 ### change the fn here
        print(dB)

        return dB


if __name__ == "__main__":
    amplitude = float(input('ENTER AMP: '))
    test('calibration_4s.wav', amplitude)