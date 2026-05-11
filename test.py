from psychopy import sound
from psychopy import core, event, visual, logging, prefs, gui
import os
prefs.hardware['audioLib'] = ['sounddevice', 'ptb', 'pyo']
prefs.hardware['audioLatencyMode'] = 0
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # get current path of this script
RESULT_PATH = os.path.join(SCRIPT_DIR, "results")

sound_file = os.path.join(SCRIPT_DIR, "400Hz_300ms_10ms_fadeinout_44100rate.wav")
probe = sound.Sound (value = sound_file, stereo = True, sampleRate = 44100)
probe.setVolume(1)
probe.play()

