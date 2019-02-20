'''
Convolves two signals. Used to apply a Room Impulse Response to audio.
'''

from scipy.io import wavfile
import numpy as np
import time

def conv_circ( signal, ker ):
    '''
        signal: real 1D array
        ker: real 1D array
        signal and ker must have same shape
    '''
    t0 = time.perf_counter()
    result = np.real(np.fft.ifft( np.fft.fft(signal)*np.fft.fft(ker) ))
    t1 = time.perf_counter()
    print('time taken:', t1-t0)
    return result, t1-t0

fs_rir, rir = wavfile.read('/Users/ericcarb/Downloads/MARDY/ir_3_L_7.wav') #ir_1_C_1.wav')
fs_audio, audio = wavfile.read('/Users/ericcarb/p254_010_48k.wav') #speaker_48k.wav')
audio_duration = len(audio)/fs_audio
print('audio duration', audio_duration)
rir = rir.astype(float)
audio = audio.astype(float)
assert fs_rir == fs_audio

if audio.shape[0] > rir.shape[0]:
	padding = np.zeros(audio.shape[0] - rir.shape[0])
	new_rir = np.concatenate((rir, padding))
	new_audio = audio
else:
	padding = np.zeros(rir.shape[0] - audio.shape[0])
	new_audio = np.concatenate((audio, padding))
	new_rir = rir

reverb_audio, t = conv_circ(new_audio, new_rir)
time_scale_factor = t/audio_duration
print('time scale factor', time_scale_factor, time_scale_factor**(-1))

# normalize (approximately)
reverb_audio /= max(reverb_audio)

wavfile.write('/Users/ericcarb/p254_010_ir_3_L_7.wav', fs_audio, reverb_audio)
