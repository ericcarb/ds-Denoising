'''
README

A basic script for running t-SNE and hyperopt on mfcc features extracted from denoised audio.
Simply searches for best t-SNE parameters for given results and accompanying audio wave files and
plots the embedding using best found parameters.
Directories containing the wave files are coded as strings in the main function.

=====
(all_results.txt files are the combination of out.txt and dout.txt from get_performance_by_words_dfs.py)
all_results.txt contains benchmark results of clean, noisy and denoised audio,
which looks like:

p243_009_opera_clapping
		SNR	Acc.		Prec.
  noisy		-15SNR	NaN	NaN
  denoised	-15SNR	0.0	0.0
  denoised v2	-15SNR	0.0	0.0

  noisy		-10SNR	NaN	NaN
  denoised	-10SNR	7.692308	25.0
  denoised v2	-10SNR	7.692308	20.0

  noisy		-5SNR	NaN	NaN
  denoised	-5SNR	7.692308	20.0
  denoised v2	-5SNR	0.0	0.0

  noisy		0SNR	NaN	NaN
  denoised	0SNR	7.692308	7.692308
  denoised v2	0SNR	0.0	0.0

  noisy		5SNR	30.769231	30.769231
  denoised	5SNR	0.0	0.0
  denoised v2	5SNR	0.0	0.0

  noisy		10SNR	61.538462	61.538462
  denoised	10SNR	23.076923	30.0
  denoised v2	10SNR	23.076923	30.0


p236_006_opera_clapping
  ...
=====

'''

from scipy.signal import spectrogram
from scipy.io import wavfile
import numpy as np
import os
from collections import defaultdict
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
import librosa
import math


def get_results_from_txt (filepath):
	'''
	Reads an all_results.txt file and returns a dictionary its contents
	'''
	with open(filepath, 'r') as inf:
		lines = inf.readlines()
	results = defaultdict(dict)
	noisy_filename_dict = {}
	denoised_filename_dict = {}
	denoisedv2_filename_dict = {}
	for line in lines:
		if line[0] == 'p':
			if len(noisy_filename_dict) > 0:
				results['noisy'][filename] = noisy_filename_dict
				results['denoised'][filename] = denoised_filename_dict
				results['denoisedv2'][filename] = denoisedv2_filename_dict
			filename = line.rstrip()
			noisy_filename_dict = {}
			denoised_filename_dict = {}
			denoisedv2_filename_dict = {}
		elif line.startswith('  noisy'):
			parts = line.split()
			snr = parts[1]
			acc = parts[2]
			prec = parts[3]
			noisy_filename_dict[snr] = (acc, prec)
		elif line.startswith('  denoised') and 'v2' not in line:
			parts = line.split()
			snr = parts[1]
			acc = parts[2]
			prec = parts[3]
			denoised_filename_dict[snr] = (acc, prec)
		elif line.startswith('  denoised v2'):
			parts = line.split()
			snr = parts[2]
			acc = parts[3]
			prec = parts[4]
			denoisedv2_filename_dict[snr] = (acc, prec)
	return results

def normalize (signal, level=0.5):
	'''
	Divides all samples by the max value, scaled to level
	(the maximum amplitude of abs(any sample) will be equal to level)
	(this is not the best way to normalize, see NN Denoiser Complete Breakdown on Confluence)

	signal should be a numpy array of floats
	'''
	normalized = signal / (max(abs(signal)) / level)
	return normalized

def convert_pcm (signal, level=0.5):
	'''
	converts int array to float array and normalizes the array
	'''
	if signal.dtype == np.dtype('int64') or signal.dtype == np.dtype('int32') or signal.dtype == np.dtype('int16'):
		# convert int to range [-1,1]
		converted = np.float32(signal) / 32767
	else:
		converted = np.float32(np.array(signal))
	return normalize(converted, level)

def difference (clean_signal, denoised_signal):
	'''
	Just subtracts two arrays
	'''
	return clean_signal - denoised_signal

def filepaths_dict (clean_dir, denoised_dirs):
	'''
	Associates the filepath of each clean speech wave file to
	the filepaths of denoised wave files containing the clean speech

	denoised_dirs should be a list of full filepaths to the directories containing the denoised wav files
	clean_dir should be the directory containing all the clean files
	'''
	result = defaultdict(list)
	all_denoised = []
	for denoised_dir in denoised_dirs:
		all_denoised.extend([os.path.join(denoised_dir, f) for f in os.listdir(denoised_dir)])
	for clean in os.listdir(clean_dir):
		name = clean.replace('.wav', '')
		for denoised in all_denoised:
			if denoised.split('/')[-1].startswith(name):
				result[os.path.join(clean_dir, clean)].append(denoised)
	return result

def zero_fill (array, target_size):
	'''
	pads the left and right sides of a 1D array with zeros
	'''
	assert len(array.shape) == 1
	target_size -= array.shape[0]
	r = target_size % 2
	lower = target_size // 2
	upper = lower + r
	# shouldn't need below two statements
	if lower < 0 or upper < 0:
		return array
	return np.concatenate((np.zeros(lower), array, np.zeros(upper)))

def get_features(y, sr):
	'''
	from ml4a/AudioTSNEViewer
	'''
	y = y[int(sr):int(sr*2)] 	# analyze just second second
	S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
	log_S = librosa.amplitude_to_db(S, ref=np.max)
	mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
	delta_mfcc = librosa.feature.delta(mfcc, mode='nearest')
	delta2_mfcc = librosa.feature.delta(mfcc, order=2, mode='nearest')
	feature_vector = np.concatenate((np.mean(mfcc,1), np.mean(delta_mfcc,1), np.mean(delta2_mfcc,1)))
	feature_vector = (feature_vector-np.mean(feature_vector)) / np.std(feature_vector)
	return feature_vector

def run_tsne (X, n_components=2, perplexity=30.0, early_exaggeration=12.0, learning_rate=200.0, n_iter=1000, n_iter_without_progress=300, min_grad_norm=1e-07, metric='euclidean', init='random', verbose=0, random_state=42, method='barnes_hut', angle=0.5):
	'''
	calls t-SNE and returns a loss metric defined in https://arxiv.org/pdf/1708.03229.pdf
	'''
	tsne = TSNE(n_components=n_components, perplexity=perplexity, early_exaggeration=early_exaggeration, learning_rate=learning_rate, n_iter=n_iter, n_iter_without_progress=n_iter_without_progress, min_grad_norm=min_grad_norm, metric=metric, init=init, verbose=verbose, random_state=random_state, method=method, angle=angle)
	embedded = tsne.fit_transform(X)
	to_minimize = 2*tsne.kl_divergence_ + (math.log(X.shape[0])*perplexity/X.shape[0])
	return to_minimize, embedded

def f(params):
	'''
	the minimization function passed to hyperopt
	just calls t-SNE and returns a dictionary to hyperopt.fmin
	'''
	p = params['perplexity']
	e = params['early_exaggeration']
	lr = params['learning_rate']
	a = params['angle']
	X = params['X']
	loss, e = run_tsne(X, perplexity=p, early_exaggeration=e, learning_rate=lr, angle=a)
	print('returning', loss)
	'''if loss < 1e-4:
		loss = 1'''
	return {'loss': loss, 'status': STATUS_OK}

def hypo (X, space):
	'''
	calls hyperopt.fmin and returns the best parameters

	need to add X into the parameter space so it will be passed to f and then to run_tsne
	'''
	space['X'] = X
	trials = Trials()
	best = fmin(fn=f, space=space, algo=tpe.suggest, max_evals=10, trials=trials)
	'''print('best:', best)
	print('trials:')
	for trial in trials.trials[:2]:
		print(trial)'''
	return space_eval(space, best)


def main (index_to_color):
	'''
	Performs hyperopt search and plots t-SNE results
	'''
	clean_dir = '/Users/ericcarb/testing_clean/'
	music_denoised_dir = '/Users/ericcarb/denoised_uk_testset_music_120e/'
	locomotive_denoised_dir = '/Users/ericcarb/denoised_uk_testset_locomotive_80e/'
	clapping_denoised_dir = '/Users/ericcarb/denoised_uk_testset_clapping_80e/'
	vacuuming_denoised_dir = '/Users/ericcarb/denoised_uk_testset_vacuuming_80e/'
	filepaths = filepaths_dict(clean_dir, [music_denoised_dir, locomotive_denoised_dir, clapping_denoised_dir, vacuuming_denoised_dir])
	#print(len(filepaths))
	filenames_dict = {}
	all_subtracted_audio_features = np.empty((0, 39))
	color_map = {'denoised_uk_testset_music_120e' : 'g',
				 'denoised_uk_testset_locomotive_80e' : 'r',
				 'denoised_uk_testset_clapping_80e' : 'b',
				 'denoised_uk_testset_vacuuming_80e' : 'y'}

	# planned on color-coding based on accuracy/lift
	music_results = get_results_from_txt('/Users/ericcarb/locomotive_uk_all_results.txt')
	locomotive_results = get_results_from_txt('/Users/ericcarb/music_uk_all_results.txt')
	clapping_results = get_results_from_txt('/Users/ericcarb/clapping_uk_all_results.txt')
	vacuuming_results = get_results_from_txt('/Users/ericcarb/vacuuming_uk_all_results.txt')

	for clean_path, denoised_list in filepaths.items():
		for denoised_path in denoised_list:
			fs_clean, clean_audio = wavfile.read(clean_path)
			fs_denoised, denoised_audio = wavfile.read(denoised_path)
			# denoised may have extra samples from FaNT, realigned by denoiser
			if denoised_audio.shape[0] > clean_audio.shape[0]:
				denoised_audio = denoised_audio[:-22]
			
			clean_audio = convert_pcm(clean_audio)
			denoised_audio = convert_pcm(denoised_audio)
			
			subtracted_audio = difference(clean_audio, denoised_audio)
			# make mfcc
			subtracted_audio_features = get_features(subtracted_audio, fs_denoised)
			# cache color
			index_to_color[len(all_subtracted_audio_features)] = color_map[denoised_path.split('/')[3]]
			
			all_subtracted_audio_features = np.append(all_subtracted_audio_features, [subtracted_audio_features], axis=0)
			# should use shape instead of len
			filenames_dict[len(all_subtracted_audio_features)] = (clean_path, denoised_path)

	#print(all_subtracted_audio_features.shape)
	space = {
		'perplexity': hp.choice('perplexity', np.arange(99+1-10)+10), # [10,99]
		'early_exaggeration': hp.uniform('early_exaggeration', 6, 24), # [6, 24]
		'learning_rate': hp.choice('learning_rate', np.arange(499+1-30)+30), # [30, 499]
		'angle': hp.uniform('angle', 0.4, 0.6) # [0.4, 0.6]
	}
	best_params = hypo(all_subtracted_audio_features, space)
	print(best_params)
	kl, embedded = run_tsne(all_subtracted_audio_features, perplexity=best_params['perplexity'], early_exaggeration=best_params['early_exaggeration'], learning_rate=best_params['learning_rate'], angle=best_params['angle'])
	for i, e in enumerate(embedded):
		plt.scatter(e[0], e[1], c=index_to_color[i])
	plt.show()

	return all_subtracted_audio_features, embedded, filenames_dict


if __name__ == '__main__':
	index_to_color = {}
	X, e, names = main(index_to_color)
	x = e[:,0]
	y = e[:,1]
	xstd = np.std(x)
	ystd = np.std(y)
	xmean = np.mean(x)
	ymean = np.mean(y)
	x_range = (xmean-(2*xstd), xmean+(2*xstd))
	y_range = (ymean-(2*ystd), ymean+(2*ystd))
	print('x range:', x_range)
	print('y range:', y_range)
	print('\noutliers:\n')
	for i, exy in enumerate(e):
		xbool = exy[0] < x_range[0] or exy[0] > x_range[1]
		ybool = exy[1] < y_range[0] or exy[1] > y_range[1]
		if xbool or ybool:
			print(exy)
			print(names[i])
			print()
