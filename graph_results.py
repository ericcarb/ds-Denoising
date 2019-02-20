'''
README

Reads an all_results.txt (combination of dout.txt and out.txt from get_performance_by_words_dfs.txt)
and plots noisy vs denoised accuracy.
'''

from collections import defaultdict
from pprint import pprint
import math
import matplotlib.pyplot as plt
import numpy as np

def read_polqa_results (filepath):
	'''
	Reads text output from POLQA.
	'''
	with open(filepath, 'r') as inf:
		lines = inf.readlines()
	results = {} # {filename : (polqa results,)}
	name = ''
	mos = 0.0
	rfactor = 0.0
	for line in lines:
		if 'Processing Time' in line:
			results[name] = (mos, rfactor)
		elif 'Reference File' in line:
			name = line.split()[3][6:-4]
		elif 'MOS-LQO' in line:
			mos = float(line.split()[2])
		elif 'R-Factor' in line:
			rfactor = float(line.split()[2])
	return results

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

def compute_metrics (results):
	'''
	Computes metrics of results.

	lift_snr			{SNR : [ (lift, filename), ]}
	drop_snr			{SNR : [ (drop, filename), ]}
	percentage_lift		number of files with lift (including no change in accuracy) / total number of files
	avg_lift			of files with lift, this is their average lift in accuracy
	avg_drop			of files with drop, this is their average drop in accuracy
	noisy_avg_acc		of all noisy files, this is their average accuracy
	denoised_avg_acc	of all denoised files, this is their average accuracy
	denoisedv2_avg_acc	of all denoisedv2 files, this is their average accuracy
	'''
	lift_snr = defaultdict(list)
	drop_snr = defaultdict(list)
	percentage_lift = 0.0
	num_files = 0
	avg_lift = 0.0
	num_lift = 0
	avg_drop = 0.0
	num_drop = 0
	noisy_avg_acc = 0.0
	num_noisy = 0
	denoised_avg_acc = 0.0
	num_denoised = 0
	denoisedv2_avg_acc = 0.0
	num_denoisedv2 = 0
	for name, snr_dict in results['noisy'].items():
		for snr, acc_prec in snr_dict.items():
			noisy_avg_acc += float(acc_prec[0]) if not math.isnan(float(acc_prec[0])) else 0.0
			num_noisy += 1
			d_acc_prec = results['denoised'][name][snr]
			denoised_avg_acc += float(d_acc_prec[0]) if not math.isnan(float(d_acc_prec[0])) else 0.0
			num_denoised += 1
			d2_acc_prec = results['denoisedv2'][name][snr]
			denoisedv2_avg_acc += float(d2_acc_prec[0]) if not math.isnan(float(d2_acc_prec[0])) else 0.0
			num_denoisedv2 += 1
			diff_acc = float(d2_acc_prec[0]) - float(acc_prec[0])
			if diff_acc >= 0: # lift
				lift_snr[snr].append((diff_acc, name))
				percentage_lift += 1 if not math.isnan(diff_acc) else 0
				avg_lift += diff_acc
				num_lift += 1
			else: # drop
				drop_snr[snr].append((diff_acc, name))
				if not math.isnan(diff_acc): # needed to exclude NaN from avg calculation
					avg_drop += diff_acc
				num_drop += 1
			num_files += 1
	percentage_lift /= num_files
	avg_lift /= num_lift
	avg_drop /= num_drop
	noisy_avg_acc /= num_noisy
	denoised_avg_acc /= num_denoised
	denoisedv2_avg_acc /= num_denoisedv2
	return lift_snr, drop_snr, (percentage_lift, avg_lift, avg_drop), (noisy_avg_acc, denoised_avg_acc, denoisedv2_avg_acc)




#pprint(results['denoisedv2'])

'''
x: file
y: acc (maybe F1-score or recall)
'''

def main ():
	polqa_results = read_polqa_results('/Users/ericcarb/Downloads/for_polqa_results/for_polqa_results/locomotive.txt')
	results = get_results_from_txt('/Users/ericcarb/locomotive_uk_all_results.txt')
	lift, drop, ratios, avg_accs = compute_metrics(results)
	print(ratios)
	print(avg_accs)
	for l in [lift, drop]:
		for k,v in l.items():
			print(k)
			for vv in v:
				print(' ', vv)
		print('-------------------')

	x_neg15 = []
	y_neg15 = []
	i_neg15 = []
	x_neg10 = []
	y_neg10 = []
	i_neg10 = []
	x_neg5 = []
	y_neg5 = []
	i_neg5 = []
	x_0 = []
	y_0 = []
	i_0 = []
	x_5 = []
	y_5 = []
	i_5 = []
	x_10 = []
	y_10 = []
	i_10 = []
	i_to_name = {}
	i = 0
	order = []
	for name, snr_dict in results['noisy'].items():
		for snr, acc_prec in snr_dict.items():
			polqa_name = '{}_{}_'.format(name, snr)
			ss = '_'.join(name.split('_')[:2])
			i_to_name[i] = '{},{}'.format(name, snr)
			if snr == '-15SNR':
				x_neg15.append('{},{}'.format(ss, snr))
				y_neg15.append(float(acc_prec[0]) if acc_prec[0] != 'NaN' else 0.0)
				i_neg15.append(i)
			elif snr == '-10SNR':
				x_neg10.append('{},{}'.format(ss, snr))
				y_neg10.append(float(acc_prec[0]) if acc_prec[0] != 'NaN' else 0.0)
				i_neg10.append(i)
				order.append(i)
			elif snr == '-5SNR':
				x_neg5.append('{},{}'.format(ss, snr))
				y_neg5.append(float(acc_prec[0]) if acc_prec[0] != 'NaN' else 0.0)
				i_neg5.append(i)
			elif snr == '0SNR':
				x_0.append('{},{}'.format(ss, snr))
				y_0.append(float(acc_prec[0]) if acc_prec[0] != 'NaN' else 0.0)
				i_0.append(i)
			elif snr == '5SNR':
				x_5.append('{},{}'.format(ss, snr))
				y_5.append(float(acc_prec[0]) if acc_prec[0] != 'NaN' else 0.0)
				i_5.append(i)
			elif snr == '10SNR':
				x_10.append('{},{}'.format(ss, snr))
				y_10.append(float(acc_prec[0]) if acc_prec[0] != 'NaN' else 0.0)
				i_10.append(i)
			i += 1


	dx_neg15 = []
	dy_neg15 = []
	di_neg15 = []
	dx_neg10 = []
	dy_neg10 = []
	di_neg10 = []
	dx_neg5 = []
	dy_neg5 = []
	di_neg5 = []
	dx_0 = []
	dy_0 = []
	di_0 = []
	dx_5 = []
	dy_5 = []
	di_5 = []
	dx_10 = []
	dy_10 = []
	di_10 = []
	di_to_name = {}
	di = 0
	dorder = []
	p_neg15 = []
	p_neg10 = []
	p_neg5 = []
	p_0 = []
	p_5 = []
	p_10 = []
	for name, snr_dict in results['denoisedv2'].items():
		for snr, acc_prec in snr_dict.items():
			polqa_name = '{}_{}_'.format(name, snr)
			ss = '_'.join(name.split('_')[:2])
			di_to_name[di] = '{},{}'.format(name, snr)
			if snr == '-15SNR':
				dx_neg15.append('{},{}'.format(ss, snr))
				dy_neg15.append(float(acc_prec[0]) if acc_prec[0] != 'NaN' else 0.0)
				#mos, rfactor = polqa_results[polqa_name]
				#p_neg15.append(mos)
				di_neg15.append(di)
			elif snr == '-10SNR':
				dx_neg10.append('{},{}'.format(ss, snr))
				dy_neg10.append(float(acc_prec[0]) if acc_prec[0] != 'NaN' else 0.0)
				#mos, rfactor = polqa_results[polqa_name]
				#p_neg10.append(mos)
				di_neg10.append(di)
				dorder.append(di)
			elif snr == '-5SNR':
				dx_neg5.append('{},{}'.format(ss, snr))
				dy_neg5.append(float(acc_prec[0]) if acc_prec[0] != 'NaN' else 0.0)
				#mos, rfactor = polqa_results[polqa_name]
				#p_neg5.append(mos)
				di_neg5.append(di)
			elif snr == '0SNR':
				dx_0.append('{},{}'.format(ss, snr))
				dy_0.append(float(acc_prec[0]) if acc_prec[0] != 'NaN' else 0.0)
				#mos, rfactor = polqa_results[polqa_name]
				#p_0.append(mos)
				di_0.append(di)
			elif snr == '5SNR':
				dx_5.append('{},{}'.format(ss, snr))
				dy_5.append(float(acc_prec[0]) if acc_prec[0] != 'NaN' else 0.0)
				#mos, rfactor = polqa_results[polqa_name]
				#p_5.append(mos)
				di_5.append(di)
			elif snr == '10SNR':
				dx_10.append('{},{}'.format(ss, snr))
				dy_10.append(float(acc_prec[0]) if acc_prec[0] != 'NaN' else 0.0)
				#mos, rfactor = polqa_results[polqa_name]
				#p_10.append(mos)
				di_10.append(di)
			di += 1

	labels = i_neg15 + i_neg10 + i_neg5 + i_0 + i_5 + i_10 #x_neg15 + x_neg10 + x_neg5 + x_0 + x_5 + x_10 #

	n_groups = len(y_neg15)+len(y_neg10)+len(y_neg5)+len(y_0)+len(y_5)+len(y_10)

	noisy_stats = y_neg15 + y_neg10 + y_neg5 + y_0 + y_5 + y_10

	denoised_stats = dy_neg15 + dy_neg10 + dy_neg5 + dy_0 + dy_5 + dy_10

	fig, ax = plt.subplots()
	fig.set_figwidth(16, forward=True)

	index = np.arange(n_groups)
	bar_width = 0.35

	opacity = 0.4
	error_config = {'ecolor': '0.3'}

	rects1 = ax.bar(index, noisy_stats, bar_width,
	                alpha=opacity, color='b',
	                label='Noisy')

	rects2 = ax.bar(index + bar_width, denoised_stats, bar_width,
	                alpha=opacity, color='r',
	                label='Denoisedv2')

	ax.set_xlabel('File')
	ax.set_ylabel('Accuracy (Speechmatics)')
	ax.set_title('Music Denoised (good valset, 40 epochs) vs. Denoised v2 (good valset, 120 epochs) (sorted by SNR)')
	ax.set_xticks(index + bar_width / 2)
	ax.set_xticklabels(labels)
	ax.legend()

	fig.tight_layout()
	plt.grid(b=None, which='major', axis='y')
	plt.show()


'''
This code plots results from POLQA.

x = [] # mos
y = [] # sm
z = [] # rfactor
yxz = []
for name, mos_r in polqa_results.items():
	mos, rfactor = mos_r
	#x.append(mos)
	short_name = '_'.join(name.split('_')[:-2])
	snr = name.split('_')[-2]
	dacc, dprec = results['denoisedv2'][short_name][snr]
	nacc, nprec = results['noisy'][short_name][snr]
	dacc = float(dacc) if not math.isnan(float(dacc)) else 0.0
	nacc = float(nacc) if not math.isnan(float(nacc)) else 0.0
	lift_acc = dacc - nacc
	print(lift_acc)
	#acc = float(acc)
	#prec = float(prec)
	#y.append(acc)
	yxz.append((nacc, mos, rfactor))

yxz = sorted(yxz)
for yy, xx, zz in yxz:
	x.append(xx)
	y.append(yy)
	z.append(zz)

plt.plot(y, x) #, c='b')
#plt.plot(y, z, c='g')
plt.ylabel('MOS score')
#plt.ylabel('R Factor')
plt.xlabel('SM Noisy Accuracy')
#plt.title('POLQA MOS score vs. Speechmatics Lift/Drop (Music)')
plt.title('POLQA MOS Score vs. Speechmatics Noisy Accuracy (Music)')
plt.show()

==========

lift_results = {
'p227_002_opera_clapping,0SNR' : (1.3315, 36.36),
'p227_002_opera_clapping,5SNR' : (1.8919, 27.27),
'p233_004_opera_clapping,-5SNR' : (0.6203, 13.33),
'p227_002_vacuuming_barbie_things,-10SNR' : (0.0248, 9.09),
'p231_008_vacuuming_barbie_things,10SNR' : (0.4672, 18.18),
'p254_010_vacuuming_barbie_things,5SNR' : (-0.0907, 25.0),
'p254_010_vacuuming_barbie_things,-5SNR' : (0.4781, 25),
'p226_001_beethoven_fur_elise,0SNR' : (1.4255, 0.0), # these were already 100acc, how high were their MOS scores?
'p226_001_beethoven_fur_elise,5SNR' : (1.0667, 0.0), # these were already 100acc, how high were their MOS scores?
'p226_001_beethoven_fur_elise,-10SNR' : (0.8576, 33.33),
'p233_004_beethoven_fur_elise,-10SNR' : (0.1103, 66.66),
'p254_010_beethoven_fur_elise,-5SNR' : (0.6996, 62.5),
'p230_005_real_locomotive,5SNR' : (0.6263, 42.9),
'p236_006_real_locomotive,10SNR' : (0.4249, 52.94),
'p254_010_real_locomotive,5SNR' : (0.347, 50.0),
'p256_007_real_locomotive,0SNR' : (0.206, 25.0)
}
drop_results = {
'p226_001_opera_clapping,10SNR' : (0.0864, -66.67),
'p228_003_opera_clapping,10SNR' : (0.3656, -45.0),
'p243_009_opera_clapping,5SNR' : (0.266, -30.77),
'p243_009_opera_clapping,10SNR' : (0.474, -38.5),
'p227_002_vacuuming_barbie_things,0SNR' : (0.2271, -36.36),
'p228_003_vacuuming_barbie_things,-5SNR' : (0.0383, -5.0),
'p243_009_vacuuming_barbie_things,5SNR' : (-0.3637, -76.92), # need to see why this was so terrible
'p243_009_vacuuming_barbie_things,10SNR' : (-0.4103, -46.15),
'p256_007_vacuuming_barbie_things,0SNR' : (0.8209, -33.33),
'p228_003_beethoven_fur_elise,10SNR' : (0.9523, -5.0),
'p254_010_beethoven_fur_elise,10SNR' : (0.6256, -12.5),
'p256_007_beethoven_fur_elise,10SNR' : (0.9446, -8.33),
'p230_005_real_locomotive,10SNR' : (0.449, -4.76),
'p243_009_real_locomotive,5SNR' : (0.0861, -7.69),
'p256_007_real_locomotive,10SNR' : (0.4406, -25.0),
}

x = [v[0] for k,v in lift_results.items()]
y = [v[1] for k,v in lift_results.items()]
z = np.polyfit(x, y, 2)
f = np.poly1d(z)
x_new = np.linspace(min(x), max(x), 50)
y_new = f(x_new)

optimal_y = np.poly1d(np.polyfit(np.arange(3), np.arange(3)**2*16, 3))(x_new)

plt.plot(x,y,'o', x_new, y_new, x_new, optimal_y)
plt.xlim([min(x_new)-1, max(x_new) + 1 ])
plt.title('Polqa MOS Difference vs Lift in Accuracy (SM)')
plt.xlabel('Change in MOS')
plt.ylabel('Change in Accuracy')
plt.show()

x = [v[0] for k,v in drop_results.items()]+x
y = [v[1] for k,v in drop_results.items()]+y
z = np.polyfit(x, y, 1)
f = np.poly1d(z)
x_new = np.linspace(min(x), max(x), 50)
y_new = f(x_new)

optimal_y = np.poly1d(np.polyfit([-3, -2, -1, 0, 1, 2, 3], [-100, -66, -33, 0, 33, 66, 100], 3))(x_new)

plt.plot(x,y,'o', x_new, y_new, x_new, optimal_y)
plt.xlim([min(x_new)-1, max(x_new) + 1 ])
plt.title('Polqa MOS Difference vs Accuracy Difference (SM)')
plt.xlabel('Change in MOS')
plt.ylabel('Change in Accuracy')
plt.show()
'''
