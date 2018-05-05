import numpy as np
import os

import re

from six.moves import cPickle as pickle

from mfcc2vectors import get_features
from mfcc2vectors import get_label

from scipy import stats


'''folder_mfcc_preemp = "/Volumes/Transcend/Dropbox/workspace/python/Project_Jammin/speech_emotion_classifier/vectors/english/mfcc_preemp"
folder_mfcc = "/Volumes/Transcend/Dropbox/workspace/python/Project_Jammin/speech_emotion_classifier/vectors/english/mfcc"
folder_sr_zcr = "/Volumes/Transcend/Dropbox/workspace/python/Project_Jammin/speech_emotion_classifier/vectors/english/sr_zcr"

folders_mfcc = [folder_mfcc_preemp,folder_mfcc]'''

folder_mfcc_preemp = "/Volumes/Transcend/Dropbox/workspace/python/Project_Jammin/speech_emotion_classifier/vectors/carlos/casa_may_24_2016/mfcc_preemp"
folder_mfcc = "/Volumes/Transcend/Dropbox/workspace/python/Project_Jammin/speech_emotion_classifier/vectors/carlos/casa_may_24_2016/mfcc"
folder_sr_zcr = "/Volumes/Transcend/Dropbox/workspace/python/Project_Jammin/speech_emotion_classifier/vectors/carlos/casa_may_24_2016/sr_zcr"

# If you have two folders use option 1, use option 2 otherwise
folders_mfcc = [folder_mfcc_preemp,folder_mfcc]
#folders_mfcc = [folder_mfcc_preemp]

#folder_mfcc = '/home/pi/Documents/Project_Jammin/speech_emotion_classifier/vectors/german/mfcc'
#folder_sr_zcr = '/home/pi/Documents/Project_Jammin/speech_emotion_classifier/vectors/german/sr_zcr_preemp'

# preemp 1 is no mfcc and yes sr_zcr
# preemp 2 is yes mfcc and no sr_zcr
#pickle_file = "/Users/Carlos/Desktop/emotion_audio/speech_emotion_classifier/mfcc.pickle"

#pickle_file = "/Volumes/Transcend/Dropbox/workspace/python/Project_Jammin/speech_emotion_classifier/data/english/utterance_deltas_no_mode_mixed_preemp2_en.pickle"
pickle_file = "/Volumes/Transcend/Dropbox/workspace/python/Project_Jammin/speech_emotion_classifier/data/carlos/utterance_deltas_no_mode_mixed_preemp2_en_casa_may_24_2016.pickle"

#pickle_file = "/home/pi/Documents/Project_Jammin/speech_emotion_classifier/data/german/utterance_deltas_no_mode_preemp1_ge.pickle"



#lang = "ge"
lang = "unk"

deltas = True

if deltas:
	# The number of features has grown
	#num_feats = 237
	num_feats = 198
else:
	num_feats = 81

tot_rows = 0

def load_mfcc(fname,path):

	mfcc = get_features(fname,path,lang)
	#print mfcc
	return mfcc

def load_sr_zcr(fname,path):
	sr_zcr = []
	print path
	f = open(path)
		
	for line in f:
			
		line = line.strip()
		sr_zcr.append(float(line))
	sr_zcr = np.array(sr_zcr)
	#print sr_zcr
	return sr_zcr

def features_from_mfcc(mfcc,deltas = False):
	
	if not deltas:
		print "Not using deltas"
		mfcc = mfcc[:,:13]
	else:
		print "Using deltas"
		mfcc = mfcc[:,:39]

	#mean, mode, sdv, sdv by neab (min, max)
	mean = np.mean(mfcc,axis = 0)
	median = np.median(mfcc,axis = 0)
	std = np.std(mfcc,axis = 0)
	#mode = stats.mode(mfcc)[0][0]
	mins = np.min(mfcc,axis = 0)
	maxs = np.max(mfcc,axis = 0)

	#print maxs
	#print maxs.shape
	#print mode
	#print mode.shape
	#print "std"
	#print std
	#print std.shape
	#print median
	#print median.shape
	#print mean
	#print mfcc.shape
	#print mean.shape

	#feats = np.concatenate((mean,median,mode,std,mins,maxs))
	feats = np.concatenate((mean,median,std,mins,maxs))
	#print feats
	print feats.shape
	return feats

def process_all():
	global tot_rows
	count = 0
	for folder_mfcc in folders_mfcc:
		for fname in os.listdir(folder_mfcc):
			path = os.path.join(folder_mfcc, fname)
			if "." not in fname[0]:
				print path

				mfcc = load_mfcc(fname,path)
				mfcc = features_from_mfcc(mfcc, deltas)

	                        if lang == "ge":
	                                
	                                path2 = os.path.join(folder_sr_zcr, fname.lower())
	                        else:
	                                path2 = os.path.join(folder_sr_zcr, fname)
				sr_zcr = load_sr_zcr(fname,path2)

				label, emo = get_label(fname,lang)
				print "Emotion %s - %s"%(emo,label)

				feats = np.concatenate((mfcc,sr_zcr,[label]))
				#print feats
				#print feats.shape

				count = count + 1

				# Concat the vectors
				if count > 1:
					matrix = np.vstack((matrix, feats))
				else:
					matrix = feats


	#print matrix
	print matrix.shape
	tot_rows = count
	return matrix


def randomize(dataset, labels):
	permutation = np.random.permutation(labels.shape[0])
	shuffled_dataset = dataset[permutation,:]
	shuffled_labels = labels[permutation]
	return shuffled_dataset, shuffled_labels

def make_arrays(nb_rows):
	print "%d - %d"%(nb_rows,num_feats)
	if nb_rows:
		dataset = np.ndarray((nb_rows, num_feats), dtype=np.float32)
		labels = np.ndarray(nb_rows, dtype=np.int32)
	else:
		dataset, labels = None, None
	return dataset, labels

def create_sets(matrix, data_size,train_size, valid_size=0):
	
	valid_dataset, valid_labels = make_arrays(valid_size)
	train_dataset, train_labels = make_arrays(train_size)
		
	start_v, start_t = 0, 0
	#end_v, end_t = valid_size, train_size
	end_t = valid_size + train_size
	skip = 0

	# let's shuffle the faces to have random validation and training set
	np.random.shuffle(matrix)
	if valid_dataset is not None:
		valid_dataset = matrix[:valid_size, :data_size]
		valid_labels = matrix[:valid_size, data_size]
					
									
	train_dataset = matrix[valid_size:end_t, :data_size]
	train_labels = matrix[valid_size:end_t, data_size]
				
	#print valid_dataset.shape
	#print train_dataset.shape
	#print train_dataset
	#print train_labels

	return valid_dataset, valid_labels, train_dataset, train_labels

def create_pickle():
	matrix = process_all()	
	vs = 0
	ts = tot_rows - vs
	valid_dataset, valid_labels, train_dataset, train_labels = create_sets(matrix,num_feats,ts,vs)

	#print "Max T %d"%(np.max(train_labels))
	if valid_labels is not None:
		#print "Max V %d"%(np.max(valid_labels))
		print valid_labels.shape
		print valid_labels
		print valid_dataset.shape
		print valid_dataset
	print('Training:', train_dataset.shape, train_labels.shape)
	if valid_labels is not None:
		print('Validation:', valid_dataset.shape, valid_labels.shape)

	train_dataset, train_labels = randomize(train_dataset, train_labels)
	if valid_labels is not None:
		valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

	try:
	  f = open(pickle_file, 'wb')
	  save = {
		'train_dataset': train_dataset,
		'train_labels': train_labels,
		'valid_dataset': valid_dataset,
		'valid_labels': valid_labels,
		
		}
	  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
	  f.close()
	except Exception as e:
	  print('Unable to save data to', pickle_file, ':', e)
	  raise
	statinfo = os.stat(pickle_file)
	print('Compressed pickle size:', statinfo.st_size)


create_pickle()
'''

for fname in os.listdir(folder_sr_zcr):
	path = os.path.join(folder_sr_zcr, fname)
	if "." not in fname[0]:
		print path
		sr_zcr = load_sr_zcr(fname,path)
'''
