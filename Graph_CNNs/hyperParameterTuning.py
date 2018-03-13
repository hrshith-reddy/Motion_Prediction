# from theano import function, config, shared, tensor
import numpy
import time
import subprocess as sbp
import os
import copy
import socket as soc
from datetime import datetime
import sys
import pdb
# from py_server import ssh
base_dir = '..'
gpus = [0,1,3]

params = {}
## These hyperparameters are OKAY to tweak. They will affect training, convergence etc.
params['initial_lr'] = 1e-3
params['decay_schedule'] = [1.5e3,4.5e3] # Decrease learning rate after these many iterations
params['decay_rate_schedule'] = [0.1,0.1] # Multiply the current learning rate by this factor
params['lstm_init'] = 'uniform' # Initialization of lstm weights
params['fc_init'] = 'uniform' # Initialization of FC layer weights
params['clipnorm'] = 25.0
params['use_noise'] = 1
params['noise_schedule'] = [250,0.5e3,1e3,1.3e3,2e3,2.5e3,3.3e3] # Add noise after these many iterations
params['noise_rate_schedule'] = [0.01,0.05,0.1,0.2,0.3,0.5,0.7] # Variance of noise to add
params['momentum'] = 0.99
params['g_clip'] = 25.0
params['truncate_gradient'] = 100#10 #
params['sequence_length'] = 150 # Length of each sequence fed to RNN
params['sequence_overlap'] = 50 
params['batch_size'] = 100
params['lstm_size'] = 1 #512 #
params['node_lstm_size'] = 1 #512 #
params['fc_size'] = 1 #256 #
params['snapshot_rate'] = 25 #10# Save the model after every 250 iterations
params['train_for'] = 'final' 


## Tweak these hyperparameters only if you want to try out new models etc. This is only for 'Advanced' users
params['use_pretrained'] = 0
params['iter_to_load'] = 25
params['crf'] = ''
params['copy_state'] = 0
params['full_skeleton'] = 1
params['weight_decay'] = 0.0
params['temporal_features'] = 0
params['dra_type'] = 'simple'
params['dataset_prefix'] = ''
params['drop_features'] = 0
params['drop_id'] = '9'
params['subsample_data'] = 1


def listToString(ll):
	st = '['
	for v_ in ll:
		st += str(v_) + ','
	st = st[:-1] + ']'
	return st 


my_env = os.environ

# Adding CUDA to path
my_env['PATH'] += ':/usr/local/cuda/bin'

use_gpu = 0


if(len(sys.argv)==3):
	params['checkpoint_path'] = '/new_data/gpu/siddsax/motion_pred_checkpoints/GCNN/checkpoints_{0}_T_{2}_bs_{1}_tg_{3}_'.format('gcnn',params['batch_size'],params['sequence_length'],params['truncate_gradient']) + '___' + sys.argv[2]
else:
	params['checkpoint_path'] =  '/new_data/gpu/siddsax/motion_pred_checkpoints/GCNN/checkpoints_{0}_T_{2}_bs_{1}_tg_{3}_'.format('gcnn',params['batch_size'],params['sequence_length'],params['truncate_gradient'])
params['dump_path'] = '../dump'

if params['weight_decay'] > 1e-6:
	params['checkpoint_path'] += '_wd_{0}'.format(params['weight_decay'])
if params['drop_features']:
	params['checkpoint_path'] += '_df_' + params['drop_id']
path_to_checkpoint = '{0}/'.format(params['checkpoint_path'])
print(path_to_checkpoint)

# script = "'if [ ! -d \"" + path_to_checkpoint + "\" ]; \n then mkdir " + path_to_checkpoint + "\nfi'"
# ssh( "echo " + script + " > file.sh")
# ssh("bash file.sh")
# if not os.path.exists(path_to_checkpoint):
# 	os.mkdir(path_to_checkpoint)

# if params['use_pretrained'] == 1:
# 	if load_pretrained_model_from[-1] == '/':
# 		os.system('cp {0}checkpoint.{1} {2}.'.format(load_pretrained_model_from,params['iter_to_load'],path_to_checkpoint))
# 		os.system('cp {0}logfile {1}.'.format(load_pretrained_model_from,path_to_checkpoint))
# 		os.system('cp {0}complete_log {1}.'.format(load_pretrained_model_from,path_to_checkpoint))
# 	else:
# 		os.system('cp {0}/checkpoint.{1} {2}.'.format(load_pretrained_model_from,params['iter_to_load'],path_to_checkpoint))
# 		os.system('cp {0}/logfile {1}.'.format(load_pretrained_model_from,path_to_checkpoint))
# 		os.system('cp {0}/complete_log {1}.'.format(load_pretrained_model_from,path_to_checkpoint))

print 'Dir: {0}'.format(path_to_checkpoint)

if(int(sys.argv[1])==0):
	args = ['python','trainGCNN_NoGraph.py']
else:
	args = ['python','trainGCNN.py']
for k in params.keys():
	args.append('--{0}'.format(k))
	if not isinstance(params[k],list):
		args.append(str(params[k]))
	else:
		for x in params[k]:
			args.append(str(x))

# FNULL = open('{0}stdout.txt'.format(path_to_checkpoint),'w')
p=sbp.Popen(args,env=my_env)
pd = p.pid
p.wait()
