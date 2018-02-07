from headers import *
import numpy as np
from neuralmodels.layers.Concatenate_Node_Layers import Concatenate_Node_Layers
theano.config.optimizer='None'
theano.exception_verbosity='high'
class GCNN(object):
	def __init__(self,graphLayers,finalLayer,preGraphNets,nodeNames,temporalNodeRNNs,nodeRNNs,topLayer,cost,nodeLabels,learning_rate,adjacency,clipnorm=0.0,update_type=RMSprop(),weight_decay=0.0):
		'''
		edgeRNNs and nodeRNNs are dictionary with keys as RNN name and value is a list of layers
		
		preGraphNets is a dictionary with keys as nodeNames and value is another dictionary whose keys are types of pregraphNetworls that is 'temporal' or 'normal' and value is a list of size-2 which indicates the features to choose from the unConcatenateLayer 

		nodeLabels is a dictionary with keys as node names and values as Theano matrix
		'''
		self.settings = locals()
		del self.settings['self']
		
		self.topLayer = topLayer
		self.nodeRNNs = nodeRNNs
		self.temporalNodeRNNs = temporalNodeRNNs
		self.nodeLabels = nodeLabels
		self.graphLayers = graphLayers
		self.finalLayer = finalLayer
		self.learning_rate = learning_rate
		self.clipnorm = clipnorm
		self.weight_decay = weight_decay
		

		self.cost = {}
		self.X = {}
		self.Y_pr = {}
		self.Y_pr_all = []
		self.Y_pr_last_timestep = {}
		self.Y = []
		self.params = {}
		self.updates = {}
		# self.train_node = {}
		self.predict_node = {}
		self.predict_node_last_timestep = {}
		self.masterlayer = {}
		self.grads = {}
		self.predict_node_loss = {}
		self.grad_norm = {}
		self.norm = {}
		self.get_cell = {}
		self.params_all = []
		self.update_type = update_type
		self.update_type.lr = self.learning_rate
		self.update_type.clipnorm = self.clipnorm
		self.node_features = []
		self.std = T.scalar(dtype=theano.config.floatX)
		self.std.tag.test_value = .5
		self.preGraphNetsTypes = ['temporal', 'normal' ]
		self.adjacency = adjacency

		self.Y_all = T.dtensor3(name="labels")#, dtype=theano.config.floatX)
		self.Y_all.tag.test_value = np.random.rand(7,150, 54)
		self.masterlayer = unConcatenateVectors(preGraphNets)
		self.X_all=self.masterlayer.input#T.tensor3(name="Data", dtype=theano.config.floatX)
		# self.masterlayer.X_all = self.X_all

		pit = 0
		indv_node_layers = []
		for nm in nodeNames:
			layers = self.nodeRNNs[nm]
			for i in range(1,len(layers)):
				layers[i].connect(layers[i-1])
				if layers[i].__class__.__name__ == 'AddNoiseToInput':
					layers[i].std = self.std

			layers = self.temporalNodeRNNs[nm]
			for i in range(1,len(layers)):
				layers[i].connect(layers[i-1])
				if layers[i].__class__.__name__ == 'AddNoiseToInput':
					layers[i].std = self.std

			self.params[nm] = []
			
			layers_below = []
			for pgnT in self.preGraphNetsTypes: # ADDED BY ME!!!!!!!!!!!!!!!!!!		
				if (pgnT =='temporal'):
					nodeLayers = self.temporalNodeRNNs[nm]
				elif(pgnT == 'normal'):
					nodeLayers = self.nodeRNNs[nm]
				else:
					print("Error in file GCNN.py, the pgnT is neither temporal or normal")	
						
				layers_below.append(nodeLayers)
				nodeLayers[0].input = self.masterlayer.output(pgnT,nm,self.X_all)

				for l in nodeLayers:
					if hasattr(l,'params'):
						self.params[nm].extend(l.params)

			cv = ConcatenateVectors()
			cv.connect(layers_below)
			nodeTopLayer = self.topLayer[nm]
			nodeTopLayer[0].connect(cv)
			for i in range(1,len(nodeTopLayer)):
				nodeTopLayer[i].connect(nodeTopLayer[i-1])

			for l in nodeTopLayer:
				if hasattr(l,'params'):
					self.params[nm].extend(l.params)

# -------------------------------------------------------------------------------
			# here will be work of gaph cnns
#  -------------------------------------------------------------------------

			# self.Y_pr[nm] = nodeTopLayer[-1].output()
			# self.Y_pr[nm] = self.Y_pr[nm].reshape((x.shape[0], x.shape[1] , 1 , x.shape[2]))#, 1, x.shape[3], x.shape[4]))
			indv_node_layers.append(nodeTopLayer[-1])

			print(nm)
			# theano.printing.pydotprint(self.Y_pr[nm], outfile="pics/bare_bones_model_foul_" + str(nm) + ".png", var_with_name_simple=True)
			size_below = nodeTopLayer[-1].size


			# self.params_all.extend(self.params[nm])
			# pit = pit + 1
			# print(pit)

		cv = Concatenate_Node_Layers()
		cv.connect(indv_node_layers)
		layers = self.graphLayers
		layers[0].connect(cv)#self.node_features,size_below) 
		for i in range(1,len(layers)):
			layers[i].connect(layers[i-1])
			if layers[i].__class__.__name__ == 'AddNoiseToInput':
				layers[i].std = self.std
	
		for l in layers:
			if hasattr(l,'params'):
				self.params_all.extend(l.params)

		# out_all = layers[-1].output()


		###########################################
		
		# how data fed to graphsLayers is not correct. see various dims FIXED?
		# see how data out of it will be structured and then feed to finalLayer FIXED?
		# make a new FCLayer that has new connect and output 
		# augment cost so that it can take 2 lists as inputs
		# feed to theano.function via a list with all Xs

		#########################################
		size_below = layers[-1].size

		indx = 0
		for nm in nodeNames:
			layers = self.finalLayer[nm]
			layers[0].connect(self.graphLayers[-1],indx)
			for i in range(1,len(layers)):
				layers[i].connect(layers[i-1])
				if layers[i].__class__.__name__ == 'AddNoiseToInput':
					layers[i].std = self.std
			
			for l in layers:
				if hasattr(l,'params'):
					self.params_all.extend(l.params)

			if(indx==0):
				x = layers[-1].output()#self.Y_pr[nm]
				print(x.shape.__repr__)
				self.Y_pr_all = x#.reshape((x.shape[0], x.shape[1] , 1 , x.shape[2]))	
				# self.Y = self.nodeLabels[nm]#.reshape((x.shape[0], x.shape[1] , 1 , x.shape[2]))
				# normalizing = self.weight_decay *nodeTopLayer[-1].L2_sqr
			else:
				x = layers[-1].output()#self.Y_pr[nm]
				self.Y_pr_all = T.concatenate([self.Y_pr_all,x],axis=2)
				# self.Y = T.concatenate([self.Y,self.nodeLabels[nm]],axis=2)
				# normalizing+=self.weight_decay *nodeTopLayer[-1].L2_sqr
			indx+=1
			

		# print(self.params_all)
		print("====================")
		print(self.Y_all.shape.__repr__)
		print(self.Y_pr_all.shape.__repr__)
		self.cost = cost(self.Y_pr_all,self.Y_all)# + normalizing
		theano.printing.pydotprint(self.cost, outfile="pics/booba23" + ".png", var_with_name_simple=True)

		# ---------- Will need considerable work here joinging the backprop --------------------------- 
		[self.updates,self.grads] = self.update_type.get_updates(self.params_all,self.cost)
			
		self.train_node = theano.function([self.X_all,self.Y_all,self.adjacency,self.learning_rate,self.std],self.cost,updates=self.updates,on_unused_input='ignore')
		self.predict_node = theano.function([self.X_all,self.adjacency,self.std],self.Y_pr_all,on_unused_input='ignore')
		self.predict_node_loss = theano.function([self.X_all,self.Y_all,self.adjacency,self.std],self.cost,on_unused_input='ignore')
		self.norm = T.sqrt(sum([T.sum(g**2) for g in self.grads]))
		self.grad_norm = theano.function([self.X_all,self.Y_all,self.adjacency,self.std],self.norm,on_unused_input='ignore')
	

		self.num_params = 0
		for nm in nodeNames:
			for i in range(3):
				if(i==0):
					nodeLayers = self.nodeRNNs[nm]
				elif(i==1):
					nodeLayers = self.nodeRNNs[nm]
				elif(i==2):
					nodeLayers = self.topLayer[nm]		
				else:
					print("Error in file GCNN.py, iterator for clculating number of params is neither 0 or 1")	
				for layer in nodeLayers:
					if hasattr(layer,'params'):
						for par in layer.params:
							val = par.get_value()
							temp = 1
							for i in range(val.ndim):
								temp *= val.shape[i]		
							self.num_params += temp

		print 'Number of parameters in DRA: ',self.num_params

# --------------------------------------------------------------------------------------------------

	def fitModel(self,trX,trY,adjacency,snapshot_rate=10,path=None,epochs=30,batch_size=50,learning_rate=1e-3,
		learning_rate_decay=0.97,std=1e-5,decay_after=-1,trX_validation=None,trY_validation=None,
		trX_forecasting=None,trY_forecasting=None,trX_forecast_nodeFeatures=None,rng=np.random.RandomState(1234567890),iter_start=None,
		decay_type=None,decay_schedule=None,decay_rate_schedule=None,
		use_noise=False,noise_schedule=None,noise_rate_schedule=None,
		new_idx=None,featureRange=None,poseDataset=None,graph=None,maxiter=10000,unNormalizeData=None):
	
		from neuralmodels.loadcheckpoint import saveGCNN

		'''If loading an existing model then some of the parameters needs to be restored'''
		epoch_count = 0
		iterations = 0
		validation_set = []
		skel_loss_after_each_minibatch = []
		loss_after_each_minibatch = []
		complete_logger = ''
		if iter_start > 0:
			if path:
				lines = open('{0}logfile'.format(path)).readlines()
				for i in range(iter_start):
					line = lines[i]
					values = line.strip().split(',')
					print values
					if len(values) == 1:
						skel_loss_after_each_minibatch.append(float(values[0]))
						validation_set.append(-1)
					elif len(values) == 2:
						skel_loss_after_each_minibatch.append(float(values[0]))
						validation_set.append(float(values[1]))
				#if os.path.exists('{0}complete_log'.format(path)):
				#	complete_logger = open('{0}complete_log'.format(path)).read()
				#	complete_logger = complete_logger[:epoch_count]
			iterations = iter_start + 1

		tr_X = {}
		tr_Y = {}
		Nmax = 0
		outputDim = 0
		unequalSize = False
		numExamples = {}
		seq_length = 0
		skel_dim = 0
		

# ----------------------data cleaning related tasks ------------------------------------------------
		nodeNames = trX.keys()
		print "nodeNames: ",nodeNames

		for nm in nodeNames:
			tr_X[nm] = []
			tr_Y[nm] = []
		
		for nm in nodeNames:
			N = trX[nm].shape[1]
			seq_length = trX[nm].shape[0]
			skel_dim += trY[nm].shape[2]

			outputDim = trY[nm].ndim
			numExamples[nm] = N
			if Nmax == 0:
				Nmax = N
			if not Nmax == N:
				if N > Nmax:
					Nmax = N
				unequalSize = True
				
		if trY_forecasting is not None and new_idx is not None:
			trY_forecasting = self.convertToSingleVec(trY_forecasting,new_idx,featureRange)
			print 'trY_forecasting shape: {0}'.format(trY_forecasting.shape)
			assert(skel_dim == trY_forecasting.shape[2])

		'''Comverting validation set to a single array when doing drop joint experiments'''
		gth = None
		T1 = -1
		N1 = -1	
		if poseDataset.drop_features and unNormalizeData is not None:
			trY_validation = self.convertToSingleVec(trY_validation,new_idx,featureRange)
			[T1,N1,D1] = trY_validation.shape
			trY_validation_new = np.zeros((T1,N1,poseDataset.data_mean.shape[0]))
			for i in range(N1):
				trY_validation_new[:,i,:] = np.float32(unNormalizeData(trY_validation[:,i,:],poseDataset.data_mean,poseDataset.data_std,poseDataset.dimensions_to_ignore))
			gth = trY_validation_new[poseDataset.drop_start-1:poseDataset.drop_end-1,:,poseDataset.drop_id]

		if unequalSize:
			batch_size = Nmax
		
		batches_in_one_epoch = 1
		for nm in nodeNames:
			N = trX[nm].shape[1]
			batches_in_one_epoch = int(np.ceil(N*1.0 / batch_size))
			break

		print "batches in each epoch ",batches_in_one_epoch
		print nodeNames	
		#iterations = epoch_count * batches_in_one_epoch * 1.0
		numrange = np.arange(Nmax)
		#for epoch in range(epoch_count,epochs):
		epoch = 0
		while iterations <= maxiter:
			t0 = time.time()

			'''Learning rate decay.'''	
			if decay_type:
				if decay_type == 'continuous' and decay_after > 0 and epoch > decay_after:
					learning_rate *= learning_rate_decay
				elif decay_type == 'schedule' and decay_schedule is not None:
					for i in range(len(decay_schedule)):
						if decay_schedule[i] > 0 and iterations > decay_schedule[i]:
							learning_rate *= decay_rate_schedule[i]
							decay_schedule[i] = -1

			'''Set noise level.'''	
			if use_noise and noise_schedule is not None:
				for i in range(len(noise_schedule)):
					if noise_schedule[i] > 0 and iterations >= noise_schedule[i]:
						std = noise_rate_schedule[i]
						noise_schedule[i] = -1

			'''Loading noisy data'''
			noisy_data = graph.readCRFgraph(poseDataset,noise=std)
			trX = noisy_data[8]
			trY = noisy_data[4]
			trX_validation = noisy_data[9]
			trY_validation = noisy_data[5]



			'''Permuting before mini-batch iteration'''
			if not unequalSize:
				shuffle_list = rng.permutation(numrange)
				for nm in nodeNames:
					trX[nm] = trX[nm][:,shuffle_list,:]
					if outputDim == 2:
						trY[nm] = trY[nm][:,shuffle_list]
					elif outputDim == 3:
						trY[nm] = trY[nm][:,shuffle_list,:]

			for j in range(batches_in_one_epoch):

				examples_taken_from_node = 0	
				for nm in nodeNames:
					# nt = nm.split(':')[1]
					examples_taken_from_node = min((j+1)*batch_size,numExamples[nm]) - j*batch_size
					tr_X[nm] = copy.deepcopy(trX[nm][:,j*batch_size:min((j+1)*batch_size,numExamples[nm]),:])
					if outputDim == 2:
						tr_Y[nm] = copy.deepcopy(trY[nm][:,j*batch_size:min((j+1)*batch_size,numExamples[nm])])
					elif outputDim == 3:
						tr_Y[nm] = copy.deepcopy(trY[nm][:,j*batch_size:min((j+1)*batch_size,numExamples[nm]),:])

				loss = 0.0
				skel_loss = 0.0
				grad_norms = []
# ---------------------------------------------------------------------------------------------
# ------------------------------ Model relted tasks -------------------------------------------
				# for nm in nodeNames:

				tr_X_all = tr_X[nodeNames[0]]
				tr_Y_all = tr_Y[nodeNames[0]]

				for i in range(1,len(nodeNames)):
					tr_X_all =  np.concatenate([tr_X_all,tr_X[nodeNames[i]]],axis=2)
					tr_Y_all =  np.concatenate([tr_Y_all,tr_Y[nodeNames[i]]],axis=2)

				loss_for_current_node = self.train_node(tr_X_all,tr_Y_all,adjacency,learning_rate,std)
			
				g = self.grad_norm(tr_X_all,tr_Y_all,adjacency,std)
				grad_norms.append(g)
				loss += loss_for_current_node

				iterations += 1
				loss_after_each_minibatch.append(loss)
				validation_set.append(-1)
				skel_loss_after_each_minibatch.append(skel_loss)
				termout = 'e={1} iter={8} m={2} lr={5} g_l2={4} noise={7} loss={0} normalized={3} skel_err={6}'.format(loss,epoch,j,(skel_loss*1.0/(seq_length*skel_dim)),grad_norms,learning_rate,np.sqrt(skel_loss*1.0/seq_length),std,iterations)
				complete_logger += termout + '\n'
				print termout

# --------------------------- SAVING PERFORMACE CHECKING ET CETRA --------------------------------------------------------
				

				del tr_X
				del tr_Y
							
				tr_X = {}
				tr_Y = {}

				for nm in nodeNames:
					tr_X[nm] = []
					tr_Y[nm] = []

				if int(iterations) % snapshot_rate == 0:
					print 'saving snapshot checkpoint.{0}'.format(int(iterations))
					saveGCNN(self,"{0}checkpoint.{1}".format(path,int(iterations)))
		
				'''Trajectory forecasting on validation set'''
				if (trX_forecasting is not None) and (trY_forecasting is not None) and path and (int(iterations) % snapshot_rate == 0):
					forecasted_motion_o = self.predict_sequence(trX_forecasting,trX_forecast_nodeFeatures,sequence_length=trY_forecasting.shape[0],poseDataset=poseDataset,graph=graph)

					forecasted_motion = self.convertToSingleVec(forecasted_motion_o,new_idx,featureRange)
					forecasted_motion_2 = self.convertToSingleLongVec(forecasted_motion_o,poseDataset, new_idx, featureRange)

					fname = 'forecast_iteration_{0}'.format(int(iterations))
					# self.saveForecastedMotion(forecasted_motion,path,fname)
					self.saveForecastedMotion(forecasted_motion_2,path,fname)

			'''Computing error on validation set'''
			if (trX_validation is not None) and (trY_validation is not None) and (not poseDataset.drop_features):
				validation_error = 0.0
				Tvalidation = 0

				trX_validation_all = trX_validation[nodeNames[0]]
				trY_validation_all = trY_validation[nodeNames[0]]

				for i in range(1,len(nodeNames)):
					trX_validation_all =  np.concatenate([trX_validation_all,trX_validation[nodeNames[i]]],axis=2)
					trY_validation_all =  np.concatenate([trY_validation_all,trY_validation[nodeNames[i]]],axis=2)


				validation_error = self.predict_node_loss(trX_validation_all,trY_validation_all,adjacency,std)
				# [self.X_all,self.Y_all,self.adjacency,self.std]
				Tvalidation = trX_validation[nm].shape[0]

				validation_set[-1] = validation_error
				termout = 'Validation: loss={0} '.format(validation_error)#,(validation_error*1.0/(Tvalidation*skel_dim)),np.sqrt(validation_error*1.0/Tvalidation))
				complete_logger += termout + '\n'
				print termout
				print("________________!11111111@@@@@#\\\______//////")
		
			# if (trX_validation is not None) and (trY_validation is not None) and (poseDataset.drop_features) and (unNormalizeData is not None):
			# 	prediction = self.predict_nextstep(trX_validation)
			# 	prediction = self.convertToSingleVec(prediction,new_idx,featureRange)
			# 	prediction_new = np.zeros((T1,N1,poseDataset.data_mean.shape[0]))
			# 	for i in range(N1):
			# 		prediction_new[:,i,:] = np.float32(unNormalizeData(prediction[:,i,:],poseDataset.data_mean,poseDataset.data_std,poseDataset.dimensions_to_ignore))
			# 	predict = prediction_new[poseDataset.drop_start-1:poseDataset.drop_end-1,:,poseDataset.drop_id]
			# 	joint_error = np.linalg.norm(predict - gth)
			# 	validation_set[-1] = joint_error
			# 	termout = 'Missing joint error {0}'.format(joint_error )
			# 	complete_logger += termout + '\n'
			# 	print termout

			'''Saving the learned model so far'''
			if path:
				
				print 'Dir: ',path				
				'''Writing training error and validation error in a log file'''
				f = open('{0}logfile'.format(path),'w')
				for l,v in zip(skel_loss_after_each_minibatch,validation_set):
					f.write('{0},{1}\n'.format(l,v))
				f.close()
				f = open('{0}complete_log'.format(path),'w')
				f.write(complete_logger)
				f.close()
			

			# t1 = time.time()
			# termout = 'Epoch took {0} seconds'.format(t1-t0)
			# complete_logger += termout + '\n'
			print termout
			epoch += 1


	def saveForecastedMotion(self,forecast,path,fname):
		T = forecast.shape[0]
		N = forecast.shape[1]
		D = forecast.shape[2]
		for j in range(N):
			motion = forecast[:,j,:]
			f = open('{0}{2}_N_{1}'.format(path,j,fname),'w')
			for i in range(T):
				st = ''
				for k in range(D):
					st += str(motion[i,k]) + ','
				st = st[:-1]
				f.write(st+'\n')
			f.close()


# ==============================================================================================
	def predict_sequence(self,teX_original,teX_original_nodeFeatures,sequence_length=100,poseDataset=None,graph=None):
	# (trX_forecasting,trX_forecast_nodeFeatures,sequence_length=trY_forecasting.shape[0],poseDataset=poseDataset,graph=graph)	
		teX = copy.deepcopy(teX_original)
		nodeNames = teX.keys()

		teY = {}
		to_return = {}
		T = 0
		nodeFeatures_t_1 = {}
		for nm in nodeNames:
			[T,N,D] = teX[nm].shape ################### ?????????????????
			to_return[nm] = np.zeros((T+sequence_length,N,D),dtype=theano.config.floatX)
			to_return[nm][:T,:,:] = teX[nm]
			teY[nm] = []
			nodeFeatures_t_1[nm] = teX_original_nodeFeatures[nm][-1:,:,:]

		tr_X_all = tr_X[nodeNames[0]]
		# tr_Y_all = tr_Y[nodeNames[0]]

		for i in range(1,len(nodeNames)):
			tr_X_all =  np.concatenate([tr_X_all,tr_X[nodeNames[i]]],axis=2)
			# tr_Y_all =  np.concatenate([tr_Y_all,tr_Y[nodeNames[i]]],axis=2)

		limits = []
		start = 0
		for i in range(sequence_length):
			for nm in nodeNames:
				input_i = T.concatenate((input_i,to_return[nm][:(T+i),:,:]),axis=2)	
				end = start + to_return[nm].shape[2]
				limits.append((start,end))
				start = end
				#nodeFeatures[nodeName] = np.array([prediction])
				

			prediction = self.predict_node(input_i,adjacency,1e-5)
			nodeFeatures = prediction[-1,:,:]

			for nm in nodeNames:
				teY[nm].append(nodeFeatures[0,:,limits[i][0]:limits[i][1]])

			for nm in nodeNames:
				features = graph.getNodeFeature(nm,nodeFeatures,nodeFeatures_t_1,poseDataset) # previously nodeRNNFeatures, they are concatenation of node and temporal features for the current time step made using nodeFeatures and nodeFeatures-1
				to_return[nm][T+i,:,:] = features[0,:,:]
			nodeFeatures_t_1 = copy.deepcopy(nodeFeatures)
		for nm in nodeNames:
			teY[nm] = np.array(teY[nm])
		del teX
		return teY

#  ============================================================================================================
	def predict_nextstep(self,teX):
		nodeNames = teX.keys()
		prediction = {}
		for nm in nodeNames:
			# nt = nm.split(':')[1]
			prediction[nm] = self.predict_node[nm](teX[nm],1e-5)
		return prediction
		
	def predict_cell(self,teX_original,teX_original_nodeFeatures,sequence_length=100,poseDataset=None,graph=None):
		teX = copy.deepcopy(teX_original)
		nodeNames = teX.keys()

		teY = {}
		to_return = {}
		T = 0
		nodeFeatures_t_1 = {}
		for nm in nodeNames:
			[T,N,D] = teX[nm].shape
			to_return[nm] = np.zeros((T+sequence_length,N,D),dtype=theano.config.floatX)
			to_return[nm][:T,:,:] = teX[nm]
			teY[nm] = []
			nodeName = nm.split(':')[0]
			nodeFeatures_t_1[nodeName] = teX_original_nodeFeatures[nm][-1:,:,:]
		for i in range(sequence_length):
			nodeFeatures = {}
			for nm in nodeNames:
				nt = nm.split(':')[1]
				nodeName = nm.split(':')[0]
				prediction = self.predict_node[nt](to_return[nm][:(T+i),:,:],1e-5)
				#nodeFeatures[nodeName] = np.array([prediction])
				nodeFeatures[nodeName] = prediction[-1:,:,:]
				teY[nm].append(nodeFeatures[nodeName][0,:,:])
			for nm in nodeNames:
				nt = nm.split(':')[1]
				nodeName = nm.split(':')[0]
				nodeRNNFeatures = graph.getNodeFeature(nodeName,nodeFeatures,nodeFeatures_t_1,poseDataset)
				to_return[nm][T+i,:,:] = nodeRNNFeatures[0,:,:]
			nodeFeatures_t_1 = copy.deepcopy(nodeFeatures)
		cellstates = {}
		for nm in nodeNames:
			nt = nm.split(':')[1]
			nodeName = nm.split(':')[0]
			cellstates[nm] = self.get_cell[nt](to_return[nm],1e-5)
		return cellstates

	def concatenateDimensions(self,dictToconcatenate,axis=2):
		conctArr = []
		for k in dictToconcatenate.keys():
			if len(conctArr) == 0:
				conctArr = copy.deepcopy(dictToconcatenate[k])	
			else:
				conctArr = np.concatenate((conctArr,dictToconcatenate[k]),axis=axis)
		return conctArr
	
	def convertToSingleVec(self,X,new_idx,featureRange):
		keys = X.keys()
		[T,N,D]  = X[keys[0]].shape
		D = len(new_idx) - len(np.where(new_idx < 0)[0])
		single_vec = np.zeros((T,N,D),dtype=np.float32)
		for k in keys:
			nm = k.split(':')[0]
			idx = new_idx[featureRange[nm]]
			insert_at = np.delete(idx,np.where(idx < 0))
			single_vec[:,:,insert_at] = X[k]
		return single_vec

	def convertToSingleLongVec(self,X,poseDataset,new_idx,featureRange):
		keys = X.keys()
		[T,N,D]  = X[keys[0]].shape
		D = len(new_idx)
		single_vec = np.zeros((T,N,D),dtype=np.float32)
		k2 = 0
		for k in keys:
			nm = k.split(':')[0]
			# idx = new_idx[featureRange[nm]]
			k1 = np.size(featureRange[nm])
			p = 0
			for i in range(k1):
				if(new_idx[featureRange[nm][i]]==-1):
					a = poseDataset.data_stats['mean'][i]
					single_vec[:,:,featureRange[nm][i]] = np.tile(a,(T,N))		
				else:
					single_vec[:,:,featureRange[nm][i]] = X[k][:,:,p]
					p+=1	

		return single_vec	