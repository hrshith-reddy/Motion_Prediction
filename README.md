# Human motion prediction with Graph-CNNs

## Dependencies

* Theano
* matplotlib ( for visualization )

Human motion prediction is the task of predicting the future movements of an object given few initial frames. It is a touch problem as movement of humans is highly asynchronous.

In this work we use **Graph CNNs** to model the motion of several mo-cap features of different body parts i.e. head, torso, arms and the relationships among them. Graph-CNNs provide a more structured way to model the spatio-temporal connections through the adjacency matrix. Here we also add connections through time over vanilla G-CNNs to model the temporal connectivity of the features.

## Code Description 

```
+---code
|    +-- Files to run the model
|    +---neuralmodels
|    |     +--- Contains layers and the main models (some layers borrowed from [NeuralModels](https://github.com/asheshjain399/NeuralModels))
|    +--- utilis
|    |      +--- Basic utility files
+---savedModels
|    +--- Directory automatically created for saving checkpoints and test projections
+---dataset
|    +--- store the provided dataset file here
+---visualize
|    +---scripts to make visualizations as above
```

The ```main.py``` file sets up the whole model and calls other helper files. Learning rates, hidden layer sizes weights initializations etc can be changed using python options. ```defineGCN.py``` consists of the architecture of the model, number of layers can be modified from here. ```trainModel.py``` calls the fit function of the model and loads the dataset. The main model is in ```neuralmodels/models/model.py```. This file is ideally not to be changed. 

## Demo
Download the dataset as
```bash
wget http://www.cs.stanford.edu/people/ashesh/h3.6m.zip
git clone https://github.com/siddsax/Motion_Prediction
cd Motion_Prediction
unzip h3.6m.zip
mv h3.6m/* .
rm -rf h3.6m
```

Now train a model as 
```bash
cd code
python -W ignore main.py
```

This will create projections in the folder savedModels periodically which can then be used as follows to produce mice movies like the following for smoking.

```bash
cd ../visualize
python forward_kinematics.py $fileNameToVisualize
```

![](https://github.com/siddsax/Motion_Prediction/blob/master/smoke.gif)

