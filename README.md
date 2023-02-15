# A-Small-Fishing-Vessel-Recognition-Method-
This project is a novel method for recognizing small fishing vessels based on laser sensors. Using four types of small fishing vessels as targets, a recognition method for small fishing vessels based on Markov transition field (MTF) time-series images and VGG-16 transfer learning is proposed. 

#Data preparation
1. The original small fishing boat contour data is in the Original Data folder.
2. The MTF image obtained from fitting output is in the MTF Data folder.
3. The formal project training data is in Dataset.


The following is the specific configuration information of the project:
# TensorFlow2.0_Fishing-Vessel_Classification
This project uses TensorFlow2.0 for image classification tasks.

## How to use
### Requirements
+ **Python 3.x** (My Python version is 3.6.8)<br/>
+ **TensorFlow version: 2.0.0-beta1**<br/> 
+ The file directory of the dataset should look like this: <br/>

### Data generation
1.Run the UCR.py file to download the complete UCR dataset and use the MTF method to encode the  UCR MTF Data.

2.Run MTF Encoding.py file in the data folder. 
The fitting data and encoded MTF image data are generated and saved in the Original Data folder and MTF Data folder.

3.The data set folder contains the enhanced fishing boat profile data, and the UCR migration learning model parameter is VGG-16 pre-training.h5 


${dataset_root}
|——train
|   |——class_name_0
|   |——class_name_1
|   |——class_name_2
|   |——class_name_3
|——valid
|   |——class_name_0
|   |——class_name_1
|   |——class_name_2
|   |——class_name_3
|——test
    |——class_name_0
    |——class_name_1
    |——class_name_2
    |——class_name_3
```
### Train
Run the script
```
train.py
```
to train the network on your image dataset, the final model will be stored. You can also change the corresponding training parameters in the `config.py`.<br/>

### 
Transfer learning pre-training model is saved as VGG-16 pre-training.h5

### Evaluate
To evaluate the model's performance on the test dataset, you can run `evaluate.py`.<br/>

The structure of the network is defined in `config。py` and `model_definition.py`, you can change the network structure to whatever you like.<br/>
