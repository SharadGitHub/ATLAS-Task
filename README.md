# ATLAS-Task

### Objective
Task is to compress the given data using Autoencoders. Tabular data is provided with 4 variables which has to be compressed to 3 variables using Autoencoder. 

### Quick Guide
__Data__: Data is available in the data directory. <br>
__Notebooks__: There are two jupyter notebooks available. They are well commented and I have tried to explain the code as much as possible. 
One notebook uses given pre-trained model and finetunes it. In other notebook I have tried various models and train from scratch to compress the data so that it gives me better understanding how the network could be improved.  

There is one python file as well which contains the code for network class and other necessary functions. 

__NOTE__: The code written in the notebooks is built over https://github.com/Skelpdar/HEPAutoencoders  work as one objective of task was also to understand this code and work. 

__Requirements__:


### Training. 
As mentioned above I have fine-tuned given pre-trained network and trained a few networks from scratch with given data. I trained the networks from scratch so that I get better idea of the problem and challenges. 

__My approach__: I started from smaller networks with just 3 layers in encoder side. But this was underfitting and loss was not decreasing much. Then I gradually started increasing the layers and channels in them but this led to overfitting as data is not enough. I trained the same network which is given as pre-trained from scratch but did not get good results. <br>
<br>
I also tried to use different loss function SmoothL1Loss instead of MSELoss as former does not punish outliers much. 
To avoid overfitting, I used dropout in a few layers, but not in bottleneck layer as this will drop too much information for the data we have. Using dropout for other layers dropped the network performance, perhaps I could have found some sweet spot if I had tried more things.

### Observations
In Skelpdar's code I noticed a few things which, from my knowledge, is not so common in practice, this is why I avoided them. <br> 
1. Using tanh as activation function in layers, I used relu instead because tanh has saturation regions which makes the derivative 0. So those neurons dont learn anything as their weights dont get updated. <br>
2. In his repository test data is being used as validation data. Instead, I used 20% of train data as validation data and kept test data aside to use it later to check the performance of the network.

### Conclusion
I got better residual error by fine tuning the pre-trained model compared to training other models from scratch. However, the residual I got is not good while compared to Skelpdar's, I have also trained the same network but did not get same residuals. I think it is because of less data as same network starts to overfits on the given data. 
