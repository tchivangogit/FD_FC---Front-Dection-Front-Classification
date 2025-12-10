## Front Dection & Front Classification - FD_FC
One of the driving force responsible of weather changing are the fronts. Front is the boundary or transition zone between two air masses of different temperatures and densities, which can lead to significant weather changes or phenomena such as rain, thunderstorms, changes in wind, and shifts in temperature. This article will describe a process of applying machine learning through a Convolution Neural Network (CNN) to detect fronts locations by learning pattern from weather variables as grid of temperature, wind speed and direction, mean sea level pressure and dew point. Those patterns will be learned based on a set of a historical labeled data of meteorological weather maps from Center for Weather Forecasting and Climate Studies (CPTEC). The output of this system are not to replace human but to help the weather specialist to map the fronts and to add features to numerical weather prediction (NWP) model to detected phenomena caused by the fronts. </br>

## Scripts and directory details

* The scritp fd_fc_CNN_123.py is the script for the training
* The script fd_fc_Inference.ipynb is the script for inference. And Explains a briefly about the pre-processing of the data
* Within the folder model_weights there is the best model saved. 
