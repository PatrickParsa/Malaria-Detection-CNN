# Malaria-Detection-CNN

## Context

This notebook uses computer vision to detect the differences with healthy red blood cells and those that are infected with Malaria, and then classifies them. The goal is to classify accurately, with specific attention to the recall score of the infected cell class. In other words, we want to minimize the number of misclassified infected cells because undetected infected cells can be harmful. 

We started with an initial base model and performed very strongly, and after that we ran different methods of improving the results such as using pre trained models, tuning the model with KerasTuner, adding features such as batch normalization, as well as altering the input images to make them easier to differentiate for our model. 


## 

Batch normalization and changing the activation produced strong results but they weren't as good as the base model. Image augmentation and the pre-trained model also did not improve our performance. 

KerasTuner allowed us to find the optimal hyperparameters to increase performance. 

Image alteration to grayscale did improve our target recall score, whereas HSV conversion produced poor results across all the data accuracy metrics apart from our targeted recall score- however it doesn't not balance out the poor performance of the other metrics. 

potential future techinques: other pre trained models, transfer learning, bayesian optimization. 

