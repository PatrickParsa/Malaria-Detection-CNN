# Malaria-Detection-CNN

![Screen Shot 2021-12-07 at 1 11 17 PM](https://user-images.githubusercontent.com/88220704/145091300-3248aa17-1f21-483b-93ad-e9d4eda631ba.png)


## Summary

In this project, we built several different convolutional neural network models in an attempt to accurately classify images of red blood cells as either parasitized or uninfected. We used an array of different techniques to better our performance from the base model (three convolutional layers each followed by max pooling and dropout layers) including adding more layers, batch normalization, changing activation functions, image alterations, and hyperparameter tuning using Keras Tuner. **Our best model ended up being our base model with one added convolutional layer followed by an additional max pooling layer and dropout layer**. We then improved performance by finding the optimal hyperparameters using Keras Tuner, and then found that feeding this model grayscaled images allowed stronger classification. 


## Context

Malaria is a contagious disease caused by Plasmodium parasites that are transmitted to humans
through the bites of infected female Anopheles mosquitoes. The parasites enter the blood and begin
damaging red blood cells (RBCs) that carry oxygen, which can result in respiratory distress and other
complications. The lethal parasites can stay alive for more than a year in a person’s body without
showing any symptoms. Therefore, late treatment can cause complications and could even be fatal.
Almost 50% of the world’s population is in danger from malaria. There were more than 229 million
malaria cases and 400,000 malaria-related deaths reported over the world in 2019. Children under 5
years of age are the most vulnerable population group affected by malaria; in 2019 they accounted
for 67% of all malaria deaths worldwide.

Traditional diagnosis of malaria in the laboratory requires careful inspection by an experienced
professional to discriminate between healthy and infected red blood cells. It is a tedious,
time-consuming process, and the diagnostic accuracy (which heavily depends on human expertise)
can be adversely impacted by inter-observer variability.

An automated system can help with the early and accurate detection of malaria. Applications of
automated classification techniques using Machine Learning (ML) and Artificial Intelligence (AI) have
consistently shown higher accuracy than manual classification. It would therefore be highly beneficial
to propose a method that performs malaria detection using Deep Learning Algorithms.
Objective

## Objective

To build an efficient computer vision model to detect malaria. The model should identify whether the image of a red blood cell is that of one infected with malaria or not, and classify the name as parasitized or uninfected, respectively. 

## Technologies used

* Numpy
* Pandas
* Matplotlib
* Seaborn
* Sklearn
* TensorFlow(Keras)
* KerasTuner
* ImageDataGenerator (from Keras)
* VGG16
* OpenCV(cv2)


## The Data

The dataset was provided from the faculty members of MIT as part of the Applied Data Science program capstone project. The data involves coloured images of red blood cells that contain parasitized and uninfected instances, where: 

* The **parasitized** cells contain the Plasmodium parasite
* the **uninfected** cells are free of the Plasmodium parasites but could contain other impurities. 

<img width="439" alt="Screen Shot 2021-11-26 at 4 47 54 PM" src="https://user-images.githubusercontent.com/88220704/143659677-a095fb99-cf60-46d1-a5f9-ea0eb3a27e57.png">

## Data processing and exploration

Given that convolutional neural networks only accept 4D arrays, we had to convert the images accordingly, and then we also normalized the images by dividing each image by the max pixel count (255) which makes the data much more optimal for our models. It makes training faster and reduces the chances of getting stuck at a local minima.

## Model building

Before initializing any models, it was important that we establish the metric that we are most interested in maximizing. In our case, we wanted to minimize the number of misclassified infected cells, which means that we want to obtain the **highest recall score** for the infected class. This is because we don't want infections to go unnoticed because it would be damaging to our overarching goal of reducing the impact of malaria. 

### Base Model 

Our base model contained 3 convolutional layers, each followed by a **MaxPooling2D** and **Dropout** layer. 

* **Dropout** layers are a technique used to prevent overfitting, and it works by randomly setting the outgoing edges of hidden units to 0 at each update of the training phase. Meanwhile, the **MaxPooling2D** layers are used to reduce the dimentions of the feature maps which thus reduces the number of parameters to learn and amout nfo computation performed in the network. 

The activation function we used is ReLU (Rectified Linear Unit), which overcomes the vanishing gradient problem allowing our models to learn faster and perform better. It is a piecewise linear function that will output the input directly if it is positive, otherwise, it will output zero. 

![Screen Shot 2021-12-07 at 1 28 11 PM](https://user-images.githubusercontent.com/88220704/145093555-1db9eb44-94c4-47e7-acdd-213916c47693.png)


### Results: 

<img width="268" alt="Screen Shot 2021-11-26 at 5 18 14 PM" src="https://user-images.githubusercontent.com/88220704/143660674-488e3ce5-b942-4dbd-a985-a0d5619a5f26.png">

We got a strong performance from our base model, with high precision, recall, and f-1 scores as we can see from the confusion matrix above. In particular, we focused on the recall score of our infected class, and we can see that in our case it is a 94% score and a digit of 73. This means that only 73 images were misclassified as being uninfected when they are actually parasitizied. We then tried to improve these scores even further. 

### Improvements to base model

Our second model simply involved adding another convolutional layer folloewd by a MaxPooling2D and Dropout layer. We saw a very slight drop in performance for recall on the infected class however we did see significant improvement on the recall score of the uninfected class, from 64 misclassifications to 77. 

<img width="245" alt="Screen Shot 2021-11-26 at 5 27 05 PM" src="https://user-images.githubusercontent.com/88220704/143660915-c09c360d-9dc6-49c3-9b26-90b84f94ba23.png">

#### Batch Normalization and LeakyReLu 

In this phase, we decided to add **Batch Normalization** and changed the activation function to **LeakyReLU** to see if we could improve the performance. But first, it is important to understand what these changes actually do. 

**Batch Normalization** is a process to make neural networks faster and more stable through adding extra layers in a deep neural network. The new layer performs the standardizing and normalizing operations on the input of a layer coming from a previous layer. It is called *batch* normalization because it takes place in batches, not as a single input. 

**LeakyReLU**(Leaky Rectified Linear Unit) is a type of activation function base on a ReLU but has a small slope for negative values instead of a flat slope. Sometimes in a ReLU layer, you may come across the problem of "dying ReLU" which happens when your ReLU always has values under 0 which blocks learning in the ReLU because of gradients of 0 in the negative part. Thus, LeakyReLU solves this problem by allowing a small gradient when the unit is not active.  

![Screen Shot 2021-12-07 at 1 28 36 PM](https://user-images.githubusercontent.com/88220704/145093587-0832d864-c170-4875-995b-79f4397e7051.png)



### Results: 

<img width="238" alt="Screen Shot 2021-11-26 at 5 41 23 PM" src="https://user-images.githubusercontent.com/88220704/143661278-2f1ff402-b913-404a-b719-74db1aad6e1b.png">

We saw a decrease in performance, as we see a higher number of misclassifications for our infected class compared to our initial class. 

## Pre-Trained model 

We tried using a pre-trained model to see how it would perform on our dataset. The model in particular that we used is **VGG16**. 

**VGG16** is a convolutional neural network archictecture which was used to win the ILSVR(Imagenet) competition in 2014. It follows a specific arrangement of convolution and max pool layers consistently throughout the whole architecture.

<img width="586" alt="Screen Shot 2021-11-26 at 5 47 04 PM" src="https://user-images.githubusercontent.com/88220704/143661459-d925030d-9024-4242-9405-29bce6ca185b.png">

### Results

<img width="244" alt="Screen Shot 2021-11-26 at 5 48 23 PM" src="https://user-images.githubusercontent.com/88220704/143661486-7ebaf7ea-c59f-4ac2-bbbd-04307cbf69ca.png">

Despite a strong performance, we see that it does not have improved scores compared to our already strong base model. Thus, we moved forward with our base model to see how we can improve it. 

## Keras Tuner

One robust way to way to improve a TensorFlow convolutional neural network is by using the **Keras Tuner** which is a library that helps us pick the optimal set of hyperparameters for our model. It works by giving the tuner a set of options for values in different hyperparameters and then it automatically tests different sets of values and combinations, evaluates their performance, and then provides a list of results. We can then tune our paramaters using the ones provided by the top scoring values given in the list of results. 

<img width="265" alt="Screen Shot 2021-11-26 at 5 56 02 PM" src="https://user-images.githubusercontent.com/88220704/143661655-03ad52ba-2d0c-4902-ad15-e932012fcd02.png">

### Results

<img width="258" alt="Screen Shot 2021-11-26 at 6 01 39 PM" src="https://user-images.githubusercontent.com/88220704/143661780-846301d1-3213-4aec-ad64-b6bd2b1d2b99.png">

Even though it's only slight, we do see an improvement from our best model so far since we can see fewer misclassifications of the infected class (71). 

## Image alterations

In this phase, we shifted our focus towards the dataset itself rather than our model. This is to see if any alterations to our images make them easier to classify for our best model. 

### Image augmentation: 

**Image augmentation** is a technique of altering the existing data to create some more data for the model training process. It artificially expands the available dataset for training the deep learning model. It creates transformed versions of the images in the dataset. It can includes shifts, flips, zooms, and other alterations, such as the examples shown below. 

<img width="305" alt="Screen Shot 2021-11-26 at 6 06 48 PM" src="https://user-images.githubusercontent.com/88220704/143661887-18466dea-a712-43fb-b2d2-983bdfab61b3.png">

#### Results: 

Despite strong performance for the uninfected class, we saw poor performance for the infected class which is what we are most interested in. 

### HSV conversion

HSV or Hue Saturation Value is used to seperate image luminance from color information. By converting our images to HSV, our model might have an easier time classifying the images. 

<img width="235" alt="Screen Shot 2021-11-26 at 6 12 21 PM" src="https://user-images.githubusercontent.com/88220704/143662039-da0e09f9-61e1-4b77-87f1-3c1cd3d2fd86.png">

#### Results: 

<img width="260" alt="Screen Shot 2021-11-26 at 6 12 52 PM" src="https://user-images.githubusercontent.com/88220704/143662055-aa68349c-ca65-42a1-aa7b-de52beb22634.png">

Interestingly, we see an extremely srtong performance for our targeted recall score of the infected class (only 46 misclassifications) however the model performed very poorly on all other metrics using these images. Therefore, we did not move forward with this model. 

### Grayscale

Grayscale conversion might help remove some of the noise that may be contributing to the misclassifications of our model. 

<img width="249" alt="Screen Shot 2021-11-26 at 6 16 08 PM" src="https://user-images.githubusercontent.com/88220704/143662131-14991b3f-23cd-4b0f-82ff-5f0ffba738f5.png">

#### Results

<img width="267" alt="Screen Shot 2021-11-26 at 6 17 18 PM" src="https://user-images.githubusercontent.com/88220704/143662154-6a88c45f-fc4f-4678-9291-b9d836d4a367.png">



The metrics show that this was our best performance. As indicated by the green circle, the number of misclassified infected cells is only 68 while the rest of our performance metrics all remain very strong. 

## Conclusion:

In this project, we explored a number of ways for building a deep learning model that can
accurately detect infected red blood cells from a set of images. These techniques ranged from
image pre-processing techniques, model alterations and hyperparameter tuning, as well as the use
of a pre-trained model. After trialing these techniques, we were able to identify a model that
provides a balanced performance, meaning that the accuracy metrics were all relatively high.
Despite obtaining a strong model, we did not exhaust all the available options available to
maximize performance even further. Thus, for future improvement it would be worth exploring
options such as trying other pre-trained models and using transfer learning. Finding models that
have already been carefully crafted for a dataset similar to ours can be highly beneficial because
we could simply alter some of its features to fit our problem. 

Additionally, we can actually display the cells that are being misclassified and try to understand why, and then consult with
medical experts to obtain a better understanding or even have them correct any potential
misclassified labels of the dataset. After these improvements, our next step would be to begin
deployment of our model, so that we can use its capabilities in real time and reach our overall
goal of detecting malaria as soon as possible to help eradicate its consequences. This would
involve communication with the engineering team to create a pipeline which would grant us
greater control and flexibility for our model, as well as scalability. This is important because we
were only dealing with a single dataset for our current model whereas we will want it to be able to process much more data and in real time, so if there are issues then we want to be able to
quickly fix it and a pipeline will help greatly with that.

## Problem and Solution Summary

Our main challenge was being able to distinguish between red blood cells that are
infected, and those that are not. Different features such as color, opacity, and other types of noise
can make it difficult for our automated model to make this distinction. Thus, our main objective
was to utilize different techniques to both equip our model to be more effective at this task and
also alter the images in a way to make this goal less challenging. We tried various techniques
such as HSV conversion and augmentation for our images, as well as adding features to our
model such as batch normalization. Our final proposed solution design involves grayscale
images being fed into a keras-tuned model. The grayscale conversion helped eliminate some of
the noise of the images that may have contributed to the misclassification of some of our images,
while the keras tuner helped us obtain the best hyperparameters needed for our model to work
optimally with the data.

## Next Steps

Our grayscale technique helped make the images easier to classify for our model,
however one thing to consider is that the removal of colour may pose a potential limitation with
future data. If for example, a red blood cell has a characteristic that appears similar to that of an
infected cell but can only be distinguished by its colour, then our model might fail to make that
distinction. Thus, communication with medical experts on this topic might help us better explore this potential limitation and thus create contingencies that will help prevent it from being a
problem. Another key challenge involves the keras tuner. Firstly, the keras tuner runs a
combination of random values for the hyperparameters which may not exactly cover the most
optimal possible values, and secondly it can be very slow to run. Thus, other tuning techniques
such as Bayesian optimization may produce better results because it allows us to jointly tune
more parameters with fewer experiments and find better values, all at a potentially quicker speed.
The slow speeds of our tuning might also mean that we should consult with the engineering team
to make sure an appropriate environment can be established so that when further potential tuning
happens down the line, we won’t be limited by slow processing speeds. This is especially
important considering we will be working with live data and in high amounts. Thus, consultation
with both medical experts for further intuition on image distinction as well as with the
engineering team to establish an appropriate environment for deployment should be of priority
importance. Following that, we can explore different tuning methods such as bayesian or grid
search to find even better hyperparameters than what we found with our Keras tuner.





