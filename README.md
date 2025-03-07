# GermanTrafficSignRecognition-MachineLearning
Final Project for Machine learning course

Project #3 Proposal
I. Project Introduction        
Traffic signs are essential in our modern era because they allow us to know what rules need to be followed at specific locations when driving. Identifying them can become a repetitive task that humans can do, but current technology allows us to recognize them using cameras and image recognition. To implement image recognition, however, we need a way to make the computer understand what a traffic sign is, which is a job that a machine learning algorithm can easily handle. For this project, we’ll test two machine learning algorithms (one essential machine learning and one deep learning) to see which is better for traffic sign image classification.
II.  Dataset
We decided to use the GTSRB - German Traffic Sign Recognition Benchmark dataset. This Dataset is an engineering dataset because it has application with self-driving vehicles and their need to quickly identify street signs, so that the code within the self-driving vehicle adjusts speed, know to stop, and know about potential other things to adjust the car’s parameters for.
The dataset provides enough data and resources for us to work on this “benchmark” for our two models. The dataset contains the following folders:
Meta = Showcasing the 43 classes (types of traffic signs)
Test = Contains 12630 images from all classes and a .csv test file
Train = Contains 43 folders which contain images from each class. The ratio between the number of images per class varies. 
We’ll be using a Dropbox link for convenience of installing the dataset to Colab: https://www.dropbox.com/scl/fi/1wdngi227kabrkedmn0gb/archive.zip?rlkey=yqetdf5rfo3ro0m8y9z083gro&st=2t785k55&dl=0
III. Data Pre-processing
The Data processing techniques we are planning on using are the following:
Segmentation = Used to remove any irrelevant feature from an image
Zooming	    = Used to avoid overfitting by slightly altering our data
Encoding	    = Used to facilitate the classification task for our model
Normalization = Used to simplify the training of our dataset data
Denoising        = Used to remove or reduce noise from an image
IV. Models
IV.i Model Selection
The Models we are planning on using are CNN and SVM. The models will be used for supervised learning. 
We’re using SVM because it is one of the more powerful classical machine learning models for data classification. It has the capability to perform well on Image classification and can utilize multiple kernels, adjust the margin of error on the hyperplane, and adjust the weights of individual points of data to adjust for outliers. These should be able to help us process the data in the most optimal way possible. 
We’re using CNN because they are the most popular and influential image classification deep learning algorithm currently available. The wide variety of filters this model provides will allow us to break down our data into sections, which should help our model better identify the traffic signs.
IV.ii Model Details
Filling text. We’ll add or remove this later.
IV.ii.A CNN Architecture & Tunning
The CNN architecture implements four convolutional layers that are complemented by using MaxPooling and Batch Normalization. Then, the 
IV.ii.B SVM Architecture
The SVM architecture implements a classifier using the NuSVC algorithm, it was chosen due to it allowing more flexibility in the choice of the margin. The RBF kernel was chosen due to this being a non-linear classification problem. The linear kernel wasn’t a good choice, cause the data isn’t linearly separable, as there are often signs that have a lot of similarities between them. The Polynomial Kernel may have been a good option, but the computation time on it was not feasible with the resources we had available. For a similar reason, we chose to use a smaller Gamma value, as too high of a value would make the model too complex, also raising the training time, as well as to reduce the fear of overfitting. 

 The data was then resized to make the colored image readable to the SVM, changing the 30*30*3 into a 2700, and making the pictures 2 dimensions so that they could be run through the SVM. 

After this setup was completed, the fit command was run, training the model. This allowed the prediction to be made, so that we could Calculate the recall, precision, f1 score and accuracy of the SVM. 


V. Results & Model Evaluation
The results can be displayed in the next sections:
V.i Metrics
For both models, we used the following:
Accuracy =  Measures how often the model is correct at predicting. 
F1 score   =  Measures the model’s performance since our dataset has some imbalanced data (more training images in certain classes.)
Precision =  Measures how often the model predicted the correct image.
Bellow, there are pictures of the SVM’s metric scores (left image) and the CNN’s metric scores.
V.ii Confusion Matrix
Bellow, there is the Confusion Matrix for both 
VI. Conclusions
The results can be displayed in the next sections:
V.i Comparison
With the 2 models, we got very good results, with the SVM, we got an accuracy of 80% and for the CNN we got an accuracy of 95%. This shows that both models are good to be used for this data, as the accuracy for both of them are very high, though the CNN is the better model to use for this image classification task. 
SVM:

CNN:

This is not unexpected to us, as the CNN is the most popular choice for image classification for a reason. CNN being a deep model makes it more well suited. It is very effective with complex and non-linear patterns, and can perform very well in large datasets like this one. 
V.ii Duties
We split the work as the following:
V.ii.a Diego
He will implement data encoding, normalization, and denoising. Moreover, he will build, tun, train, test, and evaluate a CNN model. while Alex will be doing the same for SVMs.
V.ii.b Alex
He will implement the segmentation and zooming . Moreover, he will build, run, train, test, and evaluate a SVM model.
