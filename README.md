# Semantic Segmentation
### Introduction
This project relies on a pre-trained vgg model altered to be a Fully Convolutional Network (FCN) in order to label the pixels of a road in images using pre-labeled training data.  Upsampling of convlutional layers, combined with skip layers allow the FCN to maintain spacial information, unlike regular convolutional networks.

The model follows the design employed in the paper Fully Convolutional Networks for Semantic Segmentation[1](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf) by Jonathan Long, Evan Shelhamer and
Trevor Darrell.  In this paper, an FCN is used to provide per-pixel labeling.  From the Introduction: "Both learning and inference are performed whole-image-at-a-time by dense feedforward computation and backpropagation."

By importing the trained vgg network and adding skip layers and upsampling to it, we can rapidly train an FCN without needing to design the entire network from scratch.

### Setup
##### Frameworks and Packages
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
The code in the `main.py` module was implimented using the methods covered in the Udacity Self-Driving Car Semantic Segmentation lessons, as well as in the project walkthrough.

##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
To submit this project, the following targets were met:
1. The code passes all the automated unit tests.
2. The code passes all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view). 
	a. Does the project load the pretrained vgg model?  The loaf_vgg function loads the models and layers per the walkthrough process.
	b. Does the project learn the correct features from the images?  The project does appear to learn the correct features, as loss rate declaines over time and the output images show the road surface marked in green.
	c. Does the project optimize the neural network?  The project uses the AdamOptimizer to search for minimum loss.
	d. Does the project train the neural network?  The train_nn function is implimented, and the loss measure is printed during training.
	e. Does the project train the model correctly?  On avergae, loss does decrease over time.  The final loss is roughly 0.04.
	f. Does the project use reasonable hyperparameters?  Batch is set to 5, and epochs is set to 35.
	g. Does the project correctly label the road?  The accuracy of the output images is petty good, though there is still room for improvement.
3. The following files are included here in this repo:
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)


### Additional observations
The FCN's purpose is to retain spacial information, which greatly improves the granularity of labeling.  The speed benefits of this method also allow for closer to real-time subject labeling, which is critical for use in actual vehicles.

I built the skip layers based on the lesson material, which were in turn based on the FCN paper Fully Convolutional Networks for Semantic Segmentation[1](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf) by Jonathan Long, Evan Shelhamer and
Trevor Darrell.  Pages five and six are particularly important, demonstrating the model design and describing the reasoning behind that design.

I did implement simple image augmentation during development, first with randomized horizontal flipping, and brightness alterations.  I found little improvement in the final output using either method, so I eventually removed them.  Similarly, in the FCN paper, image augmentation such as jittering showed little benefit and was not employ.  I did employ the randomization of training images, to ensure that the order of the images did not impact the training.  

One of the possible augmentations which could provide for better labeling, is the birds-eye-view augmentation we performed in Lesson 1.  Looking at the kitti dataset website, many successfuly approaches rely on birds-eye skews of the training images.  One model, named uniview [UNV], managed a 96.69% Maxf with this method, and is currently at the top of the accuracy leaderboard.

FCNs appear to add a significant improvement to accuracy of object recognition and scene labeling without adding much additional overhead to training or labeling speed.  This fairly simple method appears to be a very valuable addition to the tools which can be employed by Self-Driving cars and visual camera input.
