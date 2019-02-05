## convnet
Convolutional Neural Net to classify dog vs. cat pics.

https://www.kaggle.com/stevenhurwitt/cats-vs-dogs-using-a-keras-convnet

Data available here[https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data]

Basic Convolutional Neural Network (Convnet) implemented in Keras to classify pictures of kitties & puppies :) 

Inspiration for this notebook comes from this [Keras blog post](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) 
and the [VGG ConvNet paper](https://arxiv.org/pdf/1409.1556.pdf), 
code modified from [Jeff Delaney](https://www.kaggle.com/jeffd23). 


## Files include:

**api_convnet.py** - Simple API that takes GET requests of test image number (1-12500) and returns predicted probability. Once the program is run and served on port 5000, command line input looks like this:

```
$ curl -X GET http://127.0.0.1:5000/ -d image='1.jpg'
```

**dataprep.py** - Takes directory of files and converts each image to a numpy array of rbg values.

**runmodel.py** - Trains the CNN with a specified number of epochs.

**showcatdog.py** - Shows the images via matplotlib.

**predcatdog.py** - Makes predictions on the test data.

**convnet.py** - Runs the entire program at once: data processing, model training & predictions.
