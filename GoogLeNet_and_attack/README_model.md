Following Szegedy et al[1], GoogLeNet 2014 is implemented, with some minor changes, as a binary classifier. The network is made up by 22 layers between convolutional, inception and pooling layers. The inception layers have been implemented following the architeture described in Szegedy et al[2] which factorize convolutions with large filter size in symmetric and asymmetric convolutions, making the layer computationally more efficient. Having different convolutional layers in parallel with different kernel sizes, the inception layer allow the model itself to choose which filter size is more effective. Dimensionality inside the layers is kept from getting too large thanks to the 1x1 convolutional layers which decreases is. Furthermor, including ReLU activation, they serve as dual purpose layers.

Being deep, the model is susceptible to vanishing gradient issues, which, according to [1], can be taken care of by adding two Auxiliary Classifiers along the midsection of the network to help evolve the lower features. This argument is partially examined again in [2] where Auxiliary Classifiers seem to have some influence only in the late epochs of training and almost no effect at all if we refer to the lowest output branch. Therefore, instead of pushing the gradient signal down to lowest layers, they are thought to be acting as regularizers as they generally lead to better results in the main output when having a dropout layer.

Here, this model is employed in classifying lymphs node images in those with and without metastatic tissue. Due to memory limitations I had the model trained for a limited number of epochs compared to the papers. In fact, earlystopping is implemented along with output layers with 1 unit and sigmoid activation function. 



\
\
\
\



[1] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich. In Going Deeper with Convolutions, 2014. https://arxiv.org/abs/1409.4842 \
[2] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna. In Rethinking the Inception Architecture for Computer Vision, 2015. https://arxiv.org/abs/1512.00567
