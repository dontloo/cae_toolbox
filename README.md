# cae_toolbox
Matlab implementation of convolutional auto-encoders, based on paper [Stacked Convolutional Auto-Encoders for
Hierarchical Feature Extraction](http://people.idsia.ch/~ciresan/data/icann2011.pdf)
compatible with DeepLearnToolbox https://github.com/rasmusbergpalm/DeepLearnToolbox

[test.m](https://github.com/dontloo/cae_toolbox/blob/master/test.m) is an example of how to set up and train a convolutional auto-encoder, visualize the first layer kernels and reconstruction results alongside the original input, use the training result to initialize a convolutional neural network with the same architecture, and compare error rate with random initialization. 

`cae_check_grad` method in [cae_train.m](https://github.com/dontloo/cae_toolbox/blob/master/cae_train.m) can be turned on to verify the gradients numerically.

An example of 24 first layer convolution weights trained on the KITTI image set

<img src="https://github.com/dontloo/cae_toolbox/blob/master/exmaple_kernels.png" alt="convolution" width="500"/>
