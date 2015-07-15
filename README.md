# cae_toolbox
library for convolutional auto-encoders, based on http://people.idsia.ch/~ciresan/data/icann2011.pdf
compatible with DeepLearnToolbox https://github.com/rasmusbergpalm/DeepLearnToolbox

"test.m" is an example of how to set up and train a convolutional auto-encoder, visualize the first layer kernels and reconstruction results alongside the original input, use the training result to initialize a convolutional neural network with the same architecture, and compare error rate with random initialization. A sample of 24 first layer kernels trained on the KITTI image set is provided in "exmaple_kernels.png".
