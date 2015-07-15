% "test.m" is an example of how to set up and train a convolutional auto-encoder, 
% visualize the first layer kernels, 
% use the training result to initialize a convolutional neural network with the same architecture, 
% and compare error rate with random initialization. 
% A sample of 24 first layer kernels trained on KITTI image set is provided in "exmaple_kernels.png".

load mnist_uint8;

train_x = double(reshape(train_x',28,28,60000))/255;
test_x = double(reshape(test_x',28,28,10000))/255;
train_y = double(train_y');
test_y = double(test_y');

x = train_x(:,:,1:6000);
x = align_data(x);
% set up cae
% input channels | output channels | kernel size | pool size | noise
cae = cae_setup(1,3,5,2,0);

opts.alpha = 0.03;
opts.numepochs = 8;
opts.batchsize = 100;
opts.shuffle = 1;
cae = cae_train(cae, x, opts);

% random select, display
cae_vis(cae,x);

% the following code is based on the DeepLearnToolbox library
% train with small dataset
% set up cnn
x = train_x(:,:,end-99:end);
y = train_y(:,end-99:end);
cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 3, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer
};
opts.alpha = 0.1;
opts.batchsize = 50;
opts.numepochs = 300;

% initialize using cae
cnn = cae_setup_cnn(cae,cnn,x,y);
cnn = cnntrain(cnn, x, y, opts);
figure; plot(cnn.rL);
[er1, bad1] = cnntest(cnn, test_x, test_y);

% random initialize
cnn = cnnsetup(cnn, x, y);
cnn = cnntrain(cnn, x, y, opts);
figure; plot(cnn.rL);
[er2, bad2] = cnntest(cnn, test_x, test_y);
