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