function layers = better_cnn_classifier()
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
        layers = [imageInputLayer([28 28 1]);
              convolution2dLayer(5,20);          % (Filter size = 5, No. of filter/neurons = 20)
              reluLayer;
              maxPooling2dLayer(2,'Stride',2);   % (2x2) max pooling with stride = 2
              convolution2dLayer(5,20);          % (Filter size = 5, No. of filter/neurons = 20)
              reluLayer;
              maxPooling2dLayer(2,'Stride',2);   % (2x2) max pooling with stride = 2
              fullyConnectedLayer(10);
              softmaxLayer();
              classificationLayer()];  
end

