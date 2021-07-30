function [encoderNet, decoderNet] = run_training(numEpochs, lr, momentum, encoderNet,decoderNet, XTrain, XTest)
executionEnvironment = "auto";

numTrainImages = size(XTrain,4);

miniBatchSize = 512;
numIterations = floor(numTrainImages/miniBatchSize);
iteration = 0;

avgGradientsEncoder = [];
avgGradientsSquaredEncoder = [];
avgGradientsDecoder = [];
avgGradientsSquaredDecoder = [];

if momentum
    for epoch = 1:numEpochs
        tic;
        for i = 1:numIterations
            iteration = iteration + 1;
            idx = (i-1)*miniBatchSize+1:i*miniBatchSize;
            XBatch = XTrain(:,:,:,idx);
            XBatch = dlarray(single(XBatch), 'SSCB');

            if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
                XBatch = gpuArray(XBatch);           
            end 

            [infGrad, genGrad] = dlfeval(...
                @modelGradients, encoderNet, decoderNet, XBatch);

            [decoderNet.Learnables, avgGradientsDecoder, avgGradientsSquaredDecoder] = ...
                adamupdate(decoderNet.Learnables, ...
                    genGrad, avgGradientsDecoder, avgGradientsSquaredDecoder, iteration, lr);
            [encoderNet.Learnables, avgGradientsEncoder, avgGradientsSquaredEncoder] = ...
                adamupdate(encoderNet.Learnables, ...
                    infGrad, avgGradientsEncoder, avgGradientsSquaredEncoder, iteration, lr);
        end
        elapsedTime = toc;

        [z, zMean, zLogvar] = sampling(encoderNet, XTest);
        xPred = sigmoid(forward(decoderNet, z));
        elbo = ELBOloss(XTest, xPred, zMean, zLogvar);
        fprintf("Epoch : "+epoch+" Test ELBO loss = "+gather(extractdata(elbo))+...
            ". Time taken for epoch = "+ elapsedTime + "s \n")    
    end
else
    for epoch = 1:numEpochs
    tic;
    for i = 1:numIterations
        iteration = iteration + 1;
        idx = (i-1)*miniBatchSize+1:i*miniBatchSize;
        XBatch = XTrain(:,:,:,idx);
        XBatch = dlarray(single(XBatch), 'SSCB');

        if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
            XBatch = gpuArray(XBatch);           
        end 

        [infGrad, genGrad] = dlfeval(...
            @modelGradients, encoderNet, decoderNet, XBatch);

        [decoderNet.Learnables, avgGradientsDecoder, avgGradientsSquaredDecoder] = ...
            adamupdate(decoderNet.Learnables, ...
                genGrad, [], [], iteration, lr);
        [encoderNet.Learnables, avgGradientsEncoder, avgGradientsSquaredEncoder] = ...
            adamupdate(encoderNet.Learnables, ...
                infGrad, [], [], iteration, lr);
    end
    elapsedTime = toc;

    [z, zMean, zLogvar] = sampling(encoderNet, XTest);
    xPred = sigmoid(forward(decoderNet, z));
    elbo = ELBOloss(XTest, xPred, zMean, zLogvar);
    fprintf("Epoch : "+epoch+" Test ELBO loss = "+gather(extractdata(elbo))+...
        ". Time taken for epoch = "+ elapsedTime + "s \n") 
    end
end
% ADD MOMENTUM OFF VARIABLE
end

