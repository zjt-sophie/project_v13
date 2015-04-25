%% Deep Control 
%  modified CS294A/CS294W Stacked Autoencoder Exercise
%
%%======================================================================
%% STEP 0: Here we provide the relevant parameters values that will
%  allow your sparse autoencoder to get good filters; you do not need to 
%  change the parameters below.

inputSize = 8 * 8;
numClasses = 5;
hiddenSizeL1 = 64;    % Layer 1 Hidden Size
hiddenSizeL2 = 13;    % Layer 2 Hidden Size
outputSize = 4; % 4 directions and a background colour

sparsityParam = 0.01;   % desired average activation of the hidden units.
                       % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		               %  in the lecture notes). 
lambda = 0.0001;         % weight decay parameter
beta = 3;              % weight of sparsity penalty term   


%%======================================================================
%% STEP 1: Implement sampleIMAGES
%
%  After implementing sampleIMAGES, the display_network command should
%  display a random sample of 200 patches from the dataset

patches = sampleIMAGES;
trainData = patches;
display_network(patches(:,randi(size(patches,2),200,1)),8);

%%======================================================================
%% STEP 2: Train the first sparse autoencoder
%  This trains the first sparse autoencoder on the unlabelled STL training
%  images.
%  If you've correctly implemented sparseAutoencoderCost.m, you don't need
%  to change anything here.


%  Randomly initialize the parameters
sae1Theta = initializeParameters(hiddenSizeL1, inputSize);

%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the first layer sparse autoencoder, this layer has
%                an hidden size of "hiddenSizeL1"
%                You should store the optimal parameters in sae1OptTheta

%  Use minFunc to minimize the function
addpath minFunc/
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
% function. Generally, for minFunc to work, you
% need a function pointer with two outputs: the
% function value and the gradient. In our problem,
% sparseAutoencoderCost.m satisfies this.
options.maxIter = 400;	  % Maximum number of iterations of L-BFGS to run
% options.maxIter = 1;	  % Maximum number of iterations of L-BFGS to run

options.display = 'on';

[sae1OptTheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                                           inputSize, hiddenSizeL1, ...
                                                           lambda, sparsityParam, ...
                                                           beta, trainData), ...
                               sae1Theta, options);


% -------------------------------------------------------------------------



% %%======================================================================
% %% STEP 2: Train the second sparse autoencoder
% %  This trains the second sparse autoencoder on the first autoencoder
% %  featurse.
% %  If you've correctly implemented sparseAutoencoderCost.m, you don't need
% %  to change anything here.
%  
% load('IMAGES_DTest.mat')% 4 sketches
% 
% patchNum = 12000;
% [trainDataL2,label] = testIMAGES(IMAGES_DTest,patchNum);  
% 
%   [sae1Features] = feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1, ...
%                                           inputSize, trainDataL2);
% 
% %  Randomly initialize the parameters
% sae2Theta = initializeParameters(hiddenSizeL2, hiddenSizeL1);
% 
% %% ---------------------- YOUR CODE HERE  ---------------------------------
% %  Instructions: Train the second layer sparse autoencoder, this layer has
% %                an hidden size of "hiddenSizeL2" and an inputsize of
% %                "hiddenSizeL1"
% %
% %                You should store the optimal parameters in sae2OptTheta
% 
% [sae2OptTheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
%                                                            hiddenSizeL1, hiddenSizeL2, ...
%                                                            lambda, sparsityParam, ...
%                                                            beta, sae1Features), ...
%                                sae2Theta, options);
% 
% % -------------------------------------------------------------------------
% [sae2Features] = feedForwardAutoencoder(sae2OptTheta, hiddenSizeL2, ...
%                                         hiddenSizeL1, sae1Features);

%%======================================================================
%% STEP 5: Visualization

W1L1 = reshape(sae1OptTheta(1:hiddenSizeL1*inputSize), hiddenSizeL1, inputSize);
display_network(W1L1', 12);
                
                print -djpeg weights.jpg   % save the visualization to a file
                
% W1L2 = reshape(sae2OptTheta(1:hiddenSizeL2*hiddenSizeL1), hiddenSizeL2, hiddenSizeL1);
% 
% W = W1L2*W1L1;
% display_network(W', 12);
%                                 
%                 print -djpeg weights2.jpg   % save the visualization to a file
                

% -------------------------------------------------------------------------                
%%======================================================================
% %% STEP 6: Train the Reinforcement Learning
% % Feature_DTest = sae2Features;
% % label_DTest = label;
% % r2  = sqrt(6) / sqrt(hiddenSizeL2+outputSize+1);   % we'll choose weights uniformly from the interval [-r, r]
% % R_W1 = rand(hiddenSizeL2, outputSize ) * 2 * r2 - r2;
% % trainR;
% load('IMAGES_DTestL3'); %values should be 0-1 size:n*m*num.
% 
% IMAGES_DTestL3 = max(min(IMAGES_DTestL3,0.9),0.1);
% 
% trainDataL3 = reshape(IMAGES_DTestL3,size(IMAGES_DTestL3,1)*size(IMAGES_DTestL3,2),size(IMAGES_DTestL3,3));
% [sae1FeaturesL3] = feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1, ...
%                                         inputSize, trainDataL3);
% [sae2FeaturesL3] = feedForwardAutoencoder(sae2OptTheta, hiddenSizeL2, ...
%                                         hiddenSizeL1, sae1FeaturesL3);
% 
% Feature_DTest = sae2FeaturesL3;
% label_DTest = [1,2,3,4,5];
% r2  = sqrt(6) / sqrt(hiddenSizeL2+outputSize+1);   % we'll choose weights uniformly from the interval [-r, r]
% R_W1 = rand(hiddenSizeL2, outputSize ) * 2 * r2 - r2;    
% trainR;

%----------
%%=======
%% Test 
% % Test for layer one --------------------------------------------
% load('IMAGES_DTestL3'); %values should be 0-1 size:n*m*num.
% 
% IMAGES_DTestL3 = max(min(IMAGES_DTestL3,0.9),0.1);
% 
% trainDataL3 = reshape(IMAGES_DTestL3,size(IMAGES_DTestL3,1)*size(IMAGES_DTestL3,2),size(IMAGES_DTestL3,3));
% for i= 1: 1000 
% [sae1FeaturesL3] = feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1, ...
%                                         inputSize, trainDataL3(:,2));
% [sae2FeaturesL3] = feedForwardAutoencoder(sae2OptTheta, hiddenSizeL2, ...
%                                          hiddenSizeL1, sae1FeaturesL3);                                   
%                                     
%  test_sae1FeaturesL3 (:,i)=       sae1FeaturesL3;
%  test_sae2FeaturesL3 (:,i)=       sae2FeaturesL3;
% end  
% %-------------------------------------------------------------------

% [sae2FeaturesL3] = feedForwardAutoencoder(sae2OptTheta, hiddenSizeL2, ...
%                                         hiddenSizeL1, sae1FeaturesL3);
%                                     
%   ts=1;
% Action_RTest = zeros (outputSize, ts);
% for j = 1 : ts
%         Action_RTest (:,j)  = R_W1' * Feature_DTest(:,j);
% end
% [MAX,Index] = max(Action_RTest(1:outputSize,1:5));

% -------------------------------------------------------------------------                
%%======================================================================
%% STEP 7: Test the Network
% % Test the network with 12 sketches
% load('IMAGES_RTest'); 
% % test skecthes one by one
% skpatchNum=1000;
% outputResults = zeros(skpatchNum,size(IMAGES_RTest,3));
% for i = 1 : size(IMAGES_RTest,3)
%     im = IMAGES_RTest(:,:,i) ;
%     [patches,~] = testIMAGES(im,skpatchNum); 
%     [Feature_RTest0]= feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1, ...
%                                              inputSize, patches);
%     [Feature_RTest] = feedForwardAutoencoder(sae2OptTheta, hiddenSizeL2, ...
%                                         hiddenSizeL1, Feature_RTest0);
%                                     
%     [~,testSize] = size(Feature_RTest);
%     Action_RTest = zeros (outputSize, testSize);
%     for j = 1 : testSize
%         Action_RTest (:,j)  = R_W1' * Feature_RTest(:,j);
%     end
%     [MAX,Index] = max(Action_RTest(1:outputSize,1:testSize)); 
%     outputResults(:,i) = Index'; 
% end
% testim2;
% testimR;



%======================================================================
%% STEP 6: Train the Reinforcement Learning part 1
% Feature_DTest = sae2Features;
% label_DTest = label;
% r2  = sqrt(6) / sqrt(hiddenSizeL2+outputSize+1);   % we'll choose weights uniformly from the interval [-r, r]
% R_W1 = rand(hiddenSizeL2, outputSize ) * 2 * r2 - r2;
% trainR;


outputSize =4;
load('IMAGES_DTestL3'); %values should be 0-1 size:n*m*num.


IMAGES_DTestL3 = max(min(IMAGES_DTestL3,0.9),0.1);

trainDataL3 = reshape(IMAGES_DTestL3,size(IMAGES_DTestL3,1)*size(IMAGES_DTestL3,2),size(IMAGES_DTestL3,3));
[sae1FeaturesL3] = feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1, ...
                                        inputSize, trainDataL3(:,1:4));
% [sae2FeaturesL3] = feedForwardAutoencoder(sae2OptTheta, hiddenSizeL2, ...
%                                         hiddenSizeL1, sae1FeaturesL3);
% 
% Feature_DTest = sae2FeaturesL3;

Feature_DTest = sae1FeaturesL3;
label_DTest = [1,2,3,4];
% r2  = sqrt(6) / sqrt(hiddenSizeL2+outputSize+1);   % we'll choose weights uniformly from the interval [-r, r]
% R_W1 = rand(hiddenSizeL2, outputSize ) * 2 * r2 - r2;   
r2  = sqrt(6) / sqrt(hiddenSizeL1+outputSize+1);   % we'll choose weights uniformly from the interval [-r, r]
R_W1 = rand(hiddenSizeL1, outputSize ) * 2 * r2 - r2; 
RtrainTime=1;
trainR;
save('R_W1_Train1');
% -------------------------------------------------------------------------  
%======================================================================
%% STEP 6.5: Train the Reinforcement Learning part 2
% Feature_DTest = sae2Features;
% label_DTest = label;
% r2  = sqrt(6) / sqrt(hiddenSizeL2+outputSize+1);   % we'll choose weights uniformly from the interval [-r, r]
% R_W1 = rand(hiddenSizeL2, outputSize ) * 2 * r2 - r2;
% trainR;


% load('IMAGES_DTest.mat')% 4 sketches
load('IMAGES_RTest2.mat');


IMAGES_DTest = IMAGES_RTest2(:,:,9:12);

patchNum = 12000;
[trainDataLL3,labelLL3] = testIMAGES(IMAGES_DTest,patchNum);  

[sae1FeaturesLL3] = feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1, ...
                                        inputSize, trainDataLL3);
% [sae2FeaturesL3] = feedForwardAutoencoder(sae2OptTheta, hiddenSizeL2, ...
%                                         hiddenSizeL1, sae1FeaturesL3);
% 
% Feature_DTest = sae2FeaturesL3;

Feature_DTest = sae1FeaturesLL3;
label_DTest = labelLL3;
% r2  = sqrt(6) / sqrt(hiddenSizeL2+outputSize+1);   % we'll choose weights uniformly from the interval [-r, r]
% R_W1 = rand(hiddenSizeL2, outputSize ) * 2 * r2 - r2;   
% r2  = sqrt(6) / sqrt(hiddenSizeL1+outputSize+1);   % we'll choose weights uniformly from the interval [-r, r]
% R_W1 = rand(hiddenSizeL1, outputSize ) * 2 * r2 - r2;   
RtrainTime=2;
trainR;
% -------------------------------------------------------------------------                
% 
% %%======================================================================
%% STEP 7: Test the Network
% Test the network with 12 sketches
load('IMAGES_RTest2_v2'); 
IMAGES_RTest = IMAGES_RTest2;
% test skecthes one by one
skpatchNum = 4000;
% outputResults = zeros(skpatchNum,size(IMAGES_RTest,3));
Show=zeros (24,5);
Show2=Show;
 for i =  1: size(IMAGES_RTest,3)
    im = IMAGES_RTest(:,:,i) ;
    [patches,~] = testIMAGES(im,skpatchNum); 
    [Feature_RTest]= feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1, ...
                                             inputSize, patches);
%     [Feature_RTest] = feedForwardAutoencoder(sae2OptTheta, hiddenSizeL2, ...
%                                         hiddenSizeL1, Feature_RTest0);
                                    
    [~,testSize] = size(Feature_RTest);%testSize=skpatchNum
    Action_RTest = zeros (outputSize, testSize);
    for j = 1 : testSize
        Action_RTest (:,j)  = R_W1' * Feature_RTest(:,j); 
        [MAX,Index] = max(Action_RTest(1:outputSize,j));
%         Action_RTest (:,j) = 1 ./ (1 + exp(-1* (R_W1' * Feature_RTest(:,j)))); 
%         if (MAX>0.15)
%             Show2(i,Index)= Show2(i,Index)+1;
%         end
%         if (MAX<0.15)
%             Action_RTest (:,j) =1;
%         end    
%           if (MAX>0.15)
%                Action_RTest (Index,j) =1;
%           end 
    end

    Show(i,1)= mean (Action_RTest(1,:));
    Show(i,2)= mean (Action_RTest(2,:));
    Show(i,3)= mean (Action_RTest(3,:));
    Show(i,4)= mean (Action_RTest(4,:));
    [xx,xxi]= max (Show(i,1:4));
    Show(i,5) =xxi;
    
    if (i==2)
        figure;
        for j=1:4
         subplot(4,4,j);
         plot (Action_RTest (j,:));
        end
    end
    
%     [xx2,xxi2]= max (Show2(i,1:4));
%     Show2(i,5) = xxi2;
 end


