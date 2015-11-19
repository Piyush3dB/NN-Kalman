close all;
clc;
format compact;
clear;

% Set seed for repeatability
rand( 'state', 0);
randn('state', 0);

%% Parameters
% Number of test cases
N  = 20;
% Number of sample points per test case
Ns = 100;
% Number of hidden layers
nh = 8;
% Total number of weights and initialise
%   x2h + bh + h2y + by
nX2h = nh; nBh = nh; nH2y = nh; nBy = 1;
ns = nX2h + nBh + nH2y  + nBy;

%% Generate training data
a = -3.5;
b =  3.5;
x = (b-a).*rand(N,Ns) + a;
%x = 1.2*randn(N,Ns);
noise = 0.1*randn(N,Ns);
y = cos(x) + noise;
% Initialise model prediction
z = y;

%% Initialise the estimated state and covariance
theta = randn(ns,1);
P     = diag([ 100*ones(1, nX2h+nBh) 1000*ones(1, nH2y+nBy) ]);

% State and measurement Covariance Matrices
Q =   0.0001 * eye(ns);
R = 100.0000 * eye(Ns);


%% NN
nnukfObj = c_nnukf(Q, R);
nnukfObj = nnukfObj.init(theta, P);

%% Training model
T1 = 1:N/2;

for k = T1

    % Correct weights
    nnukfObj = nnukfObj.step(x(k,:), y(k,:));
    theta = nnukfObj.x;
    P     = nnukfObj.P;
    
    % Run net and retrieve outputs
    z(k,:)= nnukfObj.ukfObj.hmeas(theta);
    
end

%% Test model
% The equation of the NN is: 
% 
%   y     = Why * tanh( Wxh * x + bh) + bo, and 
%   theta = [W1(:);b1;W2(:);b2].
%
% Therefore,
%        input + bias + output + bias
%   ns = nx*nh +  nh  + nh*ny  + ny, 
%   ns = 1 *nh +  nh  + nh* 1  + nh
%
% which gives the number of hidden nodes is 
%
%   nh = (ns - ny) / (nx + ny + 1);

% Restructure reained weights
W1 = reshape( theta(     1:nh*2), nh, [] );
W2 = reshape( theta(nh*2+1:end ), 1 , [] );


ffnnObj = ffnn(1, nh, 1);

% Select training set and run
% Each input sample produces one output sample
% 'Ns' in 'Ns' out
T2 = N/2+1:N;
for k = T2
    
    % Restructure weights and biases
    Wxh = W1(:, 1);
    bh  = W1(:, 2+zeros(1,Ns));
    Why = W2(:, 1:nh);
    bo  = W2(:, nh+ones(1,Ns));
    
    % Configure weights
    ffnnObj = ffnnObj.setWets(Wxh, bh, Why, bo);
    
    % Feed forward
    ffnnObj = ffnnObj.step(x(k,:));
    
    % Output data
    z(k,:) = ffnnObj.output;
    
end

%% Visualisations
subplot(211)
hold on;
%plot(x(T1,:),y(T1,:),'ob')
plot(x(T1,:),z(T1,:),'.r');
title('training results');

subplot(212)
hold on;
plot(x(T2,:),y(T2,:),'ob');
plot(x(T2,:),z(T2,:),'.r');
title('testing results');

z(end,end)

% z(end)
% ans =
%   -0.7820
   