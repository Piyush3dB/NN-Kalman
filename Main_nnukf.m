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
nnObj = c_nnukf(Q, R);
nnObj = nnObj.init(theta, P);

%% Training model
T1 = 1:N/2;
for k = T1
    [theta, P, z(k,:)] = nnukf(theta, P, x(k,:), y(k,:), Q, R);
    
    nnObj = nnObj.step(x(k,:), y(k,:));
    theta = nnObj.x;
    P     = nnObj.P;
    z(k,:)= nnObj.e;
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

% Select training set and run
% Each input sample produces one output sample
% 'Ns' in 'Ns' out
T2 = N/2+1:N;
for k = T2
    
    % Input data
    xx  = x(k,:);
    
    % Input to hidden weights and biases
    % 1. First column is weights
    % 2. Second column is bias. Replicate 'Ns' number of times. 
    Wxh = W1(:, 1);
    bh  = W1(:, 2+zeros(1,Ns));
    
    % Hidden to output weights and biases
    % 1. First column is weights
    % 2. Second column is bias. Replicate 'Ns' number of times.
    Why = W2(:, 1:nh);
    bo  = W2(:, nh+ones(1,Ns));
    
    % Input to hidden. Transform each input sample to 
    % a number of hidden samples
    h_tp1_ = Wxh * xx + bh;
    h_tp1  = tanh(h_tp1_);
    
    % Hidden to output. Transform a number of hidden samples
    % to an output sample
    yy = Why * h_tp1 + bo;
    
    % Output data
    z(k,:) = yy;
    
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

yy(end)

% yy(end)
% ans =
%   -0.7820
   