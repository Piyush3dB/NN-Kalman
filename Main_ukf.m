close all;
clear;
clc;
format compact;

n = 3;      %number of states
q = 0.1;    %std of process
r = 0.1;    %std of measurement

Q = q^2*eye(n); % covariance of process
R = r^2;        % covariance of measurement

% nonlinear state equations
f=@(x)[ x(2) ; x(3) ; 0.05*x(1)*(x(2)+x(3)) ];

% measurement equation
h=@(x)x(1);

s = [0 ; 0 ; 1];                                % initial state
x = s + q*randn(n,1);      % initial state with noise
P = eye(n);                               % initial state covraiance
N = 20;                                     % total dynamic steps

% allocate memory
xV = zeros(n,N);          %estmate
sV = zeros(n,N);          %actual
zV = zeros(1,N);

for k=1:N
    
    % save actual state
    sV(:,k)= s;
    
    % new measurment and save
    z = h(s) + r*randn;   zV(k) = z;

    % ekf
    [x, P] = ukf(f,x,P,h,z,Q,R);
    
    % save estimate returned from ekf
    xV(:,k) = x;
    
    % state transition and update process
    s = f(s) + q*randn(3,1);
    
end

% plot results
for k=1:3
    subplot(3,1,k)
    plot(1:N, sV(k,:), '-', 1:N, xV(k,:), '--')
    legend('actual', 'estimate')
end
