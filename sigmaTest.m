
close all;
clear;
clc;
format compact;


x = [0;0];

c = 1;
P = [1 0 ; 0 1]



L = 2;                                      %numer of states

alpha = 1e-3;                               %default, tunable
ki    = 0;                                  %default, tunable
beta  = 2;                                  %default, tunable

lambda = alpha^2*(L+ki)-L;                    %scaling factor
c = L + lambda;                                 %scaling factor

Wm    = [lambda/c 0.5/c+zeros(1,2*L)];           %weights for means
Wc    = Wm;
Wc(1) = Wc(1)+(1-alpha^2+beta);               %weights for covariance
c     = sqrt(c);
%X     = sigmas(x,P,c);                            %sigma points around x




N = numel(x);
A = c*chol(P)';
Y = x(:,ones(1,N));
X = [x Y+A Y-A]; 