classdef c_nnukf
    % NNUKF     A function using the UKF to training a MLP NN
    %
    % [theta,P,z]=nnukf(theta,P,x,y,Q,R) searches the optimal parameters,
    %
    % - theta
    % of a MLP NN based on a set of training data with input x and output y.
    %
    % Input:
    %   - theta: initial guess of MLP NN parameter. The network structure is
    %
    %   - ns : number of parameters network parameters,
    %   - nx : the number of inputs (size of x),
    %   - ny : the number of output (size of y)
    %
    % The equation of the NN is:
    %
    %   y     = W2 * tanh( W1 * x + b1) + b2, and
    %   theta = [W1(:) ; b1 ; W2(:) ; b2].
    %
    % Therefore,
    %
    %   ns = nx * nh + nh + nh * ny + ny,
    %
    % which gives the number of hidden nodes is
    %
    %   nh = (ns - ny) / (nx + ny + 1);
    %
    %   P: the covariance of the initial theta. Needs to be tuned to get good
    %   training performance.
    %   x and y: input and output data for training. For batch training, x and
    %   y should be arranged in such a way that each observation corresponds to
    %   a column.
    %   Q: the virtual process covariance for theta, normally set to very small
    %   values.
    %   R: the measurement covariance, dependen on the noise level of data, tunable.
    %
    % Example: a NN model to approximate the sin function
    
    properties
        
        
        
        % Inputs
        f;
        h;
        ukfObj;
        
        e;
        x;
        P;
        
    end
    
    methods
        function obj = c_nnukf(Q, R)
            
            % State Transition
            obj.f = @(u)u;
            
            % Measurement Equation
            obj.h = NaN;
            
            % Create instance of Filter
            obj.ukfObj = c_ukf(obj.f, obj.h, Q, R);
            
        end
        
        function obj = init(obj, x, P)
            % Initialise the Kalman filter
            obj.ukfObj = obj.ukfObj.init(x, P);
            
        end
        
        function obj = step(obj, input, output)
            
            % State Transition
            %f = @(u)u;
            
            % Measurement Equation
            h = @(u) obj.nn( u, input, size(output,1) );
            obj.ukfObj.hmeas = h;
            
            % Run UKF
            obj.ukfObj = obj.ukfObj.step(output(:));
            
            % Retrieve newly estimated parameters
            obj.x = obj.ukfObj.x;
            obj.P = obj.ukfObj.P;
            
            % Calculate net output using new weights
            obj.e = h(obj.ukfObj.x);
            
        end
        
    end
    
    methods(Static)
        % Helper functions internal to the class
        
        function y=nn(theta,x,ny)
            
            % The equation of the NN is:
            %
            %   y     = Why * tanh( Wxh * x + bh) + bo, and
            %   theta = [ Wxh(:) ; b1 ; Why(:) ; bo ].
            %
            % Therefore,
            %
            %   ns = nx * nh + nh + nh * ny + ny,
            %
            % which gives the number of hidden nodes is
            %
            %   nh = (ns - ny) / (nx + ny + 1);
            
            [nx,N] = size(x);
            
            ns = numel(theta);
            nh = (ns-ny)/(nx+ny+1);
            
            W1 = reshape(theta(1:nh*(nx+1)),nh,[]);
            Wxh = W1(:,1:nx);
            bh  = W1(:,nx+ones(1,N));
            
            W2 = reshape(theta(nh*(nx+1)+1:end),ny,[]);
            Why = W2(:,1:nh);
            bo  = W2(:,nh+ones(1,N));
            
            % Input to hidden
            h_tp1 = tanh(Wxh * x + bh);
            
            % Hidden to output
            y = Why * h_tp1 + bo;
            
            y  = y(:);
            
            
        end
    end
end