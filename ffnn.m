classdef ffnn
    
    properties
        nx;
        nh;
        ny;
        ns;
        
        Wxh;
        bh;
        Why;
        bo;
        
        output;
        
    end
    
    methods
        function obj = ffnn(nx, nh, ny)
            obj.nx = nx;
            obj.nh = nh;
            obj.ny = ny;
        end
        
        function obj = setWets(obj, theta, Ns)
            
            nh = obj.nh;
            
            % Restructure reained weights
            W1 = reshape( theta(     1:nh*2), nh, [] );
            W2 = reshape( theta(nh*2+1:end ), 1 , [] );
            
            % Restructure weights and biases
            Wxh = W1(:, 1);
            bh  = W1(:, 2+zeros(1,Ns));
            Why = W2(:, 1:obj.nh);
            bo  = W2(:, obj.nh+ones(1,Ns));
            
            obj.Wxh = Wxh;
            obj.bh  = bh;
            obj.Why = Why;
            obj.bo  = bo;
        end
        
        function obj = step(obj, xx)
            
            obj.
            
            % Input to hidden. Transform each input sample to
            % a number of hidden samples
            h_tp1_ = obj.Wxh * xx + obj.bh;
            h_tp1  = tanh(h_tp1_);
            
            % Hidden to output. Transform a number of hidden samples
            % to an output sample
            obj.output = obj.Why * h_tp1 + obj.bo;
            
        end
        
        
    end
end