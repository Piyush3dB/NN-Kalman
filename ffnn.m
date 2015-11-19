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
        
        function obj = setWets(obj, Wxh, bh, Why, bo)
            
            
            obj.Wxh = Wxh;
            obj.bh  = bh;
            obj.Why = Why;
            obj.bo  = bo;
        end
        
        function obj = step(obj, xx)
            
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