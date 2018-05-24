function [ nll, g, H ] =Loss_logistic(X,y,w )
% w(feature,1)
% X(instance,feature)
% y(instance,1)
N = length(y);
Xw=X*w;
yXw = y.*Xw;
nll=sum(log(1+exp(-yXw)))/N;
if nargout>2
    sig = 1./(1+exp(-yXw));
    g = -X.'*(y.*(1-sig))/N;
    H = X.'*diag(sparse(sig.*(1-sig)))*X/N;
end


end

