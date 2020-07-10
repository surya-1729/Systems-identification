
function [a] = myfunc(n,l,m)

W=(diag(n));
%a=pinv(W*l)*W*m;
a=((l)'*W*(l))^(-1)*(l)'*W*(m);

end