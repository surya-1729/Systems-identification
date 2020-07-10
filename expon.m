

function g = expon(N,l,m)
g=zeros(N,1);
for i =1:N
g(i,1) = (exp((-0.5).*(l(i)-m)./0.3).^2);
end
end
