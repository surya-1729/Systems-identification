clear all, close all
% n-POLYNOMIAL REGRESSION
n=3;
%   Training Data
N_train=20;
x_train=linspace(0,1,N_train)';
y1=(4*(x_train-0.5).^2);
noise_train=randn(1,N_train)';
y_train=y1+noise_train;

%  Validation Data
N_val=200;
x_val=linspace(0,1,N_val)';
y2=(4*(x_val-0.5).^2);
noise_val=randn(1,N_val)';
y_val=y2+noise_val;

X_train=[];
NRMSE_train=[];
X_val=[];
NRMSE_val=[];
for k= 1:n+1
% REGRESSION MATRICES FOR TRAINING AND VALIDATION DATA 
X_train=[X_train x_train.^(k-1)];
X_val=[X_val x_val.^(k-1)];
% LEAST SQUARES ALGORITHM
% PHI TRAINING
phi_train=((X_train'*X_train)^(-1)*X_train')*y_train;
% MODEL OUTPUT TRAINING DATA
yhat_train=X_train*phi_train;
% MODEL OUTPUT VALIDATION DATA
yhat_val= X_val*phi_train;
% NRMSE TRAINING DATA
L=sqrt(mean((yhat_train-y_train).^2));
NRMSE_train=[NRMSE_train L];
% NRMSE VALIDATION DATA
M=sqrt(mean((yhat_val - y_val).^2));
NRMSE_val=[NRMSE_val M];
end

% NRMSE FOR TRAINING AND VALIDATION DATA
figure;
plot(0:n,NRMSE_train,'g');
hold on;
plot(0:n,NRMSE_val,'r');
hold off;
legend({'NRMSE_train','NRMSE_val'})
% Model Output(yhat) and Actual Value(y) for VALIDATION DATA
figure;
plot(x_val,y_val,'*');
hold on;
plot(x_val,yhat_val);
hold off;
legend({'y_val','yhat_val'})
