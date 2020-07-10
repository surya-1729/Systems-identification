clear all, close all
% n-POLYNOMIAL WEIGHTED REGRESSION
n=2;
%   Training Data
N_train=20;
x_train=linspace(0,1,N_train)';
y1=((0.5-x_train).^2);
noise_train=x_train.*randn(1,N_train)';
y_train=y1+noise_train;

%  Validation Data
N_val=200;
x_val=linspace(0,1,N_val)';
y2=((0.5-x_val).^2);
noise_val=x_val.*randn(1,N_val)';
y_val=y2+noise_val;

X_train=[];
X_val=[];
NRMSE_train=[];
NRMSE_w_train=[];
NRMSE_val=[];
NRMSE_w_val=[];
for k= 1:n+1
    
% REGRESSION MATRICES - TRAINING & VALIDATION
X_train=[X_train x_train.^(k-1)];
X_val=[X_val x_val.^(k-1)];

% LEAST SQUARES
% PHI-TRAINING
phi_train=((X_train'*X_train)^(-1)*X_train')*y_train;
% TRAINING MODEL OUTPUT
yhat_train=X_train*phi_train;
% NRMSE-TRAINING
L=sqrt(mean((yhat_train-y_train).^2));
NRMSE_train=[NRMSE_train L];
% VALIDATION MODEL OUTPUT using phi_train
yhat_val= X_val*phi_train;
% NRMSE-VALIDATION
M=sqrt(mean((yhat_val - y_val).^2));
NRMSE_val=[NRMSE_val M];


%WEIGHTED LEAST SQUARES
% WEIGHTS USING VARIANCE OF YHAT AND Y
c=(1./((mean(y_train)-y_train).^2));
W=(diag(c));
% PHI-WEIGHTING-TRAINING 
phi_w_train = ((X_train)'*W*(X_train))^(-1)*(X_train)'*W*(y_train);
% WEIGHTING - MODEL TRAINING OUTPUT
yhat_w_train=X_train*phi_w_train;
% NRMSE -WEIGHTING TRAINING
L_w=sqrt(mean((yhat_w_train-y_train).^2));
NRMSE_w_train=[NRMSE_w_train L_w];
% VALIDATION MODEL OUTPUT using phi_w_train
%d=(1./(yhat_val-y_val).^2);
%D=sqrt(diag(d));
%Wval=(diag(d));
yhat_w_val= X_val*phi_w_train;
% NRMSE-WEIGHTING VALIDATION
M_w=sqrt(mean((yhat_w_val - y_val).^2));
NRMSE_w_val=[NRMSE_w_val M_w];

end

% COMPARE NRMSE TRAINING AND NRMSE WEIGHTED TRAINING
figure;
plot(0:n,NRMSE_train,'r');
hold on;
plot(0:n,NRMSE_w_train,'c')
hold off;
legend({'NRMSE_train','NRMSE_w_train'})

% COMPARE NRMSE VALIDATION AND NRMSE WEIGHTED VALIDATION
figure;
plot(0:n,NRMSE_val,'r');
hold on;
plot(0:n,NRMSE_w_val,'c')
hold off;
legend({'NRMSE_val','NRMSE_w_val'})

% COMPARE MODEL OUTPUT AND WEIGHTED MODEL OUTPUT of VALIDATION DATA
figure;
plot(x_val,y_val,'*');
hold on;
plot(x_val,yhat_val);
hold on;
plot(x_val,yhat_w_val);
hold off;
legend({'y_val','yhat_val','yhat_w_val'})
