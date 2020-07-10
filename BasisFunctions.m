clear all, close all
%radial base function (RBF)/normalized radial base function(NRBF)
n=1;
%   Training Data
N_train=20;
x_train=linspace(0,1,N_train)';
g1_train=expon(N_train,x_train,0.3);
g2_train=expon(N_train,x_train,0.5);
g3_train=expon(N_train,x_train,0.7);
g_train=g1_train+g2_train+g3_train;
n1_train=g1_train./g_train;
n2_train=g2_train./g_train;
n3_train=g3_train./g_train;
y_train=((0.5-x_train).^2)+randn(1,N_train)';

%  Validation Data
N_val=200;
x_val=linspace(0,1,N_val)';
g1_val=expon(N_val,x_val,0.3);
g2_val=expon(N_val,x_val,0.5);
g3_val=expon(N_val,x_val,0.7);
g_val = g1_val+g2_val+g3_val;
n1_val=g1_val./g_val;
n2_val=g2_val./g_val;
n3_val=g3_val./g_val;
y_val=((0.5-x_val).^2)+randn(1,N_val)';

% TRAINING
%Create a TRAINING regressor of the three functions
X_train=[g1_train g2_train g3_train];
X_n_train=[n1_train n2_train n3_train];

% PHI TRAINING DATA for RADIAL basis and normalised basis Functions
phi_train=((X_train'*X_train)^(-1)*X_train')*y_train;
phi_n_train=((X_n_train'*X_n_train)^(-1)*X_n_train')*y_train;

% MODEL TRAINING OUTPUT for RADIAL basis and normalised basis Functions
yhat_train=X_train*phi_train;
yhat_n_train=X_n_train*phi_n_train;

% NRMSE TRAINING DATA for RADIAL basis and normalised basis Functions
%NRMSE_train=sqrt(mean((yhat_train-y_train).^2));
%NRMSE_n_train=sqrt(mean((yhat_n_train-y_n_train).^2));

% VALIDATION
%Create a VALIDATION regressor of the three functions
X_val=[g1_val g2_val g3_val];
X_n_val=[n1_val n2_val n3_val];

% MODEL VALIDATION OUTPUT USING PHI TRAINING
yhat_val=X_val*phi_train;
yhat_n_val=X_n_val*phi_n_train;

% NRMSE TRAINING DATA for RADIAL basis and normalised basis Functions
%NRMSE_val=sqrt(mean((yhat_val-y_val).^2));
%NRMSE_n_val=sqrt(mean((yhat_n_val-y_n_val).^2));

% WEIGHTING_LS TAKING NORMALIZED WEIGHTS
Xw_n_train=[ones(N_train,1) x_train];
Xw_n_val=[ones(N_val,1) x_val];
a1=myfunc(n1_train,Xw_n_train,y_train);
a2=myfunc(n2_train,Xw_n_train,y_train);
a3=myfunc(n3_train,Xw_n_train,y_train);
yhatw_n_train=n1_train.*Xw_n_train*a1+n2_train.*Xw_n_train*a2+n3_train.*Xw_n_train*a3;
yhatw_n_val = n1_val.*Xw_n_val*a1+n2_val.*Xw_n_val*a2+n3_val.*Xw_n_val*a3;

% MODEL OUTPUT FOR VALIDATION DATA
% MODEL OUTPUT FOR RADIAL BASIS FUNCTION
figure;
plot(x_val,y_val,'*');
hold on;
plot(x_val,yhat_val);
hold off;
legend({'y_val','yhat_val'});
% MODEL OUTPUT FOR NORMALIZED RADIAL BASIS FUNCTION
figure;
plot(x_val,y_val,'*');
hold on;
plot(x_val,yhat_n_val);
hold off;
legend({'y_n_val','yhat_n_val'});
% MODEL OUTPUT FOR WEIGHTING NORMALIZED RADIAL BASIS FUNCTION
figure;
plot(x_val,y_val,'*');
hold on;
plot(x_val,yhatw_n_val);
hold off;
legend({'y_n_val','yhatw_n_val'});

