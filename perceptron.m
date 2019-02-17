close all;
load("data3.mat");


% separting input data from labels
b=ones(200,1);
X=[data(:,1) data(:,2) b];
Y=data(:,3);
close all;

%initializing step size ,tolerance and initial values of theta
theta=zeros(size(X,2),1);
ss=0.1;
tlr=0.001;
N=size(X,1);


% Calling the gradient descent function
[t,thetastar,llos,error]=graddes2(Y,X,ss,tlr,theta);


fx=X*thetastar;
fxp=fx;
%making predictions
fxp(fxp>=0)=1;
fxp(fxp<0)=-1;
figure, plot(1:t, error, 'g', 1:t, llos, 'b');
title('perceptron error(green) and binary clasification error(blue) vs. iterations');
%plotting the graph
x_axis=0:0.01:1;
y_axis=(-thetastar(3)-thetastar(1).*x_axis)/thetastar(2);
figure, plot(x_axis,y_axis,'g'); hold on;
plot(X(:,1),X(:,2),'.');
%title("Decision boundary");
