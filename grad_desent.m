function [t,thetastar,llos,error] = graddes2(Y,X,ss,tlr,theta)
t=0;
llos=[];
error=[];
% thetastar is the updated parameter vectors initialized to a random small number  
thetastar=rand(size(theta));
N=size(X,1);
% initializing theta such that loop is executed
theta=thetastar+tlr*3;
while norm(thetastar-theta)>tlr  
   t=t+1;
   fx=X*thetastar;
   fxp=fx;
   %making predictions
   fxp(fxp>=0)=1;
   fxp(fxp<0)=-1;
   %storing errors in an array
   error(t)=sum(fxp~=Y)/size(X,1);
   %storing risk (called llos in this code) stored in an array  
   llos(t)=-(1/N)*sum((fxp~=Y).*Y.*fx);
   
   llos(isnan(llos))=0;
   theta=thetastar;
   %calculating gradient
   tgrad=sum((fxp~=Y).*[Y Y Y].*X);
   %tgrad=(1/N)*sum(tgrad);
   tgrad=tgrad';
   tgrad=-(1/N)*tgrad;
   %updating the parameter vector
   thetastar=thetastar-ss*tgrad;
end
