function [thetaD_SMs,phiD_SMs,gamma_SMs] = generate_rand_angleD(n_SMs)

% generate random angular combination from uniformly sampled space
n_SMs_large = n_SMs*20;
x1 = rand(n_SMs_large,1)*2-1;
x2 = rand(n_SMs_large,1)*2-1;

mux = 2*x1.*sqrt(1-x1.^2-x2.^2);
muy = 2*x2.*sqrt(1-x1.^2-x2.^2);
muz = 1-2*(x1.^2+x2.^2);

indx =  muz<0 | x1.^2+x2.^2>1;
mux(indx)=[];
muy(indx)=[];
muz(indx)=[];

thetaD = acos(muz)/pi*180;
phiD = atan2(muy,mux)/pi*180;

omega = rand(n_SMs_large,1)*2*pi;
gamma = 1-3*omega/4/pi+omega.^2/8/pi^2;

thetaD_SMs = thetaD(1:n_SMs).';
phiD_SMs = phiD(1:n_SMs).';
gamma_SMs = gamma(1:n_SMs).';

%gamma_SMs(:)=1;

end

