%% generate a spherical beads
load('colorSpace.mat');
R = 1000; % nm
r = 100;
[x,y,z,thetaD,phiD] = generate_rand_SMs(1000);
X = R*x; Y = R*y; Z = R*z+R;

figure(); hold on
scatter3(X,Y,Z,[],phiD,'filled'); axis image; colormap(squeeze(colorSpace));
plot3([X-r*sind(thetaD).*cosd(phiD),X,X+r*sind(thetaD).*cosd(phiD)].',...
     [Y-r*sind(thetaD).*sind(phiD),Y,Y+r*sind(thetaD).*sind(phiD)].',...
     [Z+r*cosd(thetaD),Z,Z-r*cosd(thetaD)].','k');

% [X,Y,Z] = sphere;
% X(Z>0)=nan;Y(Z>0)=nan;Z(Z>0)=nan;
%surf(X*R,Y*R,Z*R); colormap('hot')

%%
figure();
subplot(1,3,1);
scatter(thetaD,z*R+1000);
ylim([0,700]);

subplot(1,3,2);
scatter(phiD,x*R);

subplot(1,3,3);
scatter(phiD,y*R);
%%
function [x,y,z,thetaD,phiD] = generate_rand_SMs(n_SMs)

% generate random angular combination from uniformly sampled space
n_SMs_large = n_SMs;
x1 = rand(n_SMs_large,1)*2-1;
x2 = rand(n_SMs_large,1)*2-1;
x = 2*x1.*sqrt(1-x1.^2-x2.^2);
y = 2*x2.*sqrt(1-x1.^2-x2.^2);
z = 1-2*(x1.^2+x2.^2);

indx = z>0 | x1.^2+x2.^2>1;
x(indx)=[];
y(indx)=[];
z(indx)=[];

thetaD = atan2(sqrt(x.^2+y.^2),z)/pi*180;
phiD = atan2(y,x)/pi*180+180;

phiD(thetaD>90)=phiD(thetaD>90)+180; phiD = rem(phiD,360);
thetaD(thetaD>90)=180-thetaD(thetaD>90);
%phiD =phiD+180;
phiD(phiD>180)=-360+phiD(phiD>180);


end


function [thetaD_SMs,phiD_SMs,gamma_SMs] = generate_rand_SMs2(n_SMs)

% generate random angular combination from uniformly sampled space
n_SMs_large = n_SMs*20;
x1 = rand(n_SMs_large,1)*2-1;
x2 = rand(n_SMs_large,1)*2-1;
x = 2*x1.*sqrt(1-x1.^2-x2.^2);
muy = 2*x2.*sqrt(1-x1.^2-x2.^2);
muz = 1-2*(x1.^2+x2.^2);

indx =  muz<0 | x1.^2+x2.^2>1;
mux(indx)=[];
muy(indx)=[];
muz(indx)=[];

thetaD = asin(muz)/pi*180;
phiD = atan2(muy,muyx)/pi*180;

omega = rand(n_SMs_large,1)*2*pi;
gamma = 1-3*omega/4/pi+omega.^2/8/pi^2;

thetaD_SMs = thetaD(1:n_SMs).';
phiD_SMs = phiD(1:n_SMs).';
gamma_SMs = gamma(1:n_SMs).';

%gamma_SMs(:)=1;

end