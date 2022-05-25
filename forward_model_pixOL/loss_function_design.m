syms Mxx Myy Mzz Mxy Mxz Myz


M = [Mxx, Mxy, Mxz;
      Mxy, Myy, Myz;
      Mxz, Myz, Mzz];
[V,D] = eig(M);

V = simplify(V)
D = simplify(D)


%%
thetaD_GT = 80;  phiD_GT = 120; gamma_GT = 800; 
mux_GT = sind(thetaD_GT)*cosd(phiD_GT);
muy_GT = sind(thetaD_GT)*sind(phiD_GT);
muz_GT = cosd(thetaD_GT);
[muxx_GT,muyy_GT,muzz_GT,muxy_GT,muxz_GT,muyz_GT] = Quickly_rotating_matrix_angleD_gamma_to_M_in(thetaD_GT,phiD_GT,gamma_GT);
muxx1_GT = muxx_GT-(1000-gamma_GT)/3; muyy1_GT = muyy_GT-(1000-gamma_GT)/3; muzz1_GT = muzz_GT-(1000-gamma_GT)/3;
muxy1_GT = muxy_GT; muxz1_GT = muxz_GT; muyz1_GT = muyz_GT;


n_SMs = 1000;
gamma_sim = gamma_GT;
[thetaD_SMs,phiD_SMs,gamma_SMs,mux_SMs,muy_SMs] = generate_angle_uniform_in(gamma_sim );
mux_SMs = sind(thetaD_SMs).*cosd(phiD_SMs);
muy_SMs = sind(thetaD_SMs).*sind(phiD_SMs);
muz_SMs = cosd(thetaD_SMs);
[muxx,muyy,muzz,muxy,muxz,muyz] = Quickly_rotating_matrix_angleD_gamma_to_M_in(thetaD_SMs,phiD_SMs,gamma_SMs);
muxx1 = muxx-(1000-gamma_SMs)/3; muyy1 = muyy-(1000-gamma_SMs)/3; muzz1 = muzz-(1000-gamma_SMs)/3;
muxy1 = muxy; muxz1 = muxz; muyz1 = muyz;



lossf = sqrt((muxx_GT-muxx).^2+(muyy_GT-muyy).^2+(muzz_GT-muzz).^2+(muxy_GT-muxy).^2+(muxz_GT-muxz).^2+(muyz_GT-muyz).^2);
lossf1 = sqrt(abs((muxx1_GT.*muxx1+muyy1_GT.*muyy1+muzz1_GT.*muzz1+2*muxy1_GT.*muxy1+2*muxz1_GT.*muxz1+2*muyz1_GT.*muyz1)-gamma_GT.*gamma_sim));
temp1 = abs((muxx1_GT.*muxx1+muyy1_GT.*muyy1+muzz1_GT.*muzz1+2*muxy1_GT.*muxy1+2*muxz1_GT.*muxz1+2*muyz1_GT.*muyz1));
temp2 = sqrt(gamma_GT.*gamma_sim);
lossf2 = abs(acos(sqrt(temp1)/1000)-acos(temp2/1000));
lossf2_v2 = abs((pi/2-(temp1-temp1.^3/6-3*temp1.^5/40)/1000)-(pi/2-((temp2-temp2.^3/6-3*temp2.^5/40)/1000)));
lossf3 = acos(abs(mux_GT.*mux_SMs+muy_GT.*muy_SMs+muz_GT.*muz_SMs))*gamma_GT/1000;

%lossf1 = muxx.^2+muyy.^2+muzz.^2+2*muxy.^2+2*muxz.^2+2*muyz.^2;

% figure();
% scatter(lossf(:), temp(:),3,'filled','MarkerEdgeAlpha',0.02,'MarkerFaceAlpha',0.02);
% xlabel('2nd moment space'); ylabel('angle space');

x = [-1,1]; y = [-1,1];
figure(); subplot(1,5,1);
imagesc(x,y,lossf); 
hold on; scatter(mux_GT,muy_GT,100,'r+');
title('2nd moment space');
axis image; colorbar;

subplot(1,5,2);
imagesc(x,y,lossf1); 
hold on; scatter(mux_GT,muy_GT,100,'r+');
title('angle space');
axis image; colorbar;
% figure();
% histogram(lossf);

subplot(1,5,3);
imagesc(x,y,lossf2); 
hold on; scatter(mux_GT,muy_GT,100,'r+');
title('angle distance space');
axis image; colorbar;

% subplot(1,5,4);
% imagesc(x,y,lossf2_v2); 
% hold on; scatter(mux_GT,muy_GT,100,'r+');
% title('angle distance space');
% axis image; colorbar;

subplot(1,5,5);
imagesc(x,y,lossf3); 
hold on; scatter(mux_GT,muy_GT,100,'r+');
title('angle distance space');
axis image; colorbar;


%%
function [muxx,muyy,muzz,muxy,muxz,muyz] = Quickly_rotating_matrix_angleD_gamma_to_M_in(polar,azim,gamma)
% transfer the angle from degree unit to the radial unit
%polar = polar/180*pi;
%azim = azim/180*pi;

mux = cosd(azim).*sind(polar);
muy = sind(azim).*sind(polar);
muz = cosd(polar);

muxx = gamma.*mux.^2+(1000-gamma)./3;
muyy = gamma.*muy.^2+(1000-gamma)./3;
muzz = gamma.*muz.^2+(1000-gamma)./3;
muxy = gamma.*mux.*muy;
muxz = gamma.*mux.*muz;
muyz = gamma.*muz.*muy;

end



function [thetaD,phiD,gamma,mux,muy] = generate_angle_uniform_in(gammaValue)

    
    x = linspace(-1,1,1000);
    y = linspace(-1,1,1000);
    [X,Y] = meshgrid(x,y);
    indx = (X.^2+Y.^2)>1;
    X(indx)=nan;
    Y(indx)=nan;
    mux = X;
    muy = Y;
    %updown = sign(rand(size(X))-0.5);
    muz = real(sqrt(1-X.^2-Y.^2));

  
    
    thetaD = acos(muz)/pi*180;
    phiD = atan2(muy,mux)/pi*180;
    gamma = phiD; gamma(:)=gammaValue;
    

    
end
