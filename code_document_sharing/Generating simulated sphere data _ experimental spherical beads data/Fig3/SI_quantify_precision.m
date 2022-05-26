%
x1 = linspace(-1,1,35);
x2 = linspace(-1,1,35);
[X,Y] = meshgrid(x1,x2);
indx  = (X.^2+Y.^2)>=1;
X(indx)=nan;
Y(indx)=nan;

mux_sim = 2*X.*sqrt(1-X.^2-Y.^2);
muy_sim = 2*Y.*sqrt(1-X.^2-Y.^2);
muz_sim = 1-2*(X.^2+Y.^2);
indx = muz_sim<0;
mux_sim(indx) = nan;
muy_sim(indx) = nan;
muz_sim(indx) = nan;

load('data_for_DPPC_chol_v28.mat');
load('est_xyCentered_v28.mat');
%load('est_xyzCentered_v29.mat')
%load('data_for_DPPC_chol_xyz_centered_v29.mat')

%%
mux_sim2 = mux_sim(:);
muy_sim2 = muy_sim(:);
muz_sim2 = muz_sim(:);
mu_grid = [mux_sim2.';muy_sim2.';muz_sim2.'];

for ii = 1:length(mux_sim2)  
   
        mux_cur = mux_sim2(ii);
        muy_cur = muy_sim2(ii);
        muz_cur = muz_sim2(ii);
        thetaD_cur = acos(muz_cur)/pi*180;
        phiD_cur = atan2(muy_cur,mux_cur)/pi*180;

        if isnan(mux_cur)==0 
            
            
            mu_est_ref = [cosd(phiD_ref).*sind(thetaD_ref), sind(phiD_ref).*sind(thetaD_ref), cosd(thetaD_ref)];
            mu_est = [cosd(phiD_cord).*sind(thetaD), sind(phiD_cord).*sind(thetaD), cosd(thetaD)];
            
            distance = mu_est_ref*mu_grid;
            [~,distance_indx] = nanmax(distance,[], 2);
                   
            indx = distance_indx==ii & SM_est_final(:,5)>1000;
            indx_count(ii) = sum(indx);
            mu_GT = [mux_cur,muy_cur,muz_cur];
            mu_est = [cosd(phiD_cord(indx)).*sind(thetaD(indx)), sind(phiD_cord(indx)).*sind(thetaD(indx)), cosd(thetaD(indx))];
            delta_est = acos(mu_est*mu_GT.')/pi*180;     
            mu_est(delta_est>90,:) = -mu_est(delta_est>90,:);
            delta_est = acos(mu_est*mu_GT.')/pi*180; 
            mu_est_mean = median(mu_est,1);
            mu_est_mean = mu_est_mean./norm(mu_est_mean);
    
            delta_std(ii) = std(delta_est);    
            delta_bias(ii) = acos(mu_est_mean*mu_GT.')/pi*180;          
            
            omega_std(ii) = std(omega(indx));
            omega_bias(ii) = median(omega(indx));
        else
            delta_std(ii) = nan;    
            delta_bias(ii) = nan;          
            
            omega_std(ii) = nan;
            omega_bias(ii) = nan;
     
    end
end
mean(indx_count)
%%
delta_std2 = reshape(delta_std,size(mux_sim,1),size(mux_sim,2));
delta_bias2 = reshape(delta_bias,size(mux_sim,1),size(mux_sim,2));
omega_std2 = reshape(omega_std,size(mux_sim,1),size(mux_sim,2));
omega_bias2 = reshape(omega_bias,size(mux_sim,1),size(mux_sim,2));
mux_sim3 = reshape(mux_sim2,size(mux_sim,1),size(mux_sim,2));
muy_sim3= reshape(muy_sim2,size(mux_sim,1),size(mux_sim,2));
muz_sim3= reshape(muz_sim2,size(mux_sim,1),size(mux_sim,2));
Fig1 = figure();
%exportgraphics(Fig1,'BarChart.pdf','ContentType','vector')
subplot(1,8,[1,2]+0);
surf(mux_sim3,muy_sim3,muz_sim3,delta_bias2,'EdgeColor','none'); axis image; view([0 90]); axis off; hold on;
zz = plotCoord(delta_bias2); h = colorbar; caxis([min(delta_bias2,[],'all'),max(delta_bias2,[],'all')]);
h.Title.String = '\theta_{\perp,bias} (\circ)';
subplot(1,8,[1,2]+2);
surf(mux_sim3,muy_sim3,muz_sim3,omega_bias2,'EdgeColor','none'); axis image; view([0 90]); axis off; hold on;
zz = plotCoord(omega_bias2); h = colorbar; caxis([min(omega_bias2,[],'all'),max(omega_bias2,[],'all')]);
h.Title.String = '\Omega_{bias} (sr)';
subplot(1,8,[1,2]+4);
surf(mux_sim3,muy_sim3,muz_sim3,delta_std2,'EdgeColor','none'); axis image; view([0 90]); axis off; hold on; 
zz = plotCoord(delta_std2); h = colorbar; caxis([min(delta_std2,[],'all'),max(delta_std2,[],'all')]);
h.Title.String = '\sigma_{\delta} (\circ)'; 
subplot(1,8,[1,2]+6);
surf(mux_sim3,muy_sim3,muz_sim3,omega_std2,'EdgeColor','none'); axis image; view([0 90]); axis off; hold on;
zz = plotCoord(omega_std2); h = colorbar; caxis([min(omega_std2,[],'all'),max(omega_std2,[],'all')]);
h.Title.String = '\sigma_{\Omega} (sr)';



%%

function zz = plotCoord(data1)

minV = min(reshape([data1],1,[]));
maxV = max(reshape([data1],1,[]));
if minV==maxV
    minV = maxV-1;
end
caxis([minV,maxV]);

zz = max(reshape([data1],1,[]))*40;
if zz<=0
    zz=1;
end
zlim([0,zz]);

plot3(sin(linspace(-pi,pi,100)),cos(linspace(-pi,pi,100)),zz*ones(1,100),'k','linewidth',.2);
plot3(sind(30)*sin(linspace(-pi,pi,100)),sind(30)*cos(linspace(-pi,pi,100)),zz*ones(1,100),'k','linewidth',.2);
plot3(sind(60)*sin(linspace(-pi,pi,100)),sind(60)*cos(linspace(-pi,pi,100)),zz*ones(1,100),'k','linewidth',.2);
for p = 0:30:150
    plot3(sind(p)*[-1,1],cosd(p)*[-1,1],zz*[1,1],'k','linewidth',.2);
end
set(gca,'visible','off');

text(1.2*sind(60),1.2*cosd(60),zz,'$60^\circ$','interpreter','latex','fontsize',9,'HorizontalAlignment','center','VerticalAlignment','middle','rotation',-60)
text(1.2*sind(120),1.2*cosd(120),zz,'$120^\circ$','interpreter','latex','fontsize',9,'HorizontalAlignment','center','VerticalAlignment','middle','rotation',-120)
text(1.2*sind(0),1.2*cosd(0),zz,'$\phi_\textrm{sphere}=0^\circ$','interpreter','latex','fontsize',9,'HorizontalAlignment','center','VerticalAlignment','middle')
text(1.2*sind(240),1.2*cosd(240),zz,'$240^\circ$','interpreter','latex','fontsize',9,'HorizontalAlignment','center','VerticalAlignment','middle','rotation',-240)
text(1.2*sind(300),1.2*cosd(300),zz,'$300^\circ$','interpreter','latex','fontsize',9,'HorizontalAlignment','center','VerticalAlignment','middle','rotation',-300)
text(sind(175)*sind(10),cosd(175)*sind(10),zz,'$0^\circ$','interpreter','latex','fontsize',9,'HorizontalAlignment','left','VerticalAlignment','middle')
text(sind(175)*sind(30),cosd(175)*sind(30),zz,'$30^\circ$','interpreter','latex','fontsize',9,'HorizontalAlignment','left','VerticalAlignment','middle')
text(sind(175)*sind(60),cosd(175)*sind(60),zz,'$60^\circ$','interpreter','latex','fontsize',9,'HorizontalAlignment','left','VerticalAlignment','middle')
text(sind(175)*1.2,cosd(175)*1.2,zz,'$\theta_\textrm{sphere}=90^\circ$','interpreter','latex','fontsize',9,'HorizontalAlignment','left','VerticalAlignment','middle')


end
