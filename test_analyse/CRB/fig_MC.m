%exportgraphics(Fig1,'BarChart.pdf','ContentType','vector')
%%
frame_count = 0;
thetaD_simulate = (5:4:90);
phiD_simulate = (0:5:180);
omega_simulate = 0;
gamma_simulate = 1-3*omega_simulate/4+omega_simulate.^2/8/pi^2;
frames_per_state = 200;


% figure();
% scatter3(reshape(mux_sim,[],1),reshape(muy_sim,[],1),reshape(muz_sim,[],1));  axis image;

%%
%exportgraphics(Fig1,'BarChart.pdf','ContentType','vector')
locGT = [0,0,0];

state_cur = 0;
for ii = 1:length(thetaD_simulate)  
    for jj = 1:length(phiD_simulate)
        for kk = 1:length(omega_simulate)
            state_cur = state_cur+1;
                omega_cur = omega_simulate(kk);
                theta_cur = thetaD_simulate(ii)/pi*180;
                phi_cur = phiD_simulate(jj)/pi*180;

                mux_cur = sin(theta_cur)*cos(phi_cur);
                muy_cur = sin(theta_cur)*sin(phi_cur);
                muz_cur = cos(theta_cur);
                
                frame_min = (state_cur-1)*frames_per_state+1;
                frame_max = (state_cur)*frames_per_state;
                
                indx = est(:,1)>=frame_min & est(:,1)<=frame_max;
                est_cur = est(indx,:);
                SM_est_cur = est_cur(:,1:4);
                Angle_save_cur = est_cur(:,[1,5,6,7]);
                Angle_save_cur(:,4)=[3/sqrt(8)-sqrt(Angle_save_cur(:,4)+1/8)]*sqrt(8)*pi;

                if sum(indx)>0
                    
                           
                    [gc,grps] = groupcounts(SM_est_cur(:,1));
                    grps_name = grps(gc==1);
                    sigma_x = 30; sigma_y = 30;
                    
                    indx1 = ismember(SM_est_cur(:,1),grps_name);      
                    indx = ismember(SM_est_cur(:,1),grps_name)...
                            & abs((SM_est_cur(:,4)-locGT(2)))<3*sigma_y & abs((SM_est_cur(:,3)-locGT(1)))<3*sigma_y;
                    %
                    %indx = indx1;
                    theta_est = Angle_save_cur(indx,2);
                    phi_est = Angle_save_cur(indx,3);
                    %histogram(phi_est);
                    omega_est = Angle_save_cur(indx,4);
                   
                    
                    % calculate the angle between the GT and the estimations
                    mu_GT = [cosd(phi_cur)*sind(theta_cur), sind(phi_cur)*sind(theta_cur), cosd(theta_cur)];
                    mu_est = [cosd(phi_est).*sind(theta_est), sind(phi_est).*sind(theta_est), cosd(theta_est)];
                    
                    delta_est = acos(mu_est*mu_GT.')/pi*180;     
                    mu_est(delta_est>90,:) = -mu_est(delta_est>90,:);
                    delta_est = acos(mu_est*mu_GT.')/pi*180; 
                    mu_est_mean = median(mu_est,1);
                    mu_est_mean = mu_est_mean./norm(mu_est_mean);
                    
        
                    delta_est_post = delta_est;
                    delta_est_post(mu_est(:,1)<mu_GT(:,1)) = -delta_est_post(mu_est(:,1)<mu_GT(:,1));
            
                 
            
                    delta_std(ii,jj,kk) = std(delta_est);
                    omega_std(ii,jj,kk) = std(omega_est);
                    
                    %delta_bias(ii,jj,kk) = median(delta_est);
                    delta_bias(ii,jj,kk) = acos(mu_est_mean*mu_GT.')/pi*180; 
                    omega_bias(ii,jj,kk) = median(omega_est)-omega_cur;
                    
                    x_std(ii,jj,kk) = std(SM_est_cur(indx,2));
                    y_std(ii,jj,kk) = std(SM_est_cur(indx,3));
                    %z_std(ii,jj,kk) = std(SM_est_cur(indx,4));
                    
                    x_mean(ii,jj,kk) = median(SM_est_cur(indx,2));
                    y_mean(ii,jj,kk) = median(SM_est_cur(indx,3));
                    %z_mean(ii,jj,kk) = median(SM_est_cur(indx,4));
                    %jaccard_count(ii,jj,kk) = sum(indx)/sum(indx1);
                    %jaccard_count(ii,jj,kk) = sum(indx)/200;

                else 

                    delta_std(ii,jj,kk) = nan;
                    omega_std(ii,jj,kk) = nan;
                    
                    %delta_bias(ii,jj,kk) = mean(delta_est);
                    delta_bias(ii,jj,kk) = nan; 
                    omega_bias(ii,jj,kk) = nan;
                    
                    x_std(ii,jj,kk) = nan;
                    y_std(ii,jj,kk) = nan;
                    %z_std(ii,jj,kk) = nan;
                    
                    x_mean(ii,jj,kk) = nan;
                    y_mean(ii,jj,kk) = nan;
                    %z_mean(ii,jj,kk) = nan;
                    
                    %jaccard_count(ii,jj,kk) = nan;

                    
                end
            
        end
    end
end

%%












%% 
%C:\Users\wu.t\OneDrive - Washington University in St. Louis\github\PSF-optimization\opt PSF characterization
%example point for correlation analysis
% ii = 17,jj=11
figure(); scatter(SM_est_cur(indx1,4),Angle_save_cur(indx1,2),4,'filled'); box on; xlabel('h(nm)'); ylabel('\theta(\circ)'); hold on; scatter(700,theta_cur,50,'r+','LineWidth',2);
hold on; scatter(800,66.39,50,'+','MarkerEdgeColor','#7E2F8E','LineWidth',2);

figure(); scatter(SM_est_cur(indx1,2)-58.5,Angle_save_cur(indx1,2),4,'filled'); box on; xlabel('x(nm)'); ylabel('\theta(\circ)'); hold on; scatter(0,theta_cur,50,'r+','LineWidth',2);
hold on; scatter(-195,66.39,50,'+','MarkerEdgeColor','#7E2F8E','LineWidth',2);
legend('','ground truth','biased estimation');

%% biased NLL
load('biased_NLL.mat');
figure(); histogram(NLL_GT_save-NLL_cur_out_save,10);
xlabel('l_{NLL}^{GT}-l_{NLL}^{est}'); ylabel('count');
%% Jaccard
Fig1 = figure('Units','inches','InnerPosition',[1,1,6.84/4*3,3]);
panelNum = [1,3];
for ii = 1:3
omega_indx = ii;
panelCoulumnIndx = 1;
subplot(1,3,ii);
%imagesc(x1,x2, delta_std(:,:,omega_indx)); axis image; colorbar; xlabel('\phi(\circ)'); ylabel('\theta(\circ)'); title('\sigma_\delta (\circ)');
surf(mux_sim,muy_sim,muz_sim,jaccard_count(:,:,omega_indx)*100,'EdgeColor','none'); axis image; view([0 90]); axis off; hold on;
zz = plotCoord(jaccard_count(:,:,omega_indx)*100); h = colorbar;
h.Title.String = 'percentile';
end

%% std of angle estimations for omega, and angle

Fig1 = figure('Units','inches','InnerPosition',[1,1,6.84,3]);
panelNum = [3,4];

for ii = 1:3
omega_indx = ii;
panelCoulumnIndx = 1;
subplot(panelNum(1),panelNum(2),1+(ii-1)*panelNum(2));
%imagesc(x1,x2, delta_std(:,:,omega_indx)); axis image; colorbar; xlabel('\phi(\circ)'); ylabel('\theta(\circ)'); title('\sigma_\delta (\circ)');
surf(mux_sim,muy_sim,muz_sim,delta_std(:,:,omega_indx),'EdgeColor','none'); axis image; view([0 90]); axis off; hold on;
zz = plotCoord(delta_std(:,:,omega_indx)); h = colorbar;
h.Title.String = '\sigma_\delta(\circ)'; caxis([min(delta_std,[],'all'),max(delta_std,[],'all')]);

subplot(panelNum(1),panelNum(2),2+(ii-1)*panelNum(2));
%imagesc(x1,x2, omega_std(:,:,omega_indx)); axis image; colorbar; xlabel('\phi(\circ)'); ylabel('\theta(\circ)'); title('\sigma_\Omega (sr)');
surf(mux_sim,muy_sim,muz_sim,omega_std(:,:,omega_indx),'EdgeColor','none'); axis image; view([0 90]); axis off; hold on;
zz = plotCoord(omega_std(:,:,omega_indx)); h = colorbar;
h.Title.String = '\sigma_\Omega (sr)'; caxis([min(omega_std,[],'all'),max(omega_std,[],'all')]);

subplot(panelNum(1),panelNum(2),3+(ii-1)*panelNum(2));
%imagesc(x1,x2, sqrt(x_std(:,:,omega_indx).^2+y_std(:,:,omega_indx))); axis image; colorbar; xlabel('\phi(\circ)'); ylabel('\theta(\circ)'); title('\sigma_L (nm)');
surf(mux_sim,muy_sim,muz_sim,sqrt(x_std(:,:,omega_indx).^2+y_std(:,:,omega_indx).^2),'EdgeColor','none'); axis image; view([0 90]); axis off; hold on;
zz = plotCoord(sqrt(x_std(:,:,omega_indx).^2+y_std(:,:,omega_indx).^2)); h = colorbar;
h.Title.String = '\sigma_L (nm)'; caxis([min(sqrt(x_std.^2+y_std.^2),[],'all'),max(sqrt(x_std.^2+y_std.^2),[],'all')]);

subplot(panelNum(1),panelNum(2),4+(ii-1)*panelNum(2));
%imagesc(x1,x2, z_std(:,:,omega_indx)); axis image; colorbar;  xlabel('\phi(\circ)'); ylabel('\theta(\circ)'); title('\sigma_z (nm)'); 
surf(mux_sim,muy_sim,muz_sim,z_std(:,:,omega_indx),'EdgeColor','none'); axis image; view([0 90]); axis off; hold on;
zz = plotCoord(z_std(:,:,omega_indx)); h = colorbar;
h.Title.String = '\sigma_z (nm)'; caxis([min(z_std,[],'all'),max(z_std,[],'all')]);
end
whitebg('w');
%%
%-----------------------------
Fig1 = figure('Units','inches','InnerPosition',[1,1,6.84,3]);
panelNum = [3,4];

for ii = 1:3
omega_indx = ii;
panelCoulumnIndx = 1;

subplot(panelNum(1),panelNum(2),1+(ii-1)*panelNum(2));
%imagesc(x1,x2, delta_bias(:,:,omega_indx)); axis image; colorbar; xlabel('\phi(\circ)'); ylabel('\theta(\circ)'); title('\delta-\delta_0 (\circ)');
surf(mux_sim,muy_sim,muz_sim,delta_bias(:,:,omega_indx),'EdgeColor','none'); axis image; view([0 90]); axis off; hold on;
zz = plotCoord(delta_bias(:,:,omega_indx)); h = colorbar;
h.Title.String = '\delta-\delta_0 (\circ)'; caxis([min(delta_bias,[],'all'),max(delta_bias,[],'all')]);

subplot(panelNum(1),panelNum(2),2+(ii-1)*panelNum(2));
%imagesc(x1,x2, omega_bias(:,:,omega_indx)); axis image; colorbar; xlabel('\phi(\circ)'); ylabel('\theta(\circ)'); title('\Omega-\Omega_0 (sr)');
surf(mux_sim,muy_sim,muz_sim,omega_bias(:,:,omega_indx),'EdgeColor','none'); axis image; view([0 90]); axis off; hold on;
zz = plotCoord(omega_bias(:,:,omega_indx)); h = colorbar;
h.Title.String = '\Omega-\Omega_0 (sr)';  caxis([min(omega_bias,[],'all'),max(omega_bias,[],'all')]);
 
subplot(panelNum(1),panelNum(2),3+(ii-1)*panelNum(2));
%imagesc(x1,x2, sqrt((x_mean(:,:,omega_indx)-locGT(1)).^2+(y_mean(:,:,omega_indx)-locGT(2)).^2)); axis image; colorbar;  xlabel('\phi(\circ)'); ylabel('\theta(\circ)'); title('L-L_0 (nm)');
surf(mux_sim,muy_sim,muz_sim,sqrt((x_mean(:,:,omega_indx)-locGT(1)).^2+(y_mean(:,:,omega_indx)-locGT(2)).^2),'EdgeColor','none'); axis image; view([0 90]); axis off; hold on;
zz = plotCoord(sqrt((x_mean(:,:,omega_indx)-locGT(1)).^2+(y_mean(:,:,omega_indx)-locGT(2)).^2)); h = colorbar;
h.Title.String = 'L-L_0 (nm)';  caxis([min(sqrt((x_mean-locGT(1)).^2+(y_mean-locGT(2)).^2),[],'all'),max(sqrt((x_mean-locGT(1)).^2+(y_mean-locGT(2)).^2),[],'all')]);

subplot(panelNum(1),panelNum(2),4+(ii-1)*panelNum(2));
%imagesc(x1,x2, z_mean(:,:,omega_indx)-locGT(3)); axis image; colorbar;  xlabel('\phi(\circ)'); ylabel('\theta(\circ)'); title('z-z_0 (nm)');
surf(mux_sim,muy_sim,muz_sim,z_mean(:,:,omega_indx)-locGT(3),'EdgeColor','none'); axis image; view([0 90]); axis off; hold on;
zz = plotCoord(z_mean(:,:,omega_indx)-locGT(3)); h = colorbar; caxis([min(z_mean-locGT(3),[],'all'),max(z_mean-locGT(3),[],'all')]);
h.Title.String = 'z-z_0 (nm)';

end

%% quantification
figure(); subplot(2,3,1); histogram(SM_est_cur(:,6),10);
subplot(2,3,2); histogram(SM_est_cur(:,7),10);
subplot(2,3,3); histogram(SM_est_cur(:,8),10);
subplot(2,3,4); histogram(SM_est_cur(:,9));
subplot(2,3,5); histogram(SM_est_cur(:,10));
subplot(2,3,6); histogram(SM_est_cur(:,11));



%% function 

function visualizeEstimation(mu_est)

figure(); hold on;
[X,Y,Z] = sphere(100);
surf(X,Y,Z,'FaceAlpha',0.5,'EdgeColor','none');

scatter3(mu_est(:,1),mu_est(:,2),mu_est(:,3),5,'filled');

end


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

text(1.2*sind(60),1.2*cosd(60),zz,'$60^\circ$','interpreter','latex','fontsize',7,'HorizontalAlignment','center','VerticalAlignment','middle','rotation',-60)
text(1.2*sind(120),1.2*cosd(120),zz,'$120^\circ$','interpreter','latex','fontsize',7,'HorizontalAlignment','center','VerticalAlignment','middle','rotation',-120)
text(1.2*sind(0),1.2*cosd(0),zz,'$\phi=0^\circ$','interpreter','latex','fontsize',7,'HorizontalAlignment','center','VerticalAlignment','middle')
text(1.2*sind(240),1.2*cosd(240),zz,'$240^\circ$','interpreter','latex','fontsize',7,'HorizontalAlignment','center','VerticalAlignment','middle','rotation',-240)
text(1.2*sind(300),1.2*cosd(300),zz,'$300^\circ$','interpreter','latex','fontsize',7,'HorizontalAlignment','center','VerticalAlignment','middle','rotation',-300)
text(sind(175)*sind(10),cosd(175)*sind(10),zz,'$0^\circ$','interpreter','latex','fontsize',7,'HorizontalAlignment','left','VerticalAlignment','middle')
text(sind(175)*sind(30),cosd(175)*sind(30),zz,'$30^\circ$','interpreter','latex','fontsize',7,'HorizontalAlignment','left','VerticalAlignment','middle')
text(sind(175)*sind(60),cosd(175)*sind(60),zz,'$60^\circ$','interpreter','latex','fontsize',7,'HorizontalAlignment','left','VerticalAlignment','middle')
text(sind(175)*1.2,cosd(175)*1.2,zz,'$\theta=90^\circ$','interpreter','latex','fontsize',7,'HorizontalAlignment','left','VerticalAlignment','middle')


end
