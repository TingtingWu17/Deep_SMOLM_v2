load('1SM_est_results_list_atCenter.mat');
dens = 0.2;
size = 5;
figure();
scatter(est(:,3),est(:,4),size,'filled','MarkerFaceAlpha',dens,'MarkerEdgeAlpha',dens);

% hold on;
% load('1SM_est_results_list_0303upsampled.mat');
% scatter(est(:,3),est(:,4),size,'filled','MarkerFaceAlpha',dens,'MarkerEdgeAlpha',dens);

hold on;
load('1SM_est_results_list_0606upsampled.mat');
scatter(est(:,3),est(:,4),size,'filled','MarkerFaceAlpha',dens,'MarkerEdgeAlpha',dens);

% hold on;
% load('1SM_est_results_list_0909upsampled.mat');
% scatter(est(:,3),est(:,4),size,'filled','MarkerFaceAlpha',dens,'MarkerEdgeAlpha',dens);

axis image

%%
%exportgraphics(Fig1,'BarChart.pdf','ContentType','vector')

Fig1 = figure('Units','inches','InnerPosition',[1,1,6.84/2,2]);

pixel_size = 58.5/6;
edges = linspace(-20,50,80);
%exportgraphics(Fig1,'BarChart.pdf','ContentType','vector')

load('1SM_est_results_list_atCenter.mat');
histogram(est(:,3),edges,'EdgeColor','none','FaceColor','#0072BD');


hold on;
load('1SM_est_results_list_0303upsampled.mat');
histogram(est(:,3),edges,'EdgeColor','none','FaceColor','#A2142F');



load('1SM_est_results_list_0606upsampled.mat');
histogram(est(:,3),edges,'EdgeColor','none','FaceColor','#EDB120');


load('1SM_est_results_list_0909upsampled.mat');
histogram(est(:,3),edges,'EdgeColor','#77AC30','FaceColor','#FFFFFF');



plot(0*[pixel_size,pixel_size],[0,135],'LineWidth',4,'Color','#0072BD');
plot(0.3*[pixel_size,pixel_size],[0,135],'LineWidth',4,'Color','#A2142F');
plot(0.6*[pixel_size,pixel_size],[0,135],'LineWidth',4,'Color','#EDB120');
plot(0.9*[pixel_size,pixel_size],[0,135],'LineWidth',4,'Color','#77AC30');


xlim([-20,30]);
xlabel('x (nm)');
ylabel('relative occurance')
legend('','','','','0D','0.3D','0.6D','0.9D','EdgeColor','none','Color','none');

ax = gca;
ax.FontSize = 10; 
