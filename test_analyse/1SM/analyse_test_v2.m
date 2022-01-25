%exportgraphics(Fig1,'BarChart.pdf','ContentType','vector')
%% DPPC+chol xy view
edges=linspace(-50,100,200);
pixel_size = 58.5/6;
dens = 0.5;
Fig1 = figure('Units','inches','InnerPosition',[1,1,6.84/2,2]);
box on;


load('1SM_est_results_list_atCenter.mat');
histogram(est(:,3),edges,'FaceColor','#0072BD','EdgeColor','none','FaceAlpha',dens);
hold on;
load('1SM_est_results_list_0303upsampled.mat');
histogram(est(:,3),edges,'FaceColor','#EDB120','EdgeColor','none','FaceAlpha',dens);

load('1SM_est_results_list_0606upsampled.mat');
histogram(est(:,3),edges,'FaceColor','#A2142F','EdgeColor','none','FaceAlpha',dens);

load('1SM_est_results_list_0909upsampled.mat');
histogram(est(:,3),edges,'FaceColor','#FFFFFF','EdgeColor','#00FF00','FaceAlpha',0.2);

h = 120;
plot(0*[pixel_size,pixel_size],[0 h],'LineWidt',3,'Color','#0072BD');
plot(0.3*[pixel_size,pixel_size],[0 h],'LineWidt',3,'Color','#EDB120');
plot(0.6*[pixel_size,pixel_size],[0 h],'LineWidt',3,'Color','#A2142F');
plot(0.9*[pixel_size,pixel_size],[0 h],'LineWidt',3,'Color','#00FF00');

xlim([-20,30]);
xlabel('x (nm)');
ylabel('relative occurance');

ax = gca;
ax.FontSize = 10; 
legend('','','','','0D','0.3D','0.6D','0.9D','Color','none','EdgeColor','none');

%%

figure(); 
dens = 0.1;
scatter(est(:,3),est(:,4),10,'filled','MarkerFaceAlpha',dens,'MarkerEdgeAlpha',dens);
xlim([-30,30]);
ylim([-30,30]);

xlabel('x (nm)');
ylabel('y (nm)');
title('estimation for GT=[0,0]');
ax = gca;
ax.FontSize = 15; 