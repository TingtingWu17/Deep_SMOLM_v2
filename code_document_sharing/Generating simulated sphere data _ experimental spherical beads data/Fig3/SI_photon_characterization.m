%exportgraphics(Fig1,'BarChart.pdf','ContentType','vector')
%%
Fig1 = figure('Units','inches','InnerPosition',[1,1,3.4,3]);

s1 = fig_a();
s2 = fig_b();

s3 = fig_c();
s4 = fig_d();



function s1 = fig_a()

%load('20210504_DPPC_40chol_est_retrieval_1.1_xy_centered_v27.mat');
%load('est_xyCentered_v28.mat');
load('est_xyzCentered_v29.mat')
photons = SM_est_final(:,5);

s1 = subplot(2,2,1);
histogram(photons,'EdgeColor','none'); hold on;
plot([1000,1000],[0,2200],'r--')


xlabel({['Photons/localization'];['(photons)']});
ylabel('# localizations');
text(5000,1000,{['s_{median}=']; [num2str(median(round(photons)))]});
xlim([0,15000]);
ax = gca;
ax.FontSize =9; 
whitebg('w');
end


function s2 = fig_b()

%load('20210504_DPPC_40chol_est_retrieval_1.1_xy_centered_v27.mat');
%load('est_xyCentered_v28.mat');
load('est_xyzCentered_v29.mat')
frameID = SM_est_final(:,1);
frameID2 = (1:length(frameID)).';
IDname = [15:24];

frameCount = 0;
framCount_cur = 0;
for ii = 1:length(frameID)-1
   
    if frameID(ii+1)>=frameID(ii)
        framCount_cur = framCount_cur+1;
    else
     frameCount = [frameCount,framCount_cur];
     framCount_cur = 0;
    end  
end

frameCount = [frameCount,length(frameID)-sum(frameCount)];
for ii = 1:length(IDname)
for jj = 1:60:2000
    count(ii,ceil(jj/60))=sum(frameID2>sum(frameCount(1:ii))& frameID2<=sum(frameCount(1:ii+1)) & frameID>=jj & frameID<=jj+60);
end
end


count = reshape(count.',[],1);
x = 0:0.11:0.11*(length(count)-1);
s2 = subplot(2,2,2);
plot(x,count);
text(3,110,{['mean is ' num2str(round(mean(count)))]});
ylabel('# localizations');
xlabel('Measurement time (min)');
ax = gca;
ax.FontSize =9; 
ylim([0,120]);
end



function s1 = fig_c()

%load('20210524_DPPC_est_xy_centered_v14.mat');
load('est_retrieval_1.1_v15.mat');
photons = SM_est_final(:,5);

s1 = subplot(2,2,3);
histogram(photons,'EdgeColor','none'); hold on;
plot([1000,1000],[0,3000],'r--')

xlabel({['Photons/localization'];['(photons)']});
ylabel('# localizations');
text(5000,1000,{['s_{median}='];[num2str(median(round(photons)))]});
xlim([0,18000]);
ax = gca;
ax.FontSize =9; 
whitebg('w');
end


function s2 = fig_d()

%load('20210524_DPPC_est_xy_centered_v14.mat');
load('est_retrieval_1.1_v15.mat');
frameID = SM_est_final(:,1);
frameID2 = (1:length(frameID)).';
IDname = [2:10];

frameCount = 0;
framCount_cur = 0;
for ii = 1:length(frameID)-1
   
    if frameID(ii+1)>=frameID(ii)
        framCount_cur = framCount_cur+1;
    else
     frameCount = [frameCount,framCount_cur];
     framCount_cur = 0;
    end  
end

frameCount = [frameCount,length(frameID)-sum(frameCount)];
for ii = 1:length(IDname)
for jj = 1:60:2000
    count(ii,ceil(jj/60))=sum(frameID2>sum(frameCount(1:ii))& frameID2<=sum(frameCount(1:ii+1)) & frameID>=jj & frameID<=jj+60);
end
end


count = reshape(count.',[],1);
x = 0:0.11:0.11*(length(count)-1);
s2 = subplot(2,2,4);
plot(x,count);
text(3,110,{['mean is ' num2str(round(mean(count)))]});

ylabel('# localizations');
xlabel('Measurement time (min)');
ax = gca;
ax.FontSize =9; 
ylim([0,120]);
end