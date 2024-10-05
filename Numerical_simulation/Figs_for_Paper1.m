clc
clear all
close all

%% Fig 2a:
load Results_reg_5600;

names = {'t'; 'h'; 'e'; 'z'; 'r'; 'O0'; '*'; 'Y'};
%names = {'O', 't', 's', 'h'}
N = 8;
M = 560;

A = zeros(N, N);
B = zeros(N);

for i = 1:M
    Preds(i,:) = int64(find(Predictions(i,:) == max(Predictions(i,:))));
    GT(i) = int64(GroundTruth(i)) + 1;
    A(GT(i), Preds(i)) = A(GT(i), Preds(i)) + 1;
    B(GT(i)) = B(GT(i)) + 1;
end

for i = 1:N
    C(i,:) = A(i,:)/B(i,1);
end

figure(99)
imagesc(C)
colorbar

% Right order of shape classes - for the paper:
% ii = [3, 1, 5, 2, 6, 7, 4, 8];
% for i = 1:N
%     
%     for j = 1:N
%         aj = int16(a(j));
%         ConfMat(i, j) = C(int16(ii(i)), int16(ii(j)));
%     end
%     
% end

ConfMat = C;

% Plotting the confusion matrix:
figure(1);
imagesc(flipud(ConfMat'));
colormap gray
cmap=colormap;
cmap=flipud(cmap);
colormap(cmap)
caxis([0,1]);
set(gca, 'xtick', [])
set(gca, 'ytick', [])
colorbar;
set(gca,'FontName','Arial','FontSize',14)

C = ConfMat;

counter = 0;
for i = 1:M
    
    if GT(i) == Preds(i)
        counter = counter + 1;
    end
    
end
accuracy_no_rot = counter/M

fname = 'Confusion_matrix.mat';
save(fname, 'C');
%% Fig 2c:
df = zeros(3,M);

df(1,:) = GT;
df(2,:) = Preds;
df(3,:) = size;

df01 = df(:, df(3,:) < 0.2);
df02 = df(:, df(3,:) >= 0.2 & df(3,:) < 0.3);
df03 = df(:, df(3,:) >= 0.3 & df(3,:) < 0.4);
df04 = df(:, df(3,:) >= 0.4 & df(3,:) < 0.5);

df1_01 = df01(:, df01(1,:) == df01(2,:));
df1_02 = df02(:, df02(1,:) == df02(2,:));
df1_03 = df03(:, df03(1,:) == df03(2,:));
df1_04 = df04(:, df04(1,:) == df04(2,:));

acc_size(1) = numel(df1_01(1,:))/numel(df01(1,:))*100;
acc_size(2) = numel(df1_02(1,:))/numel(df02(1,:))*100;
acc_size(3) = numel(df1_03(1,:))/numel(df03(1,:))*100;
acc_size(4) = numel(df1_04(1,:))/numel(df04(1,:))*100;

sz = [0.15, 0.25, 0.35, 0.45];

figure(2)
hold on;
set(gca,'FontName','Arial','FontSize',16)
%ylabel('dataset size')
ylabel('Size (\lambda)')
zlabel('Accuracy (%)')
barr=bar3(sz, acc_size, 0.95)
set(gca,'XTick', [11:15],'XTickLabel',{'350','700','1400','2800','5600'})
set(gca,'YTick', [0.1, 0.2, 0.3, 0.4, 0.5],'YTickLabel',{'0.1','0.2','0.3','0.4','0.5'})
box on
ylim([0.09 0.51])
colormap gray
cmap=colormap;
cmap=flipud(cmap);
colormap(cmap)
barr.FaceColor = "interp";
for i = 1:length(barr)
    barr(i).CData = ((barr(i).ZData));
end
caxis([0 100])
view(90,0)
colorbar
%% Fig 2d:

ds = [1, 2, 3, 4, 5] - 1;
acc1 = [61.11, 70.42, 80.00, 82.86, 89.11];

figure(3); 
hold on;
set(gca,'FontName','Arial','FontSize',16)
ylabel('dataset size')
%ylabel('Size (\lambda)')
zlabel('Accuracy (%)')
barr=bar3(ds, acc1)
set(gca,'XTick', [11:15],'XTickLabel',{'350','700','1400','2800','5600'})
set(gca,'YTick', [0:4],'YTickLabel',{'350','700','1400','2800','5600'})
box on
ylim([-0.55 4.55])
colormap gray
cmap=colormap;
cmap=flipud(cmap);
colormap(cmap)
barr.FaceColor = "interp";
for i = 1:length(barr)
    barr(i).CData = ((barr(i).ZData));
end
caxis([0 100])
view(90,0)
colorbar
%% Fig 3a:
load Results_reg_rot_x_22400;

names = {'t'; 'h'; 'e'; 'z'; 'r'; 'O0'; '*'; 'Y'};
%names = {'O', 't', 's', 'h'}
N = 8;
M = 2240;

A_rot = zeros(N, N);
B_rot = zeros(N);

for i = 1:M
    Preds_rot(i) = int64(find(Predictions(i,:) == max(Predictions(i,:))));
    GT_rot(i) = int64(GroundTruth(i)) + 1;
    A_rot(GT_rot(i), Preds_rot(i)) = A_rot(GT_rot(i), Preds_rot(i)) + 1;
    B_rot(GT_rot(i)) = B_rot(GT_rot(i)) + 1;
end

counter = 0;
for i = 1:M
    
    if GT_rot(i) == Preds_rot(i)
        counter = counter + 1;
    end
    
end
accuracy = counter/M

for i = 1:N
    C_rot(i,:) = A_rot(i,:)/B_rot(i,1);
end

% Right order of shape classes - for the paper:
% ii = [3, 1, 5, 2, 6, 7, 4, 8];
% for i = 1:N
%     
%     for j = 1:N
%         aj = int16(a(j));
%         ConfMat_rot(j, i) = C_rot(int16(ii(i)), int16(ii(j)));
%     end
%     
% end
% Plotting the confusion matrix:
figure(4);
imagesc(flipud(C_rot));
colormap gray
cmap=colormap;
cmap=flipud(cmap);
colormap(cmap)
set(gca, 'xtick', [])
set(gca, 'ytick', [])
colorbar;
caxis([0 1])
set(gca,'FontName','Arial','FontSize',18)

fname = 'Confusion_matrix_rot.mat';
save(fname, 'C_rot');

%% Fig. 3b:

ds = [1, 2, 3, 4, 5, 6]/2 - 0.5;
acc2 = [89.11, 72.77, 53.35, 45.94, 47.00, 37.81];

figure(31); 
hold on;
set(gca,'FontName','Arial','FontSize',16)
%ylabel('L(\lambda)')
%ylabel('Size (\lambda)')
zlabel('Accuracy (%)')
barr=bar3(ds, acc2)
set(gca,'XTick', [11:16],'XTickLabel',{'350','700','1400','2800','5600'})
set(gca,'YTick', [0, 0.5, 1, 1.5, 2, 2.5],'YTickLabel',{'0','0.5','1','1.5','2', '2.5'})
box on
ylim([-0.3, 2.8])
colormap gray
cmap=colormap;
cmap=flipud(cmap);
colormap(cmap)
barr.FaceColor = "interp";
for i = 1:length(barr)
    barr(i).CData = ((barr(i).ZData));
end
caxis([0 100])
view(90,0)
colorbar

%% Fig S5:
ds = [350, 700, 1400, 2800, 5600, 11200, 22400];
ds1 = [350, 700, 1400, 2800, 5600];
%acc1 = [52.78, 80.28, 82.14, 83.93, 90.54];
acc1 = [61.11, 70.42, 80.00, 82.86, 89.11];
acc_rot = [27.78, 26.76, 35.00, 38.93, 43.93, 48.75, 67.46];
acc_pos = [27.78, 33.80, 37.14, 45.71, 53.04, 65.36, 72.77];

figure(5);
semilogx(ds1, acc1, 'bs-', 'markerfacecolor', 'b', 'markersize', 5)
hold on
semilogx(ds, acc_rot, 'ro-', 'markerfacecolor', 'r', 'markersize', 5)
hold on
semilogx(ds, acc_pos, 'kd-', 'markerfacecolor', 'k', 'markersize', 5)
hold on
set(gca, 'xtick', [])
set(gca,'FontName','Arial','FontSize',14)
xlim([250 30000])
ylim([0 100])
xticks([256, 1024, 4096, 16384])
xticklabels({'2^8','2^{10}','2^{12}','2^{14}'})
xlabel('Dataset size')
ylabel('Accuracy, %')

figure(6);
% semilogx(ds1, acc1, 'bs-', 'markerfacecolor', 'b', 'markersize', 8)
% hold on
semilogx(ds, acc_rot, 'ro-', 'markerfacecolor', 'r', 'markersize', 8)
hold on
%semilogx(ds, acc_pos, 'kd-', 'markerfacecolor', 'k', 'markersize', 8)
%hold on
set(gca, 'xtick', [])
set(gca,'FontName','Arial','FontSize',16)
xlim([250 30000])
ylim([35 75])
xticks([256, 1024, 4096, 16384])
xticklabels({'2^8','2^{10}','2^{12}','2^{14}'})
xlabel('Dataset size')
ylabel('Accuracy, %')