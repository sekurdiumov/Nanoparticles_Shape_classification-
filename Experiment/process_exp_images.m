clc
clear all
close all

%% Read tiff files:
eps0 = ones(2160, 2544)*5;
eps0(800:1500, 1100:1500) = 0.7;

% figure(99)
% pcolor(eps0)
% shading flat
% set(gcf, 'Color', 'none')
% colorbar

counter = 0;
for i = 1:24
    for j = 1:6
        for k = 1:6
            counter = counter + 1
            fname = strcat(num2str(i - 1), '-', num2str(j - 1), '-000', num2str(k - 1), '.tif');
            
            t = Tiff(fname);
            A = read(t);
            
            
            
            A = A - min(min(A));
            
            x1 = 700;
            x2 = x1 + 701;
            
            y1 = 800;
            y2 = y1 + 701;
            
                        
            A1 = double(A(x1:x2, y1:y2));
            A1 = A1/max(max(A1));
            
%             figure(24)
%             pcolor(double(A))
%             shading flat
%             colorbar
            

            %eps = 0.7;
            
            eps = ones(702, 702)*5;
            eps(200:end - 200, 200:end - 200) = 0.6;
            
            for kk = 1:5
                A1(2:699, 2:699) = (A1(1:698, 1:698) + A1(2:699, 1:698) + A1(3:700, 1:698) + A1(1:698, 2:699) + A1(2:699, 2:699) + A1(3:700, 2:699) + A1(1:698, 3:700) + A1(2:699, 3:700) + A1(3:700, 3:700))/9;
            end
            
            Gx = (A1 - eps);
            
%             figure(25)
%             pcolor(Gx)
%             shading flat
%             colorbar

            [x, y] = find(Gx >= 0);
            
            x = x + x1;
            y = y + y1;

            x0 = round((max(x) + min(x))/2);
            y0 = round((max(y) + min(y))/2);
            
            TF = isempty(x0);
            
            if TF == 1
                x0 = 990
                y0 = 1210
            end

            
            
            E = A(x0 - 256 : x0 + 255, y0 - 256 : y0 + 255);
            
            SNR(counter) = max(max(double(E))) / min(min(double(E)));
            
            fname1 = strcat('fig', num2str(i - 1), '-', num2str(j - 1), '-', num2str(k - 1), '.png');
            
            figure(1)
            pcolor(double(E))
            set(gcf, 'color', 'black')
            set(gca, 'FontSize',15, 'FontName','MS Trubuchet','Xcolor','w','Ycolor','w')
            shading flat;
            colorbar('Color', 'white');
            saveas(gcf, fname1);
            
            figure(2)
            plot(E(256,:))
            
            fname1 = strcat('dp', num2str(i - 1), '-', num2str(j - 1), '-', num2str(k - 1), '.mat');
            save(fname1, 'E');
        end
    end
end

figure(3)
plot(SNR)
xlabel('element')
ylabel('SNR')