function Copy_2_of_twoChrom_constraint_TCSVT()
s=23;
clc;format compact;
src='kodak24\';

src1='rdoea_fr\kodak24_0.4\';
% src='DIV2K_valid_HR\';
% src1='rdoea_fr\DIV2K_valid_HR\';
filename=['kodim',num2str(s,'%d'),'.png'];
inputName=fullfile(src,filename);
filename1=['kodim',num2str(s,'%d'),'_truevalue.xlsx'];
rdoea_fr_truevalue=fullfile(src1,filename1);
filename2=['kodim',num2str(s,'%d'),'_population.xlsx'];
rdoea_fr_population=fullfile(src1,filename2);


% filename=['08',num2str(s,'%d'),'.png'];
% inputName=fullfile(src,filename);
% filename1=['08',num2str(s,'%d'),'_truevalue.xlsx'];
% rdoea_fr_truevalue=fullfile(src1,filename1);
% filename2=['08',num2str(s,'%d'),'_population.xlsx'];
% rdoea_fr_population=fullfile(src1,filename2);


% inputName = 'kodak24\kodim2.png';
%inputName = 'M10.bmp';
% inputName = 'DIV2K_valid_HR\0812.png';
outputName = '01.jpg';
tic;
CELL_SIZE = 8; %greater than 4
img = imread(inputName);
%%% MAPPER RGB -> YCbCr
ycbcr_img = rgb2ycbcr(img);
y_image =ycbcr_img(:, :, 1);
cb_image = ycbcr_img(:, :, 2);
cr_image = ycbcr_img(:, :, 3);
%%% Turn into cells 8x8
repeat_height = size(y_image, 1)/CELL_SIZE;
repeat_width = size(y_image, 2)/CELL_SIZE;
repeat_height_mat = repmat(CELL_SIZE, [1 repeat_height]);
repeat_width_mat = repmat(CELL_SIZE, [1 repeat_width]);
y_sub_image = mat2cell(y_image, repeat_height_mat, repeat_width_mat);
cb_sub_image = mat2cell(cb_image, repeat_height_mat, repeat_width_mat);
cr_sub_image = mat2cell(cr_image,  repeat_height_mat,repeat_width_mat);

y_sub_dct = y_sub_image;
for i=1:repeat_height
    for j=1:repeat_width
         y_sub_image{i, j} = dcTransform(y_sub_image{i, j});
        cb_sub_image{i,j} = dcTransform(cb_sub_image{i,j});
        cr_sub_image{i,j} = dcTransform(cr_sub_image{i,j});
        y_sub_dct{i,j} = getDCT(y_sub_image{i,j});
        cb_sub_dct{i,j} = getDCT(cb_sub_image{i,j});
        cr_sub_dct{i,j} = getDCT(cr_sub_image{i,j});
    end
end

y_hist_out = GetInvCoffHist(y_sub_dct, repeat_height, repeat_width);
cb_hist_out = GetInvCoffHist(cb_sub_dct, repeat_height, repeat_width);
cr_hist_out = GetInvCoffHist(cr_sub_dct, repeat_height, repeat_width);
mse_table = estBlockDistortionInd(y_hist_out,cb_hist_out,cr_hist_out);
rate_table = estBlockRateInd(y_hist_out,cb_hist_out,cr_hist_out, repeat_height, repeat_width);
%---初始化/参数设定
generations=20;                                %迭代次数
population1=30;                                     %种群1大小(须为偶数)
global poplength
poplength=128;                                            %个体长度
lumMat = [16 11 10 16 24 40 51 61 ...
    12 12 14 19 26 58 60 55 ...
    14 13 16 24 40 57 69 56 ...
    14 17 22 29 51 87 80 62 ...
    18 22 37 56 68 109 103 77 ...
    24 35 55 64 81 104 113 92 ...
    49 64 78 87 103 121 120 101 ...
    72 92 95 98 112 100 103 99 ...
     17 18 24 47 99 99 99 99 ...
   18 21 26 66 99 99 99 99 ...
   24 26 56 99 99 99 99 99 ...
   47 99 99 99 99 99 99 99 ...
   99 99 99 99 99 99 99 99 ...
   99 99 99 99 99 99 99 99 ...
   99 99 99 99 99 99 99 99 ...
   99 99 99 99 99 99 99 99]; 
%  psyq =[16 14 13 15 19 28 37 55 14 13 15 19 28 37 55 64 13 15 19 28 37 55 64 83 15 19 28 37 55 64 83 103 19 28 37 55 64 83 103 117 28 37 55 64 83 103 117 117  37  55 64 83 103 117 117 111 55 64 83 103 117 117 111 90];
%  population = zeros(population1,64);
  for i = 1:population1%种群1初始化   
     out = quantLumMat(lumMat, i*2);
     population(i,:) = out;
  end
  pop = 30;
   population2 = zeros(pop,128);
  for i = 1:pop%种群2初始化   
     out = quantLumMat(lumMat, (50 - ((i) - pop/2)*5));
     population2(i,:) = out;
  end 
%   for i = 1:pop%种群2初始化   
%      out = quantLumMat(lumMat, i);
%      population2(i,:) = out;    
%   end 
%     functionvalue = MSE(population2, y_image, y_sub_image, y_sub_dct, repeat_height, repeat_width, outputName, 0, mse_table, w, rate_table);
%    dlmwrite('M100/functionvalue.txt',functionvalue);
%    dlmwrite('M100/qtable.txt',population2);
%    figure;
%    plot(functionvalue(:,1),functionvalue(:,2),'*r');
%    xlabel('Rate(bpp)');ylabel('1/mse');
  


populationlum1 = population;
%populationlum = populationlum1;
 populationlum2 = population2;
%---开始迭代进化
 for gene=1:generations                      %开始迭代
%     populationlum2 = [populationlum2;populationlum1];   
  populationlum = [populationlum1;populationlum2];
    px=size(populationlum,1);
    if mod(px,2)~=0
        populationlum(px+1,:)=lumMat;
    end
    newpopulation=ones(size(populationlum,1),poplength);
    for i = 1:2:px-1
        a=rand();
        if(a<=0.2) %交叉
            cpoint = 16;
            index = getTopKPostion(mse_table, rate_table,populationlum(i,:), populationlum(i+1,:), cpoint);
            newpopulation(i,:) = populationlum(i,:);
            newpopulation(i, index) = populationlum(i+1, index);
            index = getTopKPostion(mse_table, rate_table,populationlum(i+1,:), populationlum(i,:), cpoint);
            newpopulation(i+1,:) = populationlum(i+1,:);
            newpopulation(i+1, index) = populationlum(i, index);              %交叉
        else
            b=rand();
            if b<=0.5 %变异1
                if gene<generations/2
                   temp =round(rand.*63)+1;
                else
                   temp = 16;
                end
                newpopulation(i,:) = populationlum(i,:);
                newpopulation(i+1,:) = populationlum(i+1,:);
                %            [add_index, minus_index] = getTopKPostionForMut(mse_table, rate_table, populationlum(i,:), temp);%lp
                [ minus_index,index_new] = getTopKPostionForMut(mse_table, rate_table, populationlum(i,:), temp);
                populationlum(i,index_new(1,:)) = minus_index(1,index_new(1,:));
                [ minus_index,index_new] = getTopKPostionForMut(mse_table, rate_table,populationlum(i+1,:), temp);
                populationlum(i+1,index_new(1,:)) = minus_index(1,index_new(1,:));
            else
                newpopulation(i,:) = populationlum(i,:);              %变异二
                newpopulation(i+1,:) = populationlum(i+1,:);
                q=round(rand.*100)+1;
                if q<1
                    q=1;
                    scale=50*100/q;
                end
                if q>100
                    q=100;
                    scale=200-q*2;
                end
                if q<50
                    scale=50*100/q;
                else
                    scale=200-q*2;
                end
                for j=1:128
                    newpopulation(i,j)=round((newpopulation(i,j)*scale+50)/100);
                    if newpopulation(i,j)<1
                        newpopulation(i,j)=1;
                    else
                        if newpopulation(i,j)>255
                            newpopulation(i,j)=255;
                        end
                    end
                end
                for k=1:128
                    newpopulation(i+1,k)=round((newpopulation(i+1,k)*scale+50)/100);
                    if newpopulation(i+1,k)<1
                        newpopulation(i+1,k)=1;
                    else
                        if newpopulation(i+1,k)>255
                            newpopulation(i+1,k)=255;
                        end
                    end
                end
            end
        end
    end
    newpopulation = [populationlum;newpopulation];  %合并父子种群1
    functionvalue = MSE(newpopulation,y_sub_image,cb_sub_image,cr_sub_image, y_sub_dct,cb_sub_dct,cr_sub_dct, repeat_height, repeat_width, outputName, 1, mse_table, rate_table);
    
    X = [0.4,0.8,1.2,1.6];   %可行域
    Y = abs(X - functionvalue(:,1));
    Y(Y(:,1) <=0.05,1)= 0; % [0.2-0.2*10%,0.2+0.2*10%]
    Y(Y(:,2) <=0.1,2)= 0;  %[0.6-0.6*10%,0.6+0.6*10%]
    Y(Y(:,3) <=0.1,3)= 0;  %[1-1*10%,1+1*10%]
    Y(Y(:,4) <=0.2,4)= 0;  %[1.4-1.4*10%,1.4+1.4*10%]
    [cov,~]=min(Y,[],2);   %每一行选择最小的保存
    CV = CalCV(cov);       %归一化
    Fitness = CalFitness(functionvalue,CV); %计算适应度值
      
    [populationlum1,Fitness] = EnvironmentalSelection1(Fitness,newpopulation,functionvalue,population1);   
    functionvalue_gen = MSE(populationlum1, y_sub_image,cb_sub_image,cr_sub_image, y_sub_dct,cb_sub_dct,cr_sub_dct, repeat_height, repeat_width, outputName, 1, mse_table, rate_table); 
    
%    figure;
%     plot(functionvalue_gen(:,1),functionvalue_gen(:,2),'*r');
    count = ones(1,4);
    k=1;
    Y = abs(X - functionvalue_gen(:,1));
    Y(Y(:,1) <=0.05,1)= 0; % [0.2-0.2*10%,0.2+0.2*10%]
    count(1,1)=size(find(Y(:,1)==0),1);
%     populationlum1_new = populationlum1(find(Y(:,1)==0),:);
    Y(Y(:,2) <=0.1,2)= 0;  %[0.6-0.6*10%,0.6+0.6*10%]
    count(1,2)=size(find(Y(:,2)==0),1);
%     populationlum1_new = populationlum1(find(Y(:,2)==0),:);
    Y(Y(:,3) <=0.1,3)= 0;  %[1-1*10%,1+1*10%]
    count(1,3)=size(find(Y(:,3)==0),1);
%     populationlum1_new = populationlum1(find(Y(:,3)==0),:);
    Y(Y(:,4) <=0.2,4)= 0;
    count(1,4)=size(find(Y(:,4)==0),1);
%     populationlum1_new = populationlum1(find(Y(:,4)==0),:);
    
    g = size(functionvalue_gen,1)+1;
    x=1;
    for i=1:4
        if count(1,i)<10
            populationlum1_new = populationlum1(find(Y(:,i)==0),:);
            functionvalue_gen1 = functionvalue_gen(find(Y(:,i)==0),:);
           for r =1:(size(functionvalue_gen1,1)-1)
              c= (functionvalue_gen1(r+1,1)- functionvalue_gen1(r,1)).^2+(functionvalue_gen1(r+1,2)- functionvalue_gen1(r,2)).^2;
             if c>0.01       
                 popolationlum2_new= populationlum1_new(r+1,:);
                 ratelow = functionvalue_gen1(r+1,1);
                 ratehigh = functionvalue_gen1(r,1);
    %                   quality = 50 - (ratelow - ratehigh)*50/ratelow
                 quality = 50 - (ratelow - ratehigh)*50/ratelow;
                populationlumnew1= quantLumMat(popolationlum2_new, quality); 
%                 populationlumnew2(x,:) = populationlumnew1;
                x=x+1;
                 populationlum1(g,:)=populationlumnew1;  
                 g=g+1;           
             end     
           end 
            
        end
    end

%    functionvalue_gen = MSE(populationlumnew2, y_image, y_sub_image, y_sub_dct, repeat_height, repeat_width, outputName, 1, mse_table, w, rate_table);
%    hold on;
%    plot(functionvalue_gen(:,1),functionvalue_gen(:,2),'og');
%     xlabel('Rate(bpp)');ylabel('1/mse');
%      legend('Environmental Selection','Local Search');
%      for r =1:(size(functionvalue_gen,1)-3)
%          c= (functionvalue_gen(r+1,1)- functionvalue_gen(r,1)).^2+(functionvalue_gen(r+1,2)- functionvalue_gen(r,2)).^2;
%          if c>0.1        
%              popolationlum_new= populationlum1(r+1,:);
%              ratelow = functionvalue_gen(r+1,1);
%              ratehigh = functionvalue_gen(r,1);
% %                   quality = 50 - (ratelow - ratehigh)*50/ratelow
%              quality = 50 - (ratelow - ratehigh)*50/ratelow
%              populationlum1(g,:)=quantLumMat(popolationlum_new, quality);  
%              g=g+1;
%           end     
%      end  
    populationlum2 = [populationlum2;populationlum1]; 
    px=size(populationlum2,1);           %种群2无约束
    if mod(px,2)~=0
        populationlum2(px+1,:)=lumMat;
    end
    newpopulation=ones(size(populationlum2,1),poplength);
    for i = 1:2:px-1
        a=rand();
        if(a<=0.2)
            cpoint = 16;
            index = getTopKPostion(mse_table, rate_table,populationlum2(i,:), populationlum2(i+1,:), cpoint);
            newpopulation(i,:) = populationlum2(i,:);
            newpopulation(i, index) = populationlum2(i+1, index);
            index = getTopKPostion(mse_table, rate_table,populationlum2(i+1,:), populationlum2(i,:), cpoint);
            newpopulation(i+1,:) = populationlum2(i+1,:);
            newpopulation(i+1, index) = populationlum2(i, index);              %交叉
        else
            b=rand();
            if b<=0.5
                if gene<generations/2
                   temp =round(rand.*63)+1;
                else
                   temp = 16;
                end
                newpopulation(i,:) = populationlum2(i,:);
                newpopulation(i+1,:) = populationlum2(i+1,:);
                %            [add_index, minus_index] = getTopKPostionForMut(mse_table, rate_table, populationlum(i,:), temp);%lp
                [ minus_index,index_new] = getTopKPostionForMut(mse_table, rate_table, populationlum2(i,:), temp);
                populationlum2(i,index_new(1,:)) = minus_index(1,index_new(1,:));
                [ minus_index,index_new] = getTopKPostionForMut(mse_table, rate_table,populationlum2(i+1,:), temp);
                populationlum2(i+1,index_new(1,:)) = minus_index(1,index_new(1,:));
            else
                newpopulation(i,:) = populationlum2(i,:);              %变异二
                newpopulation(i+1,:) = populationlum2(i+1,:);
                q=round(rand.*100)+1;
                if q<1
                    q=1;
                    scale=50*100/q;
                end
                if q>100
                    q=100;
                    scale=200-q*2;
                end
                if q<50
                    scale=50*100/q;
                else
                    scale=200-q*2;
                end
                for j=1:128
                    newpopulation(i,j)=round((newpopulation(i,j)*scale+50)/100);
                    if newpopulation(i,j)<1
                        newpopulation(i,j)=1;
                    else
                        if newpopulation(i,j)>255
                            newpopulation(i,j)=255;
                        end
                    end
                end
                for k=1:128
                    newpopulation(i+1,k)=round((newpopulation(i+1,k)*scale+50)/100);
                    if newpopulation(i+1,k)<1
                        newpopulation(i+1,k)=1;
                    else
                        if newpopulation(i+1,k)>255
                            newpopulation(i+1,k)=255;
                        end
                    end
                end
            end
        end
    end
    newpopulation = [populationlum2;newpopulation];

%  Q=newpopulation;
%     functionvalue_Q=MSE(Q, y_sub_image,cb_sub_image,cr_sub_image, y_sub_dct,cb_sub_dct,cr_sub_dct, repeat_height, repeat_width, outputName, 1, mse_table, rate_table);
%     YQ = abs(X - functionvalue_Q(:,1));
%     YQ(YQ(:,1) <=0.05,1)= 0; % [0.2-0.2*10%,0.2+0.2*10%]
%     YQ(YQ(:,2) <=0.05,2)= 0;  %[0.6-0.6*10%,0.6+0.6*10%]
%     YQ(YQ(:,3) <=0.05,3)= 0;  %[1-1*10%,1+1*10%]
%     YQ(YQ(:,4) <=0.05,4)= 0;  %[1.4-1.4*10%,1.4+1.4*10%]
%     [covQ,~]=min(YQ,[],2);   %每一行选择最小的保存  
%      [covQ_sort,covQ_sortindex]=sortrows(covQ);%对适应度进行排序
%    feasible_index=find(covQ==0);
%    feasible_population=Q(feasible_index,:);
%    notfeasible_index=find(covQ>0);%找到适应度值大于0的解的下标
%    notfeasible_population=Q(notfeasible_index,:);
%    notfeasible_solution=functionvalue_Q(notfeasible_index,:);
%    for i=1:size(X,2)
%    areas(i,:)=[X(:,i)-0.05,X(:,i)+0.05];
%    end
%    [nearest_points,population_new] = find_nearest_points(notfeasible_solution, notfeasible_population,areas);
%    newpopulation=[feasible_population;population_new];
%    functionvalue=MSE(Q, y_sub_image,cb_sub_image,cr_sub_image, y_sub_dct,cb_sub_dct,cr_sub_dct, repeat_height, repeat_width, outputName, 1, mse_table, rate_table);
    functionvalue = MSE(newpopulation, y_sub_image,cb_sub_image,cr_sub_image, y_sub_dct,cb_sub_dct,cr_sub_dct, repeat_height, repeat_width, outputName, 1, mse_table, rate_table);
 
    [frontvalue, ~] = NDSort(functionvalue, inf);
     newnum=numel(frontvalue,frontvalue<=1);                              %前fnum个面的个体数   
    if(newnum <50)
        populationlum2(1:newnum,:)=newpopulation(frontvalue<=1,:);
    else
        populationlum_temp(1:newnum,:)=newpopulation(frontvalue<=1,:);
%         % convex hull
        point = functionvalue(frontvalue<=1,:);
        point(newnum + 1,:)=[point(1,1)+10, point(size(point,1),2)+10];
        [point, index]=unique( point,'rows');
       
        populationlum_temp_new(1:size(index,1)-1,:)= populationlum_temp(index(1:size(index,1)-1),:);
       
  
        dt = delaunayTriangulation(point(:,1),point(:,2));
        k = convexHull(dt); 
        functionvalue_gen = [dt.Points(k,1) dt.Points(k,2)];   
        populationlum2 = populationlum_temp(1,:);%初始化populationlum
        populationlum2(1:size(k,1)-2,:) = populationlum_temp_new(k(k(1:size(k,1)-1,1)<size(dt.Points,1)),:);
        functionvalue_gen_new = functionvalue_gen(1:size(functionvalue_gen(:,1))-2,:);
        n=size(k,1)-1;        
       if gene<20
          for r =1:(size(functionvalue_gen,1)-3)
              c= (functionvalue_gen(r+1,1)- functionvalue_gen(r,1)).^2+(functionvalue_gen(r+1,2)- functionvalue_gen(r,2)).^2;
              if c>0.1        
                  popolationlum_new= populationlum2(r+1,:);
                  ratelow = functionvalue_gen(r+1,1);
                  ratehigh = functionvalue_gen(r,1);
%                   quality = 50 - (ratelow - ratehigh)*50/ratelow
                  quality = 50 - (ratelow - ratehigh)*50/ratelow;
                  populationlum2(n,:)=quantLumMat(popolationlum_new, quality);  
              end     
          end
       end
        end
    
%     fprintf('已完成,耗时%4s秒\n',num2str(toc));
%   population2 = populationlum2;
%      populationlum2 = [populationlum1;populationlum2];
%      functionvalue = MSE(populationlum1, y_image, y_sub_image, y_sub_dct, repeat_height, repeat_width, outputName, 1, mse_table, w, rate_table);
%      if gene==10
%          dlmwrite('mseresult/kodim04_10.txt',functionvalue);
%      else if gene==30
%              dlmwrite('mseresult/kodim04_30.txt',functionvalue);
%          else if gene ==50
%                  dlmwrite('mseresult/kodim04_50.txt',functionvalue);
%              end
%          end
%      end
% figure;
%  plot(functionvalue(:,1),functionvalue(:,2),'^b');
    
 end
 
%    truevalue = MSE(population_new, y_sub_image,cb_sub_image,cr_sub_image, y_sub_dct,cb_sub_dct,cr_sub_dct, repeat_height, repeat_width, outputName, 0, mse_table, rate_table);


% populationlum2 = populationlum1;
%     truevalue = MSE(populationlum2(1,:), y_image, y_sub_image, y_sub_dct, repeat_height, repeat_width, outputName, 0, mse_table,w, rate_table);
    fprintf('已完成,耗时%4s秒\n',num2str(toc));          %程序最终耗时

    truevalue = MSE(populationlum2, y_sub_image,cb_sub_image,cr_sub_image, y_sub_dct,cb_sub_dct,cr_sub_dct, repeat_height, repeat_width, outputName, 0, mse_table, rate_table);
%     estvalue = MSE(populationlum2, y_sub_image,cb_sub_image,cr_sub_image, y_sub_dct,cb_sub_dct,cr_sub_dct, repeat_height, repeat_width, outputName, 1, mse_table, rate_table);

%     Y = abs(X - truevalue(:,1));
%     Y_est = abs(X - estvalue(:,1));
%     Y_est(Y_est(:,1) <=0.05,1)= 0;
%     Y_est(Y_est(:,2) <=0.1,2)= 0;
%     Y_est(Y_est(:,3) <=0.1,3)= 0;
%     Y_est(Y_est(:,4) <=0.1,4)= 0;
%     [cov_est,~]=min(Y_est,[],2);
%     index_est = find(cov_est==0);
%     
%     Y(Y(:,1) <=0.05,1)= 0;
%     Y(Y(:,2) <=0.1,2)= 0;
%     Y(Y(:,3) <=0.1,3)= 0;
%     Y(Y(:,4) <=0.1,4)= 0;
%     [cov,~]=min(Y,[],2);
%     Yvalue = X - truevalue(:,1);
%     index1 = find(cov==0);
%     truevalue_new=truevalue(index1,:);
% 
%     Yest = abs(X - estvalue(:,1));
%     Yest(Yest(:,1) <=0.05,1)= 0;
%     Yest(Yest(:,2) <=0.1,2)= 0;
%     Yest(Yest(:,3) <=0.1,3)= 0;
%     Yest(Yest(:,4) <=0.1,4)= 0;
%     [covest,~]=min(Yest,[],2);
%     indexest = find(covest==0);
%     output=estvalue(indexest,:);
%   
%     Ytrue = abs(X - truevalue_new(:,1));
%     Ytrue(Ytrue(:,1) <=0.05,1)= 0;
%     Ytrue(Ytrue(:,2) <=0.05,2)= 0;
%     Ytrue(Ytrue(:,3) <=0.05,3)= 0;
%     Ytrue(Ytrue(:,4) <=0.05,4)= 0;
%     [covtrue,~]=min(Ytrue,[],2);
%     indextrue = find(covtrue==0);
%     truevalue_new=truevalue_new(indextrue,:);
   

%      truevalue_new=sortrows(truevalue_new);
 [truevalue,b] = sortrows(truevalue);
     populationlum2_1 = (populationlum2(b,:));
     xlswrite(rdoea_fr_truevalue,truevalue);
     xlswrite(rdoea_fr_population,populationlum2_1);


 end


function CV = CalCV(CV_Original)
    CVmax = max(CV_Original);
    CVmin = min(CV_Original);
    CV = (CV_Original-CVmin)./(CVmax - CVmin);
%     CV(:,isnan(CV(1,:))) = 0;%12.23
%     CV = mean(CV,2);%12.23
end


function out = getTopKPostion(mse_whole, rate_whole, lumMat1, lumMat2, k)
value = zeros(128,1);
value(:) = 10000000;
for i = 1:128
    if i>64
        m = floor((i-64-1)/8)+1;
    n = mod(i-64-1, 8) + 1;
    y=2;
    else 
        m = floor((i-1)/8)+1;
    n = mod(i-1, 8) + 1;
        y=1;
    end

    q1 = lumMat1(i);
    q2 = lumMat2(i);
    if q1 ~= q2
        if q1 < q2
            dist1 = mse_whole(m,n,q1,y);
            dist2 = mse_whole(m,n,q2,y);
            rate1 = rate_whole(m,n,q1,y);
            rate2 = rate_whole(m,n,q2,y);
            if rate1 ~= rate2
                slope = (dist2 - dist1)/(rate1 - rate2); %smaller better
                value(i) = slope;
            end
        else
            dist1 = mse_whole(m,n,q1,y);
            dist2 = mse_whole(m,n,q2,y);
            rate1 = rate_whole(m,n,q1,y);
            rate2 = rate_whole(m,n,q2,y);
            if dist1 ~= dist2
                slope = (rate2 - rate1)/(dist1 - dist2); %smaller better
                value(i) = slope;
            end
        end
    end
   
end

[~, index]=sort(value);
out = index(1:k);
end

function [ minus_index,index_new] = getTopKPostionForMut(mse_whole, rate_whole, lumMat1, k)
 minus_value = zeros(1,128);
 minus_index = zeros(1,128);
 for i = 1:128
     value(1:255) = 0;
     if i>64
        m = floor((i-64-1)/8)+1;
    n = mod(i-64-1, 8) + 1;
    y=2;
    else 
        m = floor((i-1)/8)+1;
    n = mod(i-1, 8) + 1;
        y=1;
    end

     q1 = lumMat1(i); 
     q2 = q1 + 1;
     if q1 < 205&&q1>50
         q_high=q1+50;
         for j=q2:q_high
             if q1 < q2
                 dist1 = mse_whole(m,n,q1,y);
                 dist2 = mse_whole(m,n,j,y);
                 rate1 = rate_whole(m,n,q1,y);
                 rate2 = rate_whole(m,n,j,y);
                 if dist2~=dist1
                     slope = (rate2 - rate1)/(dist1 - dist2); %bigger better
                     value(j) = slope;
                 end
             end
         end
     else
         if q1>205
             for j=q2:255
                 if q1 < q2
                     dist1 = mse_whole(m,n,q1,y);
                     dist2 = mse_whole(m,n,j,y);
                     rate1 = rate_whole(m,n,q1,y);
                     rate2 = rate_whole(m,n,j,y);
                     if dist2~=dist1
                         slope = (rate2 - rate1)/(dist1 - dist2); %bigger better
                         value(j) = slope;
                     end
                 end
             end
         else
             if q1<50
                 for j=q2:(q1+50)
                     if q1 < q2
                         dist1 = mse_whole(m,n,q1,y);
                         dist2 = mse_whole(m,n,j,y);
                         rate1 = rate_whole(m,n,q1,y);
                         rate2 = rate_whole(m,n,j,y);
                         if dist2~=dist1
                             slope = (rate2 - rate1)/(dist1 - dist2); %bigger better
                             value(j) = slope;
                         end
                     end
                 end
             end
         end 
       end
         q2 = q1 - 1;
         if q1>50&&q1<205
             q_low=q1-50;
             for j=q_low:q2
                 dist1 = mse_whole(m,n,q1,y);
                 dist2 = mse_whole(m,n,j,y);
                 rate1 = rate_whole(m,n,q1,y);
                 rate2 = rate_whole(m,n,j,y);
                 if rate2~=rate1
                     slope = (dist1 - dist2)/(rate2 - rate1); %bigger better
                     value(j) = slope;
                 end
             end
         else
             if q1<=50
                 for j=1:q2
                     dist1 = mse_whole(m,n,q1,y);
                     dist2 = mse_whole(m,n,j,y);
                     rate1 = rate_whole(m,n,q1,y);
                     rate2 = rate_whole(m,n,j,y);
                     if rate2~=rate1
                         slope = (dist1 - dist2)/(rate2 - rate1); %bigger better
                         value(j) = slope;
                     end
                 end
             else
                 if q1>=205
                     q_low=q1-50;
                     for j=q_low:q2
                         dist1 = mse_whole(m,n,q1,y);
                         dist2 = mse_whole(m,n,j,y);
                         rate1 = rate_whole(m,n,q1,y);
                         rate2 = rate_whole(m,n,j,y);
                         if rate2~=rate1
                             slope = (dist1 - dist2)/(rate2 - rate1); %bigger better
                             value(j) = slope;
                         end
                     end
                 end
             end
         end
    [value_new, index2]=sort(value,'descend');
    minus_value(1,i)=value_new(1,1);
    minus_index(1,i) = index2(1,1); 
end
[~, index1]=sort(minus_value,'descend');
index_new=index1(1:k);
end


function out = quantLumMat(lumMat, quality)

if quality <= 50
    quality = 5000 / quality;
else
    quality = 200 - quality * 2;
end
matr = lumMat;
for i=1:128
    matr(i) = floor((matr(i) * quality + 50) / 100);
    if matr(i) <= 0
        matr(i) = 1;
    elseif matr(i) > 255
        matr(i) = 255;
    end
end
out = matr;
end