function RDOEA_FR()
s=9
clc;format compact;
src='kodak24\';
src1='Cov_truevalue\';
src2='unCov_truevalue\';
src3='rdoea\kodak24\';
filename=['kodim',num2str(s,'%d'),'.png'];
inputName=fullfile(src,filename);
filename1=['kodim',num2str(s,'%d'),'_truevalue.txt'];
unCov_truevalueName=fullfile(src2,filename1);
filename2=['jpeg_kodim0',num2str(s,'%d'),'.txt'];
jpegName=fullfile(src2,filename2);
filename3=['kodim0',num2str(s,'%d'),'_covtruevalue.txt'];
cov_truevalueName=fullfile(src1,filename3);
filename4=['rdoea_chromtruevalue_kodim',num2str(s,'%d'),'.xlsx'];
rdoea_chromName=fullfile(src3,filename4);

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
generations=100;                                %迭代次数
population1=100;                                     %种群1大小(须为偶数)
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
psyq =[16 14 13 15 19 28 37 55 14 13 15 19 28 37 55 64 13 15 19 28 37 55 64 83 15 19 28 37 55 64 83 103 19 28 37 55 64 83 103 117 28 37 55 64 83 103 117 117  37  55 64 83 103 117 117 111 55 64 83 103 117 117 111 90 ... 
 18 18 23 34 45 61 71 92 18 23 34 45 61 71 92 92 23 34 45 61 71 92 92 104 34 45 61 71 92 92 104 34 45 61 71 92 92 104 115 45 61 71 92 92 104 115 119 61 71 92 92 104 115 119 112 71 92 92 104 115 119 112 106 92 92 104 115 119 112 106 100];
%  population = zeros(population1,64);
  for i = 1:population1%种群1初始化   
     out = quantLumMat(lumMat, i*2);
     population(i,:) = out;
  end
  pop = 100;
   population2 = zeros(pop,128);
  for i = 1:pop%种群2初始化   
     out = quantLumMat(lumMat, (50 - ((i) - pop/2)*5));
     population2(i,:) = out;
  end 
  for i = 1:100%psyQ初始化   
     out = quantLumMat(psyq, i);
     population_psyQ(i,:) = out;    
  end 

 x1=zeros(40,1);
        y1=linspace(0.2,50,20)';
%         y2=linspace(50,200,10)';
        y3=linspace(50,300,20)';
        y4=[y1;y3];
%         x=linspace(0.2,4,40)';
        x2=linspace(0.2,3,20)';
%         x3=linspace(1,3,10)';
        x4=linspace(3,4,5)';
        x5=linspace(4,5,15)';
        x=[x2;x4;x5];
        y=zeros(40,1);
        W=[x1,y4;x,y];
  
  functionvalue_psyQ = MSE(population_psyQ,y_sub_image,cb_sub_image,cr_sub_image, y_sub_dct,cb_sub_dct,cr_sub_dct, repeat_height, repeat_width, outputName, 0, mse_table, rate_table);
 functionvalue = MSE(population,y_sub_image,cb_sub_image,cr_sub_image, y_sub_dct,cb_sub_dct,cr_sub_dct, repeat_height, repeat_width, outputName, 1, mse_table, rate_table);
   X = [1.4,1.6,1.8,2.0];   %可行域
    Y = abs(X - functionvalue(:,1));
    Y(Y(:,1) <=0.05,1)= 0; % [0.2-0.2*10%,0.2+0.2*10%]
    Y(Y(:,2) <=0.1,2)= 0;  %[0.6-0.6*10%,0.6+0.6*10%]
    Y(Y(:,3) <=0.1,3)= 0;  %[1-1*10%,1+1*10%]
    Y(Y(:,4) <=0.2,4)= 0;  %[1.4-1.4*10%,1.4+1.4*10%]
    [cov,~]=min(Y,[],2);   %每一行选择最小的保存
    CV = CalCV(cov);       %归一化
    Fitness = CalFitness(functionvalue,CV); %计算适应度值
      
    [populationlum1,Fitness] = EnvironmentalSelection2(Fitness,population,population1); 
%     functionvalue_init = MSE(populationlum1,y_sub_image,cb_sub_image,cr_sub_image, y_sub_dct,cb_sub_dct,cr_sub_dct, repeat_height, repeat_width, outputName, 1, mse_table, rate_table);
%     figure;
%     plot(functionvalue_init(:,1),functionvalue_init(:,2),'*b');
%populationlum = populationlum1;
 populationlum2 = population2;
%---开始迭代进化
 for gene=1:generations                      %开始迭代
     gene
%     populationlum2 = [populationlum2;populationlum1];   
  populationlum = populationlum1;
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
                for j=1:64
                    newpopulation(i,j)=round((newpopulation(i,j)*scale+50)/100);
                    if newpopulation(i,j)<1
                        newpopulation(i,j)=1;
                    else
                        if newpopulation(i,j)>255
                            newpopulation(i,j)=255;
                        end
                    end
                end
                for k=1:64
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

    px=size(populationlum2,1);           %种群2无约束
    if mod(px,2)~=0
        populationlum2(px+1,:)=lumMat;
    end
    newpopulation2=ones(size(populationlum2,1),poplength);
    for i = 1:2:px-1
        a=rand();
        if(a<=0.2)
            cpoint = 16;
            index = getTopKPostion(mse_table, rate_table,populationlum2(i,:), populationlum2(i+1,:), cpoint);
            newpopulation2(i,:) = populationlum2(i,:);
            newpopulation2(i, index) = populationlum2(i+1, index);
            index = getTopKPostion(mse_table, rate_table,populationlum2(i+1,:), populationlum2(i,:), cpoint);
            newpopulation2(i+1,:) = populationlum2(i+1,:);
            newpopulation2(i+1, index) = populationlum2(i, index);              %交叉
        else
            b=rand();
            if b<=0.5
                if gene<generations/2
                   temp =round(rand.*63)+1;
                else
                   temp = 16;
                end
                newpopulation2(i,:) = populationlum2(i,:);
                newpopulation2(i+1,:) = populationlum2(i+1,:);
                %            [add_index, minus_index] = getTopKPostionForMut(mse_table, rate_table, populationlum(i,:), temp);%lp
                [ minus_index,index_new] = getTopKPostionForMut(mse_table, rate_table, populationlum2(i,:), temp);
                populationlum2(i,index_new(1,:)) = minus_index(1,index_new(1,:));
                [ minus_index,index_new] = getTopKPostionForMut(mse_table, rate_table,populationlum2(i+1,:), temp);
                populationlum2(i+1,index_new(1,:)) = minus_index(1,index_new(1,:));
            else
                newpopulation2(i,:) = populationlum2(i,:);              %变异二
                newpopulation2(i+1,:) = populationlum2(i+1,:);
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
                for j=1:64
                    newpopulation2(i,j)=round((newpopulation2(i,j)*scale+50)/100);
                    if newpopulation2(i,j)<1
                        newpopulation2(i,j)=1;
                    else
                        if newpopulation2(i,j)>255
                            newpopulation2(i,j)=255;
                        end
                    end
                end
                for k=1:64
                    newpopulation2(i+1,k)=round((newpopulation2(i+1,k)*scale+50)/100);
                    if newpopulation2(i+1,k)<1
                        newpopulation2(i+1,k)=1;
                    else
                        if newpopulation2(i+1,k)>255
                            newpopulation2(i+1,k)=255;
                        end
                    end
                end
            end
        end
    end
     Q=newpopulation2;
    functionvalue_Q=MSE(Q, y_sub_image,cb_sub_image,cr_sub_image, y_sub_dct,cb_sub_dct,cr_sub_dct, repeat_height, repeat_width, outputName, 1, mse_table, rate_table);
    YQ = abs(X - functionvalue_Q(:,1));
    YQ(YQ(:,1) <=0.05,1)= 0; % [0.2-0.2*10%,0.2+0.2*10%]
    YQ(YQ(:,2) <=0.1,2)= 0;  %[0.6-0.6*10%,0.6+0.6*10%]
    YQ(YQ(:,3) <=0.1,3)= 0;  %[1-1*10%,1+1*10%]
    YQ(YQ(:,4) <=0.2,4)= 0;  %[1.4-1.4*10%,1.4+1.4*10%]
    [covQ,~]=min(YQ,[],2);   %每一行选择最小的保存  
%      [covQ_sort,covQ_sortindex]=sortrows(covQ);%对适应度进行排序
   feasible_index=find(covQ==0);
   feasible_population=Q(feasible_index,:);
%    functionvalue_feaible=MSE( feasible_population, y_sub_image,cb_sub_image,cr_sub_image, y_sub_dct,cb_sub_dct,cr_sub_dct, repeat_height, repeat_width, outputName, 1, mse_table, rate_table);
%    figure;
%    plot(functionvalue_feaible(:,1),functionvalue_feaible(:,2),'*b');
   notfeasible_index=find(covQ>0);%找到适应度值大于0的解的下标
   notfeasible_population=Q(notfeasible_index,:);
   notfeasible_solution=functionvalue_Q(notfeasible_index,:);
   for i=1:size(X,2)
   areas(i,:)=[X(:,i)-0.05,X(:,i)+0.05];
   end
   [~,population_new] = find_nearest_points(notfeasible_solution, notfeasible_population,areas);
%     newpopulation=[feasible_population;population_new];
%    functionvalue=MSE(population_new, y_sub_image,cb_sub_image,cr_sub_image, y_sub_dct,cb_sub_dct,cr_sub_dct, repeat_height, repeat_width, outputName, 1, mse_table, rate_table);
   newpopulation3=[feasible_population;population_new];
   populationlum1=[populationlum1;newpopulation;newpopulation3];
   populationlum2=[populationlum2;newpopulation;newpopulation2];
 functionvalue = MSE(populationlum1,y_sub_image,cb_sub_image,cr_sub_image, y_sub_dct,cb_sub_dct,cr_sub_dct, repeat_height, repeat_width, outputName, 1, mse_table, rate_table);
    
%     X = [1.4,1.6,1.8,2.0];   %可行域
    Y = abs(X - functionvalue(:,1));
    Y(Y(:,1) <=0.05,1)= 0; % [0.2-0.2*10%,0.2+0.2*10%]
    Y(Y(:,2) <=0.1,2)= 0;  %[0.6-0.6*10%,0.6+0.6*10%]
    Y(Y(:,3) <=0.1,3)= 0;  %[1-1*10%,1+1*10%]
    Y(Y(:,4) <=0.2,4)= 0;  %[1.4-1.4*10%,1.4+1.4*10%]
    [cov,~]=min(Y,[],2);   %每一行选择最小的保存
    CV = CalCV(cov);       %归一化
    Fitness = CalFitness(functionvalue,CV); %计算适应度值
      
    [populationlum1,Fitness] = EnvironmentalSelection2(Fitness,populationlum1,population1);   
%     functionvalue_gen = MSE(populationlum1, y_sub_image,cb_sub_image,cr_sub_image, y_sub_dct,cb_sub_dct,cr_sub_dct, repeat_height, repeat_width, outputName, 1, mse_table, rate_table); 
%     
% 
%     count = ones(1,4);
%     k=1;
%     Y = abs(X - functionvalue_gen(:,1));
%     Y(Y(:,1) <=0.05,1)= 0; % [0.2-0.2*10%,0.2+0.2*10%]
%     count(1,1)=size(find(Y(:,1)==0),1);
% 
%     Y(Y(:,2) <=0.1,2)= 0;  %[0.6-0.6*10%,0.6+0.6*10%]
%     count(1,2)=size(find(Y(:,2)==0),1);
% 
%     Y(Y(:,3) <=0.1,3)= 0;  %[1-1*10%,1+1*10%]
%     count(1,3)=size(find(Y(:,3)==0),1);
% 
%     Y(Y(:,4) <=0.2,4)= 0;
%     count(1,4)=size(find(Y(:,4)==0),1);
% 
%     
%     g = size(functionvalue_gen,1)+1;
%     x=1;
%     for i=1:4
%         if count(1,i)<10
%             populationlum1_new = populationlum1(find(Y(:,i)==0),:);
%             functionvalue_gen1 = functionvalue_gen(find(Y(:,i)==0),:);
%            for r =1:(size(functionvalue_gen1,1)-1)
%               c= (functionvalue_gen1(r+1,1)- functionvalue_gen1(r,1)).^2+(functionvalue_gen1(r+1,2)- functionvalue_gen1(r,2)).^2;
%              if c>0.01       
%                  popolationlum2_new= populationlum1_new(r+1,:);
%                  ratelow = functionvalue_gen1(r+1,1);
%                  ratehigh = functionvalue_gen1(r,1);
%     %                   quality = 50 - (ratelow - ratehigh)*50/ratelow
%                  quality = 50 - (ratelow - ratehigh)*50/ratelow;
%                 populationlumnew1= quantLumMat(popolationlum2_new, quality); 
% %                 populationlumnew2(x,:) = populationlumnew1;
%                 x=x+1;
%                  populationlum1(g,:)=populationlumnew1;  
%                  g=g+1;           
%              end     
%            end 
%             
%         end
%     end

functionvalue = MSE(newpopulation2, y_sub_image,cb_sub_image,cr_sub_image, y_sub_dct,cb_sub_dct,cr_sub_dct, repeat_height, repeat_width, outputName, 1, mse_table, rate_table);
 
    [frontvalue, ~] = NDSort(functionvalue, inf);
%     B=pdist2(W,functionvalue);%计算每个参考点和functionvalue的欧氏距离
%     [b1,m] = sort(B,2);
%     n=m(:,1);
%     n=unique(n,'rows');
%      for z=1:size(n,1)  
%          population_near(z,:)=newpopulation2(n(z),:);
%         
%         %  populationlum=[populationlum;newpopulation(n(z),:)];%保存40个距离参考点最近的点
%          
%      end
%      populationlum2=population_near;
     newnum=numel(frontvalue,frontvalue<=1);                              %前fnum个面的个体数   
    if(newnum <50)
        populationlum2(1:newnum,:)=newpopulation2(frontvalue<=1,:);
    else
        populationlum_temp(1:newnum,:)=newpopulation2(frontvalue<=1,:);
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
  
 
    
 end
 
%    truevalue = MSE(population_new, y_sub_image,cb_sub_image,cr_sub_image, y_sub_dct,cb_sub_dct,cr_sub_dct, repeat_height, repeat_width, outputName, 0, mse_table, rate_table);


% populationlum2 = populationlum1;
%     truevalue = MSE(populationlum2(1,:), y_image, y_sub_image, y_sub_dct, repeat_height, repeat_width, outputName, 0, mse_table,w, rate_table);
    fprintf('已完成,耗时%4s秒\n',num2str(toc));          %程序最终耗时

 
   
    truevalue = MSE(populationlum1, y_sub_image,cb_sub_image,cr_sub_image, y_sub_dct,cb_sub_dct,cr_sub_dct, repeat_height, repeat_width, outputName, 0, mse_table, rate_table);
    estvalue = MSE(populationlum1, y_sub_image,cb_sub_image,cr_sub_image, y_sub_dct,cb_sub_dct,cr_sub_dct, repeat_height, repeat_width, outputName, 1, mse_table, rate_table);
 [frontvalue, ~] = NDSort(truevalue, inf);
 truevalue=truevalue(frontvalue<=1,:);
%     Y = abs(X - truevalue(:,1));
%     Y_est = abs(X - estvalue(:,1));
%     Y_est(Y_est(:,1) <=0.05,1)= 0;
%     Y_est(Y_est(:,2) <=0.1,2)= 0;
%     Y_est(Y_est(:,3) <=0.1,3)= 0;
%     Y_est(Y_est(:,4) <=0.1,4)= 0;
%     [cov_est,~]=min(Y_est,[],2);
%     index_est = find(cov_est==0);
    
%     Y(Y(:,1) <=0.05,1)= 0;
%     Y(Y(:,2) <=0.1,2)= 0;
%     Y(Y(:,3) <=0.1,3)= 0;
%     Y(Y(:,4) <=0.1,4)= 0;
%     [cov,~]=min(Y,[],2);
%     Yvalue = X - truevalue(:,1);
%     index1 = find(cov==0);
%     truevalue_new=truevalue(index1,:);

    Yest = abs(X - estvalue(:,1));
    Yest(Yest(:,1) <=0.05,1)= 0;
    Yest(Yest(:,2) <=0.1,2)= 0;
    Yest(Yest(:,3) <=0.1,3)= 0;
    Yest(Yest(:,4) <=0.1,4)= 0;
    [covest,~]=min(Yest,[],2);
    indexest = find(covest==0);
    output=estvalue(indexest,:);
  
    Ytrue = abs(X - truevalue(:,1));
    Ytrue(Ytrue(:,1) <=0.05,1)= 0;
    Ytrue(Ytrue(:,2) <=0.05,2)= 0;
    Ytrue(Ytrue(:,3) <=0.05,3)= 0;
    Ytrue(Ytrue(:,4) <=0.05,4)= 0;
    [covtrue,~]=min(Ytrue,[],2);
    indextrue = find(covtrue==0);
    truevalue_new=truevalue(indextrue,:);
    figure;
    
    y1 = 50;
    y2 = 50;
    y3 = 50;
    y4 = 50;
    % bar(x,y);
    hold on
    bar(1.4,y1,0.05,'FaceColor',[0.75 0.75 0.75]);
    hold on
    bar(1.6,y2,0.05,'FaceColor',[0.75 0.75 0.75]);
    hold on
    bar(1.8,y3,0.05,'FaceColor',[0.75 0.75 0.75]);
    hold on
    bar(2.0,y4,0.05,'FaceColor',[0.75 0.75 0.75]);
%     unCov_truevalue =load(unCov_truevalueName);
   for i=1:100   %JPEG的对比算法不要在这个代码里面写   这样影响你算法的时间复杂度
    out = quantLumMat(lumMat, i);
      populationlum_jpeg(i,:) = out;
end
   jpeg = MSE(populationlum_jpeg, y_sub_image,cb_sub_image,cr_sub_image, y_sub_dct,cb_sub_dct,cr_sub_dct, repeat_height, repeat_width, outputName, 0, mse_table, rate_table);
%      jpeg = load(jpegName);
     
%      cov_truevalue=load(cov_truevalueName);
%      unCov_truevalue=sortrows(unCov_truevalue);
%      cov_truevalue=sortrows(cov_truevalue);
     truevalue_new=sortrows(truevalue_new);
     plot(truevalue_new(:,1),truevalue_new(:,2),'*r');%双种群的解
     hold on;
%      plot(output(:,1),output(:,2),'HM');%双种群估算的解
%      hold on;
     plot(jpeg(:,1),jpeg(:,2),'og');
     
     hold on;
     rdoea_chrom=xlsread(rdoea_chromName);
     plot( rdoea_chrom(:,1), rdoea_chrom(:,2),'*b');
     hold on;
     plot(functionvalue_psyQ(:,1),functionvalue_psyQ(:,2),'^b');
% 
     hold on;
%      plot(unCov_truevalue(:,1),unCov_truevalue(:,2),'hm');%没有约束的单种群的解
%      hold on;
%      plot(cov_truevalue(:,1),cov_truevalue(:,2),'hm');%带约束的单种群的解
     xlabel('Rate(bpp)');ylabel('mse');
%      legend('Feasible regions','Feasible regions','Feasible regions','Feasible regions','Obtained solutions','Est with constraints','jpeg','PF without constraints','constraint with single');
% legend('Feasible regions','Feasible regions','Feasible regions','Feasible regions','VOCEA','JPEG','VOCEA w/o constrained pop','VOCEA w/o unconstrained pop');
legend('Feasible regions','Feasible regions','Feasible regions','Feasible regions','MY','JPEG','RDOEA','PSYQ');
 end


function CV = CalCV(CV_Original)
    CVmax = max(CV_Original);
    CVmin = min(CV_Original);
    CV = (CV_Original-CVmin)./(CVmax - CVmin);
%     CV(:,isnan(CV(1,:))) = 0;%12.23
%     CV = mean(CV,2);%12.23
end


function out = getTopKPostion(mse_whole, rate_whole, lumMat1, lumMat2, k)
value = zeros(64,1);
value(:) = 10000000;
for i = 1:64
   m = floor((i-1)/8)+1;
    n = mod(i-1, 8) + 1;
    q1 = lumMat1(i);
    q2 = lumMat2(i);
    if q1 ~= q2
        if q1 < q2
            dist1 = mse_whole(m,n,q1);
            dist2 = mse_whole(m,n,q2);
            rate1 = rate_whole(m,n,q1);
            rate2 = rate_whole(m,n,q2);
            if rate1 ~= rate2
                slope = (dist2 - dist1)/(rate1 - rate2); %smaller better
                value(i) = slope;
            end
        else
            dist1 = mse_whole(m,n,q1);
            dist2 = mse_whole(m,n,q2);
            rate1 = rate_whole(m,n,q1);
            rate2 = rate_whole(m,n,q2);
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
 minus_value = zeros(1,64);
 minus_index = zeros(1,64);
 for i = 1:64
     value(1:255) = 0;
     m = floor((i-1)/8)+1;
     n = mod(i-1, 8) + 1;
     q1 = lumMat1(i); 
     q2 = q1 + 1;
     if q1 < 205&&q1>50
         q_high=q1+50;
         for j=q2:q_high
             if q1 < q2
                 dist1 = mse_whole(m,n,q1);
                 dist2 = mse_whole(m,n,j);
                 rate1 = rate_whole(m,n,q1);
                 rate2 = rate_whole(m,n,j);
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
                     dist1 = mse_whole(m,n,q1);
                     dist2 = mse_whole(m,n,j);
                     rate1 = rate_whole(m,n,q1);
                     rate2 = rate_whole(m,n,j);
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
                         dist1 = mse_whole(m,n,q1);
                         dist2 = mse_whole(m,n,j);
                         rate1 = rate_whole(m,n,q1);
                         rate2 = rate_whole(m,n,j);
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
                 dist1 = mse_whole(m,n,q1);
                 dist2 = mse_whole(m,n,j);
                 rate1 = rate_whole(m,n,q1);
                 rate2 = rate_whole(m,n,j);
                 if rate2~=rate1
                     slope = (dist1 - dist2)/(rate2 - rate1); %bigger better
                     value(j) = slope;
                 end
             end
         else
             if q1<=50
                 for j=1:q2
                     dist1 = mse_whole(m,n,q1);
                     dist2 = mse_whole(m,n,j);
                     rate1 = rate_whole(m,n,q1);
                     rate2 = rate_whole(m,n,j);
                     if rate2~=rate1
                         slope = (dist1 - dist2)/(rate2 - rate1); %bigger better
                         value(j) = slope;
                     end
                 end
             else
                 if q1>=205
                     q_low=q1-50;
                     for j=q_low:q2
                         dist1 = mse_whole(m,n,q1);
                         dist2 = mse_whole(m,n,j);
                         rate1 = rate_whole(m,n,q1);
                         rate2 = rate_whole(m,n,j);
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
for i=1:64
    matr(i) = floor((matr(i) * quality + 50) / 100);
    if matr(i) <= 0
        matr(i) = 1;
    elseif matr(i) > 255
        matr(i) = 255;
    end
end
out = matr;
end