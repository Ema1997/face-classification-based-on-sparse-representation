%% 导入数据
clc;clear all;close all;
addpath('./ompbox10');
data = load('DataBase.mat');
DataBase = data.DataBase;
param.sparsity = 50;

%% 数据初始化
number_class = 38;%总共38类
sparsity = param.sparsity;
D = normc(DataBase.training_feats);
Y = DataBase.testing_feats;
Y_num = size(Y, 2); %总共Y_num个测试样例
D_label = DataBase.H_train;
Y_label = DataBase.H_test;

%% OMP求解
tic %开始计算求解时间
DtX = D' * Y;
G = D' * D;
alpha = omp(DtX, G, sparsity);

%% 图像分类
Y_label_pred = zeros(Y_num, 1);
for i = 1 : Y_num
    Yi = Y(:, i);
    alphai = alpha(:, i);%与当前样本对应的参数
    
    alphai_m = repmat(alphai, 1, number_class);%为了方便矩阵运算，将当前样本对应的参数扩展
    Yi_m = repmat(Yi, 1, number_class);
    e_m = Yi_m - D * (alphai_m .* D_label');
    
    res = zeros(number_class, 1);%初始化res
    for j = 1 : number_class
        res(j) = norm(e_m(:,j), 2);
    end
    
    [ma, c] = min(res);
    Y_label_pred(i) = c;
end
time = toc; %计时结束
st = ['计算时间是', num2str(time), 's'];
disp(st);

%% 计算错误率
error_number = 0;
for i = 1 : Y_num
    if Y_label(Y_label_pred(i), i) ~= 1
        error_number = 1 + error_number;
    end
end
accuracy_SRC = (Y_num - error_number) / Y_num * 100;
st = ['SRC的分类准确率是',  num2str(accuracy_SRC), '%'];
disp(st);