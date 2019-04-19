%% ��������
clc;clear all;close all;
addpath('./ompbox10');
data = load('DataBase.mat');
DataBase = data.DataBase;
param.sparsity = 50;

%% ���ݳ�ʼ��
number_class = 38;%�ܹ�38��
sparsity = param.sparsity;
D = normc(DataBase.training_feats);
Y = DataBase.testing_feats;
Y_num = size(Y, 2); %�ܹ�Y_num����������
D_label = DataBase.H_train;
Y_label = DataBase.H_test;

%% OMP���
tic %��ʼ�������ʱ��
DtX = D' * Y;
G = D' * D;
alpha = omp(DtX, G, sparsity);

%% ͼ�����
Y_label_pred = zeros(Y_num, 1);
for i = 1 : Y_num
    Yi = Y(:, i);
    alphai = alpha(:, i);%�뵱ǰ������Ӧ�Ĳ���
    
    alphai_m = repmat(alphai, 1, number_class);%Ϊ�˷���������㣬����ǰ������Ӧ�Ĳ�����չ
    Yi_m = repmat(Yi, 1, number_class);
    e_m = Yi_m - D * (alphai_m .* D_label');
    
    res = zeros(number_class, 1);%��ʼ��res
    for j = 1 : number_class
        res(j) = norm(e_m(:,j), 2);
    end
    
    [ma, c] = min(res);
    Y_label_pred(i) = c;
end
time = toc; %��ʱ����
st = ['����ʱ����', num2str(time), 's'];
disp(st);

%% ���������
error_number = 0;
for i = 1 : Y_num
    if Y_label(Y_label_pred(i), i) ~= 1
        error_number = 1 + error_number;
    end
end
accuracy_SRC = (Y_num - error_number) / Y_num * 100;
st = ['SRC�ķ���׼ȷ����',  num2str(accuracy_SRC), '%'];
disp(st);