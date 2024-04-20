#include "lstmlib.h"
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>

// 函数用于创建一个lstmlib结构体实例，该结构体用于存储LSTM网络的状态和参数
struct lstmlib* lstmlib_create(int length)
{
    struct lstmlib* unit; // 定义一个指向lstmlib结构体的指针
    int n_classes = 3;// 类别数：3个

    if (length < 1) {
        return NULL; // 如果输入的长度小于1，返回NULL，表示创建失败
    }
    unit = (struct lstmlib*)malloc(sizeof (struct lstmlib)); // 动态分配内存给lstmlib结构体
    if (!unit) {
        return NULL; // 如果内存分配失败，返回NULL
    }
    (*unit).error_no = 0; // 初始化错误编号为0，表示没有错误
    (*unit).error_msg = "\0"; // 初始化错误信息为空字符串
    (*unit).length = length; // 设置LSTM网络的长度（时间步长）
    (*unit).n_classes = n_classes;//类别数3
    (*unit).x = (double*)calloc(length, sizeof (double)); // 分配内存用于存储输入x，并初始化为0
    if (NULL == (*unit).x) {
        free(unit); // 如果内存分配失败，释放之前分配的内存并返回NULL
        return NULL;
    }
    (*unit).h = (double*)calloc(length, sizeof (double)); // 分配内存用于存储隐藏状态h，并初始化为0
    if (NULL == (*unit).h) {
        free((*unit).x); // 如果内存分配失败，释放输入x的内存
        free(unit); // 释放lstmlib结构体的内存
        return NULL;
    }
    (*unit).f = (double*)calloc(length, sizeof (double)); // 分配内存用于存储遗忘门f的值，并初始化为0
    if (NULL == (*unit).f) {
        free((*unit).h); // 如果内存分配失败，释放隐藏状态h的内存
        free((*unit).x); // 释放输入x的内存
        free(unit); // 释放lstmlib结构体的内存
        return NULL;
    }
    (*unit).i = (double*)calloc(length, sizeof (double)); // 分配内存用于存储输入门i的值，并初始化为0
    if (NULL == (*unit).i) {
        free((*unit).f); // 如果内存分配失败，释放遗忘门f的内存
        free((*unit).h); // 释放隐藏状态h的内存
        free((*unit).x); // 释放输入x的内存
        free(unit); // 释放lstmlib结构体的内存
        return NULL;
    }
    (*unit).tilde_C = (double*)calloc(length, sizeof (double)); // 分配内存用于存储候选记忆细胞C'，并初始化为0
    if (NULL == (*unit).tilde_C) {
        free((*unit).i); // 如果内存分配失败，释放输入门i的内存
        free((*unit).f); // 释放遗忘门f的内存
        free((*unit).h); // 释放隐藏状态h的内存
        free((*unit).x); // 释放输入x的内存
        free(unit); // 释放lstmlib结构体的内存
        return NULL;
    }
    (*unit).C = (double*)calloc(length, sizeof (double)); // 分配内存用于存储记忆细胞C，并初始化为0
    if (NULL == (*unit).C) {
        free((*unit).tilde_C); // 如果内存分配失败，释放候选记忆细胞C'的内存
        free((*unit).i); // 释放输入门i的内存
        free((*unit).f); // 释放遗忘门f的内存
        free((*unit).h); // 释放隐藏状态h的内存
        free((*unit).x); // 释放输入x的内存
        free(unit); // 释放lstmlib结构体的内存
        return NULL;
    }
    (*unit).o = (double*)calloc(length, sizeof (double)); // 分配内存用于存储输出门o的值，并初始化为0
    if (NULL == (*unit).o) {
        free((*unit).C); // 如果内存分配失败，释放记忆细胞C的内存
        free((*unit).tilde_C); // 释放候选记忆细胞C'的内存
        free((*unit).i); // 释放输入门i的内存
        free((*unit).f); // 释放遗忘门f的内存
        free((*unit).h); // 释放隐藏状态h的内存
        free((*unit).x); // 释放输入x的内存
        free(unit); // 释放lstmlib结构体的内存
        return NULL;
    }
    (*unit).hat_h = (double*)calloc(length, sizeof (double)); // 分配内存用于存储目标隐藏状态，并初始化为0
    if (NULL == (*unit).hat_h) {
        free((*unit).o); // 如果内存分配失败，释放输出门o的内存
        free((*unit).C); // 释放记忆细胞C的内存
        free((*unit).tilde_C); // 释放候选记忆细胞C'的内存
        free((*unit).i); // 释放输入门i的内存
        free((*unit).f); // 释放遗忘门f的内存
        free((*unit).h); // 释放隐藏状态h的内存
        free((*unit).x); // 释放输入x的内存
        free(unit); // 释放lstmlib结构体的内存
        return NULL;
    }
    (*unit).softmax_output  = (double*)calloc(n_classes, sizeof (double)); // 分配内存用于分类概率状态，并初始化为0  hcz_add
    if (NULL == (*unit).softmax_output ) {
        free((*unit).hat_h); // 如果内存分配失败，释放分类概率状态hat_h的内存
        free((*unit).o); // 如果内存分配失败，释放输出门o的内存
        free((*unit).C); // 释放记忆细胞C的内存
        free((*unit).tilde_C); // 释放候选记忆细胞C'的内存
        free((*unit).i); // 释放输入门i的内存
        free((*unit).f); // 释放遗忘门f的内存
        free((*unit).h); // 释放隐藏状态h的内存
        free((*unit).x); // 释放输入x的内存
        free(unit); // 释放lstmlib结构体的内存
        return NULL;
    }
    (*unit).W_hy  = (double*)calloc(n_classes, sizeof (double)); // 分配内存用于输出权重Wh，并初始化为0  hcz_add
    if (NULL == (*unit).W_hy ) {
        free((*unit).softmax_output); // 如果内存分配失败，释放输出权重softmax的内存
        free((*unit).hat_h); // 如果内存分配失败，释放目标隐藏状态hat_h的内存
        free((*unit).o); // 如果内存分配失败，释放输出门o的内存
        free((*unit).C); // 释放记忆细胞C的内存
        free((*unit).tilde_C); // 释放候选记忆细胞C'的内存
        free((*unit).i); // 释放输入门i的内存
        free((*unit).f); // 释放遗忘门f的内存
        free((*unit).h); // 释放隐藏状态h的内存
        free((*unit).x); // 释放输入x的内存
        free(unit); // 释放lstmlib结构体的内存
        return NULL;
    }
    (*unit).b_y  = (double*)calloc(n_classes, sizeof (double)); // 分配内存用于输出偏置by，并初始化为0  hcz_add
    if (NULL == (*unit).b_y ) {
        free((*unit).W_hy); // 如果内存分配失败，释放输出权重Wh的内存
        free((*unit).softmax_output); // 如果内存分配失败，释放输出权重softmax的内存
        free((*unit).hat_h); // 如果内存分配失败，释放目标隐藏状态hat_h的内存
        free((*unit).o); // 如果内存分配失败，释放输出门o的内存
        free((*unit).C); // 释放记忆细胞C的内存
        free((*unit).tilde_C); // 释放候选记忆细胞C'的内存
        free((*unit).i); // 释放输入门i的内存
        free((*unit).f); // 释放遗忘门f的内存
        free((*unit).h); // 释放隐藏状态h的内存
        free((*unit).x); // 释放输入x的内存
        free(unit); // 释放lstmlib结构体的内存
        return NULL;
    }
    lstmlib_random_params(unit, -1, 1); // 调用函数随机初始化网络参数
    return unit; // 返回创建的lstmlib结构体实例
}

// 函数用于随机初始化LSTM网络的参数
char lstmlib_random_params(struct lstmlib *unit, double min, double max)
{
    int i; // 循环变量
    double diff; // 用于存储参数的范围
    // 如果传入的LSTM网络单元是NULL或者最大值小于最小值，返回0表示失败
    if (NULL == unit) {
        return 0;
    }
    if (max < min) {
        return 0;
    }
    diff = max - min; // 计算参数的范围
    // 从后向前遍历网络的长度，为每个时间步的输入、隐藏状态、门参数和目标隐藏状态
    // 以及其他网络参数赋予一个[min, max]范围内的随机值
    i = (*unit).length - 1;
    // 为网络的时间序列参数赋予随机值
    for (int i = 0; i < unit->length; i++) {
        unit->x[i] = 0.0; // 可以选择初始化输入为0或赋予随机值
        unit->h[i] = 0.0; // 可以选择初始化隐藏状态为0或赋予随机值
        unit->f[i] = (double)rand() / RAND_MAX * diff + min;
        unit->i[i] = (double)rand() / RAND_MAX * diff + min;
        unit->tilde_C[i] = (double)rand() / RAND_MAX * diff + min;
        unit->C[i] = (double)rand() / RAND_MAX * diff + min;
        unit->o[i] = (double)rand() / RAND_MAX * diff + min;
        unit->hat_h[i] = (double)rand() / RAND_MAX * diff + min;
    }
    // 为网络的权重参数赋予随机值，范围同样是[min, max]
    (*unit).W_fh = (double)rand() / RAND_MAX * diff + min; // 遗忘门到隐藏状态的权重
    (*unit).W_fx = (double)rand() / RAND_MAX * diff + min; // 输入到遗忘门的权重
    (*unit).b_f = (double)rand() / RAND_MAX * diff + min; // 遗忘门的偏置
    (*unit).W_ih = (double)rand() / RAND_MAX * diff + min; // 隐藏状态到输入门的权重
    (*unit).W_ix = (double)rand() / RAND_MAX * diff + min; // 输入到输入门的权重
    (*unit).b_i = (double)rand() / RAND_MAX * diff + min; // 输入门的偏置
    (*unit).W_Ch = (double)rand() / RAND_MAX * diff + min; // 隐藏状态到记忆细胞的权重
    (*unit).W_Cx = (double)rand() / RAND_MAX * diff + min; // 输入到记忆细胞的权重
    (*unit).b_C = (double)rand() / RAND_MAX * diff + min; // 记忆细胞的偏置
    (*unit).W_oh = (double)rand() / RAND_MAX * diff + min; // 隐藏状态到输出门的权重
    (*unit).W_ox = (double)rand() / RAND_MAX * diff + min; // 输入到输出门的权重
    (*unit).b_o = (double)rand() / RAND_MAX * diff + min; // 输出门的偏置
    
    // 初始化W_hy和b_y（随机初始权重）
    for (int i = 0; i < unit->n_classes; i++) {
        unit->W_hy[i] = (double)rand() / RAND_MAX * diff + min;
        unit->b_y[i] = (double)rand() / RAND_MAX * diff + min;
    }
    
    // 如果所有参数都成功赋予了随机值，返回1表示成功
    return 1;
}

// 函数用于执行LSTM单元的前向传播过程
char lstmlib_run_unit(struct lstmlib *unit, int *target_labels)
{
    if (unit == NULL) {
        return 0;
    }

    int length = unit->length;
    for (int i = 0; i < length; i++) {
        double x_i = (i == 0) ? 0.0 : unit->x[i - 1]; // 输入值，对于第一步可以考虑为0或从x[0]开始
        double h_prev = (i == 0) ? 0.0 : unit->h[i - 1]; // 前一个隐藏状态，第一步时为0

        // 计算门的值
        double s = unit->W_fh * h_prev + unit->W_fx * x_i + unit->b_f;
        unit->f[i] = 1.0 / (1.0 + exp(-s));

        s = unit->W_ih * h_prev + unit->W_ix * x_i + unit->b_i;
        unit->i[i] = 1.0 / (1.0 + exp(-s));

        s = unit->W_Ch * h_prev + unit->W_Cx * x_i + unit->b_C;
        unit->tilde_C[i] = tanh(s);

        unit->C[i] = unit->f[i] * ((i == 0) ? 0.0 : unit->C[i - 1]) + unit->i[i] * unit->tilde_C[i];

        s = unit->W_oh * h_prev + unit->W_ox * x_i + unit->b_o;
        unit->o[i] = 1.0 / (1.0 + exp(-s));

        unit->h[i] = unit->o[i] * tanh(unit->C[i]);
    }

    double z = 0.0;
    for (int i = 0; i < unit->n_classes; i++) {
        unit->softmax_output[i] = unit->W_hy[i] * unit->h[length - 1] + unit->b_y[i];
        z += exp(unit->softmax_output[i]);
    }

    for (int i = 0; i < unit->n_classes; i++) {
        unit->softmax_output[i] = exp(unit->softmax_output[i]) / z;
    }

    unit->Loss = 0.0;
    for (int i = 0; i < unit->n_classes; i++) {
        unit->Loss -= target_labels[i] * log(unit->softmax_output[i]);
    }
    /*假设 target_labels 是一个整数数组，
    其中包含了每个类别是否是目标类别的one-hot编码
    （例如，如果有3个类别，而实际类别是第二个，
    则 target_labels 为 [0, 1, 0]）*/

    return 1;
}

// 函数用于执行LSTM单元的后向传播过程，并更新网络参数
char lstmlib_fit_unit(struct lstmlib *unit, double lr, int *target_labels) {
    int i, t;
    int length = unit->length;
    double d_h_next = 0.0, d_C_next = 0.0;
    double d_W_hy[unit->n_classes];
    double d_b_y[unit->n_classes];
    memset(d_W_hy, 0, sizeof(d_W_hy));
    memset(d_b_y, 0, sizeof(d_b_y));
    // 输出层梯度计算
    for (i = 0; i < unit->n_classes; i++) {
        double d_z = unit->softmax_output[i] - ((target_labels[i] == 1) ? 1.0 : 0.0);
        d_W_hy[i] += d_z * unit->h[length - 1];  // 直接计算 W_hy 的梯度
        d_b_y[i] = d_z;  // 误差
        d_h_next += unit->W_hy[i] * d_z;  // 累加来自输出层的梯度
    }
    // 反向传播通过时间
    for (t = length - 1; t >= 0; t--) {
        double d_h = d_h_next + ((t == length - 1) ? 0 : unit->hat_h[t + 1] * d_C_next);
        double d_o = d_h * tanh(unit->C[t]);
        double d_net_o = d_o * unit->o[t] * (1 - unit->o[t]);
        double d_C = d_C_next + d_h * unit->o[t] * (1 - pow(tanh(unit->C[t]), 2));
        if (t > 0) {
            double d_f = d_C * unit->C[t - 1];
            double d_net_f = d_f * unit->f[t] * (1 - unit->f[t]);
            double d_i = d_C * unit->tilde_C[t];
            double d_net_i = d_i * unit->i[t] * (1 - unit->i[t]);
            double d_tilde_C = d_C * unit->i[t];
            double d_net_C = d_tilde_C * (1 - pow(unit->tilde_C[t], 2));
            // 更新遗忘门、输入门、候选单元状态权重的梯度
            unit->W_fh -= lr * d_net_f * unit->h[t - 1];
            unit->W_fx -= lr * d_net_f * unit->x[t];
            unit->b_f -= lr * d_net_f;

            unit->W_ih -= lr * d_net_i * unit->h[t - 1];
            unit->W_ix -= lr * d_net_i * unit->x[t];
            unit->b_i -= lr * d_net_i;

            unit->W_Ch -= lr * d_net_C * unit->h[t - 1];
            unit->W_Cx -= lr * d_net_C * unit->x[t];
            unit->b_C -= lr * d_net_C;

            d_h_next = d_net_f * unit->W_fh + d_net_i * unit->W_ih + d_net_C * unit->W_Ch;
            d_C_next = d_f * unit->f[t];
        }
        // 更新权重的梯度
        unit->W_oh -= lr * d_net_o * unit->h[t > 0 ? t - 1 : 0];
        unit->W_ox -= lr * d_net_o * unit->x[t];
        unit->b_o -= lr * d_net_o;
    }
    // 更新输出层权重和偏置
    for (i = 0; i < unit->n_classes; i++) {
        unit->W_hy[i] -= lr * d_W_hy[i];
        unit->b_y[i] -= lr * d_b_y[i];
    }
    return 1;
}

// 函数用于将LSTM网络的参数保存到文件中
int lstmlib_save(struct lstmlib *unit, char *file_name)
{
    FILE *file; // 文件指针，用于打开文件进行写入
    int write = 0; // 记录写入文件的字节数
    // 如果传入的LSTM网络单元是NULL或者存在错误，返回0表示失败
    if (NULL == unit) {
        return 0;
    }
    if (0 != (*unit).error_no) {
        return 0;
    }
    // 尝试打开指定的文件用于写入，如果文件打开失败，返回0表示失败
    file = fopen(file_name, "w");
    if (NULL == file) {
        return 0;
    }
    // 写入版本号到文件，这里固定为0
    write += fprintf(file, "0\n");
    // 写入网络的长度到文件
    write += fprintf(file, "%d\n", (*unit).length);
    // 依次写入网络的权重和偏置参数到文件
    write += fprintf(file, "%lf\n", (*unit).W_fh);
    write += fprintf(file, "%lf\n", (*unit).W_fx);
    write += fprintf(file, "%lf\n", (*unit).b_f);
    write += fprintf(file, "%lf\n", (*unit).W_ih);
    write += fprintf(file, "%lf\n", (*unit).W_ix);
    write += fprintf(file, "%lf\n", (*unit).b_i);
    write += fprintf(file, "%lf\n", (*unit).W_Ch);
    write += fprintf(file, "%lf\n", (*unit).W_Cx);
    write += fprintf(file, "%lf\n", (*unit).b_C);
    write += fprintf(file, "%lf\n", (*unit).W_oh);
    write += fprintf(file, "%lf\n", (*unit).W_ox);
    write += fprintf(file, "%lf\n", (*unit).b_o);
    // 保存输出层的权重和偏置：W_hy, b_y
    for (int i = 0; i < unit->n_classes; i++) {
        write += fprintf(file, "%lf\n", unit->W_hy[i]);
    }
    for (int i = 0; i < unit->n_classes; i++) {
        write += fprintf(file, "%lf\n", unit->b_y[i]);
    }
    // 关闭文件
    fclose(file);
    // 返回写入文件的总字节数
    return write;
}

//加载训练模型参数
struct lstmlib* lstmlib_load(struct lstmlib *unit, char *file_name) 
{
    FILE *file = fopen(file_name, "r");
    if (!file) {
        fprintf(stderr, "Failed to open file: %s\n", file_name);
        return NULL;
    }

    if (!unit) {
        fclose(file);
        return NULL;
    }
    char buffer[1024];  // 假设一行的数据不会超过 1024 个字符
    fgets(buffer, sizeof(buffer), file);// 读取并忽略第一行,第一行为版本号，这里固定为0
    
    fscanf(file, "%d\n", &unit->length);
    fscanf(file, "%lf\n", &unit->W_fh);
    fscanf(file, "%lf\n", &unit->W_fx);
    fscanf(file, "%lf\n", &unit->b_f);
    fscanf(file, "%lf\n", &unit->W_ih);
    fscanf(file, "%lf\n", &unit->W_ix);
    fscanf(file, "%lf\n", &unit->b_i);
    fscanf(file, "%lf\n", &unit->W_Ch);
    fscanf(file, "%lf\n", &unit->W_Cx);
    fscanf(file, "%lf\n", &unit->b_C);
    fscanf(file, "%lf\n", &unit->W_oh);
    fscanf(file, "%lf\n", &unit->W_ox);
    fscanf(file, "%lf\n", &unit->b_o);
    // 新增加的部分: 加载W_hy和b_y
    for (int i = 0; i < unit->n_classes; i++) {
        fscanf(file, "%lf\n", &unit->W_hy[i]);
    }
    for (int i = 0; i < unit->n_classes; i++) {
        fscanf(file, "%lf\n", &unit->b_y[i]);
    }

    fclose(file);
    return unit;
}

//加载csv训练数据
int load_csv_data(const char *file_name, double *data, int length) 
{
    FILE *file = fopen(file_name, "r");
    if (!file) {
        fprintf(stderr, "Failed to open file: %s\n", file_name);
        return 0;
    }
    //printf("File opened successfully.\n");
    
    int i;
    for (i = 0; i < length; i++) {
        if (fscanf(file, "%lf\n", &data[i]) != 1) {
            fprintf(stderr, "Failed to read data at index %d\n", i);
            fclose(file);
            return 0;
        }
    }
    if (i != length) {
        fprintf(stderr, "File does not contain enough data entries. Expected %d entries.\n", length);
        fclose(file);
        return 0;
    }
    fclose(file);
    return 1;
}

//one-hot对标签进行编码
void one_hot_encode(int class_id, int num_classes, int *encoded_label) 
{
    for (int i = 0; i < num_classes; i++) {
        encoded_label[i] = (i == class_id - 1) ? 1.0 : 0.0;
    }
}
