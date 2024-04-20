#ifndef LSTMLIB
#define LSTMLIB

#include <stdlib.h>
#include <math.h>
#include <stdio.h>

struct lstmlib
{
    int length; // 网络的长度（时间步长）
    double *x; // 输入序列的指针
    double *h; // 隐藏状态序列的指针
    double *f; // 遗忘门参数的指针
    double *i; // 输入门参数的指针
    double *tilde_C; // 候选记忆细胞的指针  = ~C_t
    double *C; // 记忆细胞的指针  = C_t
    double *o; // 输出门参数的指针
    double *hat_h; // 目标隐藏状态的指针
    double W_fh; // 遗忘门到隐藏状态的权重
    double W_fx; // 输入到遗忘门的权重
    double b_f; // 遗忘门的偏置
    double W_ih; // 隐藏状态到输入门的权重
    double W_ix; // 输入到输入门的权重
    double b_i; // 输入门的偏置
    double W_Ch; // 隐藏状态到记忆细胞的权重
    double W_Cx; // 输入到记忆细胞的权重
    double b_C; // 记忆细胞的偏置
    double W_oh; // 隐藏状态到输出门的权重
    double W_ox; // 输入到输出门的权重
    double b_o; // 输出门的偏置
    int error_no; // 错误编号
    char *error_msg; // 错误信息
    double *W_hy; // 隐藏状态到输出层的权重
    double *b_y; // 输出层的偏置
    double *softmax_output; // 经softmax处理后的输出层结果，即类别的概率值
    int n_classes; // 类别数
    double Loss; // 交叉熵损失值
};

// 创建LSTM网络实例
struct lstmlib* lstmlib_create(int length);
// 随机初始化LSTM网络的参数
char lstmlib_random_params(struct lstmlib *unit, double min, double max);
// 执行LSTM单元的前向传播过程
char lstmlib_run_unit(struct lstmlib *unit, int *target_labels);
// 执行LSTM单元的后向传播过程，并更新网络参数
char lstmlib_fit_unit(struct lstmlib *unit, double lr, int *target_labels);
// 将LSTM网络的参数保存到文件中
int lstmlib_save(struct lstmlib *unit, char *file_name);
//加载训练模型参数
struct lstmlib* lstmlib_load(struct lstmlib *unit, char *file_name);
//加载csv训练数据
int load_csv_data(const char *file_name, double *data, int length);
//one-hot对标签进行编码
void one_hot_encode(int class_id, int num_classes, int *encoded_label);


// 结束防止重复包含的宏定义
#endif

