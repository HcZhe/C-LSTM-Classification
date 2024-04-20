#include "lstmlib.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define NUM_CLASSES 3  //类别个数
#define NUM_SAMPLES_PER_CLASS 100   //每个类别的训练样本个数
#define DATA_LENGTH 400  //时间序列数据长度
#define NUM_EPOCHS 50  //训练次数
#define IF_TRAIN 0 //IF_TRAIN=1 训练模型；IF_TRAIN=0 测试模型
#define NUM_SAMPLES_PER_CLASS_TEST 20//每个类别的预测样本个数

typedef struct {
    double data[DATA_LENGTH];
    int label;
} Sample;

void shuffle_samples(Sample *samples, int n) {
    if (samples == NULL || n <= 0) return;

    for (int i = 0; i < n - 1; i++) {
        int j = i + rand() / (RAND_MAX / (n - i) + 1);
        Sample temp = samples[i];
        samples[i] = samples[j];
        samples[j] = temp;
    }
}

int main(int argc, char *argv[])
{
    struct lstmlib *unit;
    /* length是输入和输出序列点的个数，cyc是函数在序列上重复在多少个周期 */
    int i, length = DATA_LENGTH;

    srand((unsigned int)time(NULL));  // 初始化随机种子 以便加载参数
    unit = lstmlib_create(length);
    
    char *file_train = "your/path/Train/train.txt";
    char *file_loss = "your/path/Train/loss_data.csv";
    FILE *loss_file = NULL;
    double data[DATA_LENGTH]; // 存储加载的数据
    int encoded_label[NUM_CLASSES];

    // 准备好所有的samples
    int total_samples = NUM_CLASSES * NUM_SAMPLES_PER_CLASS;
    Sample *samples = (Sample *)malloc(total_samples * sizeof(Sample));

    if(IF_TRAIN){
    // 将数据加载到samples
    for (int class_id = 1; class_id <= NUM_CLASSES; class_id++) {
        for (int sample_id = 1; sample_id <= NUM_SAMPLES_PER_CLASS; sample_id++) {
            char file_name[100];
            snprintf(file_name, sizeof(file_name), "your/path/Train/class_%d/%d.csv", class_id, sample_id);
            
            if (load_csv_data(file_name, samples[(class_id - 1) * NUM_SAMPLES_PER_CLASS + sample_id - 1].data, DATA_LENGTH) != 1) {
                fprintf(stderr, "Error loading data from %s\n", file_name);
                free(samples);
                return 1;
            }

            samples[(class_id - 1) * NUM_SAMPLES_PER_CLASS + sample_id - 1].label = class_id;
        }
    }

    loss_file = fopen("your/path/Train/loss_data.csv", "w");
    if (loss_file == NULL) {
        fprintf(stderr, "Error opening file for writing loss data.\n");
        return 1;
    }
    fprintf(loss_file, "Epoch,Sample,Loss\n");  // 写入表头
    // 训练过程
    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        printf("epoch %d\n", epoch);
      
        shuffle_samples(samples, total_samples);
        srand((unsigned int)time(NULL));  // 更新随机种子
        for (int i = 0; i < total_samples; i++) {
            int encoded_label[NUM_CLASSES];
            shuffle_samples(samples, total_samples);
            one_hot_encode(samples[i].label, NUM_CLASSES, encoded_label);
            //设置LSTM单元输入时间序列数据
            for (int j = 0; j < length; j++) {
                // unit->x[j] = samples[i].data[j]+ 0.4 * (rand() / (double)RAND_MAX) - 0.2;//加入随机噪声进行训练
                unit->x[j] = samples[i].data[j];
            }

            lstmlib_run_unit(unit, encoded_label);
            lstmlib_fit_unit(unit, 0.0005, encoded_label);
            fprintf(loss_file, "%d,%d,%.5f\n", epoch, i, unit->Loss);//将交叉熵损失值写入文件

            // printf("sample:%d, label:%d, Loss:%.5f\n", i, samples[i].label, unit->Loss);
        }

    }
    if (loss_file != NULL) {
        fclose(loss_file);
    }
    printf("训练完成\n");
    lstmlib_save(unit,file_train);
    // 释放内存
    free(samples);

    }
    //验证测试
    else {
    lstmlib_load(unit, file_train); // 加载训练模型参数
    int correct_predictions = 0;
    int total_predictions = 0;
    double max_prob;
    int predicted_class;

    for (int class_id = 1; class_id <= NUM_CLASSES; class_id++) {
        for (int sample_id = 1; sample_id <= NUM_SAMPLES_PER_CLASS_TEST; sample_id++) {
            char file_name[100];
            snprintf(file_name, sizeof(file_name), "your/path/Test/class_%d/%d.csv", class_id, sample_id);
            if (load_csv_data(file_name, data, DATA_LENGTH) != 1) {
                fprintf(stderr, "Error loading data from %s\n", file_name);
                continue;
            }

            // 这里可能需要真实的one-hot编码标签来与预测进行比较
            int real_label[NUM_CLASSES];
            one_hot_encode(class_id, NUM_CLASSES, real_label);

            for (int i = 0; i < length; i++) {
                unit->x[i] = data[i]; //加载时间序列数据，即将数据data赋值给lstm单元的输入“x”
            }

            lstmlib_run_unit(unit, real_label);  // 这里执行前向传播得到预测结果

            // 找出概率最大的类别作为预测类别
            max_prob = unit->softmax_output[0];
            predicted_class = 1;
            for (int i = 0; i < NUM_CLASSES; i++) {
                if (unit->softmax_output[i] > max_prob) {
                    max_prob = unit->softmax_output[i];
                    predicted_class = i + 1;
                }
            }

            if (predicted_class == class_id) {
                correct_predictions++;
                printf("正确：第%d次预测: softmax: %.3f\n预测类别: %d\n", (class_id - 1)*20 + sample_id, unit->softmax_output[predicted_class-1],predicted_class);
            }
            else{
                printf("【错误】第%d次预测: softmax: %.3f\n预测类别: %d\n", (class_id - 1)*20 + sample_id, unit->softmax_output[predicted_class-1],predicted_class);
            }
            total_predictions++;
        }
    }
    double accuracy = (double)correct_predictions / total_predictions;
    printf("总测试数：%d,模型准确率: %.2f%%\n", total_predictions, accuracy * 100);
        }
    return 0;
}
