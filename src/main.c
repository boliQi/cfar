/*
 * cfar_main.c
 *
 *  Created on: 2025��6��24��
 *      Author: wuying
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <cfar.h>

#define MAX_SIZE 1024*1024	//����������ݳ���
#define TIMES 	1000   // ����ִ�д���
unsigned int N; 		//�źų���,ͨ����ȡ�ļ�ȷ�����鳤��

void readData(float* data, const char* input_filename)
{
    FILE* fp = fopen(input_filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error opening file: %s\n", input_filename);
        exit(EXIT_FAILURE);
    }

    //��ȡ�ļ������С
    fseek(fp, 0 , SEEK_END);
    N = ftell(fp) / sizeof(float);

    rewind(fp);	//�ص��ļ���λ
    fread(data, sizeof(float), N, fp);
    fclose(fp);
}

void writeData(float* data, const char* input_filename, int data_size)
{
    FILE* fp = fopen(input_filename, "wb+");
    if (!fp) {
        fprintf(stderr, "Error opening file: %s\n", input_filename);
        exit(EXIT_FAILURE);
    }

    fwrite(data, sizeof(float), data_size, fp);
    fclose(fp);
}

int main(int argc, char** argv)
{
    unsigned int train = 32; 	//�ο����볤
    unsigned int guard = 4; 	    //�������볤
    float pfa = 0.0001; 	//Ŀ���龯��
    float *inputdata;	    //��������
    unsigned int *dec;      //����ж�����
    float *threshold;	    //�����ֵ����
    const size_t lws = 128; //ÿ��������ı��ع���������

    inputdata = (float *)malloc(MAX_SIZE * sizeof(float));	//Ϊ�������ݷ���ռ�
    //��ȡ��������
    readData(inputdata, "../data/cfar_data.bin");

    //����ռ�
    dec = (unsigned int *)malloc(N * sizeof(unsigned int));
    threshold = (float *)malloc(N * sizeof(float));

    //��ʼ��OPENCL����
    initOpenCL();

    //1����ȷ��У��
    CA_CFAR(inputdata, N, guard, train, pfa, dec, threshold);

    //�����д���ļ���
    writeData(dec, "../data/dec_actual.bin", N);
    writeData(threshold, "../data/threshold_actual.bin", N);

    int ret = compare_data_int("../data/standard_data/dec.bin", "../data/dec_actual.bin", N);
    if(ret != 0)
    {
        printf("CFAR data valid fail!!\n");
        return;
    }

    ret = compare_data("../data/standard_data/threshold.bin", "../data/threshold_actual.bin", N, 1e-6);
    if(ret == 0)
    {
        printf("CFAR data valid success!!\n");
    }
    else
    {
        printf("CFAR data valid fail!!\n");
        return;
    }

    //2��ͳ�����ܼ��ȶ���
    struct timeval tv1, tv2;
    double timeSpend[TIMES];
    double batchTimeMax[4] = {0.0};
    double batchTimeMin[4] = {0.0};
    double batchTimeAvg[4] = {0.0};
    int batch = TIMES / 4;
    for(int b = 0; b < 4; b++) {
        for(int i = 0; i < batch; i++) {
            int idx = b * batch + i;
            gettimeofday(&tv1, NULL);
            CA_CFAR(inputdata, N, guard, train, pfa, dec, threshold);
            gettimeofday(&tv2, NULL);
            timeSpend[idx] = time_used(tv1, tv2);
        }
        batchTimeMax[b] = time_max(&timeSpend[b * batch], batch);
        batchTimeMin[b] = time_min(&timeSpend[b * batch], batch);
        batchTimeAvg[b] = time_avg(&timeSpend[b * batch], batch);
        printf("Batch %d: Max=%fus, Min=%fus, Avg=%fus\n", b+1, batchTimeMax[b], batchTimeMin[b], batchTimeAvg[b]);
    }

    double timeMax = time_max(timeSpend, TIMES);
    double timeMin = time_min(timeSpend, TIMES);
    double timeAvg = time_avg(timeSpend, TIMES);
    printf("Times=%d, timeMax=%fus, timeMin=%fus, timeAve=%fus\n", TIMES, timeMax, timeMin, timeAvg);

    float delta = 0.0;
    delta = (timeMax - timeMin) / timeMin * 100;
    printf("\n===========================CFAR RESULT==========================================\n");
    printf("CFAR algorithm stability delta = %f%%, performance average time = %fus\n", delta, timeAvg);

    free(inputdata);
    free(dec);
    free(threshold);
    //��������
    cleanup();
}
