/*
 * cfar_main.c
 *
 *  Created on: 2025年6月24日
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

#define MAX_SIZE 1024*1024	//最大输入数据长度
#define TIMES 	1000   // 算子执行次数
unsigned int N; 		//信号长度,通过读取文件确定数组长度

void readData(float* data, const char* input_filename)
{
    FILE* fp = fopen(input_filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error opening file: %s\n", input_filename);
        exit(EXIT_FAILURE);
    }

	//获取文件数组大小
	fseek(fp, 0 , SEEK_END);
	N = ftell(fp) / sizeof(float);

	rewind(fp);	//回到文件首位
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
	unsigned int train = 32; 	//参考窗半长
	unsigned int guard = 4; 	    //保护窗半长
	float pfa = 0.0001; 	//目标虚警率
	float *inputdata;	    //输入数据
	unsigned int *dec;      //输出判断序列
	float *threshold;	    //输出阈值序列
	const size_t lws = 128; //每个工作组的本地工作项数量

	inputdata = (float *)malloc(MAX_SIZE * sizeof(float));	//为输入数据分配空间
	//读取输入数据
	readData(inputdata, "../data/cfar_data.bin");

	//分配空间
	dec = (unsigned int *)malloc(N * sizeof(unsigned int));
	threshold = (float *)malloc(N * sizeof(float));

	//初始化OPENCL环境
	initOpenCL();

    //1、正确性校验
    CA_CFAR(inputdata, N, guard, train, pfa, dec, threshold);

	//将结果写入文件中
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

    //2、统计性能及稳定性
    struct timeval tv1, tv2;
    double timeSpend[TIMES];
    double timeMax = 0.0;
    double timeMin = 0.0;
    double timeAvg = 0.0;

    for(int i = 0; i < TIMES; i++)
    {
        gettimeofday(&tv1, NULL);
        //  运算
        CA_CFAR(inputdata, N, guard, train, pfa, dec, threshold);
        gettimeofday(&tv2, NULL);
        timeSpend[i] = time_used(tv1, tv2);
    }


    timeMax = time_max(timeSpend, TIMES);
    timeMin = time_min(timeSpend, TIMES);
    timeAvg = time_avg(timeSpend, TIMES);
    printf("Times=%d, timeMax=%fus, timeMin=%fus, timeAve=%fus\n", TIMES, timeMax, timeMin, timeAvg);

    float delta = 0.0;
    delta = (timeMax - timeMin) / timeMin * 100;  //单位：%
    printf("\n===========================CFAR RESULT==========================================\n");
    printf("CFAR algorithm stability delta = %f%%, performance average time = %fus\n", delta, timeAvg);

	free(inputdata);
	free(dec);
	free(threshold);
	//清理环境
	cleanup();
}
