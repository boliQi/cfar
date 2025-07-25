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
#define TIMES 	10   // ����ִ�д���
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
    double timeMax = 0.0;
    double timeMin = 0.0;
    double timeAvg = 0.0;

    for(int i = 0; i < TIMES; i++)
    {
        gettimeofday(&tv1, NULL);
        //  ����
        CA_CFAR(inputdata, N, guard, train, pfa, dec, threshold);
        gettimeofday(&tv2, NULL);
        timeSpend[i] = time_used(tv1, tv2);
    }


    timeMax = time_max(timeSpend, TIMES);
    timeMin = time_min(timeSpend, TIMES);
    timeAvg = time_avg(timeSpend, TIMES);
    printf("Times=%d, timeMax=%fus, timeMin=%fus, timeAve=%fus\n", TIMES, timeMax, timeMin, timeAvg);

    float delta = 0.0;
    delta = (timeMax - timeMin) / timeMin * 100;  //��λ��%
    printf("\n===========================CFAR RESULT==========================================\n");
    printf("CFAR algorithm stability delta = %f%%, performance average time = %fus\n", delta, timeAvg);

	free(inputdata);
	free(dec);
	free(threshold);
	//��������
	cleanup();
}
