#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cfar.h>

#define EPSILON 1e-8f

int compare_data(const char* realdataString, const char* actuldataString, int size, float delta)
{
	int ret = 0;
	int i = 0;
	float *realdata, *actuldata;
	FILE *pFile_real, *pFile_actul;

	pFile_real = fopen(realdataString, "rb+");		//读取的标准数据，由matlab生成
	pFile_actul = fopen(actuldataString, "rb+");	// 读取实际计算数据，由算法生成
	if (!pFile_real || !pFile_actul)
	{
		printf("fopen error!\n");
		return -1;
	}

	//获取文件数组大小
	fseek(pFile_actul, 0, SEEK_END);
	int file_size_actual = ftell(pFile_actul) / sizeof(float);

	if (file_size_actual != size)
	{
		printf("dataSize is not right!\n");
		ret = -1;
		return ret;
	}

	realdata  = (float *)malloc(sizeof(float) * size);
	actuldata = (float *)malloc(sizeof(float) * size);

	rewind(pFile_real);	//回到文件首位
	rewind(pFile_actul);	//回到文件首位
	fread(realdata, sizeof(float), size, pFile_real);
	fread(actuldata, sizeof(float), size, pFile_actul);

	float temp = 0.0f;
	for (i = 0; i < size; i++)
	{
		temp = fabs(actuldata[i] - realdata[i])/(realdata[i]+EPSILON);
		if (temp > delta)
		{
			printf("data error!!\n");
			ret = -1;
			break;
		}
	}

	fclose(pFile_real);
	fclose(pFile_actul);
	free(realdata);
	free(actuldata);

	printf("compare finished!\n");
	return ret;
}

int compare_data_int(const char* realdataString, const char* actuldataString, int size)
{
	int ret = 0;
	int i = 0;
	unsigned int *realdata, *actuldata;
	FILE *pFile_real, *pFile_actul;

	pFile_real = fopen(realdataString, "rb+");		//读取的标准数据，由matlab生成
	pFile_actul = fopen(actuldataString, "rb+");	// 读取实际计算数据，由算法生成
	if (!pFile_real || !pFile_actul)
	{
		printf("fopen error!\n");
		return -1;
	}

	//获取文件数组大小
	fseek(pFile_actul, 0, SEEK_END);
	int file_size_actual = ftell(pFile_actul) / sizeof(unsigned int);

	if (file_size_actual != size)
	{
		printf("dataSize is not right!\n");
		ret = -1;
		return ret;
	}

	realdata  = (unsigned int *)malloc(sizeof(unsigned int) * size);
	actuldata = (unsigned int *)malloc(sizeof(unsigned int) * size);

	rewind(pFile_real);	//回到文件首位
	rewind(pFile_actul);	//回到文件首位
	fread(realdata, sizeof(unsigned int), size, pFile_real);
	fread(actuldata, sizeof(unsigned int), size, pFile_actul);

	for (i = 0; i < size; i++)
	{
		if (actuldata[i] != realdata[i])
		{
			printf("data error!!\n");
			ret = -1;
			break;
		}
	}

	fclose(pFile_real);
	fclose(pFile_actul);
	free(realdata);
	free(actuldata);

	printf("compare finished!\n");
	return ret;
}
