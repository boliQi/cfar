/*
 * cfar.h
 *
 *  Created on: 2025年6月24日
 *      Author: wuying
 */

#ifndef INCLUDE_CFAR_H_
#define INCLUDE_CFAR_H_
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

/****************************************************************
 * 函数名：CA_CFAR
 * 函数功能：CFAR算法实现
 * 		  采用单元平均法进行目标检测，在待检测单元两侧各取N个参考单元，
 * 		  计算总共2N个参考单元的平均值，将其当作背景杂波功率估计值，
 * 		 再将该估计值乘以门限乘积因子k，得到待检测单元的门限值。
 * 		 再将待检测单元的信号功率与门限值进行比较，若待检测单元的信号
 * 		 功率超过门限值，则认为该单元存在目标，否则认为不存在目标。
 *
 * 输入参数：
 * 		 x: 待处理序列
 * 		 N: 待处理序列长度
 * 		 ref_len:  参考窗半长
 * 		 guard_len: 保护窗半长
 * 		 pfa: 目标虚警率
 * 输出参数：
 * 		 dec:  检测结果序列，取值为：0表示未检测到目标；1表示检测到目标。
 * 		 threshold: CA-CFAR检测阈值
 * 返回值：无
 * 其他说明：无
 * **************************************************************/
void CA_CFAR(float *x, unsigned int N, unsigned int ref_len,  unsigned int  guard_len, float pfa, unsigned int *dec, float *threshold);

/****************************************************************
 * 函数名：vsip_vmean_f
 * 函数功能：实向量求平均数
 * 输入参数：
 * 			a:输入向量1，；
 *          n:向量长度，；
 * 输出参数：
 *
 * 返回值：   实向量平均数
 * 其他说明：无
 * **************************************************************/
float (vsip_vmean_f)(float *a, int n);

//统计时间
double time_used(struct timeval tvStart, struct timeval tvEnd);
double time_max(double *t, int n);
double time_min(double *t, int n);
double time_avg(double *t, int n);

/*
 * 初始化OPENCL环境
 *
 * 函数功能：
 * 完成获取显卡设备、创建上下文、命令队列、加载核函数，创建内核、clblas库初始化、clfft初始化等操作。
 * */
void initOpenCL();

/*
 * 清理环境，释放内存空间
 * */
void cleanup();

/****************************************************************
 * 函数名：compare_data
 * 函数功能：比较数据
 *
 * 输入参数：
 * 		 realdataString: 标准数据文件名
 * 		 actuldataString: 实际数据文件名
 * 		 size：比较的数据大小
 * 		 delta：精度要求
 * 输出参数：
 * 		 无
 * 返回值：
 * 		-1表示数据校验失败，0表示校验成功。
 * 其他说明：无
 * **************************************************************/
int compare_data(const char* realdataString, const char* actuldataString, int size, float delta);

int compare_data_int(const char* realdataString, const char* actuldataString, int size);

#endif /* INCLUDE_CFAR_H_ */
