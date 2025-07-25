/*
 * cfar.h
 *
 *  Created on: 2025��6��24��
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
 * ��������CA_CFAR
 * �������ܣ�CFAR�㷨ʵ��
 * 		  ���õ�Ԫƽ��������Ŀ���⣬�ڴ���ⵥԪ�����ȡN���ο���Ԫ��
 * 		  �����ܹ�2N���ο���Ԫ��ƽ��ֵ�����䵱�������Ӳ����ʹ���ֵ��
 * 		 �ٽ��ù���ֵ�������޳˻�����k���õ�����ⵥԪ������ֵ��
 * 		 �ٽ�����ⵥԪ���źŹ���������ֵ���бȽϣ�������ⵥԪ���ź�
 * 		 ���ʳ�������ֵ������Ϊ�õ�Ԫ����Ŀ�꣬������Ϊ������Ŀ�ꡣ
 *
 * ���������
 * 		 x: ����������
 * 		 N: ���������г���
 * 		 ref_len:  �ο����볤
 * 		 guard_len: �������볤
 * 		 pfa: Ŀ���龯��
 * ���������
 * 		 dec:  ��������У�ȡֵΪ��0��ʾδ��⵽Ŀ�ꣻ1��ʾ��⵽Ŀ�ꡣ
 * 		 threshold: CA-CFAR�����ֵ
 * ����ֵ����
 * ����˵������
 * **************************************************************/
void CA_CFAR(float *x, unsigned int N, unsigned int ref_len,  unsigned int  guard_len, float pfa, unsigned int *dec, float *threshold);

/****************************************************************
 * ��������vsip_vmean_f
 * �������ܣ�ʵ������ƽ����
 * ���������
 * 			a:��������1����
 *          n:�������ȣ���
 * ���������
 *
 * ����ֵ��   ʵ����ƽ����
 * ����˵������
 * **************************************************************/
float (vsip_vmean_f)(float *a, int n);

//ͳ��ʱ��
double time_used(struct timeval tvStart, struct timeval tvEnd);
double time_max(double *t, int n);
double time_min(double *t, int n);
double time_avg(double *t, int n);

/*
 * ��ʼ��OPENCL����
 *
 * �������ܣ�
 * ��ɻ�ȡ�Կ��豸�����������ġ�������С����غ˺����������ںˡ�clblas���ʼ����clfft��ʼ���Ȳ�����
 * */
void initOpenCL();

/*
 * ���������ͷ��ڴ�ռ�
 * */
void cleanup();

/****************************************************************
 * ��������compare_data
 * �������ܣ��Ƚ�����
 *
 * ���������
 * 		 realdataString: ��׼�����ļ���
 * 		 actuldataString: ʵ�������ļ���
 * 		 size���Ƚϵ����ݴ�С
 * 		 delta������Ҫ��
 * ���������
 * 		 ��
 * ����ֵ��
 * 		-1��ʾ����У��ʧ�ܣ�0��ʾУ��ɹ���
 * ����˵������
 * **************************************************************/
int compare_data(const char* realdataString, const char* actuldataString, int size, float delta);

int compare_data_int(const char* realdataString, const char* actuldataString, int size);

#endif /* INCLUDE_CFAR_H_ */
