/*
 * cfar_main_gpu.c
 *
 *  Created on: 2025��7��14��
 *      Author: zhangtianyi
 */
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <float.h>
#include <cfar.h>

cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue;
cl_int err;
cl_program program;
cl_kernel kernel;
#define PRINT_DEVICE_INFO 1

/*
 * ��ʼ��OPENCL����
 * */
void initOpenCL()
{
    // ��ȡƽ̨
    err = clGetPlatformIDs(1, &platform, NULL);

    // ��ȡ�豸
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err == CL_DEVICE_NOT_FOUND) {
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
    }

    #if PRINT_DEVICE_INFO
	//查询设备详细信息
	size_t cb;
	clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &cb);
	char* devname = (char *)malloc(cb);
	clGetDeviceInfo(device, CL_DEVICE_NAME, cb, devname, 0);
	printf("Device:  %s\n", devname);
	free(devname);
	int nsize[2] = { 0,0 };
	clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, 0, NULL, &cb);
	printf("CL_DEVICE_LOCAL_MEM_SIZE num = %d\n", cb / sizeof(int));
	clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, 2 * sizeof(int), nsize, 0);
	printf("CL_DEVICE_LOCAL_MEM_SIZE nsize = {%d, %d}\n", nsize[0], nsize[1]);
	cl_ulong local_mem_size = 0;
	clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &local_mem_size, NULL);
	printf("CL_DEVICE_LOCAL_MEM_SIZE = %llu bytes\n", local_mem_size);

	clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, 0, NULL, &cb);
	printf("CL_DEVICE_GLOBAL_MEM_SIZE num = %d\n", cb / sizeof(int));
	clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, 2 * sizeof(int), nsize, 0);
	printf("CL_DEVICE_GLOBAL_MEM_SIZE nsize = {%d, %d}\n", nsize[0], nsize[1]);

	clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, 0, NULL, &cb);
	printf("CL_DEVICE_MAX_MEM_ALLOC_SIZE num = %d\n", cb / sizeof(int));
	clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, 2 * sizeof(int), nsize, 0);
	printf("CL_DEVICE_MAX_MEM_ALLOC_SIZE nsize = {%d, %d}\n", nsize[0], nsize[1]);

	clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, 0, NULL, &cb);
	printf("CL_DEVICE_MAX_COMPUTE_UNITS num = %d\n", cb / sizeof(int));
	clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, 2 * sizeof(int), nsize, 0);
	printf("CL_DEVICE_MAX_COMPUTE_UNITS nsize = {%d, %d}\n", nsize[0], nsize[1]);

	clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, 0, NULL, &cb);
	printf("CL_DEVICE_MAX_WORK_GROUP_SIZE num = %d\n", cb / sizeof(int));
	clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, 2 * sizeof(int), nsize, 0);
	printf("CL_DEVICE_MAX_WORK_GROUP_SIZE nsize = {%d, %d}\n", nsize[0], nsize[1]);

	clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, 0, NULL, &cb);
	printf("CL_DEVICE_MAX_WORK_ITEM_SIZES num = %d\n", cb / sizeof(int));
	clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, 3 * sizeof(int), nsize, 0);
	printf("CL_DEVICE_MAX_WORK_ITEM_SIZES nsize = {%d, %d, %d}\n", nsize[0], nsize[1], nsize[2]);
	printf("\n");
#endif

    // ����������
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);

    // �����������
    queue = clCreateCommandQueue(context, device, 0, &err);

    // ��ȡ�豸��Ϣ
    char device_name[128];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);

    size_t sizes;
    FILE *file = fopen("../src/cfar/cfar.cl", "r");

    fseek(file, 0, SEEK_END);
    sizes = ftell(file);
    rewind(file);
    char * filestr = (char*)malloc(sizes + 1);
    filestr[sizes] = '\0';
    sizes = fread(filestr, sizeof(char), sizes, file);
    fclose(file);
    program = clCreateProgramWithSource(context, 1, (const char**)&filestr, NULL, NULL);
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t log_size;
        char *program_log;
        clGetProgramBuildInfo(program, &device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        program_log = (char*)malloc(log_size + 1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(program, &device, CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL);
        printf("%s\n", program_log);
        free(program_log);
        return -1;
    }
    free(filestr);

    // �����ں�
    kernel = clCreateKernel(program, "cfar_ca", &err);
}

/*
 * �����������ͷ��ڴ�ռ�
 * */
void cleanup()
{
    clReleaseKernel(kernel);

    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
	clReleaseDevice(device);
}

/*
  CA_CFAR.c
  ����˵����
  float *x ��������
  unsigned int N ���г���
  unsigned int  guard_len ���ౣ����Ԫ����
  unsigned int ref_len ����ο���Ԫ����
  float pfa �龯��
  unsigned int *dec ���������
  float *threshold �����ֵ����
 */
void CA_CFAR(float *x,
			 unsigned int N,
	         unsigned int  guard_len,
	         unsigned int ref_len,
			 float pfa,
			 unsigned int *dec, 
			 float *threshold)
{
	//����������ں�
	const size_t lws = 128;
	const size_t gws = ((N + lws - 1) / lws) * lws;
	// ����������
	cl_mem d_pwr = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * N, NULL, &err);
	cl_mem d_dec = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(unsigned int) * N, NULL, &err);
	cl_mem d_threshold = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * N, NULL, &err);

	clEnqueueWriteBuffer(queue, d_pwr, CL_TRUE, 0, sizeof(float) * N, x, 0, NULL, NULL);

	// ����Kernel����
	float K = sqrt(-log(pfa)); // �����������

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_pwr);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_dec);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_threshold);
	clSetKernelArg(kernel, 3, sizeof(float), &K);
	clSetKernelArg(kernel, 4, sizeof(unsigned int), &guard_len);
	clSetKernelArg(kernel, 5, sizeof(unsigned int), &ref_len);
	clSetKernelArg(kernel, 6, sizeof(unsigned int), &N);

	// ����NDRangeִ�м���
	clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &gws, &lws, 0, NULL, NULL);

	// ��ȡ���
	clEnqueueReadBuffer(queue, d_dec, CL_TRUE, 0, sizeof(unsigned int) * N, dec, 0, NULL, NULL);
	clEnqueueReadBuffer(queue, d_threshold, CL_TRUE, 0, sizeof(float) * N, threshold, 0, NULL, NULL);

	// �����ڴ�
	clReleaseMemObject(d_pwr);
	clReleaseMemObject(d_dec);
	clReleaseMemObject(d_threshold);
}

