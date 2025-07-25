/*
 * cfar_main_gpu.c
 *
 *  Created on: 2025年7月14日
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

/*
 * 初始化OPENCL环境
 * */
void initOpenCL()
{
    // 获取平台
    err = clGetPlatformIDs(1, &platform, NULL);

    // 获取设备
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err == CL_DEVICE_NOT_FOUND) {
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
    }

    // 创建上下文
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);

    // 创建命令队列
    queue = clCreateCommandQueue(context, device, 0, &err);

    // 获取设备信息
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

    // 创建内核
    kernel = clCreateKernel(program, "cfar_ca", &err);
}

/*
 * 清理环境，释放内存空间
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
  参数说明：
  float *x 输入序列
  unsigned int N 序列长度
  unsigned int  guard_len 单侧保护单元长度
  unsigned int ref_len 单侧参考单元长度
  float pfa 虚警率
  unsigned int *dec 检测结果序列
  float *threshold 检测阈值序列
 */
void CA_CFAR(float *x,
			 unsigned int N,
	         unsigned int  guard_len,
	         unsigned int ref_len,
			 float pfa,
			 unsigned int *dec, 
			 float *threshold)
{
	//创建程序和内核
	const size_t lws = 128;
	const size_t gws = ((N + lws - 1) / lws) * lws;
	// 创建缓冲区
	cl_mem d_pwr = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * N, NULL, &err);
	cl_mem d_dec = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(unsigned int) * N, NULL, &err);
	cl_mem d_threshold = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * N, NULL, &err);

	clEnqueueWriteBuffer(queue, d_pwr, CL_TRUE, 0, sizeof(float) * N, x, 0, NULL, NULL);

	// 设置Kernel参数
	float K = sqrt(-log(pfa)); // 计算调节因子

	clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_pwr);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_dec);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_threshold);
	clSetKernelArg(kernel, 3, sizeof(float), &K);
	clSetKernelArg(kernel, 4, sizeof(unsigned int), &guard_len);
	clSetKernelArg(kernel, 5, sizeof(unsigned int), &ref_len);
	clSetKernelArg(kernel, 6, sizeof(unsigned int), &N);

	// 调度NDRange执行计算
	clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &gws, &lws, 0, NULL, NULL);

	// 读取结果
	clEnqueueReadBuffer(queue, d_dec, CL_TRUE, 0, sizeof(unsigned int) * N, dec, 0, NULL, NULL);
	clEnqueueReadBuffer(queue, d_threshold, CL_TRUE, 0, sizeof(float) * N, threshold, 0, NULL, NULL);

	// 清理内存
	clReleaseMemObject(d_pwr);
	clReleaseMemObject(d_dec);
	clReleaseMemObject(d_threshold);
}

