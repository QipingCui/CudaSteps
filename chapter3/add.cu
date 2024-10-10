#include <math.h>
//#include <stdlib.h>
#include <stdio.h>


const float EPSILON = 1.0e-15;
const float a = 1.23;
const float b = 2.34;
const float c = 3.57;


// 核函数。
__global__ void add(const float *x, const float *y, float *z, const int N);

// 重载设备函数。
__device__ float add_in_device(const float x, const float y);
__device__ void add_in_device(const float x, const float y, float &z);

// 主机函数。
void check(const float *z, const int N);


int main()
{
    const int N = 1e8;
    const int M = sizeof(float) * N;

    // 申请主机内存。
    // float *h_x = (float*) malloc(M);
    // float *h_y = (float*) malloc(M);
    // float *h_z = (float*) malloc(M);
    // 支持使用 new-delete 方式创建和释放内存。
    float *h_x = new float[N];
    float *h_y = new float[N];
    float *h_z = new float[N];

    // 初始化主机数据。
    for (int i = 0; i < N; ++i)
    {
        h_x[i] = a;
        h_y[i] = b;
    }

    // 申请设备内存。
    // cudeError_t cudaMalloc(void **address, size_t size);
    float *d_x, *d_y, *d_z;
    cudaMalloc((void**)&d_x, M);
    cudaMalloc((void**)&d_y, M);
    cudaMalloc((void**)&d_z, M);

    // 从主机复制数据到设备。
    // cudaError_t cudaMemcpy(void *dst, void *src, size_t count, enum cudaMemcpyKind kind);
    // kind 可以简化使用 `cudaMemcpyDefault`，由系统自动判断拷贝方向（x64主机）。
    cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice);

    // 在设备中执行计算。
    const int block_size = 128;
    // const int grid_size = (N % block_size == 0) ? (N / block_size) : (N / block_size + 1);
    const int grid_size = (N - 1) / block_size + 1; // 线程数应该不少于计算数目。
    add<<<grid_size, block_size>>>(d_x, d_y, d_z, N);

    // 从设备复制数据到主机。
    cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost);
    check(h_z, N);

    // 释放主机内存。
    // free(h_x);
    // free(h_y);
    // free(h_z);
    if (h_x) delete[] h_x;
    if (h_y) delete[] h_y;
    if (h_z) delete[] h_z;

    // 释放设备内存。
    // cudaError_t cudaFree(void *address);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);

    return 0;
}

// __global__ void add(const float *x, const float *y, float *z, const int N)
// {
//     const int n = blockDim.x * blockIdx.x + threadIdx.x;
//     if (n >= N) return;
//     z[n] = x[n] + y[n];
// }

__global__ void add(const float *x, const float *y, float *z, const int N)
{
    // 在主机函数中需要依次对每个元素进行操作，需要使用一个循环。
    // 在设备函数中，因为采用“单指令-多线程”方式，所以可以去掉循环、只要将数组元素索引和线程索引一一对应即可。

    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n > N) return;

    if (n%5 == 0)
    {
        z[n] = add_in_device(x[n], y[n]);
    }
    else
    {
        add_in_device(x[n], y[n], z[n]);
    }
}

__device__ float add_in_device(const float x, const float y)
{
    return x + y;
}

__device__ void add_in_device(const float x, const float y, float &z)
{
    z = x + y;
}

void check(const float *z, const int N)
{
    bool has_error = false;
    for (int i = 0; i < N ;++i)
    {
        if (fabs(z[i] - c) > EPSILON)
        {
            //printf("%d, %f, %f\n", i, z[i], c);
            has_error = true;
        }
    }

    printf("cuda; %s\n", has_error ? "has error" : "no error");
}
