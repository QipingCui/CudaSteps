# include <iostream>

using namespace std;


__global__ void hello_from_gpu()
{
    printf("Hello World from the GPU!\n");
}


int main()
{
    hello_from_gpu<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}