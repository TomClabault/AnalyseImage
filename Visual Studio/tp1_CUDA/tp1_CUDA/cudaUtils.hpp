#include <cuda_runtime.h>

#define NB_BLOCKS 128
#define NB_THREADS 256

#define HANDLE_CUDA_ERROR(errorCode) \
if(errorCode == cudaErrorMemoryAllocation) \
{\
	std::cout << "Not enough memory on the GPU." << std::endl;\
}

typedef struct {
	unsigned char* imgData;

	unsigned int* rows;
	unsigned int* cols;
} GPU_img_data;
