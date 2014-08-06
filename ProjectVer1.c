__global__ void matrix_mult(float* C, float* A, float* B, int n)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float sum = 0.0f;
	for (int k = 0; k < n; k++) {
		sum += A[row*n+k] * B[k * n + col];
	}
	C[row*n+col] = sum;
}