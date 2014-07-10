/* Author: Christopher Mitchell <chrism@lclark.edu>
 * Date: 2011-07-15
 *
 * Compile with `gcc gol.c`.
 */

#include <stdlib.h> // for rand
#include <string.h> // for memcpy
#include <stdio.h> // for printf
#include <time.h> // for nanosleep

#define WIDTH 60
#define HEIGHT 30

// The two boards 
int current[WIDTH * HEIGHT];
int next[WIDTH * HEIGHT];
unsigned char *current_dev;
unsigned char *next_dev;
const dim3 gridDim(8, 8, 1);
const dim3 blocksDim(16, 16, 1); // 256 threads per block
extern const size_t field_size;

/*
const int offsets[8][2] = {{-1, 1},{0, 1},{1, 1},
                           {-1, 0},       {1, 0},
                           {-1,-1},{0,-1},{1,-1}};


*/
void fill_board(int *board) {
    int i;
    for (i=0; i<WIDTH*HEIGHT; i++)
        board[i] = rand() % 2;
}

void print_board(int *board) {
    int x, y;
    for (y=0; y<HEIGHT; y++) {
        for (x=0; x<WIDTH; x++) {
            char c = board[y * WIDTH + x] ? '#':' ';
            printf("%c", c);
        }
        printf("\n");
    }
    printf("-----\n");
}

extern __device__ void step(const unsigned char *current, unsigned char *next) {
    // coordinates of the cell we're currently evaluating
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    // offset index, neighbor coordinates, alive neighbor count
    int i, nx, ny, num_neighbors;
	
	const int offsets[8][2] = {{-1, 1},{0, 1},{1, 1},
                           {-1, 0},       {1, 0},
                           {-1,-1},{0,-1},{1,-1}};

    // write the next board state
    for (y=0; y<HEIGHT; y++) {
        for (x=0; x<WIDTH; x++) {
            // count this cell's alive neighbors
            num_neighbors = 0;
            for (i=0; i<8; i++) {
                // To make the board torroidal, we use modular arithmetic to
                // wrap neighbor coordinates around to the other side of the
                // board if they fall off.
                nx = (x + offsets[i][0] + WIDTH) % WIDTH;
                ny = (y + offsets[i][1] + HEIGHT) % HEIGHT;
				num_neighbors += current[ny * WIDTH + nx]==1;
                //if (current[ny * WIDTH + nx]) {
                //    num_neighbors++;
                //}
            }

            // apply the Game of Life rules to this cell
            next[y * WIDTH + x] = 0;
			/*
            if ((current[y * WIDTH + x] && num_neighbors==2) ||
                    num_neighbors==3) {
                next[y * WIDTH + x] = 1;
            }
			*/
        }
    }
}

void animate(void (*)(void), const unsigned char *board) {
    struct timespec delay = {0, 125000000}; // 0.125 seconds
    struct timespec remaining;
    while (1) {
        print_board(current);
        step(current_dev, next_dev);
        // Copy the next state, that step() just wrote into, to current state
        memcpy(current, next, sizeof(int) * WIDTH * HEIGHT);
        // We sleep only because textual output is slow and the console needs
        // time to catch up. We don't sleep in the graphical X11 version.
        nanosleep(&delay, &remaining);
    }
}

void do_loop() {
    step<<<gridDim, blocksDim>>>(current_dev, next_dev);
    //cudaCheckError("kernel execution");

    cudaMemcpy(current, next_dev, field_size, cudaMemcpyDeviceToHost);
    //cudaCheckError("Device->Host memcpy");

    cudaMemcpy(current_dev, next_dev, field_size, cudaMemcpyDeviceToDevice);
    //cudaCheckError("Device->Device memcpy");
}

int main(void) {
    // Initialize the global "current".
	
	cudaMalloc((void **)&current_dev, field_size);
	cudaMalloc((void **)&next_dev, field_size);
	
    fill_board(current);
	cudaMemcpy(current_dev, current, field_size, cudaMemcpyHostToDevice);
    animate(do_loop, current);

	cudaFree(current_dev);
	cudaFree(next_dev);
    return 0;
}

