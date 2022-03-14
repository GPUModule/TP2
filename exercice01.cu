#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utils.h"


#define A_WIDTH 1024
#define A_HEIGHT 1024
#define B_WIDTH 1024
#define B_HEIGHT 1024
#define C_WIDTH B_WIDTH
#define C_HEIGHT A_HEIGHT

#define BLOCK_SIZE 8
#define NUM_SUBS (A_WIDTH / BLOCK_SIZE)

__device__ float d_A[A_HEIGHT][A_WIDTH];
__device__ float d_B[B_HEIGHT][B_WIDTH];
__device__ float d_C[C_HEIGHT][C_WIDTH];

float h_A[A_HEIGHT][A_WIDTH];
float h_B[B_HEIGHT][B_WIDTH];
float h_C[C_HEIGHT][C_WIDTH];
float h_C_ref[C_HEIGHT][C_WIDTH];

void checkCUDAError(const char *msg);
void matMulCPU(float A[A_HEIGHT][A_WIDTH], float B[B_HEIGHT][B_WIDTH], float C[C_HEIGHT][C_WIDTH]);
int matMulValidate(float C[C_HEIGHT][C_WIDTH], float Cref[C_HEIGHT][C_WIDTH]);

__global__ void matMulKernel()
{
    // Index des blocs et des threads
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	// 1.2.1 Indexation globale des threads
	int x = 
	int y = 
    

	float Csub = 0;
	// On itere sur A_WIDTH (meme que B_HEIGHT) pour calculer le produit
	for (int k = 0; k < A_WIDTH; k++){
		// 1.2.2 Multiplication matricielle entre une ligne de A et une colonne de B
		Csub += 
	}

	// On stocke le resultat dans la matrice C
	d_C[y][x] = Csub;
}

__global__ void matMulKernelSharedMemory()
{
    // Creation d'une sous-matrice de A et de B dans la memoire partagee.
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
	// Index des blocs et des threads
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
    //Variable permettant de stocker la valeur du produit matricielle entre As et Bs
    float Csub = 0;
 
	//On itere sur le nombre de sous-matrices de A et B
	for (int i = 0; i < NUM_SUBS; i++){
		//2.1: Calculer les indices globaux des threads des matrices A et B requis pour faire la copie depuis la memoire globale vers la memoire partagee. 
        int a_x = ;
		int a_y = ;
		int b_x = ;
		int b_y = ;
        
        //2.2: Chaque thread doit charger un seul element de A et B dans les sous_matrices As et Bs
        As[A remplir][A remplir] =
		Bs[A remplir][A remplir] =

        // Synchronisation pour attendre la fin du chargement des elements par les threads 
		__syncthreads();
        
        //2.3: Produit matricielle de la As et Bs
		for (int k = 0; k < BLOCK_SIZE; ++k)
		{
			
		}
        
        // Synchronisation pour attendre la fin du produit matricielle entre les deux sous-matrices As et Bs
		__syncthreads();
        
	}

    //2.4: Calculer les indices globaux des threads de la matrice C
	int c_x = ;
	int c_y = ;
    
	// On stocke le resultat dans la matrice C
	d_C[c_y][c_x] = Csub;
}


int main(int argc, char **argv)

{
	unsigned int octets_A, octets_B, octets_C;
	unsigned int x, y, errors;
	int maxActiveBlocks;

	event_pair timer;

	if (A_WIDTH != B_HEIGHT){
		printf("Error: A_HEIGHT and B_WIDTH do not match\n");
	}

	// 1.1.1 Recuperer la taille en octets des matrices A, B et C.
	octets_A = 
	octets_B =
	octets_C =

	// Initialisation de A
	for (y = 0; y < A_HEIGHT; y++)
	for (x = 0; x <A_WIDTH; x++)
		h_A[y][x] = (float)rand() / RAND_MAX;
	// Initialisation de B
	for (y = 0; y < B_HEIGHT; y++)
	for (x = 0; x <B_WIDTH; x++)
		h_B[y][x] = (float)rand() / RAND_MAX;

	// 1.1.2 Copie de la memoire Host sur le Device
	cudaMemcpyToSymbol(A remplir, A remplir, A remplir); //cudaMemcpyToSymbol(destination, source, taille en octets);
	cudaMemcpyToSymbol(A remplir, A remplir, A remplir); //cudaMemcpyToSymbol(destination, source, taille en octets);
	checkCUDAError("CUDA memcpy");

	// Setup execution parameters
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid(C_WIDTH / BLOCK_SIZE, C_HEIGHT / BLOCK_SIZE);
	
    start_timer(&timer);
    matMulKernel << < grid, threads >> >();
	
    //2.5 lancement du kernel
	//start_timer(&timer);
    //matMulKernelSharedMemory << < grid, threads >> >();
  	cudaDeviceSynchronize();  
	stop_timer(&timer,"Produit matriciel GPU"); // Rajouter Shared memory dans la chaine de caractères pour l'exo 2.5
	checkCUDAError("CUDA kernel execution");

	// 1.1.3 Copie du resultat depuis le device vers l'host
	cudaMemcpyFromSymbol(A remplir, A remplir, A remplir); // cudaMemcpyFromSymbol(destination, source, taille en octets);
	checkCUDAError("CUDA memcpy results");

	// Version de la multiplication matricielle sur CPU
	// Timer Host
	clock_t start, end;
	start = clock(); // start_timer(&timer);
	matMulCPU(h_A, h_B, h_C_ref);
	end = clock(); // stop_timer(&timer,"Produit matriciel CPU
	printf ("max_index: %0.8f sec\n",
		   ((float) end - start)/CLOCKS_PER_SEC);
	// Check les erreurs
	errors = matMulValidate(h_C, h_C_ref);
	if (errors)
		printf("%d Nombre total d'erreur\n", errors);
	else
		printf("La validation est un succes !\n");

}


void matMulCPU(float A[A_HEIGHT][A_WIDTH], float B[C_HEIGHT][C_WIDTH], float C[C_HEIGHT][C_WIDTH])
{
	int x, y, k;
	for (y = 0; y < C_HEIGHT; y++){
		for (x = 0; x < C_WIDTH; x++){
			C[y][x] = 0;
			for (k = 0; k < A_WIDTH; k++){
				C[y][x] += A[y][k] * B[k][x];
			}
		}
	}

}

int matMulValidate(float C[C_HEIGHT][C_WIDTH], float Cref[C_HEIGHT][C_WIDTH])
{
	int errors = 0;
	int y, x;
	float epsilon = 0.001;
	for (y = 0; y < C_HEIGHT; y++){
		for (x = 0; x < C_WIDTH; x++){
			if (!(C[y][x] >= Cref[y][x] - epsilon && C[y][x] <= Cref[y][x] + epsilon)){
				errors++;
				printf("Device item c[%d][%d] = %f does not mach host result %f\n", y, x, C[y][x], Cref[y][x]);
			}
		}
	}

	return errors;
}

void checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}
