#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <time.h>

#define MAX_TEXT 10000000        // Ukuran maksimum teks DNA
#define MAX_PATTERNS 2048        // Jumlah maksimum pattern
#define PATTERN_LEN 10           // Panjang masing-masing pattern
#define d 256                    // Basis hash
#define q 101                    // Modulus hash (bilangan prima)

// Fungsi hashing Rabin-Karp di device
__device__ int compute_hash(const char *str, int length) {
    int hash = 0;
    for (int i = 0; i < length; i++) {
        hash = (d * hash + str[i]) % q; // Proses hashing karakter demi karakter
    }
    return hash;
}

// Kernel CUDA Rabin-Karp
__global__ void rabin_karp_kernel(char *text, int text_len, char *patterns, int *result, int pattern_count) {
    int id = blockIdx.x * blockDim.x + threadIdx.x; // Hitung index global thread
    if (id >= pattern_count) return; // Lewati jika index melebihi jumlah pattern

    char *pattern = patterns + id * PATTERN_LEN;    // Ambil pattern ke-id
    int pHash = compute_hash(pattern, PATTERN_LEN); // Hitung hash pattern

    for (int i = 0; i <= text_len - PATTERN_LEN; i++) {
        int tHash = compute_hash(text + i, PATTERN_LEN); // Hash teks bagian saat ini
        if (tHash == pHash) { // Jika hash cocok, periksa karakter satu per satu
            bool match = true;
            for (int j = 0; j < PATTERN_LEN; j++) {
                if (text[i + j] != pattern[j]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                atomicAdd(&result[id], 1); // Tambah jumlah kemunculan dengan aman
                printf("[CUDA] Pattern ke-%d ditemukan di index %d\n", id, i); // Tampilkan posisi match
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printf("Usage: %s <text_file> <pattern_file>\n", argv[0]); // Validasi input argumen
        return 1;
    }

    // Baca file DNA panjang
    FILE *ftext = fopen(argv[1], "r");
    if (!ftext) { perror("DNA file not found"); return 1; }
    char *h_text = (char *)malloc(MAX_TEXT);
    fscanf(ftext, "%s", h_text); // Baca string DNA
    fclose(ftext);
    int text_len = strlen(h_text); // Hitung panjang DNA

    // Baca file patterns
    FILE *fpat = fopen(argv[2], "r");
    if (!fpat) { perror("Pattern file not found"); return 1; }
    char (*h_patterns)[PATTERN_LEN + 1] = (char (*)[PATTERN_LEN + 1])malloc(MAX_PATTERNS * (PATTERN_LEN + 1));
    int pattern_count = 0;
    while (fscanf(fpat, "%s", h_patterns[pattern_count]) != EOF && pattern_count < MAX_PATTERNS) {
        pattern_count++; // Hitung banyak pattern yang dibaca
    }
    fclose(fpat);

    // Flatten pattern agar sesuai untuk memory CUDA
    char *h_flat_patterns = (char *)malloc(pattern_count * PATTERN_LEN);
    for (int i = 0; i < pattern_count; i++) {
        memcpy(h_flat_patterns + i * PATTERN_LEN, h_patterns[i], PATTERN_LEN);
    }

    // Alokasi memori di device
    char *d_text, *d_patterns;
    int *d_result;
    int *h_result = (int *)calloc(pattern_count, sizeof(int)); // Inisialisasi hasil di host

    cudaMalloc(&d_text, text_len);
    cudaMalloc(&d_patterns, pattern_count * PATTERN_LEN);
    cudaMalloc(&d_result, pattern_count * sizeof(int));
    cudaMemset(d_result, 0, pattern_count * sizeof(int)); // Set hasil awal = 0

    cudaMemcpy(d_text, h_text, text_len, cudaMemcpyHostToDevice); // Copy DNA ke GPU
    cudaMemcpy(d_patterns, h_flat_patterns, pattern_count * PATTERN_LEN, cudaMemcpyHostToDevice); // Copy pattern ke GPU

    // Mulai pengukuran waktu CUDA
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    printf("[HOST] Menjalankan kernel Rabin-Karp CUDA...\n");
    int blockSize = 64; // Jumlah thread per block
    int gridSize = (pattern_count + blockSize - 1) / blockSize; // Hitung jumlah block
    rabin_karp_kernel<<<gridSize, blockSize>>>(d_text, text_len, d_patterns, d_result, pattern_count);
    cudaDeviceSynchronize(); // Sinkronisasi thread

    // Stop pengukuran waktu
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_ms;
    cudaEventElapsedTime(&time_ms, start, stop);

    cudaMemcpy(h_result, d_result, pattern_count * sizeof(int), cudaMemcpyDeviceToHost); // Copy hasil ke host

    // Print hasil akhir
    int total_matches = 0;
    for (int i = 0; i < pattern_count; i++) {
        printf("Pattern ke-%d ditemukan sebanyak %d kali\n", i, h_result[i]);
        total_matches += h_result[i];
    }

    printf("\nTotal pattern matches: %d\n", total_matches);
    printf("CUDA execution time: %.4f ms\n", time_ms); // Tampilkan waktu eksekusi CUDA

    // Bersihkan memori
    free(h_text); free(h_patterns); free(h_flat_patterns); free(h_result);
    cudaFree(d_text); cudaFree(d_patterns); cudaFree(d_result);
    return 0;

