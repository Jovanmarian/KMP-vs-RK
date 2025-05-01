#include <iostream>                     // Library input/output
#include <fstream>                      // Untuk baca file
#include <cstring>                      // Untuk fungsi string
#include <cuda_runtime.h>               // CUDA runtime API
using namespace std;

#define MAX_TEXT 10000000              // Ukuran maksimum teks DNA
#define MAX_PATTERNS 2048              // Maksimum jumlah pola
#define PATTERN_LEN 10                 // Panjang setiap pola

// Fungsi prefix KMP versi device
__device__ void preKMP_device(const char *pattern, int *f) {
    f[0] = -1;
    for (int i = 1; i < PATTERN_LEN; i++) {
        int k = f[i - 1];
        while (k >= 0 && pattern[k] != pattern[i - 1])  // Cari prefix yang cocok
            k = f[k];
        f[i] = k + 1;                                   // Simpan hasil ke table prefix
    }
}

// Kernel KMP CUDA
__global__ void kmp_kernel(char *text, int text_len, char *patterns, int *results, int pattern_count) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;     // Setiap thread memproses 1 pattern
    if (id >= pattern_count) return;

    char *pattern = patterns + id * PATTERN_LEN;        // Ambil pattern ke-id
    int f[PATTERN_LEN];
    preKMP_device(pattern, f);                          // Hitung prefix table

    int i = 0, k = 0;
    while (i < text_len) {
        if (k == -1 || text[i] == pattern[k]) {
            i++; k++;
            if (k == PATTERN_LEN) {
                atomicAdd(&results[id], 1);              // Jika cocok, tambahkan jumlah
                k = f[k-1];                              // Lanjut dari prefix sebelumnya
            }
        } else {
            k = f[k];                                    // Lanjutkan pencarian dengan prefix
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cout << "Usage: " << argv[0] << " <text_file> <pattern_file>\n";
        return 1;
    }

    // Baca DNA panjang
    ifstream ftext(argv[1]);
    if (!ftext) { perror("DNA file not found"); return 1; }
    string text_str;
    ftext >> text_str;
    ftext.close();
    int text_len = text_str.length();

    // Baca daftar pola
    ifstream fpat(argv[2]);
    if (!fpat) { perror("Pattern file not found"); return 1; }
    string patterns_vec[MAX_PATTERNS];
    int pattern_count = 0;
    while (fpat >> patterns_vec[pattern_count] && pattern_count < MAX_PATTERNS) {
        pattern_count++;                                // Hitung jumlah pola
    }
    fpat.close();

    // Flatten: gabungkan semua pattern ke satu array
    char *h_flat_patterns = (char *)malloc(pattern_count * PATTERN_LEN);
    for (int i = 0; i < pattern_count; i++)
        memcpy(h_flat_patterns + i * PATTERN_LEN, patterns_vec[i].c_str(), PATTERN_LEN);

    // Alokasi memori device
    char *d_text, *d_patterns;
    int *d_results, *h_results;
    h_results = (int *)calloc(pattern_count, sizeof(int)); // Inisialisasi hasil = 0

    cudaMalloc(&d_text, text_len);                         // Alokasi teks
    cudaMalloc(&d_patterns, pattern_count * PATTERN_LEN);  // Alokasi pola
    cudaMalloc(&d_results, pattern_count * sizeof(int));   // Alokasi hasil

    cudaMemcpy(d_text, text_str.c_str(), text_len, cudaMemcpyHostToDevice);          // Copy teks ke device
    cudaMemcpy(d_patterns, h_flat_patterns, pattern_count * PATTERN_LEN, cudaMemcpyHostToDevice); // Copy pola
    cudaMemset(d_results, 0, pattern_count * sizeof(int)); // Reset hasil

    // Ukur waktu CUDA
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int blockSize = 64;                                     // 64 thread per block
    int gridSize = (pattern_count + blockSize - 1) / blockSize; // Banyak block
    kmp_kernel<<<gridSize, blockSize>>>(d_text, text_len, d_patterns, d_results, pattern_count);
    cudaDeviceSynchronize();                                // Tunggu semua thread selesai

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_ms;
    cudaEventElapsedTime(&time_ms, start, stop);            // Hitung waktu eksekusi CUDA

    cudaMemcpy(h_results, d_results, pattern_count * sizeof(int), cudaMemcpyDeviceToHost); // Ambil hasil

    int total_matches = 0;
    for (int i = 0; i < pattern_count; i++) {
        printf("Pattern ke-%d ditemukan sebanyak %d kali\n", i, h_results[i]);
        total_matches += h_results[i];
    }

    printf("\nTotal pattern matches: %d\n", total_matches);
    printf("CUDA execution time: %.4f ms\n", time_ms);       // Cetak waktu CUDA

    // Bersihkan memori
    free(h_flat_patterns);
    free(h_results);
    cudaFree(d_text);
    cudaFree(d_patterns);
    cudaFree(d_results);

    return 0;
}
