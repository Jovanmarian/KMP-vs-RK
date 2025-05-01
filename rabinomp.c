#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_PATTERNS 1024
#define PATTERN_LENGTH 100 // Disesuaikan dengan panjang maksimal pattern

// Fungsi membaca isi DNA dari file
char* read_dna_sequence(const char* filename) {
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        printf("File tidak ditemukan: %s\n", filename);
        exit(1);
    }

    fseek(fp, 0, SEEK_END);
    long length = ftell(fp);
    rewind(fp);

    char *buffer = (char*)malloc(sizeof(char) * (length + 1));
    fread(buffer, sizeof(char), length, fp);
    buffer[length] = '\0';

    fclose(fp);
    return buffer;
}

// Fungsi membaca patterns dari file
int read_patterns(const char* filename, char patterns[][PATTERN_LENGTH]) {
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        printf("File pattern tidak ditemukan: %s\n", filename);
        exit(1);
    }

    int count = 0;
    while (fgets(patterns[count], PATTERN_LENGTH, fp) != NULL) {
        // Hilangkan newline di akhir pattern
        patterns[count][strcspn(patterns[count], "\r\n")] = '\0';
        count++;
        if (count >= MAX_PATTERNS) break;
    }

    fclose(fp);
    return count;
}

// Fungsi Rabin-Karp matcher
void rk_matcher(char *text, char *pattern, int d, int q) {
    int n = strlen(text);
    int m = strlen(pattern);
    int i, j;
    int h = 1;
    int p = 0;  // hash untuk pattern
    int t = 0;  // hash untuk teks

    // h = pow(d, m-1) % q
    for (i = 0; i < m - 1; i++)
        h = (h * d) % q;

    // Hitung hash awal pattern dan teks
    for (i = 0; i < m; i++) {
        p = (d * p + pattern[i]) % q;
        t = (d * t + text[i]) % q;
    }

    int found = 0;

    // Slide pola ke seluruh teks
    for (i = 0; i <= n - m; i++) {
        // Jika hash cocok, cek karakter satu per satu
        if (p == t) {
            for (j = 0; j < m; j++) {
                if (text[i + j] != pattern[j])
                    break;
            }
            if (j == m) {
                // printf("Pola ditemukan di posisi: %d\n", i); // Optional
                found = 1;
            }
        }

        // Hitung hash untuk window teks berikutnya
        if (i < n - m) {
            t = (d * (t - text[i] * h) + text[i + m]) % q;

            // pastikan t tidak negatif
            if (t < 0)
                t = (t + q);
        }
    }

    if (!found) {
        // printf("Pola tidak ditemukan.\n"); // Optional
    }
}

int main() {
    const char *dna_filename = "sequence_100k1.txt";    // Ganti dengan path file DNA jika perlu
    const char *pattern_filename = "patterns_1024.txt";    // Ganti dengan path file pattern jika perlu

    printf("\nMembaca file DNA...\n");
    char *dna = read_dna_sequence(dna_filename);

    printf("\nMembaca file patterns...\n");
    char patterns[MAX_PATTERNS][PATTERN_LENGTH];
    int num_patterns = read_patterns(pattern_filename, patterns);

    printf("\nMelakukan pencocokan untuk %d pola...\n", num_patterns);

    // Mulai hitung waktu
    clock_t start = clock();

    for (int i = 0; i < num_patterns; i++) {
        rk_matcher(dna, patterns[i], 256, 101); // base 256, modulus q=101
    }

    // Akhir hitung waktu
    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

    printf("\nWaktu eksekusi untuk %d pola: %.6f detik\n", num_patterns, elapsed);

    free(dna);
}
