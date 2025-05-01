#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <vector>

using namespace std;

// Fungsi untuk membaca DNA sequence dari file
char* read_dna_sequence(const char* filename) {
    ifstream file(filename, ios::binary | ios::ate);  // Buka file dalam mode biner dan langsung ke akhir untuk tahu ukurannya

    if (!file.is_open()) {                            // Jika file gagal dibuka
        cerr << "File tidak ditemukan: " << filename << endl;
        exit(1);                                       // Keluar program
    }

    streamsize size = file.tellg();                   // Ambil ukuran file dalam byte
    file.seekg(0, ios::beg);                          // Kembalikan posisi baca ke awal file

    char* buffer = new char[size + 1];                // Alokasikan buffer sebesar ukuran file + 1 byte untuk null terminator

    if (!file.read(buffer, size)) {                   // Baca isi file ke buffer
        cerr << "Gagal membaca file\n";
        exit(1);                                       // Keluar jika gagal membaca
    }

    buffer[size] = '\0';                              // Tambahkan null terminator di akhir buffer agar menjadi string valid
    return buffer;                                    // Kembalikan pointer ke buffer
}

// Fungsi untuk membaca semua pola dari file teks
vector<string> read_patterns(const char* filename) {
    ifstream file(filename);                          // Buka file teks dalam mode default (teks)

    vector<string> patterns;                          // Vector untuk menyimpan semua pola dari file
    string line;

    if (!file.is_open()) {
        cerr << "Gagal membuka file pola: " << filename << endl;
        exit(1);                                       // Keluar jika file tidak bisa dibuka
    }

    while (getline(file, line)) {                      // Baca file baris per baris
        if (!line.empty())                             // Jika baris tidak kosong
            patterns.push_back(line);                  // Simpan pola ke vector
    }

    return patterns;                                   // Kembalikan vector berisi pola
}

// Fungsi untuk melakukan preprocessing pada pattern agar KMP lebih efisien
void preKMP(char* pattern, int f[]) {
    int m = strlen(pattern);                           // Panjang dari pola
    f[0] = -1;                                         // Nilai awal untuk indeks ke-0

    for (int i = 1; i < m; i++) {                      // Iterasi dari indeks 1 hingga akhir pola
        int k = f[i - 1];                              // Ambil nilai fallback sebelumnya

        while (k >= 0 && pattern[k] != pattern[i - 1]) // Selama tidak cocok, fallback mundur
            k = f[k];

        f[i] = k + 1;                                  // Simpan posisi fallback yang baru
    }
}

// Fungsi utama untuk menjalankan pencocokan KMP
void KMP(char* pattern, char* text, int* f) {
    int m = strlen(pattern);                           // Panjang pola
    int n = strlen(text);                              // Panjang teks (DNA)
    int i = 0, k = 0;                                  // i = indeks teks, k = indeks pola

    while (i < n) {                                    // Selama belum mencapai akhir teks
        if (k == -1 || text[i] == pattern[k]) {        // Jika karakter cocok atau fallback (k = -1)
            i++;                                       // Lanjut ke karakter berikutnya di teks
            k++;                                       // Lanjut ke karakter berikutnya di pola

            if (k == m) {                              // Jika seluruh pola cocok
                // Posisi pencocokan = i - m (bisa ditampilkan kalau mau)
                k = f[k - 1];                          // Reset k ke posisi fallback untuk pencarian berikutnya
            }
        } else {
            k = f[k];                                  // Fallback: lompat ke posisi sebelumnya di pola
        }
    }
}

// Fungsi utama (entry point program)
int main() {
    const char* dna_filename = "C:\\Users\\Jonathan\\Downloads\\Parallel 1\\sequence_100k1.txt";       // Lokasi file DNA
    const char* pattern_filename = "C:\\Users\\Jonathan\\Downloads\\Parallel 1\\patterns_1024.txt";    // Lokasi file pola

    cout << "Membaca DNA dari file..." << endl;
    char* dna = read_dna_sequence(dna_filename);       // Baca isi DNA dan simpan ke dalam buffer

    cout << "Membaca pola dari file..." << endl;
    vector<string> patterns = read_patterns(pattern_filename); // Baca semua pola ke vector

    cout << "\nMenjalankan KMP untuk " << patterns.size() << " pola..." << endl;
    clock_t start = clock();                           // Mulai stopwatch untuk ukur waktu eksekusi

    // Loop untuk mencocokkan setiap pola ke DNA
    for (const auto& pattern_str : patterns) {
        const char* pattern = pattern_str.c_str();     // Ubah string STL ke C-style string
        int m = strlen(pattern);                       // Dapatkan panjang pola

        int* f = new int[m + 1];                       // Alokasi array untuk tabel fallback
        preKMP((char*)pattern, f);                     // Lakukan preprocessing KMP
        KMP((char*)pattern, dna, f);                   // Jalankan algoritma KMP
        delete[] f;                                    // Hapus array setelah dipakai
    }

    clock_t end = clock();                             // Ambil waktu selesai
    double time_taken = double(end - start) / CLOCKS_PER_SEC; // Hitung waktu dalam detik
    cout << "Waktu eksekusi: " << time_taken << " detik" << endl;

    delete[] dna;                                      // Bebaskan memori DNA
    return 0;                                          // Program selesai
}