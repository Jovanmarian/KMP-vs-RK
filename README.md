#  Parallel String Matching Using CUDA: KMP vs Rabin-Karp on DNA Sequence

##  Deskripsi Proyek
Repositori ini berisi implementasi dan perbandingan dua algoritma pencocokan string — **Knuth-Morris-Pratt (KMP)** dan **Rabin-Karp (RK)** — untuk pencocokan pola DNA. Algoritma ini diimplementasikan dalam dua versi:
-  Versi sekuensial (CPU)
-  Versi paralel menggunakan CUDA (GPU)
- https://docs.google.com/document/d/1xZc1V3vqB4yin6NjN_uEihYQc8wv3opvMNl8nEYz89w/edit


Dataset menggunakan genom **Mus musculus (tikus)** dari NCBI:  
[GCF_000001635.27](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000001635.27/)

---

##  Tujuan
- Menganalisis efisiensi algoritma string matching untuk urutan DNA.
- Mengimplementasikan CUDA untuk mempercepat proses pencocokan string.
- Membandingkan hasil dan kecepatan eksekusi antara versi CPU dan GPU.
- Membuat laporan ilmiah dalam format IEEE berdasarkan hasil eksperimen.
- Mengimplementasikan algoritma KMP dan Rabin-Karp menggunakan CUDA untuk pencocokan string pada data DNA.  
- Menganalisis waktu eksekusi dan penggunaan memori pada berbagai ukuran dataset.  
- Membandingkan efisiensi dan skalabilitas kedua algoritma dalam eksekusi secara paralel.

---

##  Struktur Proyek

```
Parallel 1/
├── .vscode/                        # Configuration files for Visual Studio Code.
│   ├── c_cpp_properties.json       # Configures IntelliSense for C++ development.
│   ├── launch.json                # Debugging configurations for VSCode.
│   └── settings.json              # VSCode workspace settings.
├── Nsys-rep/                       # Nsight Systems profiling results.
│   ├── kmp_profile_1024.nsys-rep   # Profiling results for KMP algorithm with pattern size 1024.
│   ├── kmp_profile_128.nsys-rep    # Profiling results for KMP algorithm with pattern size 128.
│   ├── kmp_profile_16.nsys-rep     # Profiling results for KMP algorithm with pattern size 16.
│   ├── kmp_profile_256.nsys-rep    # Profiling results for KMP algorithm with pattern size 256.
│   ├── kmp_profile_32.nsys-rep     # Profiling results for KMP algorithm with pattern size 32.
│   ├── kmp_profile_512.nsys-rep    # Profiling results for KMP algorithm with pattern size 512.
│   ├── kmp_profile_64.nsys-rep     # Profiling results for KMP algorithm with pattern size 64.
│   ├── kmp_profile_8.nsys-rep      # Profiling results for KMP algorithm with pattern size 8.
│   ├── rabin_profile_1024.nsys-rep # Profiling results for Rabin-Karp algorithm with pattern size 1024.
│   ├── rabin_profile_128.nsys-rep  # Profiling results for Rabin-Karp algorithm with pattern size 128.
│   ├── rabin_profile_16.nsys-rep   # Profiling results for Rabin-Karp algorithm with pattern size 16.
│   ├── rabin_profile_256.nsys-rep  # Profiling results for Rabin-Karp algorithm with pattern size 256.
│   ├── rabin_profile_32.nsys-rep   # Profiling results for Rabin-Karp algorithm with pattern size 32.
│   ├── rabin_profile_512.nsys-rep  # Profiling results for Rabin-Karp algorithm with pattern size 512.
│   ├── rabin_profile_64.nsys-rep   # Profiling results for Rabin-Karp algorithm with pattern size 64.
│   └── rabin_profile_8.nsys-rep    # Profiling results for Rabin-Karp algorithm with pattern size 8.
├── Pattern/                        # Directory containing pattern files for matching.
│   └── kmp rk.xlsx                # Excel file with patterns for KMP and Rabin-Karp algorithms.
├── Sqlite/                         # SQLite database files for profiling data.
│   ├── kmp_profile_1024.sqlite     # SQLite database for KMP algorithm profiling with pattern size 1024.
│   ├── kmp_profile_128.sqlite      # SQLite database for KMP algorithm profiling with pattern size 128.
│   ├── kmp_profile_16.sqlite       # SQLite database for KMP algorithm profiling with pattern size 16.
│   ├── kmp_profile_256.sqlite      # SQLite database for KMP algorithm profiling with pattern size 256.
│   ├── kmp_profile_32.sqlite       # SQLite database for KMP algorithm profiling with pattern size 32.
│   ├── kmp_profile_512.sqlite      # SQLite database for KMP algorithm profiling with pattern size 512.
│   ├── kmp_profile_64.sqlite       # SQLite database for KMP algorithm profiling with pattern size 64.
│   ├── kmp_profile_8.sqlite        # SQLite database for KMP algorithm profiling with pattern size 8.
│   ├── rabin_profile_1024.sqlite   # SQLite database for Rabin-Karp algorithm profiling with pattern size 1024.
│   ├── rabin_profile_128.sqlite    # SQLite database for Rabin-Karp algorithm profiling with pattern size 128.
│   ├── rabin_profile_16.sqlite     # SQLite database for Rabin-Karp algorithm profiling with pattern size 16.
│   ├── rabin_profile_256.sqlite    # SQLite database for Rabin-Karp algorithm profiling with pattern size 256.
│   ├── rabin_profile_32.sqlite     # SQLite database for Rabin-Karp algorithm profiling with pattern size 32.
│   ├── rabin_profile_512.sqlite    # SQLite database for Rabin-Karp algorithm profiling with pattern size 512.
│   ├── rabin_profile_64.sqlite     # SQLite database for Rabin-Karp algorithm profiling with pattern size 64.
│   └── rabin_profile_8.sqlite      # SQLite database for Rabin-Karp algorithm profiling with pattern size 8.
├── cut_100k_chars.py               # Python script to trim large DNA sequence data.
├── dna.py                          # Python script for manipulating DNA sequence data.
├── gen_patterns.exe                # Executable to generate patterns for matching.
├── generate_data.exe               # Executable to generate test data for matching algorithms.
├── generate_patterns_from_dna.cpp  # C++ program to generate patterns from DNA data.
├── kmp.cpp                         # C++ implementation of the KMP string matching algorithm.
├── kmp.exe                         # Executable for running the KMP algorithm.
├── kmp.exp                         # Exported results for KMP execution.
├── kmp.lib                         # Library file for the KMP algorithm.
├── kmp_CUDA.cu                     # CUDA implementation of the KMP algorithm.
├── kmp_CUDA.exe                    # Executable for running the KMP CUDA implementation.
├── kmp_CUDA.exp                    # Exported results for KMP CUDA execution.
├── kmp_CUDA.lib                    # Library file for the KMP CUDA implementation.
├── output/                         # Directory containing the outputs from execution.
│   ├── kmp.exe                     # Output for KMP algorithm execution.
│   └── rabinomp.exe                # Output for Rabin-Karp algorithm (OpenMP) execution.
├── patterns_1024.txt               # Text file with patterns of size 1024 for matching.
├── patterns_128.txt                # Text file with patterns of size 128 for matching.
├── patterns_16.txt                 # Text file with patterns of size 16 for matching.
├── patterns_256.txt                # Text file with patterns of size 256 for matching.
├── patterns_32.txt                 # Text file with patterns of size 32 for matching.
├── patterns_512.txt                # Text file with patterns of size 512 for matching.
├── patterns_64.txt                 # Text file with patterns of size 64 for matching.
├── patterns_8.txt                  # Text file with patterns of size 8 for matching.
├── rabin.cu                        # CUDA implementation of the Rabin-Karp algorithm.
├── rabin.exe                       # Executable for running the Rabin-Karp algorithm.
├── rabin.exp                       # Exported results for Rabin-Karp execution.
├── rabin.lib                       # Library file for the Rabin-Karp algorithm.
├── rabinomp.c                      # C implementation of the Rabin-Karp algorithm using OpenMP.
├── rabinomp.exe                    # Executable for running the Rabin-Karp OpenMP implementation.
└── sequence_100k1.txt              # Large DNA sequence file for testing string matching algorithms.

```

---
---

##  Compilation and Execution

Make sure you have **NVIDIA CUDA Toolkit** installed.

### KMP (CUDA)
```bash
nvcc kmp.cu -o kmp
nsys profile -o kmp_profile_8 ./kmp sequence_100k1.txt patterns_8.txt
nsys profile -o kmp_profile_16 ./kmp sequence_100k1.txt patterns_16.txt
nsys profile -o kmp_profile_32 ./kmp sequence_100k1.txt patterns_32.txt
nsys profile -o kmp_profile_64 ./kmp sequence_100k1.txt patterns_64.txt
nsys profile -o kmp_profile_128 ./kmp sequence_100k1.txt patterns_128.txt
nsys profile -o kmp_profile_256 ./kmp sequence_100k1.txt patterns_256.txt
nsys profile -o kmp_profile_512 ./kmp sequence_100k1.txt patterns_512.txt
nsys profile -o kmp_profile_1024 ./kmp sequence_100k1.txt patterns_1024.txt
```

### Rabin-Karp (CUDA)
```bash
nvcc rabin.cu -o rabin
nsys profile -o rabin_profile_8 ./rabin sequence_100k1.txt patterns_8.txt
nsys profile -o rabin_profile_16 ./rabin sequence_100k1.txt patterns_16.txt
nsys profile -o rabin_profile_32 ./rabin sequence_100k1.txt patterns_32.txt
nsys profile -o rabin_profile_64 ./rabin sequence_100k1.txt patterns_64.txt
nsys profile -o rabin_profile_128 ./rabin sequence_100k1.txt patterns_128.txt
nsys profile -o rabin_profile_256 ./rabin sequence_100k1.txt patterns_256.txt
nsys profile -o rabin_profile_512 ./rabin sequence_100k1.txt patterns_512.txt
nsys profile -o rabin_profile_1024 ./rabin sequence_100k1.txt patterns_1024.txt
```

---

## 
Output

- Results will be written to `output.txt`.
- You may compare GPU performance (execution time and memory usage) using:
  - `cudaEventRecord` timing (embedded in code), or
  - `nsys` for detailed profiling:
    ```bash
    nsys profile ./kmp_cuda input.txt
    ```

---


---

## References

- CUDA Toolkit Documentation: https://docs.nvidia.com/cuda/
- NCBI Genome Data: https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000001635.27/
- KMP CUDA GitHub: https://github.com/Fang-Haoshu/KMP-on-CUDA
- Rabin-Karp CUDA GitHub: https://github.com/mstftrn/rabin-karp-for-CUDA

---

## Authors

- Vincent Arianto (01082220009)
- Jonathan Tiong (01082220017)
- Jovan Marian Winarko (01082220003)
- Mikhail Keene Grady (01082220012)

Fakultas Teknologi Informasi, Universitas Pelita Harapan

