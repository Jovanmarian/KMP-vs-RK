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
KMP-on-CUDA-master/
│
├── kmp_CUDA.cu            # CUDA implementation of KMP algorithm
├── direct_match.cu        # CUDA implementation of brute-force matching (optional reference)
├── kmp.cpp                # Sequential version of KMP (for comparison)
├── input.txt              # Input file (DNA sequence + pattern)
├── output.txt             # Output file
├── README.md              # Project description
```

```
rabin-karp-for-CUDA-master/
│
├── rabin_karp.cu          # CUDA implementation of Rabin-Karp
├── rk_seq.cpp             # Sequential version of Rabin-Karp
├── input.txt              # Input file (DNA sequence + pattern)
├── output.txt             # Output file
├── README.md              # Project description
```

---
---

##  Compilation and Execution

Make sure you have **NVIDIA CUDA Toolkit** installed.

### KMP (CUDA)
```bash
nvcc -o kmp_cuda kmp_CUDA.cu
./kmp_cuda input.txt
```

### Rabin-Karp (CUDA)
```bash
nvcc -o rk_cuda rabin_karp.cu
./rk_cuda input.txt
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

