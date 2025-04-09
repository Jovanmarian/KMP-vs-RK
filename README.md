# 🔬 Parallel String Matching Using CUDA: KMP vs Rabin-Karp on DNA Sequence

## 📌 Deskripsi Proyek
Repositori ini berisi implementasi dan perbandingan dua algoritma pencocokan string — **Knuth-Morris-Pratt (KMP)** dan **Rabin-Karp (RK)** — untuk pencocokan pola DNA. Algoritma ini diimplementasikan dalam dua versi:
- ✅ Versi sekuensial (CPU)
- ⚡ Versi paralel menggunakan CUDA (GPU)

Dataset menggunakan genom **Mus musculus (tikus)** dari NCBI:  
[GCF_000001635.27](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000001635.27/)

---

## 🎯 Tujuan
- Menganalisis efisiensi algoritma string matching untuk urutan DNA.
- Mengimplementasikan CUDA untuk mempercepat proses pencocokan string.
- Membandingkan hasil dan kecepatan eksekusi antara versi CPU dan GPU.
- Membuat laporan ilmiah dalam format IEEE berdasarkan hasil eksperimen.

---

## 📁 Struktur Proyek

