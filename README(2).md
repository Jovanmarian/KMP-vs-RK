
# 🔬 Parallel String Matching using CUDA: KMP and Rabin-Karp

This project implements and compares two classic string-matching algorithms — **Knuth-Morris Pratt (KMP)** and **Rabin-Karp (RK)** — using **CUDA** to accelerate computation for large-scale DNA datasets. The aim is to evaluate and benchmark their performance in a GPU-parallel environment, particularly for bioinformatics applications.

---

## 📁 Project Structure

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

## 🧬 Dataset

Dataset used:  
Human genome reference: [GCF_000001635.27](https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000001635.27/)

To prepare input:
- Extract a portion of DNA sequence (e.g., 1MB chunk).
- Use a 10-character DNA pattern.
- Format the input file `input.txt` to contain:
  ```
  <TEXT>
  <PATTERN>
  ```

---

## 🚀 Compilation and Execution

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

## 📊 Output

- Results will be written to `output.txt`.
- You may compare GPU performance (execution time and memory usage) using:
  - `cudaEventRecord` timing (embedded in code), or
  - `nsys` for detailed profiling:
    ```bash
    nsys profile ./kmp_cuda input.txt
    ```

---

## 🎯 Research Objectives

- Implement KMP and Rabin-Karp using CUDA for string matching on DNA.
- Analyze execution time and memory usage on various dataset sizes.
- Compare efficiency and scalability of both algorithms in parallel execution.

---

## 📚 References

- CUDA Toolkit Documentation: https://docs.nvidia.com/cuda/
- NCBI Genome Data: https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000001635.27/
- KMP CUDA GitHub: https://github.com/Fang-Haoshu/KMP-on-CUDA
- Rabin-Karp CUDA GitHub: https://github.com/mstftrn/rabin-karp-for-CUDA

---

## 🧠 Authors

- Vincent Arianto (01082220009)
- Jonathan Tiong (01082220017)
- Jovan Marian Winarko (01082220003)
- Mikhail Keene Grady (01082220012)

Fakultas Teknologi Informasi, Universitas Pelita Harapan
