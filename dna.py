def clean_fna_file():
    input_file = 'GCF_000001635.27_GRCm39_genomic.fna'
    output_file = 'cleaned_sequence1.txt'

    print(f"Mulai membersihkan file: {input_file}")
    sequence_parts = []
    line_count = 0

    # Membuka file input
    with open(input_file, 'r') as infile:
        for line in infile:
            line = line.strip()
            if line and not line.startswith('>'):
                # Hanya simpan A, C, G, T, hapus N, n, atau karakter lainnya
                filtered_line = ''.join([char for char in line.upper() if char in 'ACGT'])
                sequence_parts.append(filtered_line)

            line_count += 1
            if line_count % 100000 == 0:
                print(f"Processed {line_count} lines...")

    print("Menggabungkan seluruh DNA sequence...")
    cleaned_sequence = ''.join(sequence_parts)

    # Menyimpan hasil ke output file
    with open(output_file, 'w') as outfile:
        outfile.write(cleaned_sequence)

    print(f"Selesai! Hasil disimpan ke: {output_file}")

# Jalankan fungsi
clean_fna_file()
