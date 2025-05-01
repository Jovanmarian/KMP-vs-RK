def cut_first_100k_chars(input_file, output_file):
    with open(input_file, 'r') as infile:
        data = infile.read()
        first_100k = data[:100000]  # Ambil hanya 100.000 karakter pertama

    with open(output_file, 'w') as outfile:
        outfile.write(first_100k)

    print(f"100.000 karakter pertama telah disimpan di: {output_file}")

# Contoh penggunaan
cut_first_100k_chars('cleaned_sequence1.txt', 'sequence_100k1.txt')
