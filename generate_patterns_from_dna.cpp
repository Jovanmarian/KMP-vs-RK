#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <string>

using namespace std;

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <jumlah_pattern>\n";
        return 1;
    }

    int total_patterns = atoi(argv[1]);
    const char* input_file = "sequence_100k1.txt";
    string output_file = "patterns_" + to_string(total_patterns) + ".txt";

    const int pattern_len = 10;

    ifstream infile(input_file);
    ofstream outfile(output_file);

    if (!infile) {
        cout << "Gagal membuka file DNA.\n";
        return 1;
    }

    string dna;
    infile >> dna;
    infile.close();

    srand(time(0));
    int dna_len = dna.length();

    for (int i = 0; i < total_patterns; i++) {
        int start = rand() % (dna_len - pattern_len);
        string pattern = dna.substr(start, pattern_len);
        outfile << pattern << endl;
    }

    outfile.close();
    cout << "Berhasil generate " << total_patterns << " pattern ke " << output_file << endl;

    return 0;
}
