#include <stdio.h>
#include <stdlib.h>

int * create_binary_matrix(int rows, int cols);
int div_ceil(int x, int y);
int ints_per_row(int rows, int cols);
struct BinaryCMM create_binary_cmm(int rows, int cols);
void print_binary_cmm(struct BinaryCMM cmm);
char *int2bin(int a, char *buffer, int buf_size);

struct BinaryCMM {
    int * matrix;
    int rows;
    int cols;
};

// For a binary matrix, the number of 'int' types needed per row
// in the matrix.
int ints_per_row(int rows, int cols) {
    return div_ceil(cols, sizeof(int) * 8);
}

int * create_binary_matrix(int rows, int cols) {
    printf("create_binary_matrix(%d, %d)\n", rows, cols);
    return malloc(ints_per_row(rows, cols) * rows);
}

int div_ceil(int x, int y) {
    return x/y + (x % y != 0);
}

struct BinaryCMM create_binary_cmm(int rows, int cols) {
    struct BinaryCMM result;
    result.matrix = create_binary_matrix(rows, cols);
    result.rows = rows;
    result.cols = cols;
    return result;
}

void print_binary_cmm(struct BinaryCMM cmm) {
    int ipr = ints_per_row(cmm.rows, cmm.cols);
    char row_string[cmm.cols+1];
    char int_string[33];
    row_string[cmm.cols] = '\0';
    int_string[32] = '\0';

    for (int row=0; row < cmm.rows; row++) {
        for (int col_block=0; col_block < ipr; col_block++) {
            // Since 'int' types are used to represent groups of bits
            // col_block refers to the index of a particular int
            int data = cmm.matrix[row*ipr + col_block]; 
            int2bin(data, int_string, 32);
            for (int i=0; i < 32; i++) {
                if (i >= cmm.cols) break;
                row_string[col_block*32 + i] = int_string[i];
            }
        }
        printf("%s", row_string);
        printf("\n");
    }
}

// buffer must have length >= sizeof(int) + 1
// Write to the buffer backwards so that the binary representation
// is in the correct order i.e.  the LSB is on the far right
// instead of the far left of the printed string
char *int2bin(int a, char *buffer, int buf_size) {
    buffer += (buf_size - 1);

    for (int i = 31; i >= 0; i--) {
        *buffer-- = (a & 1) + '0';
        a >>= 1;
    }

    return buffer;
}

int main(int argc, char** argv) {
    printf("main\n");
    struct BinaryCMM cmm = create_binary_cmm(10, 10);
    print_binary_cmm(cmm);
    return 0;
}
