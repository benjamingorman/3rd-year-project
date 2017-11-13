#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define IRIS_ENCODING_SIZE 147

struct CMM create_cmm(int rows, int cols);
int * create_matrix(int rows, int cols);
void print_cmm(struct CMM cmm);
void insert(struct CMM cmm, int* data_vec, int* key_vec);
int * recall(struct CMM cmm, int * key_vector, int * output_vector);
float iris_train_set[120][5] = {
    {5.4,3.7,1.5,0.2,1},
    {4.8,3.4,1.6,0.2,1},
    {4.8,3.0,1.4,0.1,1},
    {4.3,3.0,1.1,0.1,1},
    {5.8,4.0,1.2,0.2,1},
    {5.7,4.4,1.5,0.4,1},
    {5.4,3.9,1.3,0.4,1},
    {5.1,3.5,1.4,0.3,1},
    {5.7,3.8,1.7,0.3,1},
    {5.1,3.8,1.5,0.3,1},
    {5.4,3.4,1.7,0.2,1},
    {5.1,3.7,1.5,0.4,1},
    {4.6,3.6,1.0,0.2,1},
    {5.1,3.3,1.7,0.5,1},
    {4.8,3.4,1.9,0.2,1},
    {5.0,3.0,1.6,0.2,1},
    {5.0,3.4,1.6,0.4,1},
    {5.2,3.5,1.5,0.2,1},
    {5.2,3.4,1.4,0.2,1},
    {4.7,3.2,1.6,0.2,1},
    {4.8,3.1,1.6,0.2,1},
    {5.4,3.4,1.5,0.4,1},
    {5.2,4.1,1.5,0.1,1},
    {5.5,4.2,1.4,0.2,1},
    {4.9,3.1,1.5,0.1,1},
    {5.0,3.2,1.2,0.2,1},
    {5.5,3.5,1.3,0.2,1},
    {4.9,3.1,1.5,0.1,1},
    {4.4,3.0,1.3,0.2,1},
    {5.1,3.4,1.5,0.2,1},
    {5.0,3.5,1.3,0.3,1},
    {4.5,2.3,1.3,0.3,1},
    {4.4,3.2,1.3,0.2,1},
    {5.0,3.5,1.6,0.6,1},
    {5.1,3.8,1.9,0.4,1},
    {4.8,3.0,1.4,0.3,1},
    {5.1,3.8,1.6,0.2,1},
    {4.6,3.2,1.4,0.2,1},
    {5.3,3.7,1.5,0.2,1},
    {5.0,3.3,1.4,0.2,1},
    {5.0,2.0,3.5,1.0,2},
    {5.9,3.0,4.2,1.5,2},
    {6.0,2.2,4.0,1.0,2},
    {6.1,2.9,4.7,1.4,2},
    {5.6,2.9,3.6,1.3,2},
    {6.7,3.1,4.4,1.4,2},
    {5.6,3.0,4.5,1.5,2},
    {5.8,2.7,4.1,1.0,2},
    {6.2,2.2,4.5,1.5,2},
    {5.6,2.5,3.9,1.1,2},
    {5.9,3.2,4.8,1.8,2},
    {6.1,2.8,4.0,1.3,2},
    {6.3,2.5,4.9,1.5,2},
    {6.1,2.8,4.7,1.2,2},
    {6.4,2.9,4.3,1.3,2},
    {6.6,3.0,4.4,1.4,2},
    {6.8,2.8,4.8,1.4,2},
    {6.7,3.0,5.0,1.7,2},
    {6.0,2.9,4.5,1.5,2},
    {5.7,2.6,3.5,1.0,2},
    {5.5,2.4,3.8,1.1,2},
    {5.5,2.4,3.7,1.0,2},
    {5.8,2.7,3.9,1.2,2},
    {6.0,2.7,5.1,1.6,2},
    {5.4,3.0,4.5,1.5,2},
    {6.0,3.4,4.5,1.6,2},
    {6.7,3.1,4.7,1.5,2},
    {6.3,2.3,4.4,1.3,2},
    {5.6,3.0,4.1,1.3,2},
    {5.5,2.5,4.0,1.3,2},
    {5.5,2.6,4.4,1.2,2},
    {6.1,3.0,4.6,1.4,2},
    {5.8,2.6,4.0,1.2,2},
    {5.0,2.3,3.3,1.0,2},
    {5.6,2.7,4.2,1.3,2},
    {5.7,3.0,4.2,1.2,2},
    {5.7,2.9,4.2,1.3,2},
    {6.2,2.9,4.3,1.3,2},
    {5.1,2.5,3.0,1.1,2},
    {5.7,2.8,4.1,1.3,2},
    {6.3,3.3,6.0,2.5,3},
    {6.4,2.7,5.3,1.9,3},
    {6.8,3.0,5.5,2.1,3},
    {5.7,2.5,5.0,2.0,3},
    {5.8,2.8,5.1,2.4,3},
    {6.4,3.2,5.3,2.3,3},
    {6.5,3.0,5.5,1.8,3},
    {7.7,3.8,6.7,2.2,3},
    {7.7,2.6,6.9,2.3,3},
    {6.0,2.2,5.0,1.5,3},
    {6.9,3.2,5.7,2.3,3},
    {5.6,2.8,4.9,2.0,3},
    {7.7,2.8,6.7,2.0,3},
    {6.3,2.7,4.9,1.8,3},
    {6.7,3.3,5.7,2.1,3},
    {7.2,3.2,6.0,1.8,3},
    {6.2,2.8,4.8,1.8,3},
    {6.1,3.0,4.9,1.8,3},
    {6.4,2.8,5.6,2.1,3},
    {7.2,3.0,5.8,1.6,3},
    {7.4,2.8,6.1,1.9,3},
    {7.9,3.8,6.4,2.0,3},
    {6.4,2.8,5.6,2.2,3},
    {6.3,2.8,5.1,1.5,3},
    {6.1,2.6,5.6,1.4,3},
    {7.7,3.0,6.1,2.3,3},
    {6.3,3.4,5.6,2.4,3},
    {6.4,3.1,5.5,1.8,3},
    {6.0,3.0,4.8,1.8,3},
    {6.9,3.1,5.4,2.1,3},
    {6.7,3.1,5.6,2.4,3},
    {6.9,3.1,5.1,2.3,3},
    {5.8,2.7,5.1,1.9,3},
    {6.8,3.2,5.9,2.3,3},
    {6.7,3.3,5.7,2.5,3},
    {6.7,3.0,5.2,2.3,3},
    {6.3,2.5,5.0,1.9,3},
    {6.5,3.0,5.2,2.0,3},
    {6.2,3.4,5.4,2.3,3},
    {5.9,3.0,5.1,1.8,3}
};

float iris_test_set[30][5] = {
    {5.1,3.5,1.4,0.2,1},
    {4.9,3.0,1.4,0.2,1},
    {4.7,3.2,1.3,0.2,1},
    {4.6,3.1,1.5,0.2,1},
    {5.0,3.6,1.4,0.2,1},
    {5.4,3.9,1.7,0.4,1},
    {4.6,3.4,1.4,0.3,1},
    {5.0,3.4,1.5,0.2,1},
    {4.4,2.9,1.4,0.2,1},
    {4.9,3.1,1.5,0.1,1},
    {7.0,3.2,4.7,1.4,2},
    {6.4,3.2,4.5,1.5,2},
    {6.9,3.1,4.9,1.5,2},
    {5.5,2.3,4.0,1.3,2},
    {6.5,2.8,4.6,1.5,2},
    {5.7,2.8,4.5,1.3,2},
    {6.3,3.3,4.7,1.6,2},
    {4.9,2.4,3.3,1.0,2},
    {6.6,2.9,4.6,1.3,2},
    {5.2,2.7,3.9,1.4,2},
    {5.8,2.7,5.1,1.9,3},
    {7.1,3.0,5.9,2.1,3},
    {6.3,2.9,5.6,1.8,3},
    {6.5,3.0,5.8,2.2,3},
    {7.6,3.0,6.6,2.1,3},
    {4.9,2.5,4.5,1.7,3},
    {7.3,2.9,6.3,1.8,3},
    {6.7,2.5,5.8,1.8,3},
    {7.2,3.6,6.1,2.5,3},
    {6.5,3.2,5.1,2.0,3},
};


struct CMM {
    int * matrix;
    int rows;
    int cols;
};

struct CMM create_cmm(int rows, int cols) {
    struct CMM result;
    result.matrix = create_matrix(rows, cols);
    result.rows = rows;
    result.cols = cols;
    return result;
}

int * create_matrix(int rows, int cols) {
    return malloc(rows * cols * sizeof(int));
}

void print_cmm(struct CMM cmm) {
    for (int row=0; row < cmm.rows; row++) {
        for (int col=0; col < cmm.cols; col++) {
            printf("%d", cmm.matrix[cmm.cols*row + col]);
        }
        printf("\n");
    }
}

void insert(struct CMM cmm, int* data_vec, int* key_vec) {
    int * mat = create_matrix(cmm.rows, cmm.cols);
    
    // Multiply the data vector by the transpose of the key vector
    for (int row=0; row < cmm.rows; row++) {
        for (int col=0; col < cmm.cols; col++) {
            int ix = row * cmm.cols + col;
            mat[ix] = data_vec[row] * key_vec[col];
        }
    }

    // Copy new matrix onto existing cmm matrix
    for (int i=0; i < cmm.rows * cmm.cols; i++) {
        cmm.matrix[i] |= mat[i];
    }

    free(mat);
}

int * recall(struct CMM cmm, int * key_vector, int * output_vector) {
    for (int row = 0; row < cmm.rows; row++) {
        int sum = 0;

        for (int col = 0; col < cmm.cols; col++) {
            sum += cmm.matrix[row*cmm.cols + col] * key_vector[col];
        }

        output_vector[row] = sum >= 1;
    }
}

int binning_encode(float input, float min, float max, float bin_size) {
    /*
    printf("%f ", input);
    printf("%f ", min);
    printf("%f", max);
    printf("\n");
    */
    assert(min <= input);
    assert(input < max);
    int bin = (input - min) / bin_size;
    return bin;
}

void test() {
    printf("test\n");
    struct CMM cmm = create_cmm(3, 3);
    int data_vec[] = {1,0,1};
    int key_vec[] = {1,1,0};
    insert(cmm, data_vec, key_vec);
    print_cmm(cmm);

    printf("Recalling..\n");
    int output_vec[3];
    recall(cmm, key_vec, output_vec);
    for (int i=0; i < 3; ++i) {
        printf("%d", output_vec[i]);
    }
    printf("\n");

    printf("%d\n", binning_encode(9.99, 5.0, 10.0, 0.1));
}

void encode_iris_data(float input[], int * output, int * class_output) {
    const int bins_sepal_length = 37;
    const int bins_sepal_width = 25;
    const int bins_petal_length = 60;
    const int bins_petal_width = 25;

    int enc_sepal_length = binning_encode(input[0], 4.25, 7.95, 0.1);
    int enc_sepal_width  = binning_encode(input[1], 1.95, 4.45, 0.1);
    int enc_petal_length = binning_encode(input[2], 0.95, 6.95, 0.1);
    int enc_petal_width  = binning_encode(input[3], 0.05, 2.55, 0.1);

    printf("%d ", enc_sepal_length);
    printf("%d ", enc_sepal_width);
    printf("%d ", enc_petal_length);
    printf("%d ", enc_petal_width);
    printf("\n");

    for (int i=0; i < IRIS_ENCODING_SIZE; ++i) {
        output[i] = 0;
    }
    output[enc_sepal_length] = 1;
    output[bins_sepal_length + enc_sepal_width] = 1;
    output[bins_sepal_length + bins_sepal_width + enc_petal_length] = 1;
    output[bins_sepal_length + bins_sepal_width + bins_petal_length + enc_petal_width] = 1;

    for (int i=0; i < 3; ++i) {
        class_output[i] = 0;
        if (i == input[4]-1)
            class_output[i] = 1;
    }
}

void print_iris(float iris[]) {
    printf("%f, %f, %f, %f. Class %f\n", iris[0], iris[1], iris[2], iris[3], iris[4]);
}

void run_iris_experiment() {
    struct CMM cmm = create_cmm(3, IRIS_ENCODING_SIZE);

    // Training
    for (int i=0; i < 120; i++) {
        printf("Encoding %d\n", i);
        int encoding[IRIS_ENCODING_SIZE];
        int class_encoding[3];
        encode_iris_data(iris_train_set[i], encoding, class_encoding);
        /*
        for (int j=0; j < IRIS_ENCODING_SIZE; ++j) {
            printf("%d", encoding[j]);
        }
        printf("\n");
        for (int j=0; j < 3; ++j) {
            printf("%d", class_encoding[j]);
        }
        */
        insert(cmm, class_encoding, encoding);
    }

    // Testing
    int correct = 0;
    for (int i=0; i < 30; i++) {
        printf("Testing %d\n", i);
        int encoding[IRIS_ENCODING_SIZE];
        int class_encoding[3];
        int output[3];
        int predicted_class = 0;

        encode_iris_data(iris_test_set[i], encoding, class_encoding);
        recall(cmm, encoding, output);
        for (int j=0; j < 3; ++j) {
            if (output[j] != 0) {
                predicted_class = j+1;
                break;
            }
        }

        if (predicted_class == (int)iris_test_set[i][4])
            correct++;

        print_iris(iris_test_set[i]);
        printf("Predicted class: %d\n", predicted_class);
        printf("\n");
    }
    printf("%d out of %d correct.\n", correct, 30);

    //print_cmm(cmm);
}

int main(int argc, char** argv) {
    run_iris_experiment();
    return 0;
}
