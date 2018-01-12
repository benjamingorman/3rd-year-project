#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <regex.h>
#include <math.h>
#include <float.h>


struct SOM {
    int rows;
    int cols;
    int input_dims;
    float * data;
};

struct SOM create_SOM(int rows, int cols, int input_dims) {
    struct SOM result;
    result.rows = rows;
    result.cols = cols;
    result.input_dims = input_dims;
    result.data = (float *)malloc(sizeof(float) * input_dims * rows * cols);
    return result;
}

void free_SOM(struct SOM som) {
    free(som.data);
}

int get_num_weight_elements(struct SOM som) {
    return som.rows * som.cols * som.input_dims;
}

int save_SOM(struct SOM som, char* filepath) {
    FILE *f = fopen(filepath, "w");
    if (f == NULL) {
        printf("Error opening file: %s", filepath);
        return 1;
    }

    fprintf(f, "%d,%d,%d\n", som.rows, som.cols, som.input_dims);

    for (int i=0; i < get_num_weight_elements(som); ++i) {
        fprintf(f, "%f", som.data[i]);

        // Each neuron's weight matrix on a new line
        if ((i+1) % som.input_dims == 0)
            fprintf(f, "\n");
        else
            fprintf(f, ",");
    }

    fclose(f);
    return 0;
}

int load_SOM(struct SOM * som, char* filepath) {
    FILE *f = fopen(filepath, "r");
    if (f == NULL) {
        printf("Error opening file: %s", filepath);
        return 1;
    }


    // Treat first line separately
    char c;
    char buf[100];
    int ptr = 0;
    while ((c = fgetc(f)) != '\n') {
        buf[ptr++] = c;
    }
    buf[ptr] = '\0';
    sscanf(buf, "%d,%d,%d", &(som->rows), &(som->cols), &(som->input_dims));


    // Rest of file is just weight element data
    ptr = 0;
    int weight_element_ptr = 0;
    while ((c = fgetc(f)) != EOF) {
        //printf("%c", c);

        if (weight_element_ptr >= get_num_weight_elements(*som)) {
            printf("ERROR: number of weight elements exceeds expected amount\n");
            return 1;
        }

        if (c == ',' || c == '\n') {
            buf[ptr] = '\0';
            ptr = 0;
            sscanf(buf, "%f", &(som->data[weight_element_ptr++]));
        }
        else {
            buf[ptr++] = c;
        }
    }

    if (weight_element_ptr != get_num_weight_elements(*som)) {
        printf("ERROR: wrong number of weight elements\n");
        return 1;
    }

    return 0;
}

int read_input_file_line(FILE *f, float * input_vector, int dims) {
    char * line = NULL;
    size_t len = 0;
    ssize_t read;

    read = getline(&line, &len, f);

    int ret_code = 0;
    if (read == -1) {
        ret_code = 1;
    }
    else {
        /*
        printf("Read line of length %zu :\n", read);
        printf("%s", line);
        */

        int input_vector_ptr = 0;
        char number_buf[50];
        int number_buf_ptr = 0;
        char c;
        for (int i=0; i < read; ++i) {
            c = line[i];

            if (c == ',' || c == '\n') {
                number_buf[number_buf_ptr] = '\0';
                number_buf_ptr = 0;
                sscanf(number_buf, "%f", &(input_vector[input_vector_ptr++]));
            }
            else if (input_vector_ptr >= dims) {
                printf("ERROR: Input file has too many numbers on one row\n");
                ret_code = 1;
                break;
            }
            else {
                number_buf[number_buf_ptr++] = c;
            }
        }

        ret_code = 0;
    }

    free(line);
    return ret_code;
}

float euclidean_distance(float * vec_a, float * vec_b, int dims) {
    float sum_of_squares = 0;

    for (int i=0; i < dims; ++i) {
        sum_of_squares += powf(vec_a[i] - vec_b[i],  2.0);
    }

    return sqrtf(sum_of_squares);
}

float * get_neuron_weight_vector(struct SOM som, int index) {
    return &som.data[index * som.input_dims];
}

int find_bmu(struct SOM som, float * input_vec) {
    int num_neurons = som.rows * som.cols;
    int best_match = 0;
    float best_dist = FLT_MAX;

    for (int i=0; i < num_neurons; ++i) {
        float dist = euclidean_distance(input_vec, get_neuron_weight_vector(som, i), som.input_dims);
        //printf("neuron %d, dist %f\n", i, dist);
        if (dist < best_dist) {
            best_dist = dist;
            best_match = i;
        }
    }

    printf("bmu is %d\n", best_match);
    printf("best dist is %f\n", best_dist);

    return best_match;
}

void adjust_weights(struct SOM som, float * input_vec, int bmu, float training_rate) {

}

float neighbourhood_function(struct SOM som, int n1, int n2) {
    int n1_row = n1 / som.cols;
    int n1_col = n1 % som.cols;
    int n2_row = n2 / som.cols;
    int n2_col = n2 % som.cols;

    float n1_coords[2] = {n1_row, n1_col};
    float n2_coords[2] = {n2_row, n2_col};

    float origin[2] = {0.0, 0.0};
    float corner[2] = {som.rows, som.cols};
    
    float max_dist = euclidean_distance(origin, corner, 2);
    float n_dist = euclidean_distance(n1_coords, n2_coords, 2);
    float result = 1.0 - n_dist / max_dist;

    printf("(%d, %d) (%d, %d)\n", n1_row, n1_col, n2_row, n2_col);
    printf("n_dist %f\n", n_dist);
    printf("max_dist %f\n", max_dist);
    printf("result %f\n\n", result);

    return result;
}

// Expects CSV of floats as file format
void train_SOM(struct SOM som, char * train_set_filepath) {
    float input_vector[som.input_dims];
    float training_rate = 0.05;

    FILE *f = fopen(train_set_filepath, "r");
    if (f == NULL) {
        printf("Error opening file: %s", train_set_filepath);
        return;
    }

    int ret_code;
    while ((ret_code = read_input_file_line(f, input_vector, som.input_dims)) == 0) {
        printf("%f %f %f\n", input_vector[0], input_vector[1], input_vector[2]);

        int bmu = find_bmu(som, input_vector);
        adjust_weights(som, input_vector, bmu, training_rate);
    }
}

float rand_float_range(float min, float max) {
    float scale = rand() / (float) RAND_MAX; /* [0, 1.0] */
    return min + scale * ( max - min );      /* [min, max] */
}

void randomize_weight_vectors(struct SOM som, float min, float max) {
    for (int i=0; i < get_num_weight_elements(som); ++i) {
        som.data[i] = rand_float_range(min, max);
    }
}

void normalize_weight_vectors(struct SOM som, float value) {
    for (int i=0; i < get_num_weight_elements(som); ++i) {
        som.data[i] = value;
    }
}

int main(int argc, char** argv) {
    /*
    randomize_weight_vectors(som, -2, 2);
    train_SOM(som, "sample_input_file.txt");

    float x[3] = {0.0, 0.0, 0.0};
    find_bmu(som, x);
    */

    struct SOM som = create_SOM(10, 10, 3);
    neighbourhood_function(som, 0, 0);
    neighbourhood_function(som, 0, 1);
    neighbourhood_function(som, 0, 2);
    neighbourhood_function(som, 0, 5);
    neighbourhood_function(som, 0, 9);
    neighbourhood_function(som, 0, 10);

    return 0;
}
