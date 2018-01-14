#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <regex.h>
#include <math.h>
#include <float.h>

#define STATUS_OK 0
#define STATUS_ERROR 1
#define STATUS_EOF 2


struct SOM {
    int rows;
    int cols;
    int input_dims;
    float * data;
};

struct SOMTrainingParams {
    int iterations;
    float learn_rate_initial;
    float learn_rate_final;
    float n_radius_initial;
    float n_radius_final;
};

struct SOM create_SOM(int rows, int cols, int input_dims) {
    struct SOM result;
    result.rows = rows;
    result.cols = cols;
    result.input_dims = input_dims;
    result.data = (float *)malloc(sizeof(float) * input_dims * rows * cols);
    return result;
}

struct SOMTrainingParams create_SOMTrainingParams() {
    struct SOMTrainingParams result;
    result.iterations = 1000;
    result.learn_rate_initial = 0.10;
    result.learn_rate_final = 0.01;
    result.n_radius_initial = 5.0;
    result.n_radius_final = 2.0;
    return result;
}

void free_SOM(struct SOM som) {
    free(som.data);
}

int get_num_weight_elements(struct SOM som) {
    return som.rows * som.cols * som.input_dims;
}

// Returns a pointer to the start of a particular neuron's weight vector
float * get_neuron_weight_vector(struct SOM som, int index) {
    return &som.data[index * som.input_dims];
}

int save_SOM(struct SOM som, char* filepath) {
    FILE *f = fopen(filepath, "w");
    if (f == NULL) {
        printf("Error opening file: %s", filepath);
        return STATUS_ERROR;
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
    return STATUS_OK;
}

int load_SOM(struct SOM * som, char* filepath) {
    FILE *f = fopen(filepath, "r");
    if (f == NULL) {
        printf("Error opening file: %s", filepath);
        return STATUS_ERROR;
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
            return STATUS_ERROR;
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
        return STATUS_ERROR;
    }

    return STATUS_OK;
}

int read_input_file_line(FILE *f, float * input_vector, int dims) {
    char * line = NULL;
    size_t len = 0;
    ssize_t read;

    read = getline(&line, &len, f);

    int ret_code = 0;
    if (read == -1) {
        ret_code = STATUS_EOF;
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

// Computes b - a as vectors
void calc_vector_difference(float * a, float * b, float * out, int len) {
    for (int i=0; i < len; ++i) {
        out[i] = b[i] - a[i];
    }
}

void do_vector_add(float * a, float * b, float * out, int len) {
    for (int i=0; i < len; ++i) {
        out[i] = a[i] + b[i];
    }
}

void do_scalar_vector_mul(float c, float * vec, float * out, int len) {
    for (int i=0; i < len; ++i) {
        out[i] = c * vec[i];
    }
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

    //printf("bmu is %d\n", best_match);
    //printf("best dist is %f\n", best_dist);

    return best_match;
}

float neighbourhood_function(float dist, float r) {
    float e = (float)M_E;
    return powf(e, -0.5 * powf(dist / r, 2));
}

float neuron_distance(struct SOM som, int n1, int n2) {
    int n1_row = n1 / som.cols;
    int n1_col = n1 % som.cols;
    int n2_row = n2 / som.cols;
    int n2_col = n2 % som.cols;

    float n1_coords[2] = {n1_row, n1_col};
    float n2_coords[2] = {n2_row, n2_col};

    return euclidean_distance(n1_coords, n2_coords, 2);
}

float neuron_neighbourhood_function(struct SOM som, int n1, int n2, float n_radius) {
    float dist = neuron_distance(som, n1, n2);
    return neighbourhood_function(dist, n_radius);
}

float linear_blend(float start, float end, float pct) {
    assert(0.0 <= pct && pct <= 1.0);
    return start + (end - start) * pct;
}

// Adds vec * scalar to the given neuron's weight matrix
void adjust_neuron_weight_vector(struct SOM som, int neuron, float * vec, float scalar) {
    float vec_scaled[som.input_dims];
    float * weight_vec = get_neuron_weight_vector(som, neuron);

    do_scalar_vector_mul(scalar, vec, vec_scaled, som.input_dims);
    do_vector_add(weight_vec, vec_scaled, weight_vec, som.input_dims);
}

void adjust_weights(struct SOM som, float * input_vec, int bmu, float learn_rate, float n_radius) {
    int num_neurons = som.rows * som.cols;
    float neighbour_value;
    float delta_vec[som.input_dims];
    float * bmu_vec = get_neuron_weight_vector(som, bmu);
    float * other_vec;

    calc_vector_difference(bmu_vec, input_vec, delta_vec, som.input_dims);

    for (int neuron=0; neuron < num_neurons; neuron++) {
        neighbour_value = neuron_neighbourhood_function(som, bmu, neuron, n_radius);

        // TODO: neighbour value cutoff?
        adjust_neuron_weight_vector(som, neuron, delta_vec, learn_rate * neighbour_value);
    }
}


// Expects CSV of floats as file format
void train_SOM(struct SOM som, struct SOMTrainingParams params,
        char * train_set_filepath) {
    FILE *input_file = fopen(train_set_filepath, "r");
    if (input_file == NULL) {
        printf("Error opening file: %s", train_set_filepath);
        return;
    }

    float learn_rate;
    float n_radius;
    int ret_code;
    float progress;
    float input_vector[som.input_dims];

    for (int i=0; i < params.iterations; ++i) {
        progress = i/(params.iterations-1);
        printf("Progress: %0.0f%%, Iteration %d\r", progress*100, i);

        learn_rate = linear_blend(params.learn_rate_initial,
                                  params.learn_rate_final, progress);
        n_radius = linear_blend(params.n_radius_initial,
                                params.n_radius_final, progress);

        ret_code = read_input_file_line(input_file, input_vector, som.input_dims);
        if (ret_code == STATUS_ERROR) {
            printf("ERROR: Could not read input file\n");
            break;
        }
        else if (ret_code == STATUS_EOF) {
            // If we get to the end of the input file then cycle back to the beginning and read a new line
            rewind(input_file);
            ret_code = read_input_file_line(input_file, input_vector, som.input_dims);
        }
        assert(ret_code == STATUS_OK);

        int bmu = find_bmu(som, input_vector);
        adjust_weights(som, input_vector, bmu, learn_rate, n_radius);
    }
    printf("\n");
}

float rand_float_range(float min, float max) {
    float scale = rand() / (float) RAND_MAX; /* [0, 1.0] */
    return min + scale * (max - min);      /* [min, max] */
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

void print_neuron_weights(struct SOM som, int neuron) {
    float * weight_vec = get_neuron_weight_vector(som, neuron);

    for (int i=0; i < som.input_dims; ++i) {
        printf("%f, ", weight_vec[i]);
    }
    printf("\n");
}

int main(int argc, char** argv) {
    /*
    randomize_weight_vectors(som, -2, 2);
    train_SOM(som, "sample_input_file.txt");

    float x[3] = {0.0, 0.0, 0.0};
    find_bmu(som, x);

    struct SOM som = create_SOM(10, 10, 3);
    struct SOMTrainingParams params = create_SOMTrainingParams();
    normalize_weight_vectors(som, 0);
    train_SOM(som, params, "sample_input_file.txt");
    save_SOM(som, "foo.som");
    */

    struct SOM som = create_SOM(100, 100, 4);
    randomize_weight_vectors(som, -2, 2);
    save_SOM(som, "big_random.som");

    return 0;
}
