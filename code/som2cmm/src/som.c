#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <regex.h>
#include <math.h>
#include <float.h>
#include <stdbool.h>
#include <ctype.h>

#define STATUS_OK 0
#define STATUS_ERROR 1
#define STATUS_EMPTY_LINE 2
#define STATUS_EOF 3
#define LINE_BUF_SIZE 2048


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
        printf("Error opening file: %s\n", filepath);
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

    printf("Saved SOM to \"%s\"\n", filepath);
    fclose(f);
    return STATUS_OK;
}

int load_SOM(struct SOM * som, char* filepath) {
    FILE *f = fopen(filepath, "r");
    if (f == NULL) {
        printf("Error opening file: %s\n", filepath);
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

    fclose(f);
    return STATUS_OK;
}

bool is_line_empty(const char *s) {
    while (*s != '\0') {
        if (!isspace((unsigned char)*s))
            return false;
        s++;
    }
    return true;
}

int read_input_file_line(FILE *fp, char * line) {
    size_t buf_size = LINE_BUF_SIZE;
    ssize_t num_chars_read = getline(&line, &buf_size, fp);

    if (num_chars_read == -1) {
        return STATUS_EOF;
    }
    else if (is_line_empty(line)) {
        return STATUS_EMPTY_LINE;
    }
    else {
        return STATUS_OK;
    }
}

int parse_input_line(char * line, int class_index, float * vector, int dims) {
    char * token = strtok(line, ",");
    int token_index = 0;
    int vec_index = 0;

    while(token != NULL) {
        //printf("index %d, token: %s\n", token_index, token);
        if (token_index != class_index) {
            vector[vec_index++] = atof(token);
        }
        token = strtok(NULL, ",");
        token_index++;
    }

    return STATUS_OK;;
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
    return start + (end - start) * pct;
}

float get_linear_blend(float start, float end, float middle) {
    return (middle - start) / (end - start);
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

void find_normalize_minima_maxima(
        char * train_file,
        int train_file_class_index,
        int input_dims,
        float * normalize_minima,
        float * normalize_maxima
        ) {
    FILE *fp = fopen(train_file, "r");
    if (fp == NULL) {
        printf("Error opening file: %s\n", train_file);
        return;
    }

    for (int i = 0; i < input_dims; i++) {
        normalize_minima[i] = FLT_MAX;
        normalize_maxima[i] = FLT_MIN;
    }

    int status; 
    char * input_line_buf = (char *)malloc(LINE_BUF_SIZE);
    float input_vector[input_dims];

    while (true) {
        status = read_input_file_line(fp, input_line_buf);
        if (status == STATUS_EMPTY_LINE)
            continue;
        else if (status == STATUS_EOF)
            break;
        else {
            parse_input_line(input_line_buf, train_file_class_index, input_vector, input_dims);

            for (int i = 0; i < input_dims; i++) {
                if (input_vector[i] < normalize_minima[i])
                    normalize_minima[i] = input_vector[i];
                else if (input_vector[i] > normalize_maxima[i])
                    normalize_maxima[i] = input_vector[i];
            }
        }
    }

    free(input_line_buf);
}

void normalize_input_vector(int dims, float * input_vector, float * normalize_minima, float * normalize_maxima) {
    for (int i=0; i < dims; ++i) {
        float min = normalize_minima[i];
        float max = normalize_maxima[i];
        input_vector[i] = get_linear_blend(min, max, input_vector[i]);
    }
}

void denormalize_neuron_weight_vector(
        struct SOM som,
        int neuron,
        float * normalize_minima,
        float * normalize_maxima
        ) {
    float * weight_vec = get_neuron_weight_vector(som, neuron);

    for (int i = 0; i < som.input_dims; i++) {
        float min = normalize_minima[i];
        float max = normalize_maxima[i];
        weight_vec[i] = linear_blend(min, max, weight_vec[i]);
    }
}

void denormalize_som(struct SOM som, float * normalize_minima, float * normalize_maxima) {
    int num_neurons = som.rows * som.cols;
    for (int i = 0; i < num_neurons; i++) {
        denormalize_neuron_weight_vector(som, i, normalize_minima, normalize_maxima);
    }
}

// Expects CSV of floats as file format
// If classes are included then the index can be specified and it will be ignored
void train_SOM(
        struct SOM som,
        struct SOMTrainingParams params,
        int phase,
        char * train_filepath,
        int train_file_class_index,
        bool normalize_inputs,
        float * normalize_minima, // min values of each input dimension
        float * normalize_maxima // max values of each input dimension
        )
{
    printf("===== TRAINING Phase %d =====\n", phase);
    FILE *fp = fopen(train_filepath, "r");
    if (fp == NULL) {
        printf("Error opening file: %s\n", train_filepath);
        return;
    }

    float input_vector[som.input_dims];
    char * input_line_buf = (char *)malloc(LINE_BUF_SIZE);
    int iteration = 0;
    int file_rewinds = 0;

    for (int i=0; (iteration = i-file_rewinds) < params.iterations; ++i) {
        float progress;
        if (params.iterations == 1)
            progress = 0;
        else
            progress = iteration/(float)(params.iterations-1);

        int status = read_input_file_line(fp, input_line_buf);

        if (status == STATUS_ERROR) {
            printf("ERROR: Could not read input file\n");
            break;
        }
        else if (status == STATUS_EMPTY_LINE) {
            printf("Skipping empty line in input file...\n");
            continue;
        }
        else if (status == STATUS_EOF) {
            // If we get to the end of the input file then cycle back to the beginning and read a new line
            rewind(fp);
            file_rewinds++;
            //printf("Rewinding input file...\n");
            continue;
        }
        assert(status == STATUS_OK);

        parse_input_line(input_line_buf, train_file_class_index, input_vector, som.input_dims);

        if (normalize_inputs)
            normalize_input_vector(som.input_dims, input_vector, normalize_minima, normalize_maxima);

        float learn_rate = linear_blend(params.learn_rate_initial,
                                  params.learn_rate_final, progress);
        float n_radius = linear_blend(params.n_radius_initial,
                                params.n_radius_final, progress);

        if (iteration % 100 == 0)
            printf("Progress: %0.1f%%, Iteration %d, learn_rate %f, n_radius %f\r",
                    progress*100, iteration, learn_rate, n_radius);

        int bmu = find_bmu(som, input_vector);
        adjust_weights(som, input_vector, bmu, learn_rate, n_radius);
    }

    free(input_line_buf);
    fclose(fp);
    printf("\n\n");
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

void equalize_weight_vectors(struct SOM som, float value) {
    for (int i=0; i < get_num_weight_elements(som); ++i) {
        som.data[i] = value;
    }
}

// Distribute each weight linearly across the range of inputs for each dimension
void intelligently_randomize_weight_vectors(
        struct SOM som,
        char * input_file,
        int input_file_class_index)
{
    FILE *fp = fopen(input_file, "r");
    if (fp == NULL) {
        printf("Error opening file: %s\n", input_file);
        return;
    }

    float min_values[som.input_dims];
    float max_values[som.input_dims];

    for (int i=0; i < som.input_dims; ++i) {
        min_values[i] = FLT_MAX;
        max_values[i] = FLT_MIN;
    }

    float input_vector[som.input_dims];
    char * input_line_buf = (char *)malloc(LINE_BUF_SIZE);

    int status;
    while ((status = read_input_file_line(fp, input_line_buf)) == STATUS_OK) {
        parse_input_line(input_line_buf, input_file_class_index, input_vector, som.input_dims);

        for (int i=0; i < som.input_dims; ++i) {
            float x = input_vector[i];
            if (x < min_values[i]) {
                min_values[i] = x;
            }
            if (x > max_values[i]) {
                max_values[i] = x;
            }
        }

    }

    int num_neurons = som.rows * som.cols;
    for (int n=0; n < num_neurons; ++n) {
        float * weight_vector = get_neuron_weight_vector(som, n);
        
        for (int i=0; i < som.input_dims; ++i) {
            weight_vector[i] = rand_float_range(min_values[i], max_values[i]);
        }
    }

    free(input_line_buf);
    fclose(fp);
}

void print_neuron_weights(struct SOM som, int neuron) {
    float * weight_vec = get_neuron_weight_vector(som, neuron);

    for (int i=0; i < som.input_dims; ++i) {
        printf("%f, ", weight_vec[i]);
    }
    printf("\n");
}

void print_vector(float * vec, int dims) {
    printf("(");
    for (int i = 0; i < dims; i++) {
        printf("%.2f", vec[i]);
        if (i != dims - 1) {
            printf(", ");
        }
    }
    printf(")");
}

int main(int argc, char** argv) {
    int opt_rows = 10;
    int opt_cols = 10;
    int opt_input_dims = 3;
    int opt_train_file_class_index = -1; // the index of the class for each pattern, if used
    char opt_train_file[128] = "data/default_train_file.txt";
    char opt_save_file[128] = "trained/default_save_file.som";
    char opt_weight_init_method[128] = "intelligent";
    float opt_weight_equalize_val = 0.0;
    float opt_weight_randomize_min = -1.0;
    float opt_weight_randomize_max = 1.0;
    bool opt_normalize_inputs = true;

    // Training happens in two phases, p1 is loose and p2 is tight
    struct SOMTrainingParams p1_params = create_SOMTrainingParams();
    struct SOMTrainingParams p2_params = create_SOMTrainingParams();

    for (int i=0; i < argc; ++i) {
        char * opt = argv[i];
        char * arg;

        if (opt[0] == '-' && opt[1] == '-') {
            // It's an option so it must be followed by an argument
            assert(argc > i);
        }

        if (i + 1 < argc) {
            arg = argv[i+1];
        }

        if (strcmp(opt, "--rows") == 0) {
            opt_rows = atoi(arg);
        }
        else if (strcmp(opt, "--cols") == 0) {
            opt_cols = atoi(arg);
        }
        else if (strcmp(opt, "--input-dims") == 0) {
            opt_input_dims = atoi(arg);
        }
        else if (strcmp(opt, "--train") == 0) {
            strcpy(opt_train_file, arg);
        }
        else if (strcmp(opt, "--train-file-class-index") == 0) {
            opt_train_file_class_index = atoi(arg);
        }
        else if (strcmp(opt, "--save") == 0) {
            strcpy(opt_save_file, arg);
        }
        else if (strcmp(opt, "--weight-init-method") == 0) {
            strcpy(opt_weight_init_method, arg);
        }
        else if (strcmp(opt, "--weight-equalize-val") == 0) {
            opt_weight_equalize_val = atof(arg);
        }
        else if (strcmp(opt, "--weight-randomize-min") == 0) {
            opt_weight_randomize_min = atof(arg);
        }
        else if (strcmp(opt, "--weight-randomize-max") == 0) {
            opt_weight_randomize_max = atof(arg);
        }
        else if (strcmp(opt, "--normalize-inputs") == 0) {
            opt_normalize_inputs = true;
        }
        else if (strcmp(opt, "--p1-iterations") == 0) {
            p1_params.iterations = atoi(arg);
        }
        else if (strcmp(opt, "--p2-iterations") == 0) {
            p2_params.iterations = atoi(arg);
        }
        else if (strcmp(opt, "--p1-learn-rate-initial") == 0) {
            p1_params.learn_rate_initial = atof(arg);
        }
        else if (strcmp(opt, "--p2-learn-rate-initial") == 0) {
            p2_params.learn_rate_initial = atof(arg);
        }
        else if (strcmp(opt, "--p1-learn-rate-final") == 0) {
            p1_params.learn_rate_final = atof(arg);
        }
        else if (strcmp(opt, "--p2-learn-rate-final") == 0) {
            p2_params.learn_rate_final = atof(arg);
        }
        else if (strcmp(opt, "--p1-n-radius-initial") == 0) {
            p1_params.n_radius_initial = atof(arg);
        }
        else if (strcmp(opt, "--p2-n-radius-initial") == 0) {
            p2_params.n_radius_initial = atof(arg);
        }
        else if (strcmp(opt, "--p1-n-radius-final") == 0) {
            p1_params.n_radius_final = atof(arg);
        }
        else if (strcmp(opt, "--p2-n-radius-final") == 0) {
            p2_params.n_radius_final = atof(arg);
        }
    }

    printf("Set rows: %d\n", opt_rows);
    printf("Set input_dims: %d\n", opt_input_dims);
    printf("Set cols: %d\n", opt_cols);
    printf("Set train file: \"%s\"\n", opt_train_file);
    printf("Set train file class index to: %d\n", opt_train_file_class_index);
    printf("Set save file: \"%s\"\n", opt_save_file);
    printf("Set weight init method: %s\n", opt_weight_init_method);
    printf("Set weight equalize val: %f\n", opt_weight_equalize_val);
    printf("Set weight randomize min: %f\n", opt_weight_randomize_min);
    printf("Set weight randomize max: %f\n", opt_weight_randomize_max);
    printf("Set p1 iterations: %d\n", p1_params.iterations);
    printf("Set p2 iterations: %d\n", p2_params.iterations);
    printf("Set p1 learn_rate_initial: %f\n", p1_params.learn_rate_initial);
    printf("Set p2 learn_rate_initial: %f\n", p2_params.learn_rate_initial);
    printf("Set p1 learn_rate_final: %f\n", p1_params.learn_rate_final);
    printf("Set p2 learn_rate_final: %f\n", p2_params.learn_rate_final);
    printf("Set p1 n_radius_initial: %f\n", p1_params.n_radius_initial);
    printf("Set p2 n_radius_initial: %f\n", p2_params.n_radius_initial);
    printf("Set p1 n_radius_final: %f\n", p1_params.n_radius_final);
    printf("Set p2 n_radius_final: %f\n", p2_params.n_radius_final);
    printf("Set normalize inputs: %d\n", opt_normalize_inputs);
    printf("\n");

    struct SOM som = create_SOM(opt_rows, opt_cols, opt_input_dims);

    if (strcmp(opt_weight_init_method, "intelligent") == 0) {
        intelligently_randomize_weight_vectors(som, opt_train_file, opt_train_file_class_index);
    }
    else if (strcmp(opt_weight_init_method, "randomize") == 0) {
        randomize_weight_vectors(som, opt_weight_randomize_min, opt_weight_randomize_max);
    }
    else if (strcmp(opt_weight_init_method, "equalize") == 0) {
        equalize_weight_vectors(som, opt_weight_equalize_val);
    }

    float normalize_minima[opt_input_dims];
    float normalize_maxima[opt_input_dims];

    if (opt_normalize_inputs) {
        find_normalize_minima_maxima(opt_train_file, opt_train_file_class_index, opt_input_dims,
                                     normalize_minima, normalize_maxima);

        printf("Normalize minima: ");
        print_vector(normalize_minima, opt_input_dims);
        printf("\n");

        printf("Normalize maxima: ");
        print_vector(normalize_maxima, opt_input_dims);
        printf("\n");
    }

    train_SOM(som, p1_params, 1, opt_train_file, opt_train_file_class_index,
              opt_normalize_inputs, normalize_minima, normalize_maxima);
    train_SOM(som, p2_params, 2, opt_train_file, opt_train_file_class_index,
              opt_normalize_inputs, normalize_minima, normalize_maxima);

    if (opt_normalize_inputs) {
        denormalize_som(som, normalize_minima, normalize_maxima);
    }

    save_SOM(som, opt_save_file);

    return 0;
}
