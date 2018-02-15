set -e

# This directory
EXP_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CODE_DIR="$HOME/modules/3rd-year-project/code/som2cmm"
cd $CODE_DIR

keys="$EXP_DIR/random_keys.txt"
values="$EXP_DIR/random_values.txt"

#for bits_per_attr in 10 20 30 40 50; do
#    for bits_set_per_attr in 1 2 3 4 5; do
for bits_per_attr in 60 70 80 90 100; do
    for bits_set_per_attr in 1 2 3 4 5; do
        work="$EXP_DIR/$bits_per_attr-$bits_set_per_attr"
        mkdir -p $work
        echo "Running $work"

        python3 -m "src.encode_input" \
            --input "$keys" \
            --output "$work/enc_keys" \
            --encoding quantize \
            --quantize-bits-per-attr $bits_per_attr $bits_per_attr $bits_per_attr $bits_per_attr $bits_per_attr \
            --quantize-bits-set-per-attr $bits_set_per_attr $bits_set_per_attr $bits_set_per_attr $bits_set_per_attr $bits_set_per_attr
        python3 -m "src.encode_input" \
            --input "$values" \
            --output "$work/enc_values" \
            --encoding quantize \
            --quantize-bits-per-attr $bits_per_attr $bits_per_attr $bits_per_attr $bits_per_attr $bits_per_attr \
            --quantize-bits-set-per-attr $bits_set_per_attr $bits_set_per_attr $bits_set_per_attr $bits_set_per_attr $bits_set_per_attr

        let key_size="$bits_per_attr * 5"
        let data_size="$bits_per_attr * 5"
        let key_num_bits="$bits_set_per_attr * 5"
        printf "" > "$work/cmm_input"
        echo $key_size >> "$work/cmm_input"
        echo $data_size >> "$work/cmm_input"
        paste "$work/enc_keys" "$work/enc_values" -d : >> "$work/cmm_input"

        python3 -m "src.cmm" \
            --input "$work/cmm_input" \
            --out-dir "$work" \
            --bits-in-key $key_num_bits
    done
done

echo "DONE"
