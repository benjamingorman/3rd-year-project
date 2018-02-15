set -e

# This directory
EXP_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CODE_DIR="$HOME/modules/3rd-year-project/code/som2cmm"

#rm $EXP_DIR/tmp/*

cd $CODE_DIR

keys="$EXP_DIR/tmp/random_keys.txt"
values="$EXP_DIR/tmp/random_values.txt"
enc_keys="$EXP_DIR/tmp/encoded_random_keys.txt"
enc_values="$EXP_DIR/tmp/encoded_random_values.txt"
cmm_input="$EXP_DIR/tmp/cmm_input.txt"
cmm_out_dir="$EXP_DIR/dist"

python3 gen_random_patterns.py --out $keys --n 1000 --len 5 --min 0 --max 100
python3 gen_random_patterns.py --out $values --n 1000 --len 5 --min 0 --max 100

python3 -m "src.encode_input" \
    --input "$keys" \
    --output "$enc_keys" \
    --encoding quantize \
    --quantize-bits-per-attr 50 50 50 50 50 \
    --quantize-bits-set-per-attr 1 1 1 1 1

python3 -m "src.encode_input" \
    --input "$values" \
    --output "$enc_values" \
    --encoding quantize \
    --quantize-bits-per-attr 50 50 50 50 50 \
    --quantize-bits-set-per-attr 1 1 1 1 1

printf "" > $cmm_input
echo 250 >> $cmm_input
echo 250 >> $cmm_input
paste $enc_keys $enc_values -d : >> $cmm_input

python3 -m "src.cmm" \
    --input "$cmm_input" \
    --out-dir "$cmm_out_dir" \
    --bits-in-key 5

echo "DONE"
