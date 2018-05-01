data="/home/userfs/b/bg739/modules/3rd-year-project/code/som2cmm/experiments/mnist/"

../../c/bin/som --rows 1518 \
    --cols 1518 \
    --input-dims 784 \
    --train "$data/mnist_som_train_5000.txt" \
    --train-file-class-index 784 \
    --save mnist5000.som \
    --weight-init-method equalize \
    --weight-equalize-value 0 \
    --p1-iterations 1000 \
    --p1-learn-rate-initial 0.10 \
    --p1-learn-rate-final 0.01  \
    --p1-n-radius-initial 800.0 \
    --p1-n-radius-final 100.0 \
    --p2-iterations 1000 \
    --p2-learn-rate-initial 0.01 \
    --p2-learn-rate-final 0.01  \
    --p2-n-radius-initial 100.0 \
    --p2-n-radius-final 50.0 \
    --normalize-inputs
