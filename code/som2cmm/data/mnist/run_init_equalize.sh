../../c/bin/som --rows 50 \
    --cols 50 \
    --input-dims 784 \
    --train mnist_som_train_5000.txt \
    --train-file-class-index 784 \
    --save mnist5000.som \
    --weight-init-method equalize \
    --weight-equalize-value 0 \
    --normalize-inputs \
    --p1-iterations 1000 \
    --p1-learn-rate-initial 0.10 \
    --p1-learn-rate-final 0.01  \
    --p1-n-radius-initial 20.0 \
    --p1-n-radius-final 5.0 \
    --p2-iterations 1000 \
    --p2-learn-rate-initial 0.01 \
    --p2-learn-rate-final 0.01  \
    --p2-n-radius-initial 5.0 \
    --p2-n-radius-final 2.0 \
