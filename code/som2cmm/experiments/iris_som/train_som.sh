../../c/bin/som --rows 10 \
    --cols 10 \
    --input-dims 4 \
    --train iris.data.train120 \
    --train "/home/userfs/b/bg739/modules/3rd-year-project/code/som2cmm/data/iris/iris.data.train120" \
    --train-file-class-index 4 \
    --save iris.som \
    --weight-init-method equalize \
    --weight-equalize-value 0 \
    --normalize-inputs \
    --p1-iterations 1000 \
    --p1-learn-rate-initial 0.10 \
    --p1-learn-rate-final 0.01  \
    --p1-n-radius-initial 5.0 \
    --p1-n-radius-final 2.0 \
    --p2-iterations 50000 \
    --p2-learn-rate-initial 0.01 \
    --p2-learn-rate-final 0.01  \
    --p2-n-radius-initial 2.0 \
    --p2-n-radius-final 2.0 \
