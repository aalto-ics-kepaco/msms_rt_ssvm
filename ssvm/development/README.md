# Inspect results using Tensorboard

## Filter "Objectives" or "Top-k"

```Objective Functions/(Dual|Primal)$```

```Metric \((Training|Validation)\)/Top-1 \((training|validation)\)```

## Compare different batch sizes
```C=64_bsize=\d_ninit=1$```

## Compare different regularization parameters
```C=\d*_bsize=1_ninit=1$```

## Compare different number of initial variables
```C=64_bsize=4_ninit=\d+```
