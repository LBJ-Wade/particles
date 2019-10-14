# Particles

Code to solve for central configurations in 3 dimensions. 

## Torch

The pytorch version can be ran with e.g.

```
python minimize.py --particles 5000 --dim 3 --iters 50000 --log_dir logs/test
```

A GPU will be used if available, which greatly increases the speed. A current limitation is the memory constraint from calculating pairwise distances (around 10k particles). 

## N-Body

The N-body version uses an adapated version of the NVIDIA CUDA N-body code. First make 
```
make all
```
Can be ran with e.g.

```
./nbody -numbodies=200000 -fp64 -fullscreen
```


## Analyse 


Runs can be analysed by

```
python analyse.py --log_dir torch/logs/test
```

![positions](https://github.com/adammoss/particles/blob/master/positions.png)
![hist](https://github.com/adammoss/particles/blob/master/hist.png)
