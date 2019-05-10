# particles

Code to solve for central configurations in N dimension. Can be ran with e.g.

```
python minimize.py --particles 5000 --dim 3 --iters 50000 --log_dir logs/test
```

It will use an available GPU, and this greatly increases the speed. A current limitation is the memory constraint from calculating pairwise distances. 

Runs can be analysed by

```
python analyse.py --log_dir logs/test
```

![positions](https://github.com/adammoss/positions/blob/master/positions.png)
![hist](https://github.com/adammoss/positions/blob/master/hist.png)
