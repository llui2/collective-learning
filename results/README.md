# Results

This directory contains the compact canonical outputs produced by GPU training.

The main sweep uses five seeds and five coupling strengths:

```text
lambda = 0, 0.25, 0.5, 1, 2
seed   = 0, 1, 2, 3, 4
steps  = 4000
```

On Minerva:

```bash
./experiment.sh
git add results/sweep_lambda*_seed*.json
git commit -m "update coupling sweep"
git push
```

`experiment.sh` skips completed run files, so restarting it resumes the sweep.

On the manuscript machine:

```bash
git pull
./build.sh
```

The build only regenerates the main figure when all 25 runs are available. Model checkpoints and temporary training artifacts do not belong here.
