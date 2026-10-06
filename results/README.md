# Results

This directory contains the small canonical outputs produced by the GPU training runs.

On Minerva:

```bash
./experiment.sh
git add results/lambda0.json results/lambda1.json
git commit -m "update experiment results"
git push
```

On the manuscript machine:

```bash
git pull
./build.sh
```

The build regenerates the manuscript figure from these files. Model checkpoints and temporary training artifacts do not belong here.
