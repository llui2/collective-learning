# Results

This directory contains the small canonical outputs produced by the training runs.

On the compute machine:

```bash
./compute.sh
git add results/lambda0.json results/lambda1.json
git commit -m "update experiment results"
git push
```

The manuscript machine pulls these files and regenerates figures from them with `./build.sh`.

Model checkpoints and temporary training artifacts do not belong here.
