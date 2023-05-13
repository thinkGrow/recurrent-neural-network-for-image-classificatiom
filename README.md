#### Run Train:
<!-- --device cuda -->
```terminal
python train.py --batch_size 4 --num_epochs 4 --lr 1e-4 --num_workers 4  --device mps --save_dir results --save_interval 2
```

#### Run Test:

```terminal
python test.py --batch_size 16 --num_workers 4 --save_dir output --model_dir results/run_0/
```

#### Run Tensorboard:

```terminal
tensorboard --logdir=results/ --port=2171
```

#### Evaluate Results:

```terminal
python utils/evaluate.py --stitched_path stitched_output --hr_path data/DIV2K_train_HR
```
