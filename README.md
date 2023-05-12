#### Run Train:
<!-- --device cuda -->
```bash
python train.py --batch_size 16 --num_epochs 1 --lr 1e-4 --num_workers 4  --save_dir results --save_interval 2
```

#### Run Test:

```bash
$ python test.py --data_dir data --scale_factor 4 --patch_size 48 --batch_size 32 --num_workers 4 --device cuda --save_dir output --model_dir results/run_0/
```

#### Run Tensorboard:

```bash
tensorboard --logdir=results/ --port=2171
```

#### Evaluate Results:

```bash
$ python utils/evaluate.py --stitched_path stitched_output --hr_path data/DIV2K_train_HR
```
