#### Run Train:
<!-- --device cuda -->
```terminal
python train.py --batch_size 8 --num_epochs 4 --lr 1e-4 --num_workers 3  --device cpu --save_dir results --save_interval 2
```

#### Run Test:

```terminal
python test.py --batch_size 16 --num_epochs 2 --num_workers 4 --save_dir output --confusion_dir confusion --model_dir results/run_12/
!python3 test.py --batch_size 16 --num_epochs 2 --num_workers 4 --save_dir output --confusion_dir confusion --model_dir results/run_12/
```

#### Run Tensorboard:

```terminal
tensorboard --logdir=results/ --port=2171
tensorboard --logdir=output/ --port=2171
```

#### Evaluate Results:

```terminal
python utils/evaluate.py --stitched_path stitched_output --hr_path data/DIV2K_train_HR
```
