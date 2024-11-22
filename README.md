tst
tensorboard --logdir=runs
```bash
wget https://nlp.stanford.edu/data/glove.840B.300d.zip
```

### Arguments
```
--model-name NRMS ^
--reprocess-data ^
--regenerate-dataset ^
--use-category ^
--epochs 10 ^
--learning-rate 3e-5 ^
--train-batch-size 4 ^
--eval-batch-size 4 ^
--mode test ^
--max-dataset-size 10 ^
--ckpt-dir 2024-11-15T005604_ep10_16-16
```


## Credits
 - Dataset by **MI**crosoft **N**ews **D**ataset (MIND), see <https://msnews.github.io/>.
 - Pretrained embedding by Stanford GloVe, see <https://nlp.stanford.edu/projects/glove/>.