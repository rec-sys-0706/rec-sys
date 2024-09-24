cd .\training\
python .\train.py --model-name NRMS-BERT --epochs 10 --pretrained-model-name "distilbert-base-uncased" --train-batch-size 16 eval-batch-size 64