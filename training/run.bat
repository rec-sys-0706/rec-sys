cd .\training\
python .\main.py --model-name NRMS-BERT ^
--epochs 10 ^
--learning-rate 3e-5 ^
--pretrained-model-name "distilbert-base-uncased" ^
--train-batch-size 16 ^
--eval-batch-size 16