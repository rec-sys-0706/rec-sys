cd .\training\
python .\main.py --model-name NRMS-BERT ^
--epochs 10 ^
--learning-rate 3e-5 ^
--pretrained-model-name "distilbert-base-uncased" ^
--train-batch-size 16 ^
--eval-batch-size 32

python .\main.py --model-name NRMS-Glove --epochs 20
python .\main.py --model-name NRMS --epochs 20
python .\main.py --model-name NRMS-Glove --epochs 10
python .\main.py --model-name NRMS --epochs 10