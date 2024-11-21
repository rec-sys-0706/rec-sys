cd .\training\
python .\main.py --model-name NRMS-BERT ^
--epochs 10 ^
--learning-rate 1e-5 ^
--pretrained-model-name "distilbert-base-uncased" ^
--train-batch-size 16 ^
--eval-batch-size 16

python .\main.py --model-name NRMS-Glove --epochs 20
python .\main.py --model-name NRMS --epochs 20
python .\main.py --model-name NRMS-Glove --epochs 10
python .\main.py --model-name NRMS --epochs 10

@REM Category
python .\run_model.py ^
--reprocess-data ^
--regenerate-dataset ^
--use-category ^
--model-name NRMS-BERT ^
--epochs 10 ^
--learning-rate 1e-5 ^
--pretrained-model-name "distilbert-base-uncased" ^
--train-batch-size 16 ^
--eval-batch-size 16 ^
--mode train

python .\run_model.py 
--reprocess-data ^
--regenerate-dataset ^
--model-name NRMS-BERT ^
--epochs 10 ^
--learning-rate 1e-5 ^
--pretrained-model-name "distilbert-base-uncased" ^
--train-batch-size 16 ^
--eval-batch-size 16 ^
--mode train