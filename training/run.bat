cd .\training\
python .\main.py ^
--valid-test ^
--model-name NRMS-BERT ^
--epochs 10 ^
--learning-rate 1e-5 ^
--pretrained-model-name "distilbert-base-uncased" ^
--train-batch-size 16 ^
--mode valid ^
--max-dataset-size 10 ^
--eval-batch-size 1 ^
--ckpt-dir 2024-10-01T190040_ep10_16-32