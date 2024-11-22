cd .\training\
python .\main.py ^
--reprocess-data ^
--regenerate-dataset ^
--use-category ^
--use-full-candidate ^
--generate-tsne ^
--model-name NRMS-BERT ^
--eval-batch-size 16 ^
--mode valid ^
--ckpt-dir 2024-11-16T054407_ep10_16-16_cate

cd .\training\
python .\main.py ^
--reprocess-data ^
--regenerate-dataset ^
--use-full-candidate ^
--generate-tsne ^
--model-name NRMS-BERT ^
--eval-batch-size 16 ^
--mode valid ^
--ckpt-dir 2024-11-16T074143_ep10_16-16_uncate