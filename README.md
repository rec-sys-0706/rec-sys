# Personalized News Recommendation System
This is a FJU CSIE graduate project about a news recommendation system, with the model based on the NRMS-BERT paper. ✨ฅ^•ﻌ•^ฅ

## 1. Training Environment Setup
1. pip install -r requirements.txt
2. install CUDA for GPU training
3. Run ./run_model.py to train the model
## 2. Server Environment Setup
1. Setup environment variables of (SERVER_URL, BASE_URL, JWT_SECRET_KEY, SQL_SERVER, SQL_DATABASE, SQL_USERNAME, SQL_PASSWORD)
2. Run ./app.py to start the server
## 3. Dataset
1. The dataset is from the **MI**crosoft **N**ews **D**ataset (MIND), which can be downloaded from the [MIND website](https://msnews.github.io/).
2. The pre-trained embedding is from the Stanford GloVe, which can be downloaded from the [GloVe website](https://nlp.stanford.edu/projects/glove/).