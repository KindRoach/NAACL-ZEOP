## Environment

Linux's system with CUDA devices required. 

Note: argostranslate is not available on Windows yet.

### Python packages

```
conda create --name ZECL python=3.8
conda activate ZECL
conda install -c pytorch pytorch=1.10.0 cudatoolkit=11.3
conda install -c huggingface transformers=4.11.3 
conda install -c conda-forge spacy=3.2.0 cupy=9.6.0
conda install numpy scikit-learn tqdm pandas tensorboard
pip install argostranslate
python -m spacy download en_core_web_sm
```

### Download transformers model

download page: https://huggingface.co/models

put the mode under path ```pretrain/bert```

```
pretrain
├─bert
│  └─bert-base-uncased
```

Or you could edit code in ```tool/pretrain_model_helper.py``` to download automatically when running code.

### Download argostranslate model (optional)

Only required for data pre-process.

Download page: https://www.argosopentech.com/argospm/index/

Put the mode under path ```pretrain/argos```

```
pretrain
├─argos
│      translate-en_zh-1_1.argosmodel
│      translate-zh_en-1_1.argosmodel
```

## Experiments

### pre-process data (optional)

This step only need run onetime, and processed data would be saved to path ```out/processed_data```

Processed data is already included in git repo, and this step is skipped automatically because pt file is exist.

If you want precessed data by yourself:
- Delete all contents under ```out/processed_data``` .
- Retrive full version of data from dataset soure and put them under path ```data/Ace``` and ```data/FewShotED```.
```
data
├─Ace
│      dev_docs.json
│      test_docs.json
│      train_docs.json
│
└─FewShotED
        Few-Shot_ED.json
```

- Run follow commond.
```
export ARGOS_DEVICE_TYPE=cuda
python -m data_process.gen_train_data
```


### Train model

```
python train.py
```

Edit the code in ```train.py```  to change the setting of Experiments.

The trained model would be saved to path ```out/checkpoints```

### Test model

```
python test.py
```


You need first add model checkpoints name to main function of ```test.py``` .
