#  WORK IN PROGRESS
## TRAIN THE MODEL
Ensure the `results` directory is present. Run:
```
python run_code.py [graph_construction] [graph_weight_adjustment] [label_type]
```

This command uses the best hyperparameter for the non fine-tuned cases. For the fine_tuned ones run `train_sbert.py` to get fine_tuned sentence transformer. Define the model directory in the `--llm` arg of `run_code.py`


## ARCHITECTURE
1. Models => Contains the model itself
2. Training => Contains training/evaluation related items
3. Data => Contains data/preprocessing code
4. Graph => Contains graph construction code(modularity graph+base one)
5. Utils => For utils

Non Module Structure
1. Scripts
2. Experiments->All sorts of experiments
3. Configs->experiments and dataset yaml
4. Datasets->Raw datasets and their metadata
5. Results
