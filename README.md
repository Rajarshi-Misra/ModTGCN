## TRAIN THE MODEL
Ensure the `results` directory is present. Run:
```
python run_code.py [graph_construction] [graph_weight_adjustment] [label_type]
```

This command uses the best hyperparameter for the non fine-tuned cases. For the fine_tuned ones run `train_sbert.py` to get fine_tuned sentence transformer. Define the model directory in the `--llm` arg of `run_code.py`

