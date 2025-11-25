import re
import pandas as pd
import numpy as np


def clean_str(post: str, remove_prefixes: tuple):
    new_lines = []
    
    for line in post.splitlines():
        if line.startswith(remove_prefixes) or line.strip().startswith('>'):
            continue
        if line.strip():
            new_lines.append(line)
    
    text = '\n'.join(new_lines)
    
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    
    return text.strip().lower()


remove_prefixes = (
    'From:', 'Subject:', 'Reply-To:', 'In-Reply-To:',
    'Nntp-Posting-Host:', 'Organization:', 'X-Mailer:',
    'In article', 'Lines:', 'NNTP-Posting-Host:',
    'Summary:', 'Article-I.D.:', 'Distribution:',
    'Keywords:', 'Date:', 'Message-ID:'
)

# DEBUG: Load original CSVs
df_train = pd.read_csv("../inputs/20ng/train.csv", keep_default_na=False)
print("BEFORE CLEANING:")
print(f"Train NaN count: {df_train.isnull().sum().sum()}")
print(f"Train shape: {df_train.shape}")
print(f"Train columns: {df_train.columns.tolist()}")
print(f"NaN per column:\n{df_train.isnull().sum()}\n")

# Apply cleaning
df_train['text'] = df_train['text'].apply(lambda x: clean_str(x, remove_prefixes))

print("AFTER CLEANING:")
print(f"Train NaN count: {df_train.isnull().sum().sum()}")
print(f"NaN per column:\n{df_train.isnull().sum()}\n")

# Remove empty text rows
df_train = df_train[df_train['text'].str.strip() != '']
df_train = df_train.dropna()

print("AFTER DROPNA:")
print(f"Train NaN count: {df_train.isnull().sum().sum()}")
print(f"Train shape: {df_train.shape}\n")

# Save properly
df_train.to_csv("../inputs/20ng_cleaned/train.csv", index=False, na_rep='')

# DEBUG: Reload and check
df_reload = pd.read_csv("../inputs/20ng_cleaned/train.csv", keep_default_na=False)
print("AFTER RELOAD:")
print(f"Reloaded NaN count: {df_reload.isnull().sum().sum()}")
print(f"NaN per column:\n{df_reload.isnull().sum()}")
print(f"Reloaded shape: {df_reload.shape}")
