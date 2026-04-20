In this project you will build Named Entity Recognition (NER) and Relation Extraction (RE) models on biomedical literature. 

You are given two annotated datasets, each derived from PubMed abstracts:

- **ADKG** — Alzheimer's Disease Knowledge Graph: 8,031 sentences about Alzheimer's disease and related neurodegenerative disorders.
- **MDKG** — Mental Disorder Knowledge Graph: 6,678 sentences covering a broad range of mental disorders (schizophrenia, depression, bipolar disorder, PTSD, OCD, etc.).

Both datasets are annotated with biomedical entities and the relations between them.

**Environment Setup**

Opened a WSL terminal (not Windows PowerShell)
Created a new conda environment called bios740_final with Python 3.11
Registered it as a Jupyter kernel so VSCode can see it

**Installing SpERT**

Cloned the SpERT repo into ~/BIOS740/final_project/spert
Installed dependencies, fixing several version conflicts along the way:

numpy was too old for Python 3.11 → installed latest
jinja2==2.11.3 conflicted with markupsafe → upgraded to 3.1.6
protobuf conflicted with tensorboardX → downgraded to 3.20.3
transformers==5.5.4 removed AdamW → downgraded to 4.28.0

Installed PyTorch with CUDA 12.9 support
Downloaded the spaCy English model (en_core_web_sm)

**Testing**

Downloaded SpERT's example CoNLL04 dataset and pretrained models
Successfully ran python ./spert.py predict on the example data

**STEPS:**

Step 1. 
**convert_to_spert.py**: converts the raw ADKG/MDKG data into JSON format that SpERT can read.
1. Text -> tokens: Convert text string to a pre-tokenized list
2. Character offsets -> token indicies: Entities use character-level positions but we want token-level positions
3. Relation entity references -> list indices: relation reference entitites by ID but SpERT needs list postion numbers


NER (Named Entity Recognition) - finding WHERE entities are and WHAT type they are
NEC (Named Entity Classification) - subtask within NER where we assign the TYPE label to a detected entity span
- without NEC - only check if spans and *relation type* are correct; IGNORES entity types
- with NEC - checks everything: spans, relation type, AND entity types must ALL be correct! [STRICTER]

EXAMPLE:
Without NEC: micro F1 = 43.83%
With NEC: micro F1 = 39.75%
Interpretation: ~4% gap means some relations are being found correctly but with WRONG entity types attached; helps us understand where errors come from - is the model failing at finding relations, or at classifying the entity types involved?

**REMEBER TO RUN FILES FROM THE ENVIRONMENT 'bios740_final'**

Step 2.
- Train ADKG on train dataset, evaluate on dev dataset
> python ./spert.py train --config configs/adkg_train.conf 2>&1 | tee logs/adkg_train_log.txt

Step 3. 
- Evaluate the ADKG model on **test** dataset
> python ./spert.py eval --config configs/adkg_eval.conf

**run.sh**: run the script to run the  model for adkg and mdkg

# Make it executable
chmod +x run.sh

# Run for ADKG
bash run.sh adkg

# Run for MDKG
bash run.sh mdkg