# Language Models for Named Entity Normalization
This repository is a set of experiments written in tensorflow + pytorch to explore further in the direction of how scraping internet data and language models can be used for more enhanced and robust named entity normalization.

## Installation

Pull this repository from Git via:

```bash
git clone https://github.com/yashsmehta/named-entity-normalization.git
```

See the requirements.txt for the list of dependent packages which can be installed via:

```bash
pip -r requirements.txt
```

## Usage
First run the LM extractor code which first performs a Google search query on the result and goes to the wiki page and gets the first paragraph for a particular item. It then passes this through the language model (in this case ALBERT, simply because it consumes lesser memory) and stores the embeddings (of all layers) in a pickle file. 

This 'new dataset' is created to train the model as then we do not have to parse the information again and again for every run. Before running the code, create a data/pkl_data folder in the repo folder. All the arguments are optional and passing no arguments runs the extractor with the default values.

```bash
python LM_extractor.py -max_token_length 512 -datafile 'data/entity_class/' -batch_size 32 -op_dir 'pkl_data'
```

Next run the finetuning network which is currently a MLP.

```bash
python MLP_classifier.py 
```
