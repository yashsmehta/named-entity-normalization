# Language Models for Named Entity Normalization
This repository is a set of experiments written in tensorflow + pytorch to explore further in the direction of how scraping internet data and language models can be used for more enhanced and robust named entity normalization.

## Method
Let's say we are given a raw list of entities (e.g. Marks and Spencers Ltd, LONDON, etc.) which are not annotated with the 
entity class (e.g. Physical Goods, Company names, etc.). Let's say a new entry is 'Marks and Spencers Ltd' and we dont know that this entity comes
under the 'Company names' class. The code performs 2 tasks:
1. Predict the class of the entity
     1. Get a paragraph describing the word, get a vector representation (768 dim in this case) of this paragraph by passing it
              through an **language model** (LM) (e.g. BERT)
     2. Feed this embedding vector to a shallow MLP classifier to make the prediction of what entity class the new entity belongs to.
              
2. After predicting the entity class, use a similarity metric with the other entries within that class to determine if the new entry is of a new 'type'
     or should be stored under a particular entry. Let's say 'M&S Limited' is already present in the graph. We get the embeddings and prediction for 'Marks and  Spencers Ltd'. Now, use a similarity measure (e.g. **cosine similarity**) to see how 'close' is the current embedding and the other embeddings in the 'Company Names' class. Have a threshold value, if max similarity is less than that, create a new item, else store this under an existing class.


In this case, any new entity doesn't need to have a labelled entity class. It's also easy to incorporate new entity classes and store them in a hierarchical graph-like structure to keep a more organized track of inventory.
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
