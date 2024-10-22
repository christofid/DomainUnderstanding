Pytorch implementation of domain understaning techniques that is described in `Understood in translation, transformers for domain understanding`

Authors: Dimitrios Christofidellis, Matteo Manica, Leonidas Georgopoulos and Hans Vandierendonck

## Usage 

* To train the model:

```
cd transformer

python3 execute.py --data_path DATA_PATH --order_type bfs --root_node ROOT_NODE --embedding_path EMBEDDING_PATH  

```

 where DATA_PATH is the path where the training dataset is located, ROOT_NODE is the root node of the bfs traveral, MODEL_PATH is the path where the trained model is located and EMBEDDING_PATH is the path where the pretrained embeddings are located.



The model by default have the parameters defined in the paper. In order to change the parameters or/and specify the paths where the trained model and the output should be saved consult `python3 execute.py --help`



* To predict the relation types of a text snippet and visualize the respective attentions:

```
cd transformer

python3  attention_visualizer.py --data_path DATA_PATH  --order_type bfs --root_node ROOT_NODE --model_path MODEL_PATH --embedding_path EMBEDDING_PATH

```
 wwhere DATA_PATH is the path where the training dataset is located, ROOT_NODE is the root node of the bfs traveral, MODEL_PATH is the path where the trained model is located and EMBEDDING_PATH is the path where the pretrained embeddings are located.

Once the model is loaded, the user can type any text snippet to test the model.


* To predict the relation types and the respective entities


```
cd transformer

python3  attention_based_entities_extractor.py --data_path DATA_PATH  --order_type bfs --root_node ROOT_NODE --model_path MODEL_PATH --embedding_path EMBEDDING_PATH

```
where DATA_PATH is the path where the training dataset is located, ROOT_NODE is the root node of the bfs traveral, MODEL_PATH is the path where the trained model is located and EMBEDDING_PATH is the path where the pretrained embeddings are located.

It produces the relation types and the entities of the testing instances of the dataset.


* To predict the relation types using the Wisdom of Crowd consensus method:

```
cd ensemble

python3  execute.py --data_path DATA_PATH  --predictions_path PREDICTIONS_PATH

```

where DATA_PATH path is the path where the training dataset is located and PREDICTIONS_PATH is the path where all the predictions of the Transformers models have been stored.

In the context of our paper, we utilize the GloVE petrained word embeddings which can be downloaded from  http://nlp.stanford.edu/data/glove.6B.zip . 


