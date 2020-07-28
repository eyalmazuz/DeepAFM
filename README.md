# DeepAFM

## Training

you can train either the DeepFM the AFM or the ADFM using the train.py in the pyfiles fodler

the train.py received command line arguments using the 
```
--flag value
```
.

the arguments it can receive are: 

  embedding_size: int - the embedding size for the sparse features, default is 256.
  
  epochs: int - number of training epochs to train the models, default is 5.
  
  batch_size: int - size of the batch size in the epoch, default is 256.
 
  attention_factor: int - attention factor size for the AFM and DeepAFM models only, default value is 16.
  
  validation_size: float - the size of the validation set from all the data range between (0, 1), default is 0.2.
  
  test_size: float - the size of the test set from all the data range between (0, 1), default is 0.1.
  
  learning_rate: float - the learning rate value for the optimizer, default value is 0.1.
  
  drouput: float - dropout rate for the dnn and the last layer in the afm, default value is 0.1.
  
  regularization: float - l2 regularization value for the dnn layers in the models, default value is 0.1.
  
  model: string - the models you can choose to train from which are: AFM, DeepAFM.
  
  save_path: string - path to save model checkpoint, default value is './' which is the current directory.
  
  dataset_path: string - path to the dataset, dataset are in the data folders movielens_all.csv, frappe_all.csv.
  
  dnn: comma seperate list - number of layers and nodes in the dnn componnet of the DeepAFM, default value is 128,128.
  
  eval: boolean - if to the eval the model on the test set or not, default value is True.
  
  example running:
  ```sh
  python3 train.py --embedding size 256 --epochs 10 --batch_sizer 1024 --attention_factor 64 \
  --validation_size 0.3 --test_size 0.15 --learning_rate 0.001 --dropout 0.3 --regularization 0.5\
  --model DeepAFM  --save_path ../models/DeepAFM --dataset_path ../data/frappe_all.csv --dnn 512,256,256 --eval True
  ```
  
  notice that default values are not mandatory and the only mandatory flag is the dataset_path
  
## Sampling

  sampling can be done thought the py files in pyfiles/sampling folder, there are sampling files to sample for the movielens and frappe dataset.
  
  sampling means creating negative labels for the data, the ratio of the sampling is creating 2 negative labels per 1 positive label in the data.
  
  positive labels are the original rows in the dataset folder which located in data/frappe data/ml-20, we consider positive sample to be the user applied the tag (for movielens) or used the app (for frappe) under the context given which we give them the value of 1.
  
  negative sampling takes each example and change the context value for each row, this create new rows which means user applied the tag or used the app but those aren't real so these are our negative labels which we give the value 0
  
  
## Jupyter notebooks
  there are serveral jupyter notebooks located under the notebooks folder, these notebook have the same functionality as the train.py file
  but are seperate for each dataset and model so there are total of 6 notebooks. running the entire notebook will load the data, preprocess it, train the model and in the end evaluate on the test set.
  Jupyter notebooks is needed in order to run the DeepCTR version of DeepFM
