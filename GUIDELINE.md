# How to Add Custom Datasets and Models
We provide a skeleton code to help users add their own dataset and MLSV model to our MoLEF framework. Following the instructions will lead to build a new model within the MoLEF framework. Specifically, the following functions need to be implemented.

## Datasets 
To experiment with a custom dataset, the training and test sets need to be prepared in the JSON format.
These files should include the following attributes in the order: video identifier, duration, timestamp label, text description, tokenized words, and indices for the words. See the URL above for an example.
```
[["v_yINX46xPRf0", # video feauture name
159.99, # duration
[0, 31.2], # timestamps
"The people are in the pool wearing goggles.", # sentence
["the", "people", "are", "in", "the", "pool", "wearing", "goggles"], # words
[0, 1, 2, 3, 4, 5, 6, 7], # id2pos,
]

```

## Constructors 
### CustomDataset 
The `CustomDataset` constructor has 5 arguments, including feature path, word2vec, the number of frames, and the number of words.
In addition, custom member variables may be added to the class.

### CustomModel
The `CustomModel` constructor has the following structure, which loads `torch.nn.Module`. Following this structure, the user needs to code their own model, e.g., the visual and text encoders, the logic to produce outputs and loss, and so on.


```
class Model(nn.Module) :
  def __init__(self) :
      video_encoder = CustomVidEncoder()
      text_encoder = CustomTextEncoder()
  def forward(self, batch) :
      ....
      return outputs, loss
```

## Model Builder 
### build_model
Each model needs to be encapsulated to establish a running state.
Within the MoLEF, the `build.py` file located within the \code{runners} directory helps to integrate the new model into the structure as follows:


```
def build_model() : 
  from models.{model_name} import Model
    model = Model()
  
  return model 
```
### build_forward
Here, the custom inputs to the model are passed with the `model_inputs`. To expedite the training, accelerator setups may be needed, e.g., adding `cuda()` to the model inputs.

```
def build_forward(batch, model) :
  vid_feats, word_feats, ... = batch
  model_inputs = {'feats': vid_feats.cuda(), 'words': word_feats.cuda(), ...}
  outputs, loss = self.model(**model_inputs)

  return outputs, loss
```

## Script 
We provide a script to help training and evaluation. The script distinguishes training and evaluation by `mode`: either `training` or `evaluation`. Other hyperparameters can be configured by this command line, e.g., paths to the visual and text features, maximum number of epochs, learning rate scheduling, weight decay, and more.

### Training
```
python main.py --mode train --model model_name --word2vec-path  data/glove.840B.300d.bin \
--dataset Tacos --feature-path data/tacos/org --train-data data/tacos/train_data.json \
--val-data data/tacos/val_data.json --test-data data/tacos/test_data.json \
--max-num-epochs 20 --warmup-updates 300 --warmup-init-lr 1e-06 --lr 8e-4 \
--weight-decay 1e-7 --model-saved-path results/ --cfg code/configs/model_name.yml 
```
### Evaluation 
```
python main.py --mode evaluation --model model_name --word2vec-path  data/glove.840B.300d.bin \
--dataset Tacos --feature-path data/tacos/org  --train-data data/tacos/train_data.json \
--val-data data/tacos/val_data.json  --test-data data/tacos/test_data.json \
--model-load-path results/model_name --cfg code/configs/model_name.yml 
```
