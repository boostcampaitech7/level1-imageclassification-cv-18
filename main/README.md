# How to use this code?

## Train and Test
### Curriculum learning with Fold

``` console
python3 curriculum_train_test.py
```

### Only Fold

``` console
python3 train_test.py
```

## Revise Model

### Data 
**Fold** - [`base/dataloader.py`](./base/dataloader.py)\
**Curriculum** - [`curriculum/curriculum_dataloader.py`](./curriculum/curriculum_dataloader.py)

if you click file, you can move that file

### Model
Our `train_test.py` and `curriculum_train_test.py` support `PyTorch` model, `Timm` model, `Custom` model\
You can custom model by revising [`model/SimpleCNN.py`](./model/simpleCNN.py)\
Also, you should revise `ModelSelector` in `base/customize_layer.py` or `curriculum/curriculum_dataloader.py`

### Transfer Learning
If you want to transfer learning, simply add `--pretrained` option 'True' or change `default` parser.add_argument in `train_test.py` or `curriculum_train_test.py` 
  