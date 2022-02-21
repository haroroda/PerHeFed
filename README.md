# PerHeFed
PerHeFed: a General Framework of Personalized Federated Learning for Heterogeneous Convolutional Neural Networks

## Requirements
numpy == 1.19.5            
torch == 1.9.1         
torchvision == 0.10.1               

## How to Use?
### Test PerHeFed performence in *simple heterogeneous scenario* by using like:               
```python
python server.py -g 0 1 -d 10 0 -op adam -lr 0.0001 0.001 -td SVHN -ui 2 -se 0 -e 100 -t layer -ir 1.0 -or 0.5 -m origin -i 1 -p 0 -ha 10
```
or       
```python
python server.py -g 0 1 -d 10 0 -op adam -lr 0.0001 0.001 -td SVHN -ui 2 -se 0 -e 100 -t inner -ir 0.5 -or 1.0 -m origin -i 1 -p 0 -ha 10
```
*-d  10 0* means 10 ResNet34s and 0 DenseNet121. *ir* controls Cintra, and *or*  controls Cinter, and both of them should be at (0,1]                  

### Test PerHeFed performence in *cross-model hybrid scenario* by using like:   
```python
python server.py -g 0 1 -d 5 5 -op adam -lr 0.0001 0.001 -td SVHN -ui 2 -se 0 -e 100 -t layer -ir 0.6 -or 1.0 -m origin -i 1 -p 0 -ha 10
```

### Test FLOP performence  by using like:               
```python
python server.py -g 0 1 -d 10 0 -op adam -lr 0.0001 0.001 -td SVHN -ui 2 -se 0 -e 100 -t layer -ir 1.0 -or 1.0 -m flop -i 1 -p 0 -ha 10
```


### Test Hermes performence  by using like:               
```python
python server.py -g 0 1 -d 10 0 -op adam -lr 0.0001 0.001 -td SVHN -ui 2 -se 0 -e 100 -t layer -ir 0.5 -or 1.0 -m hermes -i 1 -p 0 -ha 10
```

If you want to test the accuracy in extreme Non-IID setting, just need to set  *-i  0*.            
