# PerHeFed
PerHeFed: a General Framework of Personalized Federated Learning for Heterogeneous Convolutional Neural Networks

## Requirements
numpy == 1.19.5            
torch == 1.9.1         
torchvision == 0.10.1               

##How to Use?
###Test PerHeFed performence in *simple heterogeneous scenario* by using:               
```python
python server.py -g 0 1 -d 10 0 -op adam -lr 0.0001 0.001 -td SVHN -ui 2 -se 0 -e 100 -t layer -ir 0.6 -or 1.0 -m origin -i 0 -p 0 -ha 10
```

*ir* controls Cintra, and *or*  controls Cinter, and both of them should be at (0,1]                  
###Test PerHeFed performence in *simple heterogeneous scenario* by using:   
