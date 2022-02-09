
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
#import models
import torch.nn as nn
import numpy as np
import argparse
import time
from tqdm import tqdm
import random
import torchvision.models as models

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-g', '--gpu', nargs='+', type=str, default='0,1,2,3,4,5', help='gpu id to use(e.g. 0,1,2,3)')
parser.add_argument('-d', '--distribution', nargs='+', type=str, default='1,1,8', help='models type to use(e.g. 1 vgg13, 1 vgg16, 8 vgg19)')
parser.add_argument('-f', '--fusion', type=int, default=1, help='fusion or not')
parser.add_argument('-op', '--optimizer', type=str, default='adam', help='optimizer')
parser.add_argument('-lr', '--learningrate', type=float, default=0.001, help='learning rate')
parser.add_argument('-td', '--trainingdata', type=str, help='training data')
parser.add_argument('-se', '--startepoch', type=int, help='start epoch')
parser.add_argument('-ui', '--uploadinterval', type=int, help='upload interval')
parser.add_argument('-p', '--pretrained', type=int, help='pretrained')
parser.add_argument('-e', '--epoch', type=int, help='epoch')

                
        

def expansion(global_params,client_params,basiclayer,net_index):
    for key,index in zip(basiclayer,range(len(global_params))):
        shape=list(client_params[key].shape)
        global_params[index][layer_offset[net_index][index][0]:layer_offset[net_index][index][0]+shape[0],layer_offset[net_index][index][1]:layer_offset[net_index][index][1]+shape[1],:shape[2],:shape[3]]+=client_params[key].cpu()*local_conv_index[net_index][key]
        
    return global_params

def compression(client_params,net_index):
    global_params={}
    index=0
    for key in client_params.keys():
        if(key in basic_layer[Net_type[net_index]]):
            shape=list(client_params[key].shape)
            global_params[key]=global_model[index][layer_offset[net_index][index][0]:layer_offset[net_index][index][0]+shape[0],layer_offset[net_index][index][1]:layer_offset[net_index][index][1]+shape[1],:shape[2],:shape[3]]
            index+=1
        else:
            global_params[key]=client_params[key]
    return global_params

def layoffset_generator():
    layer_offset=[]
    for net in Net_type:
        temp_offset=[]
        for offset_range in layer_offset_range[net]:
            temp_offset.append([random.randint(offset_range[0],offset_range[1]),random.randint(offset_range[2],offset_range[3])])
        layer_offset.append(temp_offset)
    return layer_offset
    
def validation(testset,net,device):
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,shuffle=False, num_workers=1)
    correct = 0
    total = 0
    test_net=net.eval()
    test_net.to(device)
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = test_net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_net.to("cpu")
    return ( 100 * correct / total)



args = parser.parse_args()
gpu1="cuda:"+args.gpu[0]
gpu2="cuda:"+args.gpu[1]
'''gpu3="cuda:"+args.gpu[2]
gpu4="cuda:"+args.gpu[3]'''
print('gpu',args.gpu)#,gpu2,gpu3,gpu4)
print('distribution',args.distribution)
print('fusion',args.fusion)
print('optimizer',args.optimizer)
print('learningrate',args.learningrate)
print('training data',args.trainingdata)
print('start epoch', args.startepoch)
print('upload interval', args.uploadinterval)
print('pretrained', args.pretrained)
print('epoch',args.epoch)
'''
#resnet

basic_layer=[['conv1.weight', 'layer1.0.conv1.weight', 'layer1.0.conv2.weight', 'layer1.1.conv1.weight', 'layer1.1.conv2.weight', 'layer2.0.conv1.weight', 'layer2.0.conv2.weight', 'layer2.1.conv1.weight', 'layer2.1.conv2.weight', 'layer3.0.conv1.weight', 'layer3.0.conv2.weight', 'layer3.1.conv1.weight', 'layer3.1.conv2.weight', 'layer4.0.conv1.weight', 'layer4.0.conv2.weight', 'layer4.1.conv1.weight', 'layer4.1.conv2.weight'],
             ['conv1.weight', 'layer1.0.conv1.weight', 'layer1.0.conv2.weight', 'layer1.1.conv1.weight', 'layer1.1.conv2.weight', 'layer2.0.conv1.weight', 'layer2.0.conv2.weight', 'layer2.1.conv1.weight', 'layer2.1.conv2.weight', 'layer3.0.conv1.weight', 'layer3.0.conv2.weight', 'layer3.1.conv1.weight', 'layer3.1.conv2.weight', 'layer4.0.conv1.weight', 'layer4.0.conv2.weight', 'layer4.1.conv1.weight', 'layer4.1.conv2.weight'],
             ['conv1.weight', 'layer1.0.conv1.weight', 'layer1.0.conv2.weight', 'layer1.1.conv1.weight', 'layer1.1.conv2.weight', 'layer2.0.conv1.weight', 'layer2.0.conv2.weight', 'layer2.1.conv1.weight', 'layer2.1.conv2.weight', 'layer3.0.conv1.weight', 'layer3.0.conv2.weight', 'layer3.1.conv1.weight', 'layer3.1.conv2.weight', 'layer4.0.conv1.weight', 'layer4.0.conv2.weight', 'layer4.1.conv1.weight', 'layer4.1.conv2.weight']]

global_shape=[[64, 3, 7, 7], [64, 64, 3, 3], [64, 64, 3, 3], [64, 256, 3, 3], [64, 64, 3, 3], [128, 256, 3, 3], [128, 128, 3, 3], [128, 512, 3, 3], [128, 128, 3, 3], [256, 512, 3, 3], [256, 256, 3, 3], [256, 1024, 3, 3], [256, 256, 3, 3], [512, 1024, 3, 3], [512, 512, 3, 3], [512, 2048, 3, 3], [512, 512, 3, 3]]

layer_offset_range=[[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,192],[0,0,0,0],[0,0,0,192],[0,0,0,0],[0,0,0,384],[0,0,0,0],[0,0,0,384],[0,0,0,0],[0,0,0,768],[0,0,0,0],[0,0,0,768],[0,0,0,0],[0,0,0,1536],[0,0,0,0]],
                    [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,192],[0,0,0,0],[0,0,0,192],[0,0,0,0],[0,0,0,384],[0,0,0,0],[0,0,0,384],[0,0,0,0],[0,0,0,768],[0,0,0,0],[0,0,0,768],[0,0,0,0],[0,0,0,1536],[0,0,0,0]],
                    [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
                    ]
'''
'''layer_offset=[[[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]],
              [[0,0],[0,0],[0,0],[0,32],[0,0],[0,32],[0,0],[0,64],[0,0],[0,64],[0,0],[0,128],[0,0],[0,128],[0,0],[0,256],[0,0]],
              [[0,0],[0,0],[0,0],[0,64],[0,0],[0,64],[0,0],[0,128],[0,0],[0,128],[0,0],[0,256],[0,0],[0,256],[0,0],[0,512],[0,0]],
              [[0,0],[0,0],[0,0],[0,96],[0,0],[0,96],[0,0],[0,192],[0,0],[0,192],[0,0],[0,384],[0,0],[0,384],[0,0],[0,768],[0,0]],
              [[0,0],[0,0],[0,0],[0,128],[0,0],[0,128],[0,0],[0,256],[0,0],[0,256],[0,0],[0,512],[0,0],[0,512],[0,0],[0,1024],[0,0]],
              [[0,0],[0,0],[0,0],[0,160],[0,0],[0,160],[0,0],[0,320],[0,0],[0,320],[0,0],[0,640],[0,0],[0,640],[0,0],[0,1280],[0,0]],
              [[0,0],[0,0],[0,0],[0,192],[0,0],[0,192],[0,0],[0,384],[0,0],[0,384],[0,0],[0,768],[0,0],[0,768],[0,0],[0,1536],[0,0]],
              
              [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]],
              [[0,0],[0,0],[0,0],[0,32],[0,0],[0,32],[0,0],[0,64],[0,0],[0,64],[0,0],[0,128],[0,0],[0,128],[0,0],[0,256],[0,0]],
              [[0,0],[0,0],[0,0],[0,64],[0,0],[0,64],[0,0],[0,128],[0,0],[0,128],[0,0],[0,256],[0,0],[0,256],[0,0],[0,512],[0,0]],
              [[0,0],[0,0],[0,0],[0,96],[0,0],[0,96],[0,0],[0,192],[0,0],[0,192],[0,0],[0,384],[0,0],[0,384],[0,0],[0,768],[0,0]],
              [[0,0],[0,0],[0,0],[0,128],[0,0],[0,128],[0,0],[0,256],[0,0],[0,256],[0,0],[0,512],[0,0],[0,512],[0,0],[0,1024],[0,0]],
              [[0,0],[0,0],[0,0],[0,160],[0,0],[0,160],[0,0],[0,320],[0,0],[0,320],[0,0],[0,640],[0,0],[0,640],[0,0],[0,1280],[0,0]],
              [[0,0],[0,0],[0,0],[0,192],[0,0],[0,192],[0,0],[0,384],[0,0],[0,384],[0,0],[0,768],[0,0],[0,768],[0,0],[0,1536],[0,0]],
              
              [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]],
              [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]
        ]'''


#resnet shape
basic_layer=[['conv1.weight', 'layer1.0.conv1.weight', 'layer1.0.conv2.weight', 'layer1.1.conv1.weight', 'layer1.1.conv2.weight', 'layer2.0.conv1.weight', 'layer2.0.conv2.weight', 'layer2.1.conv1.weight', 'layer2.1.conv2.weight', 'layer3.0.conv1.weight', 'layer3.0.conv2.weight', 'layer3.1.conv1.weight', 'layer3.1.conv2.weight', 'layer4.0.conv1.weight', 'layer4.0.conv2.weight', 'layer4.1.conv1.weight', 'layer4.1.conv2.weight'],
             ['conv1.weight', 'layer1.0.conv1.weight', 'layer1.0.conv2.weight', 'layer1.1.conv1.weight', 'layer1.1.conv2.weight', 'layer1.2.conv1.weight', 'layer1.2.conv2.weight', 'layer2.0.conv1.weight', 'layer2.0.conv2.weight', 'layer2.1.conv1.weight', 'layer2.1.conv2.weight', 'layer2.2.conv1.weight', 'layer2.2.conv2.weight', 'layer2.3.conv1.weight', 'layer2.3.conv2.weight', 'layer3.0.conv1.weight', 'layer3.0.conv2.weight'],
             ['conv1.weight', 'layer1.0.conv1.weight', 'layer1.0.conv2.weight', 'layer1.0.conv3.weight', 'layer1.1.conv1.weight', 'layer1.1.conv2.weight', 'layer1.1.conv3.weight', 'layer1.2.conv1.weight', 'layer1.2.conv2.weight', 'layer1.2.conv3.weight', 'layer2.0.conv1.weight', 'layer2.0.conv2.weight', 'layer2.0.conv3.weight', 'layer2.1.conv1.weight', 'layer2.1.conv2.weight', 'layer2.1.conv3.weight', 'layer2.2.conv1.weight']]
global_shape=[[64, 3, 7, 7],[64, 64, 3, 3],[64, 64, 3, 3],[256, 64, 3, 3],[64, 256, 3, 3],[128, 64, 3, 3],[256, 128, 3, 3],[128, 256, 3, 3],[128, 128, 3, 3],[256, 128, 3, 3],[256, 256, 3, 3],[256, 256, 3, 3],[512, 256, 3, 3],[512, 512, 3, 3],[512, 512, 3, 3],[512, 512, 3, 3],[512, 512, 3, 3]]
layer_offset_range=[[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,192,0,0],[0,0,0,192],[0,0,0,0],[0,128,0,0],[0,0,0,128],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,256,0,0],[0,0,0,256],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
                    [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,192,0,0],[0,0,0,192],[0,64,0,0],[0,192,0,0],[0,0,0,192],[0,0,0,0],[0,128,0,0],[0,128,0,128],[0,128,0,128],[0,384,0,128],[0,384,0,384],[0,384,0,384],[0,256,0,384],[0,256,0,256]],
                    [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,64,0,0],[0,0,0,64],[0,64,0,0],[0,64,0,64],[0,0,0,64],[0,128,0,0],[0,128,0,128],[0,0,0,128],[0,256,0,0],[0,256,0,256],[0,0,0,256],[0,256,0,0]]
                    ]

'''
layer_offset=[[[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]],
            [[0,0],[0,0],[0,0],[64,0],[0,64],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]],
            [[0,0],[0,0],[0,0],[128,0],[0,128],[0,0],[128,0],[0,128],[0,0],[0,0],[0,0],[0,0],[256,0],[0,256],[0,0],[0,0],[0,0]],
            [[0,0],[0,0],[0,0],[192,0],[0,192],[0,0],[128,0],[0,128],[0,0],[0,0],[0,0],[0,0],[256,0],[0,256],[0,0],[0,0],[0,0]],
            
            [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]],
            [[0,0],[0,0],[0,0],[64,0],[0,64],[0,0],[64,64],[0,64],[0,0],[0,0],[0,128],[128,0],[128,0],[128,128],[128,128],[0,128],[0,256]],
            [[0,0],[0,0],[0,0],[128,0],[0,128],[64,0],[128,0],[0,128],[0,0],[128,0],[128,0],[0,128],[256,128],[256,256],[256,256],[256,256],[256,0]],
            [[0,0],[0,0],[0,0],[192,0],[0,192],[64,0],[192,0],[0,192],[0,0],[128,0],[128,128],[128,128],[384,128],[384,384],[384,384],[256,384],[256,256]],
            
            [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]],
            [[0,0],[0,0],[0,0],[0,0],[0,0],[64,0],[0,64],[64,0],[64,64],[0,64],[128,0],[128,128],[0,128],[256,0],[256,256],[0,256],[256,0]],
        ]
'''

'''
#vgg
basic_layer=[['features.0.weight','features.2.weight','features.5.weight','features.7.weight','features.10.weight','features.12.weight'],
             ['features.0.weight','features.2.weight','features.5.weight','features.7.weight','features.10.weight','features.12.weight'],
             ['features.0.weight','features.2.weight','features.5.weight','features.7.weight','features.10.weight','features.12.weight']]
global_shape=[[64, 3, 3, 3],[64, 64, 3, 3],[128, 64, 3, 3],[128, 128, 3, 3],[256, 128, 3, 3],[256, 256, 3, 3]]
layer_offset_range=[[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
                    [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]],
                    [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
                    ]
'''
transform = transforms.Compose([
         # 对原始32*32图像四周各填充4个0像素（40*40），然后随机裁剪成32*32
         transforms.RandomCrop(32, padding=5),
         # 按0.5的概率水平翻转图片
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
transform_test = transforms.Compose([
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
transform_SVHN = transforms.Compose([
         # 对原始32*32图像四周各填充4个0像素（40*40），然后随机裁剪成32*32
         transforms.RandomCrop(32, padding=4),
         # 按0.5的概率水平翻转图片
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
transform_SVHN_test = transforms.Compose([
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

if args.trainingdata=='cifar10':
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform_test)
elif args.trainingdata=='cifar100':
    trainset = torchvision.datasets.CIFAR100(root='./data100', train=True,download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(root='./data100', train=False,download=True, transform=transform_test)
elif args.trainingdata=='SVHN':
    trainset = torchvision.datasets.SVHN(root='./dataSVHN', download=True, transform=transform_SVHN, split='train')
    testset = torchvision.datasets.SVHN(root='./dataSVHN', download=True, transform=transform_SVHN_test, split='test')

global_model=[]
for shape in global_shape:
    global_model.append(torch.randn(shape))
     
criterion = nn.CrossEntropyLoss()

if args.pretrained==1:
    pretrained=True
elif args.pretrained==0:
    pretrained=False

device1 = torch.device(gpu1 if torch.cuda.is_available() else "cpu")
device2 = torch.device(gpu2 if torch.cuda.is_available() else "cpu")
print(device1,device2)
'''device3 = torch.device(gpu3 if torch.cuda.is_available() else "cpu")
device4 = torch.device(gpu4 if torch.cuda.is_available() else "cpu")'''

loss_list=[]
a_all=[]
b_all=[]
c_all=[]
avg_all=[]
Net_type=[]
Net=[]
Op=[]
device=[]
for _ in range(5):
    device.append(device1)
for _ in range(5):
    device.append(device2)
'''
for _ in range(25):
    device.append(device3)
for _ in range(25):
    device.append(device4)'''


for _ in range(int(args.distribution[0])):
    print('resnet')
    Net_type.append(0)
    Net.append(models.resnet18(pretrained= pretrained))
    if args.optimizer=='SGD':
        Op.append(optim.SGD(Net[-1].parameters(), lr=args.learningrate,weight_decay=0.0001))
    elif args.optimizer=='adam' or args.optimizer=='ADAM':
        Op.append(optim.Adam(Net[-1].parameters(), lr=args.learningrate,weight_decay=0.0001))
    elif args.optimizer=='momentum':
        Op.append(optim.SGD(Net[-1].parameters(), lr=args.learningrate,momentum=0.9,weight_decay=0.0001))
for _ in range(int(args.distribution[1])):
    Net_type.append(1)
    Net.append(models.resnet34(pretrained= pretrained))
    if args.optimizer=='SGD':
        Op.append(optim.SGD(Net[-1].parameters(), lr=args.learningrate,weight_decay=0.0001))
    elif args.optimizer=='adam' or args.optimizer=='ADAM':
        Op.append(optim.Adam(Net[-1].parameters(), lr=args.learningrate,weight_decay=0.0001))
    elif args.optimizer=='momentum':
        Op.append(optim.SGD(Net[-1].parameters(), lr=args.learningrate,momentum=0.9,weight_decay=0.0001))
for _ in range(int(args.distribution[2])):
    Net_type.append(2)
    Net.append(models.resnet50(pretrained= pretrained))
    if args.optimizer=='SGD':
        Op.append(optim.SGD(Net[-1].parameters(), lr=args.learningrate,weight_decay=0.0001))
    elif args.optimizer=='adam' or args.optimizer=='ADAM':
        Op.append(optim.Adam(Net[-1].parameters(), lr=args.learningrate,weight_decay=0.0001))
    elif args.optimizer=='momentum':
        Op.append(optim.SGD(Net[-1].parameters(), lr=args.learningrate,momentum=0.9,weight_decay=0.001))
for _ in range(0):
    print('vgg')
    Net_type.append(0)
    Net.append(models.vgg13(pretrained= pretrained))

    if args.optimizer=='SGD':
        Op.append(optim.SGD(Net[-1].parameters(), lr=args.learningrate,weight_decay=0.0001))
    elif args.optimizer=='adam' or args.optimizer=='ADAM':
        Op.append(optim.Adam(Net[-1].parameters(), lr=args.learningrate,weight_decay=0.0001))
    elif args.optimizer=='momentum':
        Op.append(optim.SGD(Net[-1].parameters(), lr=args.learningrate,momentum=0.9,weight_decay=0.001))
for _ in range(0):
    Net_type.append(1)
    Net.append(models.vgg16(pretrained= pretrained))

    if args.optimizer=='SGD':
        Op.append(optim.SGD(Net[-1].parameters(), lr=args.learningrate,weight_decay=0.0001))
    elif args.optimizer=='adam' or args.optimizer=='ADAM':
        Op.append(optim.Adam(Net[-1].parameters(), lr=args.learningrate,weight_decay=0.0001))
    elif args.optimizer=='momentum':
        Op.append(optim.SGD(Net[-1].parameters(), lr=args.learningrate,momentum=0.9,weight_decay=0.001))
for _ in range(0):
    Net_type.append(2)
    Net.append(models.vgg19(pretrained= pretrained))

    if args.optimizer=='SGD':
        Op.append(optim.SGD(Net[-1].parameters(), lr=args.learningrate,weight_decay=0.0001))
    elif args.optimizer=='adam' or args.optimizer=='ADAM':
        Op.append(optim.Adam(Net[-1].parameters(), lr=args.learningrate,weight_decay=0.0001))
    elif args.optimizer=='momentum':
        Op.append(optim.SGD(Net[-1].parameters(), lr=args.learningrate,momentum=0.9,weight_decay=0.001))

print(Net_type)

node_num=len(Net_type)


data_index=list(range(len(trainset)))
data_split_len=len(data_index)/node_num
print(data_split_len)
print(int(data_split_len))
data_subset=[]
for i in range(node_num-1):
    data_subset.append(torch.utils.data.Subset(trainset, data_index[i*int(data_split_len):(i+1)*int(data_split_len)]))
data_subset.append(torch.utils.data.Subset(trainset, data_index[(i+1)*int(data_split_len):]))



random.shuffle(data_subset)

if(args.fusion==1):
    layer_offset=layoffset_generator()
    local_conv_index=[]
    for j in range(len(Net_type)):
        net_params = Net[j].state_dict()
        layer_shape={}
        i=0
        for key in net_params.keys():
            if key in basic_layer[Net_type[j]]:
                shape=list(net_params[key].shape)
                
                layer_shape[key]=shape[0]*shape[1]*shape[2]*shape[3]
                #layer_shape[key]=(Net_type[j]+1)*3
                '''
                layer_shape[key]=i+1
                '''
            i+=1
        local_conv_index.append(layer_shape)

    global_repeat=[]
    for shape in global_shape:
        global_repeat.append(torch.zeros(shape))
    for net_index in range(len(Net_type)):
        net_params = Net[net_index].state_dict()
        for key,index in zip(basic_layer[Net_type[net_index]],range(len(global_repeat))):
            onee=torch.ones(net_params[key].size())
            shape=list(net_params[key].size())
            global_repeat[index][layer_offset[net_index][index][0]:layer_offset[net_index][index][0]+shape[0],layer_offset[net_index][index][1]:layer_offset[net_index][index][1]+shape[1],:shape[2],:shape[3]]+=onee*local_conv_index[net_index][key]


    for epoch in range(args.epoch):
        start_time=time.time()
        '''
        if (epoch>200):
            for p in Op[net_index].param_groups:
                p['lr'] = 0.0001
        elif (epoch>100):
            for p in Op[net_index].param_groups:
                p['lr'] = 0.0005'''
        running_loss = 0.
        batch_size = 256
        
        add_global=[]
        for shape in global_shape:
            add_global.append(torch.zeros(shape))
    
        acc_0=[]
        acc_1=[]
        acc_2=[]
        loss_all=0.0

        for net_index in range(int(len(Net_type))):

            if(epoch>args.startepoch+1):
                net_params = Net[net_index].state_dict()
                params = compression(net_params,net_index)
                Net[net_index].load_state_dict(params)

                
            Net[net_index].to(device[net_index])
            for _ in range(args.uploadinterval):
                for i, data in enumerate(torch.utils.data.DataLoader(data_subset[net_index], batch_size=batch_size,shuffle=True), 0):
                        
                    inputs, labels = data
                    inputs, labels = inputs.to(device[net_index]), labels.to(device[net_index])
                    Op[net_index].zero_grad()
                    outputs = Net[net_index](inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    Op[net_index].step()
              
                loss_all+=loss.item()/len(Net_type)

            Net[net_index].to("cpu")
            
            print(net_index,loss.item())

            if(epoch>args.startepoch):
                net_params = Net[net_index].state_dict()
                add_global = expansion(add_global,net_params,basic_layer[Net_type[net_index]],net_index)
        loss_list.append(loss_all)
        print('[%d] loss: %.4f' %(epoch + 1, loss_all))    
        if(epoch>args.startepoch):
            for index in range(len(add_global)):
                add_global[index]/=global_repeat[index]
                global_model[index]=add_global[index]
        
        if((epoch+1)%10== 0):
            for net_index in range(len(Net_type)):
                net_params = Net[net_index].state_dict()
                if(Net_type[net_index]==0):
                    acc_0.append(validation(testset,Net[net_index],device[net_index]))
                elif(Net_type[net_index]==1):
                    acc_1.append(validation(testset,Net[net_index],device[net_index]))
                elif(Net_type[net_index]==2):
                    acc_2.append(validation(testset,Net[net_index],device[net_index]))
                Net[net_index].load_state_dict(net_params)
                Net[net_index]=Net[net_index].train()
            #print('Accuracy',sum(acc_0)/len(acc_0),sum(acc_1)/len(acc_1),sum(acc_2)/len(acc_2))
            if(len(acc_0)>0):
                a_all.append(sum(acc_0)/len(acc_0))
                print('net_index_A:Accuracy',net_index, sum(acc_0)/len(acc_0))
            if(len(acc_1)>0):
                b_all.append(sum(acc_1)/len(acc_1))
                print('net_index_B:Accuracy',net_index, sum(acc_1)/len(acc_1))
            if(len(acc_2)>0):
                c_all.append(sum(acc_2)/len(acc_2))
                print('net_index_C:Accuracy',net_index, sum(acc_2)/len(acc_2))
        end_time=time.time()
        
        print('duration time',end_time-start_time)
    print(loss_list)
    print(a_all)
    print(b_all)
    print(c_all)
    for ac in range(len(a_all)):
        avg_all.append((a_all[ac]+b_all[ac]+c_all[ac])/3)
    print(avg_all)
    print('Finished Training')

if(args.fusion!=1):
    acc_0=[]
    acc_1=[]
    acc_2=[]
    acc_3=[]
    for net_index in range(len(Net_type)):
        start_time=time.time()
        for epoch in range(args.epoch):
            
            running_loss = 0.
            batch_size = 256
            loss_all=0.0
            for i, data in enumerate(torch.utils.data.DataLoader(data_subset[net_index], batch_size=batch_size,shuffle=True), 0):
        
                inputs, labels = data
                inputs, labels = inputs.to(device[net_index]), labels.to(device[net_index])
        
                Op[net_index].zero_grad()
                outputs = Net[net_index](inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                Op[net_index].step()
            
            if((epoch+1)%10== 0):
                if(Net_type[net_index]==0):
                    acc_0.append(validation(testset,Net[net_index],device[net_index]))
                elif(Net_type[net_index]==1):
                    acc_1.append(validation(testset,Net[net_index],device[net_index]))
                elif(Net_type[net_index]==2):
                    acc_2.append(validation(testset,Net[net_index],device[net_index]))
                elif(Net_type[net_index]==3):
                    acc_3.append(validation(testset,Net[net_index],device[net_index]))
                if(len(acc_0)>0):
                    print('net_index_A:Accuracy',net_index, sum(acc_0)/len(acc_0))
                if(len(acc_1)>0):
                    print('net_index_B:Accuracy',net_index, sum(acc_1)/len(acc_1))
                if(len(acc_2)>0):
                    print('net_index_C:Testing Accuracy',net_index, sum(acc_2)/len(acc_2))
                if(len(acc_3)>0):
                    print('net_index_D:Testing Accuracy',net_index, sum(acc_3)/len(acc_3))
                print('net_index_C:Training Accuracy',net_index, validation(trainset,Net[net_index],device[net_index]))
            loss_all+=loss.item()/len(Net_type)
            print('[%d] loss: %.4f' %(epoch + 1, loss_all))
        
        end_time=time.time()
        print('duration time',end_time-start_time)



'''

基础层分开同名,10个节点,有权重,权限按layer计算,数据集分割,每5轮融合一次,第2轮开始,节点全下载
adam 0.001 not fusion


基础层分开同名,10个节点,数据集分割,节点全下载
SGD 0.005 not fusion


基础层分开同名,10个节点,有权重,权限按layer计算,数据集分割,每5轮融合一次,第2轮开始,节点全下载
adam 0.001 fusion


基础层分开同名,10个节点,有权重,权限按layer计算,数据集分割,每5轮融合一次,第2轮开始,节点全下载
adam 0.01 fusion


基础层分开同名,10个节点,有权重,权限按layer计算,数据集分割,每5轮融合一次,第2轮开始,节点全下载
adam 0.005 fusion


基础层前18层,10个节点,有权重,权限按shape计算,数据集分割,每轮融合一次,第2轮开始,节点全下载
SGD 0.001 fusion
Accuracy 43.07599999999999 41.5075 38.74


基础层前18层,10个节点,有权重,权限按shape计算,数据集分割,每5轮融合一次,第50轮开始,节点全下载
SGD 0.005 fusion
Accuracy 48.523999999999994 51.48499999999999 50.91

基础层前18层,10个节点,数据集分割,节点全下载
adam 0.005 not fusion
net_index:Accuracy 9 56.17 54.6775 49.64

基础层前18层,10个节点,数据集分割,节点全下载
adam 0.001 not fusion
net_index:Accuracy 9 55.072 54.949999999999996 52.59

基础层前18层,10个节点,有权重,权限按shape计算,数据集分割,每轮融合一次,第2轮开始,节点全下载
adam 0.001 fusion
Accuracy 59.894000000000005 62.69499999999999 59.9

基础层前18层,10个节点,有权重,权限按shape计算,数据集分割,每轮融合一次,第2轮开始,节点全下载
adam 0.005 fusion
Accuracy 60.492000000000004 57.3425 57.43

基础层前18层,10个节点,有权重,权限按不同模型手动设定的,数据集分割,每轮融合一次,第2轮开始,节点全下载
adam 0.005 fusion
Accuracy 58.818 59.402499999999996 55.16


'''
#vgg group epoch 200 cifar10 82~83%

#resnet group epoch 200 cifar10 72~73%
