import math

from torch import nn
import torch.nn.functional as F





class MLP(nn.Module):

    def __init__(self,dim_in,dim_hidden,dim_out):
        super(MLP,self).__init__()
        self.layer_input = nn.Linear(dim_in,dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden,dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        x = x.view(-1,x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)

class CNNMnist(nn.Module):

    def __init__(self,args):
        super(CNNMnist,self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels,10,kernel_size=5)
        self.conv2 = nn.Conv2d(10,20,kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320,50)
        self.fc2 = nn.Linear(50,args.num_classes)

    def forward(self,x):
        x = F.relu(F.max_pool2d(self.conv1),2)
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)),2))
        x = x.view(-1,x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x,training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x,dim=1)

def make_layers_Mnist(cfg,quant,batch_norm=False,conv = nn.Conv2d):
    layers = list()
    in_channels = 1
    n = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2,stride=2)]
        else:
            use_quant = v[-1] != 'N'
            filters = int(v) if use_quant else int(v[:-1])
            conv2d = conv(in_channels,filters,kernel_size=3,padding=1)
            if batch_norm:
                layers += [conv2d,nn.BatchNorm2d(filters),nn.ReLU]
            else:
                layers += [conv2d,nn.ReLU()]
            if quant != None:
                layers += [quant()]
            n += 1
            in_channels = filters
    return nn.Sequential(*layers)


class CNNMnist_HQ(nn.Module):
    def __init__(self,args,quant):
        super(CNNMnist_HQ,self).__init__()
        self.args = args
        self.linear = nn.Linear
        cfg = {
            16: ['16','M','32','M']
        }
        self.conv = nn.Conv2d
        self.features = make_layers_Mnist(cfg[16],quant,True,self.conv)
        self.classifier = None
        if quant != None:
            self.classifier = nn.Sequential(
                nn.Dropout(),
                self.linear(7*7*32,512),
                nn.ReLU(True),
                quant(),
                self.linear(512,10),
                nn.ReLU(True),
                quant(),
                nn.LogSoftmax(dim=1)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(),
                self.linear(7*7*32,512),
                nn.ReLU(True),
                self.linear(512,10),
                nn.ReLU(True),
                nn.LogSoftmax(dim=1)
            )
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0,math.sqrt(2. / n))
                m.bias.data.zero_()
    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x







class CNNfashion_Mnist(nn.Module):

    def __init__(self,args):
        super(CNNfashion_Mnist,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,16,kernel_size=5,padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16,32,kernel_size=5,padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32,10)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = self.fc(out)
        return out


class CNNCifar(nn.Module):
    def __init__(self,args):
        super(CNNCifar,self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,args.num_classes)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x,dim=1)

def make_layers_Cifar10(cfg,quant,batch_norm=False,conv=nn.Conv2d):
    layers = list()
    in_channels = 3
    n = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2,stride=2)]
        else:
            use_quant = v[-1] != 'N'
            filters = int(v) if use_quant else int(v[:-1])
            conv2d = conv(in_channels,filters,kernel_size=3,padding=1,bias=False)
            if batch_norm:
                layers += [conv2d,nn.BatchNorm2d(filters),nn.ReLU(True)]
            else:
                layers += [conv2d,nn.ReLU()]
            if quant != None:
                layers += filters
            n += 1
            in_channels = filters
    return nn.Sequential(*layers)

class CNNCifar_HQ(nn.Module):
    def __init__(self,args,quant):
        super(CNNCifar_HQ,self).__init__()
        self.args = args
        self.linear = nn.Linear
        cfg = {
            9: ['64', '64', 'M', '128', '128', 'M', '256', '256', 'M'],
            11: ['64', 'M', '128', 'M', '256', '256', 'M', '512', '512', 'M', '512', '512', 'M'],
            13: ['64', '64', 'M', '128', '128', 'M', '256', '256', 'M', '512', '512', 'M', '512', '512', 'M'],
            16: ['64', '64', 'M', '128', '128', 'M', '256', '256', '256', 'M', '512', '512', '512', 'M', '512', '512', '512', 'M'],
        }
        self.conv = nn.Conv2d
        self.features = make_layers_Cifar10(cfg[16],quant,True,self.conv)
        self.classifier=None
        if quant != None:
            self.classifier = nn.Sequential(
                nn.Dropout(),
                self.linear(512*1*1,4096),
                nn.ReLU(True),
                quant(),
                self.linear(4096,4096),
                nn.ReLU(True),
                quant(),
                self.linear(4096,args.num_classes),
                nn.ReLU(True),
                quant(),
                nn.LogSoftmax(dim=1)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(),
                self.linear(512*1*1,4096),
                nn.ReLU(True),
                self.linear(4096,4096),
                nn.ReLU(True),
                self.linear(4096,args.num_classes),
                nn.ReLU(True),
                nn.LogSoftmax(dim=1)
            )

class ResnetCifar18(nn.Module):
    def __init__(self,quant,quantx,in_channel,out_channel,strides):
        super(ResnetCifar18,self).__init__()
        self.block = None
        self.residual = nn.Sequential()
        self.quantx = quantx
        if quant == None:
            self.block = nn.Sequential(
                nn.Conv2d(in_channel,out_channel,kernel_size=3,stride=strides,padding=1,bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(True),
                nn.Conv2d(out_channel,out_channel,kernel_size=3,stride=1,padding=1,bias=False),
                nn.BatchNorm2d(out_channel)
            )
            if strides!=1 or in_channel!=out_channel:
                self.residual=nn.Sequential(
                    nn.Conv2d(in_channel,out_channel,kernel_size=1,stride=strides,bias=False),
                    nn.BatchNorm2d(out_channel)
                )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_channel,out_channel,kernel_size=3,stride=strides,padding=1,bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(True),
                quant(),
                nn.Conv2d(out_channel,out_channel,kernel_size=3,stride=1,padding=1,bias=False),
                nn.BatchNorm2d(out_channel),
                quant()
            )
            self.residual = nn.Sequential()
            if strides != 1 or in_channel != out_channel:
                self.residual = nn.Sequential(
                    nn.Conv2d(in_channel,out_channel,kernel_size=1,stride=strides,bias=False),
                    nn.BatchNorm2d(out_channel),
                    quant()
                )
    def forward(self,x):
        out = self.block(x)
        out += self.residual(x)
        out = F.relu(out)
        if self.quantx != None:
            out = self.quantx(out)
        return  out



class ResNet(nn.Module):
    def __init__(self,args,quant,quantx):
        super(ResNet,self).__init__()
        self.in_channel = 64
        self.quantx = quantx
        self.conv1 = None
        if quant == None:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True) # ,
                # nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                quant()#,
                #nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
            )
        self.layer1 = self.make_layer(quant,quantx,64,2,stride=1)
        self.layer2 = self.make_layer(quant,quantx,128,2,stride=1)
        self.layer3 = self.make_layer(quant, quantx, 256, 2, stride=1)
        self.layer4 = self.make_layer(quant, quantx, 512, 2, stride=1)
        self.fc = nn.Linear(512,args.num_classes)


    def make_layer(self,quant,quantx,channel,num_blocks,stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(ResnetCifar18(quant,quantx,self.in_channel,channel,stride))
            self.in_channel=channel
        return  nn.Sequential(*layers)

    def forward(self,x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out,4)
        if self.quantx != None:
            out = self.quantx(out)
        out = out.view(out.size(0),-1)
        out = self.fc(out)
        if self.quantx != None:
            out = self.quantx(out)
        out = F.log_softmax(out,dim=1)
        return out

class modelC(nn.Module):
    def __init__(self,input_size,n_classes=10,**kwargs):
        super(modelC,self).__init__()
        self.conv1 = nn.Conv2d(input_size,96,3,padding=1)
        self.conv2 = nn.Conv2d(96,96,3,padding=1)
        self.conv3 = nn.Conv2d(96,96,4,padding=1,stride=2)
        self.conv4 = nn.Conv2d(96,192,3,padding=1)
        self.conv5 = nn.Conv2d(192,192,3,padding=1)
        self.conv6 = nn.Conv2d(192,192,3,padding=1,stride=2)
        self.conv7 = nn.Conv2d(192,192,3,padding=1)
        self.conv8 = nn.Conv2d(192,192,1)

        self.class_conv = nn.Conv2d(192,n_classes,1)


    def forward(self,x):
        x_drop = F.dropout(x,2)
        conv1_out = F.relu(self.conv1(x_drop))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv3_out = F.relu(self.conv3(conv2_out))
        conv3_out_drop = F.dropout(conv3_out,5)
        conv4_out = F.relu(self.conv4(conv3_out_drop))
        conv5_out = F.relu(self.conv5(conv4_out))
        conv6_out = F.relu(self.conv6(conv5_out))
        conv6_out_drop = F.dropout(conv6_out,5)
        conv7_out = F.relu(self.conv7(conv6_out_drop))
        conv8_out = F.relu(self.conv8(conv7_out))

        class_out = F.relu(self.class_conv(conv8_out))
        pool_out = F.adaptive_avg_pool2d(class_out,1)
        pool_out.squeeze_(-1)
        pool_out.squeeze_(-1)
        return pool_out
        