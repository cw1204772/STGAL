import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from torchvision import transforms
import torch.nn.functional as F

class FeatureResNet(nn.Module):
    def __init__(self,n_layers=50,pretrained=True,no_avgpool=True):
        super(FeatureResNet,self).__init__()
        if n_layers == 50:
            old_model= models.resnet50(pretrained=pretrained)
        elif n_layers == 34:
            old_model= models.resnet34(pretrained=pretrained)
        elif n_layers == 18:
            old_model= models.resnet18(pretrained=pretrained)
        else:
            raise NotImplementedError('resnet%s is not found'%(n_layers))

        for name,modules in old_model._modules.items():
            if name.find('fc') == -1:
                if no_avgpool:
                    if name.find('avgpool') == -1:
                        self.add_module(name,modules) 
                else:
                    self.add_module(name,modules)
        self.add_module('avgpool', nn.AdaptiveAvgPool2d((1,1)))
        self.output_dim = old_model.fc.in_features
        self.pretrained = pretrained
        self.no_avgpool = no_avgpool

    def forward(self,x):
        for name, module in self._modules.items():
            x = nn.parallel.data_parallel(module, x)
        #if self.no_avgpool: 
        #    return x
        #else:
        #    return x.view(x.size(0), -1)
        return x.view(x.size(0), -1)


class new_FeatureResNet(nn.Module):
    def __init__(self,n_layers=50,pretrained=True,no_avgpool=False,fc_dim=0):
        super(new_FeatureResNet,self).__init__()
        if n_layers == 50:
            old_model= models.resnet50(pretrained=pretrained)
        elif n_layers == 34:
            old_model= models.resnet34(pretrained=pretrained)
        elif n_layers == 18:
            old_model= models.resnet18(pretrained=pretrained)
        else:
            raise NotImplementedError('resnet%s is not found'%(n_layers))

        if no_avgpool: k = -2
        else: k = -1
        modules = list(old_model.children())[:k]
        self.backbone = nn.Sequential(*modules)

        #for name,modules in old_model._modules.items():
        #    if name.find('fc') == -1:
        #        if no_avgpool:
        #            if name.find('avgpool') == -1:
        #                self.add_module(name,modules) 
        #        else:
        #            self.add_module(name,modules)

        if fc_dim != 0:
            self.fc = nn.Sequential(
                nn.Linear(old_model.fc.in_features, fc_dim),
                nn.ReLU(True))

        self.output_dim = old_model.fc.in_features
        self.pretrained = pretrained
        self.no_avgpool = no_avgpool
        self.fc_dim = fc_dim
    def forward(self,x):
        #for name, module in self._modules.items():
        #    x = nn.parallel.data_parallel(module, x)
        x = self.backbone(x)
        if self.no_avgpool: 
            return x
        else:
            if self.fc_dim != 0:
                return self.fc(x.view(x.size(0), -1))
            else:
                return x.view(x.size(0), -1)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2048, 512, (1,1), bias=False),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, (3,3), padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 2048, (1,1), bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(True))
        self.pool = nn.AvgPool2d(7)

    def forward(self, x):
        x = self.net(x)
        x = self.pool(x)
        return x.view(-1, 2048)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.upsample = nn.Upsample(size=(7,7))
        self.net = nn.Sequential(
            nn.Conv2d(4096, 512, (1,1), bias=False),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, (3,3), padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 2048, (1,1), bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(True))

    def forward(self, x):
        x = self.upsample(x.view(-1,4096,1,1))
        x = self.net(x)
        return x

class ResNet(nn.Module):
    def __init__(self,n_id,n_layers=50,pretrained=True):
        super(ResNet,self).__init__()
        if n_layers == 50:
            old_model= models.resnet50(pretrained=pretrained)
        elif n_layers == 34:
            old_model= models.resnet34(pretrained=pretrained)
        elif n_layers == 18:
            old_model= models.resnet18(pretrained=pretrained)
        else:
            raise NotImplementedError('resnet%s is not found'%(n_layers))

        for name,modules in old_model._modules.items():
            self.add_module(name,modules)
        self.fc = nn.Linear(self.fc.in_features,n_id)
        #########
        self.pretrained = pretrained
    def forward(self,x):
        for name,module in self._modules.items():
            if name != 'fc':
                x = module(x)
        out = self.fc(x.view(x.size(0),-1))
        return out, x.view(x.size(0), -1)

class NLayersFC(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=1, n_layers=0):
        super(NLayersFC, self).__init__()
        if n_layers == 0:
            model = [nn.Linear(in_dim, out_dim)]
        else:
            model = []
            model += [nn.Linear(in_dim, hidden_dim),
                      nn.ReLU(True)]
            for i in range(n_layers-1):
                model += [nn.Linear(hidden_dim, hidden_dim),
                          nn.ReLU(True)]
            model += [nn.Linear(hidden_dim, out_dim)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class ICT_ResNet(nn.Module):
    def __init__(self,n_id,n_color,n_type,n_layers=50,pretrained=True):
        super(ICT_ResNet,self).__init__()
        if n_layers == 50:
            old_model= models.resnet50(pretrained=pretrained)
        elif n_layers == 34:
            old_model= models.resnet34(pretrained=pretrained)
        elif n_layers == 18:
            old_model= models.resnet18(pretrained=pretrained)
        else:
            raise NotImplementedError('resnet%s is not found'%(n_layers))

        for name,modules in old_model._modules.items():
            self.add_module(name,modules)
        self.fc = nn.Linear(self.fc.in_features,n_id)
        self.fc_c = nn.Linear(self.fc.in_features,n_color)
        self.fc_t = nn.Linear(self.fc.in_features,n_type)
        #########
        self.pretrained = pretrained
    def forward(self,x):
        for name,module in self._modules.items():
            if name.find('fc')==-1:
                x = module(x)
        x = x.view(x.size(0),-1)
        x_i = self.fc(x)
        x_c = self.fc_c(x)
        x_t = self.fc_t(x)
        return x_i,x_c,x_t

class TripletNet(nn.Module):
    def __init__(self, net):
        super(TripletNet, self).__init__()
        self.net = net

    def forward(self, x, y, z):
        pred_x, feat_x = self.net(x)
        pred_y, feat_y = self.net(y)
        pred_z, feat_z = self.net(z)
        dist_pos = F.pairwise_distance(feat_x, feat_y, 2)
        dist_neg = F.pairwise_distance(feat_x, feat_z, 2)
        return dist_pos, dist_neg, pred_x, pred_y, pred_z

if __name__ == '__main__':
    netM = ICT_ResNet(n_id=1000,n_color=7,n_type=7,n_layers=18,pretrained=True).cuda()

    print(netM)
    output = netM(Variable(torch.ones(1,3,224,224).cuda()/2.))
    print(output[0].size())
    print(output[1].size())
    print(output[2].size())
