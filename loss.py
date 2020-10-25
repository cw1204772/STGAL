import math
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

class TripletLoss(nn.Module):
    '''
    Original margin ranking loss:
        loss(x1, x2, y) = max(0, -y * (x1 - x2) + margin)
    
    Let z = -y * (x1 - x2)

    Soft_margin mode:
        loss(x1, x2, y) = log(1 + exp(z))
    Batch_hard mode:
        z = -y * (x1' - x2'),
        where x1' is the max x1 within a batch,
        x2' is the min x2 within a batch
    '''
    def __init__(self, margin=0, batch_hard=False, dist_type='eucl'):
        """
        Args:
            margin: int or 'soft'
            batch_hard: whether to use batch_hard loss
        """
        super(TripletLoss, self).__init__()
        self.batch_hard = batch_hard
        self.dist_type = dist_type
        if isinstance(margin, float) or margin == 'soft':
            self.margin = margin
        else:
            raise NotImplementedError(
                'The margin {} is not recognized in TripletLoss()'.format(margin))

    def forward(self, feat, id=None, pos_mask=None, neg_mask=None, mode='id'):
        if self.dist_type == 'eucl':
            dist = self.cdist(feat, feat)
        elif self.dist_type == 'norm_eucl':
            dist = self.cdist(feat, feat, normalize=True)
        elif self.dist_type == 'cos':
            dist = self.cosdist(feat, feat)
        elif self.dist_type == 'cos_linear':
            dist = self.cosdist(feat, feat, linear=True)
        else:
            raise NotImplementedError('Unrecognized dist type "%s"' % mode)

        if mode == 'id':
            if id is None:
                 raise RuntimeError('foward is in id mode, please input id!')
            else:
                 identity_mask = Variable(torch.eye(feat.size(0)).byte())
                 identity_mask = identity_mask.cuda() if id.is_cuda else identity_mask
                 same_id_mask = torch.eq(id.unsqueeze(1), id.unsqueeze(0))
                 negative_mask = same_id_mask ^ 1
                 positive_mask = same_id_mask ^ identity_mask
        elif mode == 'mask':
            if pos_mask is None or neg_mask is None:
                 raise RuntimeError('foward is in mask mode, please input pos_mask & neg_mask!')
            else:
                 positive_mask = pos_mask
                 same_id_mask = neg_mask ^ 1
                 negative_mask = neg_mask
        else:
            raise ValueError('unrecognized mode')
        if self.batch_hard:
            max_positive = (dist * positive_mask.float()).max(1)[0]
            min_negative = (dist + 1e5*same_id_mask.float()).min(1)[0]
            z = max_positive - min_negative
        else:
            pos = positive_mask.topk(k=1, dim=1)[1].view(-1,1)
            positive = torch.gather(dist, dim=1, index=pos)
            pos = negative_mask.topk(k=1, dim=1)[1].view(-1,1)
            negative = torch.gather(dist, dim=1, index=pos)
            z = positive - negative
        if isinstance(self.margin, float):
            b_loss = torch.clamp(z + self.margin, min=0)
        elif self.margin == 'soft':
            b_loss = torch.log(1 + torch.exp(z))
        else:
            raise NotImplementedError("How do you even get here!")
        return b_loss
            
    def cdist(self, a, b, normalize=False):
        '''
        Returns euclidean distance between a and b
        
        Args:
             a (2D Tensor): A batch of vectors shaped (B1, D)
             b (2D Tensor): A batch of vectors shaped (B2, D)
        Returns:
             A matrix of all pairwise distance between all vectors in a and b,
             will be shape of (B1, B2)
        '''
        if normalize:
            a = nn.functional.normalize(a, dim=1)
            b = nn.functional.normalize(b, dim=1)
        diff = a.unsqueeze(1) - b.unsqueeze(0)
        return ((diff**2).sum(2)+1e-12).sqrt()

    def cosdist(self, feat1, feat2, linear=False):
        """Cosine distance"""
        feat1 = torch.nn.functional.normalize(feat1, dim=1)
        feat2 = torch.nn.functional.normalize(feat2, dim=1).transpose(0, 1)
        cos_sim = torch.mm(feat1, feat2)
        cos_sim = torch.clamp(cos_sim, -1+1e-4, 1-1e-4)
        if linear:
            prob = (cos_sim + 1) / 2
        else:
            prob = 1 - (torch.acos(cos_sim) / math.pi)
        if (torch.isnan(prob)).sum() != 0:
            raise ValueError('prob smaller than 0')
        return -1 * torch.log(prob + 1e-12)

if __name__ == '__main__':
    criterion0 = TripletLoss(margin=0.5, batch_hard=False)
    criterion1 = TripletLoss(margin=0.5, batch_hard=True)
    
    t = np.random.randint(3, size=(10,))
    print(t)
    
    feat = Variable(torch.rand(10, 2048), requires_grad=True).cuda()
    id = Variable(torch.from_numpy(t), requires_grad=True).cuda()
    loss0 = criterion0(feat, id)
    loss1 = criterion1(feat, id)
    print('no batch hard:', loss0)
    print('batch hard:', loss1)
    loss0.backward()
    loss1.backward()
