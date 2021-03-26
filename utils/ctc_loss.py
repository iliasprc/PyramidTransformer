import numpy as np
#import pyximport
import torch.nn as nn

#from utils.ctc_losses.ctc_ent import ctc_ent_cost

#pyximport.install(setup_args={"include_dirs": np.get_include()})

# from utils.ctc_losses import ctc_fast_cui as ctc_fast_cui, soft_alignments, alignments, dpd_decode

import torch


def calc_gradient_penalty(args, netD, real_data, fake_data, LAMBDA=1.0):
    use_cuda = torch.cuda.is_available()
    if (False):
        interpolates = real_data
    else:
        interpolates = fake_data

    if use_cuda:
        interpolates = interpolates.cuda()

    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    # TODO: Make ConvBackward diffentiable
    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(
                                        disc_interpolates.size()).cuda() if use_cuda else torch.ones(
                                        disc_interpolates.size()),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty



class Discriminator_loss(nn.Module):
    def __init__(self, args, reduction='mean', real_label=1, fake_label=0):
        super(Discriminator_loss, self).__init__()
        print('{} Discriminator loss'.format(args.adversarial_loss))
        self.args = args
        if args.adversarial_loss == 'mse':
            self.crit = nn.MSELoss(reduction=reduction)
            self.adversarial_loss = self.mse
        elif args.adversarial_loss == 'bce':

            self.crit = nn.BCELoss(reduction=reduction)
            self.adversarial_loss = self.bce_logits
        elif args.adversarial_loss == 'bcelogits':

            self.crit = nn.BCEWithLogitsLoss(reduction=reduction)
            self.adversarial_loss = self.bce_logits
        elif args.adversarial_loss == 'vanilla':
            self.adversarial_loss = self.vanilla
        elif args.adversarial_loss == 'wgan':
            self.adversarial_loss = self.wgan_loss
        self.real_target = torch.tensor([real_label], dtype=torch.float).view(1, 1)
        self.fake_target = torch.tensor([fake_label], dtype=torch.float).view(1, 1)
        if self.args.cuda:
            self.real_target = self.real_target.cuda()
            self.fake_target = self.fake_target.cuda()

    def forward(self, x_real, x_fake):

        return self.adversarial_loss(x_real, x_fake)

    def vanilla(self, x_real, x_fake):

        return torch.mean(torch.log(torch.sigmoid(x_real)) - torch.log(1 - torch.sigmoid(x_fake)))

    def wgan_loss(self, x_real, x_fake):

        return -(torch.mean(x_real) - torch.mean(x_fake))

    def bce_logits(self, x_real, x_fake):
        # print(f'real {x_real.shape} fake {x_fake.shape}')
        B, T, c = x_real.shape
        B, K, c = x_fake.shape
        # print(x_real.shape,self.real_target.repeat(B, T, c).shape)
        return self.crit(x_real, self.real_target.repeat(B, T, c)) + self.crit(x_fake, self.fake_target.repeat(B, K, c))

    def mse(self, x_real, x_fake):
        B, T, c = x_real.shape
        B, K, c = x_fake.shape
        return self.crit(x_real, self.real_target.repeat(B, T, c)) + self.crit(x_fake, self.fake_target.repeat(B, K, c))


class Generator_loss(nn.Module):
    def __init__(self, args, reduction='mean', landa=0.001, real_label=1, fake_label=0):
        super(Generator_loss, self).__init__()
        self.args = args
        self.landa = landa
        print('{} Generator loss'.format(args.adversarial_loss))

        if args.adversarial_loss == 'mse':
            self.crit = nn.MSELoss(reduction=reduction)
            self.adversarial_loss = self.mse
        elif args.adversarial_loss == 'bcelogits':

            self.crit = nn.BCEWithLogitsLoss(reduction=reduction)
            self.adversarial_loss = self.bce_logits
        elif args.adversarial_loss == 'bce':

            self.crit = nn.BCEWithLogitsLoss(reduction=reduction)
            self.adversarial_loss = self.bce_logits
        elif args.adversarial_loss == 'vanilla':
            self.adversarial_loss = self.vanilla
        elif args.adversarial_loss == 'wgan':
            self.adversarial_loss = self.wgan_loss
        self.real_target = torch.tensor([real_label], dtype=torch.float).view(1, 1)
        self.fake_target = torch.tensor([fake_label], dtype=torch.float).view(1, 1)
        if self.args.cuda:
            self.real_target = self.real_target.cuda()
            self.fake_target = self.fake_target.cuda()

    def forward(self, x_fake):

        return self.landa * self.adversarial_loss(x_fake)

    def vanilla(self, x_fake):

        return -self.landa * torch.mean(torch.log(torch.sigmoid(x_fake)))

    def wgan_loss(self, x_fake):

        return -self.landa * torch.mean(x_fake)

    def bce_logits(self, x_fake):
        # print(x_real.shape,x_fake.shape)
        # B,T,c = x_real.shape
        B, K, c = x_fake.shape
        return self.landa * self.crit(x_fake, self.real_target.repeat(B, K, c))

    def mse(self, x_fake):

        return self.landa * self.crit(x_fake, self.real_target)


class Dist(nn.Module):
    def __init__(self, crit):
        super(Dist, self).__init__()
        self.distance = torch.nn.CosineSimilarity(dim=-1)
        self.crit = crit
        if (crit == 'mse'):
            # self.crit = nn.MSELoss(reduction='sum')
            self.distance = torch.nn.PairwiseDistance()

    def forward(self, x1, x2):
        if (self.crit == 'mse'):
            return self.distance(x1, x2)
        else:
            return 1.0 - self.distance(x1, x2)


class CTC_Loss(nn.Module):
    def __init__(self, crit='normal', average=True, alpha=0.99, gamma=2.0, beta=0.1, return_ctc_cost=False):
        super(CTC_Loss, self).__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.crit = crit
        self.average = average
        self.return_ctc_cost = return_ctc_cost

        if (crit == 'normal'):
            self.loss = self.normal_ctc_loss



        elif (crit == 'focal'):
            self.loss = self.focal_ctc_loss



    def forward(self, output, target):

        cost = self.loss(output, target)

        return cost

    def normal_ctc_loss(self, log_probs, target):
        if (self.average):
            criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        else:
            criterion = nn.CTCLoss(blank=0, reduction='sum', zero_infinity=True)
        input_len = torch.tensor([log_probs.size(0)], dtype=torch.int)
        target_len = torch.tensor([target.size(1)], dtype=torch.int)
        loss = criterion(nn.functional.log_softmax(log_probs, dim=2), target, input_len, target_len)
        return loss

    def Aggregation_CE(self, outputs, target):

        Time, batch_size, N_classes = outputs.size()
        probs = nn.functional.softmax(outputs, dim=-1)
        target[:, 0] = 0.0
        input = torch.sum(probs, dim=0)
        input = input / float(Time)
        target = target / float(Time)

        loss = (-torch.sum(torch.log(input) * target))
        return loss

    def focal_ctc_loss(self, log_probs, target):

        if (self.average):
            criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
        else:
            criterion = nn.CTCLoss(blank=0, reduction='sum', zero_infinity=True)
        input_len = torch.tensor([log_probs.size(0)], dtype=torch.int)
        target_len = torch.tensor([target.size(1)], dtype=torch.int)
        loss = criterion(nn.functional.log_softmax(log_probs, dim=2), target, input_len, target_len)
        p = torch.exp((-1) * loss)
        focal_loss = self.alpha * ((1 - p) ** self.gamma) * loss
        return focal_loss







class GAN_align(nn.Module):
    def __init__(self):
        super(GAN_align, self).__init__()
        self.loss = nn.BCELoss()

    def forward(self, x1, x2, gamma):
        loss_align = 0.0
        # print(x1.shape, x2.shape,gamma.shape)
        x1g = torch.mm(x1.squeeze(-1), gamma)
        loss_align = torch.mm(x1g, x2.squeeze(0))
        # print(loss_align,loss_align.shape)
        return loss_align


class L_align(nn.Module):

    def __init__(self, crit='mse', average=True):
        super(L_align, self).__init__()
        print("Loss align criterion {}".format(crit))

        self.distance = Dist(crit)

    def forward(self, x1, x2, gamma):
        loss_align = 0.0
        # print(x1.shape, x2.shape)
        size_target = x2.size(0)
        x2 = x2[0:-1, :, :]
        x1 = x1.squeeze(1)
        x2 = x2.repeat(1, x1.size(0), 1)
        # print(x1.shape,x2.shape)
        for i in range(x2.size(0)):
            # print(gamma.shape , x1.shape, x2[i, :, :].squeeze(0).shape)
            inter = torch.sum((gamma[i, :] * self.distance(x1, x2[i, :, :].squeeze(0))), dim=-1)
            inter = torch.sum(inter)

            loss_align += inter
        # print(loss_align)
        loss_align /= x1.size(0) * size_target

        return loss_align

    def forward222(self, x1, x2, gamma):
        loss_align = 0.0
        # print(x1.shape, x2.shape)
        size_target = x2.size(0)
        x1 = x1[0:-1, :, :]
        x2 = x2.squeeze(1)
        x1 = x1.repeat(1, x2.size(0), 1)
        print(x1.shape, x2.shape)
        for i in range(x1.size(0)):
            print(gamma[i, :].shape, x2.shape, x1[i, :, :].squeeze(0).shape)
            inter = torch.sum((gamma[i, :] * self.distance(x2, x1[i, :, :].squeeze(0))), dim=-1)
            inter = torch.sum(inter)

            loss_align += inter
        # print(loss_align)
        loss_align /= x2.size(0) * size_target

        return loss_align


class L_div(nn.Module):

    def __init__(self, crit='kldiv', average=True):
        super(L_div, self).__init__()
        self.crit = crit
        if (crit == 'cosine'):
            self.loss = nn.CosineEmbeddingLoss()
        else:
            self.loss = nn.KLDivLoss(reduction='sum')

    def forward(self, x1, x_target, gamma):

        x_target = x_target[0:-1, :, :].squeeze(1)
        if (self.crit == 'kldiv'):
            x1 = torch.nn.functional.log_softmax(x1, dim=-1)
            x_target = torch.nn.functional.softmax(x_target, dim=-1)
        loss_align = 0.0

        KT = x_target.size(0) * x1.size(0)
        # x1 = x1[0:-1, :, :]
        x1 = x1.repeat(1, x_target.size(0), 1)
        gamma = gamma.t()

        for i in range(x1.size(0)):
            l = gamma[i, :] * torch.sum(x_target * x1[i, :, :])

            loss_align += l

        loss_align = (-1) * torch.sum(loss_align)
        # print(l)
        return loss_align / KT

    def forward1(self, x1, x_target, gamma):

        x_target = x_target[0:-1, :, :].squeeze(1)
        if (self.crit == 'kldiv'):
            x1 = torch.nn.functional.log_softmax(x1, dim=-1)
            x_target = torch.nn.functional.softmax(x_target, dim=-1)
        loss_align = 0.0

        KT = x_target.size(0) * x1.size(0)

        x1 = x1.repeat(1, x_target.size(0), 1)
        gamma = gamma.t()
        print(gamma.shape, x1.size, x_target.size)
        for i in range(x1.size(0)):
            loss_align += gamma[i, :] * torch.sum(self.loss(x1[i, :, :], x_target))
        loss_align = torch.sum(loss_align)
        return loss_align / KT


class FocalLoss(nn.Module):

    def __init__(self, weight=None,
                 gamma=0., reduction='mean'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        self.criterion = nn.NLLLoss(reduction=reduction)

    def forward(self, input_tensor, target_tensor):
        # log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(input_tensor)
        return self.criterion(
            ((1 - prob) ** self.gamma) * input_tensor,
            target_tensor,

        )


'''

    def custom_ctc_loss(self, probs, labels):

        probs1, labels1 = torch.softmax(probs, dim=-1).cpu().squeeze(1).permute(1, 0).detach(), labels.squeeze(
            0).detach()

        norm = labels1.size(0)

        cost, grad = normal_ctc_fast.ctc_loss(probs1.cpu().numpy().astype(np.float64, order='F'),
                                              labels1.cpu().numpy().astype(np.int32), blank=0)

        cost = torch.tensor(cost)
        grads = (torch.tensor(grad, dtype=probs1.dtype)).permute(1, 0).unsqueeze(1)

        if (self.average):
            grads = grads / norm
            cost = cost / norm

        return grads, cost



    def Aggregation_CE(self, outputs, target):

        Time, batch_size, N_classes = outputs.size()
        probs = nn.functional.softmax(outputs, dim=-1)
        target[:, 0] = 0.0
        input = torch.sum(probs, dim=0)
        input = input / float(Time)
        target = target / float(Time)


        loss = (-torch.sum(torch.log(input) * target))
        return loss




'''
