# from googlenet_cfg import googlenet as g1c
# import torch
#
#
# path='./pruned_googlenet_slr.pth.tar'
# checkpoint=torch.load(path)
# model = g1c(cfg=checkpoint['cfg'])
# model.load_state_dict(checkpoint['state_dict'])
# model = model.cuda()

# print(model)
# from torchsummary import summary
# #
# #
# summary(model, (3,224,224))