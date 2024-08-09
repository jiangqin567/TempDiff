# import torch
# h= torch.randn(15, 3, 8, 8)
# emb_out = torch.randn(3, 3, 1, 1)
# if h.shape[0] != emb_out.shape[0]:
#     emb_out = torch.repeat_interleave(emb_out,  h.shape[0] //  emb_out.shape[0], dim=0)
#
# print('Hidden state shape: {}, Time embedding shape: {}'.format(h.shape, emb_out.shape))
# h = h + emb_out
#
# print(h.shape)
#
import torch
import pyiqa

print(pyiqa.list_models())