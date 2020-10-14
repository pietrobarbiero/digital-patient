import dgl
from dgl.data import CoraGraphDataset
import torch

data = CoraGraphDataset()

g = data[0]
cuda = False

features = g.ndata['feat']
labels = g.ndata['label']
train_mask = g.ndata['train_mask']
val_mask = g.ndata['val_mask']
test_mask = g.ndata['test_mask']
in_feats = features.shape[1]
n_classes = data.num_labels
n_edges = data.graph.number_of_edges()
print("""----Data statistics------'
  #Edges %d
  #Classes %d
  #Train samples %d
  #Val samples %d
  #Test samples %d""" %
      (n_edges, n_classes,
       train_mask.int().sum().item(),
       val_mask.int().sum().item(),
       test_mask.int().sum().item()))

# add self loop
g = dgl.remove_self_loop(g)
g = dgl.add_self_loop(g)
n_edges = g.number_of_edges()

# normalization
degs = g.in_degrees().float()
norm = torch.pow(degs, -0.5)
norm[torch.isinf(norm)] = 0
if cuda:
    norm = norm.cuda()
g.ndata['norm'] = norm.unsqueeze(1)