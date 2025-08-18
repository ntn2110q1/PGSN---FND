import argparse
import random
import time
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import copy as cp
import math

import torch
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, TransformerConv,  DataParallel
from torch.utils.data import random_split
from torch_geometric.data import DataLoader, DataListLoader
import torch_geometric


from utils.data_loader import *
from utils.eval_helper import *


"""

The GCN, GAT, GraphSAGE and Graph Transformer implementation

"""

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            input: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x.permute(1, 0, 2)
        x = x + self.pe[:x.size(0)]
        x = self.dropout(x)
        x = x.permute(1, 0, 2)
        return x

class Model(torch.nn.Module):
    def __init__(self, args, concat=False):
        super(Model, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        self.model = args.model
        self.concat = concat
        self.seq_layer_type = args.seq_layer_type
        self.num_seq_layers = args.num_seq_layers if hasattr(args, 'num_seq_layers') else 2
        self.pre_padding = False

        # GNN layer
        if self.model == 'gcn':
            self.conv1 = GCNConv(self.num_features, self.nhid)
        elif self.model == 'sage':
            self.conv1 = SAGEConv(self.num_features, self.nhid)
        elif self.model == 'gat':
            self.conv1 = GATConv(self.num_features, self.nhid)
        elif self.model == 'gtn':
            self.conv1 = TransformerConv(self.num_features, self.nhid)

        # Sequential layer
        if self.seq_layer_type == 'lstm':
            self.seq_layer = torch.nn.LSTM(self.num_features, self.nhid//2, num_layers=self.num_seq_layers, batch_first=True, bidirectional=True)
        elif self.seq_layer_type == 'transformer':
            self.pe = PositionalEncoding(self.num_features, dropout=0.0)
            self.seq_layer = torch.nn.Transformer(d_model=self.num_features, nhead=2, num_encoder_layers=self.num_seq_layers, num_decoder_layers=self.num_seq_layers, dim_feedforward=self.num_features//2, batch_first=True)
            self.pre_padding = True


        # Linear layers
        self.lin_seq = torch.nn.Linear(self.num_features if self.seq_layer_type in ['transformer', 'transformer_encoder'] else self.nhid, self.nhid)
        if self.concat:
            self.lin0 = torch.nn.Linear(self.num_features, self.nhid)
            self.lin1 = torch.nn.Linear(self.nhid * 3, self.nhid)  # Adjusted for graph + sequential + news
        else:
            self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)  # Graph + sequential

        self.lin2 = torch.nn.Linear(self.nhid, self.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = None
        num_graphs = data.num_graphs

        # GNN processing
        x_graph = F.relu(self.conv1(x, edge_index, edge_attr))
        x_graph = gmp(x_graph, batch)  # Graph representation

        # Sequential processing
        seq_x = torch_geometric.utils.unbatch(data.x, batch)
        if self.pre_padding:
            seq_x = tuple(map(lambda s: s.flip(0), seq_x))
            seq_x = pad_sequence(seq_x, batch_first=True)
            seq_x = seq_x.flip(1)
        else:
            seq_x = pad_sequence(seq_x, batch_first=True)
        pad_mask = seq_x.sum(-1) == 0

        if self.seq_layer_type == 'transformer':
            seq_x = self.pe(seq_x * math.sqrt(self.num_features))
            seq_x = self.seq_layer(seq_x, seq_x, src_key_padding_mask=pad_mask, tgt_key_padding_mask=pad_mask)
        elif self.seq_layer_type == 'lstm':
            seq_x, _ = self.seq_layer(seq_x)

        # Extract sequential representation
        seq_x = seq_x[:, -1, :] if self.pre_padding else seq_x[:, 0, :]
        x_seq = F.leaky_relu(F.dropout(self.lin_seq(seq_x), p=self.dropout_ratio, training=self.training))
        x_seq = x_seq.squeeze()

    # Concatenate with news node if concat=True
        if self.concat:
            news = torch.stack([data.x[(data.batch == idx).nonzero().squeeze()[0]] for idx in range(num_graphs)])
            news = F.leaky_relu(self.lin0(news))
            x = torch.cat([x_graph, x_seq, news], dim=1)
        else:
            x = torch.cat([x_graph, x_seq], dim=1)

        x = F.relu(self.lin1(x))
        x = F.log_softmax(self.lin2(x), dim=-1)
        return x


@torch.no_grad()
def compute_test(loader, verbose=False):
	model.eval()
	loss_test = 0.0
	out_log = []
	for data in loader:
		if not args.multi_gpu:
			data = data.to(args.device)
		out = model(data)
		if args.multi_gpu:
			y = torch.cat([d.y.unsqueeze(0) for d in data]).squeeze().to(out.device)
		else:
			y = data.y
		if verbose:
			print(F.softmax(out, dim=1).cpu().numpy())
		out_log.append([F.softmax(out, dim=1), y])
		loss_test += F.nll_loss(out, y).item()
	return eval_deep(out_log, loader), loss_test


parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')

# hyper-parameters
parser.add_argument('--dataset', type=str, default='politifact', help='[politifact, gossipcop]')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--dropout_ratio', type=float, default=0.0, help='dropout ratio')
parser.add_argument('--epochs', type=int, default=35, help='maximum number of epochs')
parser.add_argument('--concat', type=bool, default=True, help='whether concat news embedding and graph embedding, sequence embedding')
parser.add_argument('--multi_gpu', type=bool, default=False, help='multi-gpu mode')
parser.add_argument('--feature', type=str, default='bert', help='feature type, [profile, spacy, bert, content]')
parser.add_argument('--model', type=str, default='sage', help='model type, [gcn, gat, sage, gtn]')
parser.add_argument('--seq_layer_type', type=str, default='transformer', help='sequential_layer type, ["transformer", "lstm"]')
parser.add_argument('--num_seq_layers', type=int, default=2)

args = parser.parse_args()
torch.manual_seed(args.seed)
if torch.cuda.is_available():
	torch.cuda.manual_seed(args.seed)

dataset = FNNDataset(root=r'your path', feature=args.feature, empty=False, name=args.dataset, transform=ToUndirected())

args.num_classes = dataset.num_classes
args.num_features = dataset.num_features

print(args)

time_dict = make_temporal_weight(dataset, name=args.dataset)

num_training = int(len(dataset) * 0.2)
num_val = int(len(dataset) * 0.1)
num_test = len(dataset) - (num_training + num_val)
training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])

if args.multi_gpu:
	loader = DataListLoader
else:
	loader = DataLoader

train_loader = loader(training_set, batch_size=args.batch_size, shuffle=True)
val_loader = loader(validation_set, batch_size=args.batch_size, shuffle=False)
test_loader = loader(test_set, batch_size=args.batch_size, shuffle=False)

model = Model(args, concat=args.concat)
if args.multi_gpu:
	model = DataParallel(model)
model = model.to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


if __name__ == '__main__':
	# Model training

	min_loss = 1e10
	val_loss_values = []
	best_epoch = 0

	t = time.time()
	model.train()
	for epoch in tqdm(range(args.epochs)):
		loss_train = 0.0
		out_log = []
		for i, data in enumerate(train_loader):
			optimizer.zero_grad()
			if not args.multi_gpu:
				data = data.to(args.device)
			out = model(data)
			if args.multi_gpu:
				y = torch.cat([d.y.unsqueeze(0) for d in data]).squeeze().to(out.device)
			else:
				y = data.y
			loss = F.nll_loss(out, y)
			loss.backward()
			optimizer.step()
			loss_train += loss.item()
			out_log.append([F.softmax(out, dim=1), y])
		acc_train, _, _, _, recall_train, auc_train, _ = eval_deep(out_log, train_loader)
		[acc_val, _, _, _, recall_val, auc_val, _], loss_val = compute_test(val_loader)
		print(f'loss_train: {loss_train:.4f}, acc_train: {acc_train:.4f},'
			  f' recall_train: {recall_train:.4f}, auc_train: {auc_train:.4f},'
			  f' loss_val: {loss_val:.4f}, acc_val: {acc_val:.4f},'
			  f' recall_val: {recall_val:.4f}, auc_val: {auc_val:.4f}')

	[acc, f1_macro, f1_micro, precision, recall, auc, ap], test_loss = compute_test(test_loader, verbose=False)
	print(f'Test set results: acc: {acc:.4f}, f1_macro: {f1_macro:.4f}, f1_micro: {f1_micro:.4f}, '
		  f'precision: {precision:.4f}, recall: {recall:.4f}, auc: {auc:.4f}, ap: {ap:.4f}')

