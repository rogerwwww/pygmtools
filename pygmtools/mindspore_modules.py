# Copyright (c) 2022 Thinklab@SJTU
# pygmtools is licensed under Mulan PSL v2.

import math
import numpy as np
import mindspore
import mindspore.ops as ops


def relu(x):
    return ops.relu(x)


def normalize_abs(array, axis):
    denom = ops.sum(ops.abs(array), axis=axis, keepdims=True)
    return ops.div_no_nan(array, denom)


class WeightedInnerProdAffinity:
    def __init__(self, d):
        self.d = d
        stdv = 1. / math.sqrt(d)
        init = np.random.uniform(-stdv, stdv, (d, d)).astype(np.float32) + np.eye(d, dtype=np.float32)
        self.A = mindspore.Tensor(init)

    def forward(self, X, Y):
        return ops.matmul(ops.matmul(X, self.A), Y.swapaxes(1, 2))


class Linear:
    def __init__(self, in_features, out_features, bias=True):
        bound = 1 / math.sqrt(in_features) if in_features > 0 else 0
        weight = np.random.uniform(-bound, bound, (out_features, in_features)).astype(np.float32)
        self.weight = mindspore.Tensor(weight)
        self.bias = mindspore.Tensor(np.random.uniform(-bound, bound, (out_features,)).astype(np.float32)) if bias else None

    def forward(self, x):
        y = ops.matmul(x, self.weight.swapaxes(-1, -2))
        if self.bias is not None:
            y = y + self.bias
        return y


class Sequential:
    def __init__(self, *args):
        self._modules = {idx: module for idx, module in enumerate(args)}

    def getitem(self, idx):
        return self._modules[idx]

    def forward(self, inputs):
        for module in self._modules.values():
            inputs = module.forward(inputs)
        return inputs


class ReLU:
    def forward(self, x):
        return relu(x)


class Gconv:
    def __init__(self, in_features, out_features):
        self.a_fc = Linear(in_features, out_features)
        self.u_fc = Linear(in_features, out_features)

    def forward(self, A, x, norm=True):
        if norm:
            A = normalize_abs(A, axis=-2)
        return ops.matmul(A, relu(self.a_fc.forward(x))) + relu(self.u_fc.forward(x))


class ChannelIndependentConv:
    def __init__(self, in_features, out_features, in_edges, out_edges=None):
        if out_edges is None:
            out_edges = out_features
        self.node_fc = Linear(in_features, out_features)
        self.node_sfc = Linear(in_features, out_features)
        self.edge_fc = Linear(in_edges, out_edges)

    def forward(self, A, emb_node, emb_edge, mode=1):
        if mode != 1:
            raise ValueError(f'Unknown mode {mode}. Possible options: 1 or 2')
        node_x = self.node_fc.forward(emb_node)
        node_sx = self.node_sfc.forward(emb_node)
        edge_x = self.edge_fc.forward(emb_edge)
        node_x = ops.einsum('bijf,bjf->bif', ops.expand_dims(A, -1) * edge_x, node_x)
        return relu(node_x) + relu(node_sx), relu(edge_x)


class Siamese_Gconv:
    def __init__(self, in_features, num_features):
        self.gconv = Gconv(in_features, num_features)

    def forward(self, g1, *args):
        emb1 = self.gconv.forward(*g1)
        if len(args) == 0:
            return emb1
        returns = [emb1]
        for g in args:
            returns.append(self.gconv.forward(*g))
        return returns


class Siamese_ChannelIndependentConv:
    def __init__(self, in_features, num_features, in_edges, out_edges=None):
        self.gconv = ChannelIndependentConv(in_features, num_features, in_edges, out_edges)

    def forward(self, g1, *args):
        emb1, emb_edge1 = self.gconv.forward(*g1)
        embs, emb_edges = [emb1], [emb_edge1]
        for g in args:
            emb2, emb_edge2 = self.gconv.forward(*g)
            embs.append(emb2)
            emb_edges.append(emb_edge2)
        return embs + emb_edges


class NGMConvLayer:
    def __init__(self, in_node_features, in_edge_features, out_node_features, out_edge_features, sk_channel=0):
        self.sk_channel = sk_channel
        assert out_node_features == out_edge_features + sk_channel
        if sk_channel > 0:
            self.out_nfeat = out_node_features - sk_channel
            self.classifier = Linear(self.out_nfeat, sk_channel)
        else:
            self.out_nfeat = out_node_features
            self.classifier = None
        self.n_func = Sequential(
            Linear(in_node_features, self.out_nfeat),
            ReLU(),
            Linear(self.out_nfeat, self.out_nfeat),
            ReLU(),
        )
        self.n_self_func = Sequential(
            Linear(in_node_features, self.out_nfeat),
            ReLU(),
            Linear(self.out_nfeat, self.out_nfeat),
            ReLU(),
        )

    def forward(self, A, W, x, n1=None, n2=None, norm=True, sk_func=None):
        W_new = W
        if norm:
            A = normalize_abs(A, axis=2)
        x1 = self.n_func.forward(x)
        x2 = ops.einsum('bijd,bjd->bid', ops.expand_dims(A, -1) * W_new, x1) + self.n_self_func.forward(x)
        if self.classifier is None:
            return W_new, x2
        x3 = self.classifier.forward(x2)
        n1_rep = ops.tile(n1, (self.sk_channel,))
        n2_rep = ops.tile(n2, (self.sk_channel,))
        max_n1 = n1.max()
        max_n2 = n2.max()
        x4 = x3.swapaxes(1, 2).reshape((-1, max_n2, max_n1)).swapaxes(1, 2)
        x5 = sk_func(x4, n1_rep, n2_rep, dummy_row=True).swapaxes(1, 2)
        x6 = x5.reshape((x.shape[0], self.sk_channel, max_n1 * max_n2)).swapaxes(1, 2)
        return W_new, ops.concat((x2, x6), axis=-1)
