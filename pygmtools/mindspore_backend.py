from multiprocessing import Pool
import numpy as np
import mindspore
import mindspore.nn as nn
from mindspore.ops import stop_gradient
import math
import os
import itertools

import inspect
import functools
from pygmtools.mindspore_modules import WeightedInnerProdAffinity, Linear, Siamese_Gconv, \
    Siamese_ChannelIndependentConv, NGMConvLayer
_max_signature = inspect.signature(mindspore.ops.max)
if 'keep_dims' in _max_signature.parameters:
    def _ms_max(*args, keep_dims=False, **kwargs):
        return mindspore.ops.max(*args, keep_dims=keep_dims, **kwargs)
elif 'keepdims' in _max_signature.parameters:
    def _ms_max(*args, keep_dims=False, **kwargs):
        max, indices = mindspore.ops.max(*args, keepdims=keep_dims, **kwargs)
        return indices, max
else:
    raise ValueError('Mindspore function mindspore.ops.max has unsupported signature. It is likely you are working with '
                     'a new Mindspore version which breaks backward compatibility. Please report your Mindspore version '
                     'to GitHub issues.')

_logsumexp_signature = inspect.signature(mindspore.ops.logsumexp)
if 'keep_dims' in _logsumexp_signature.parameters:
    _ms_logsumexp_keepdim = functools.partial(mindspore.ops.logsumexp, keep_dims=True)
elif 'keepdim' in _logsumexp_signature.parameters:
    _ms_logsumexp_keepdim = functools.partial(mindspore.ops.logsumexp, keepdim=True)
else:
    raise ValueError('Mindspore function mindspore.ops.logsumexp has unsupported signature. It is likely you are '
                     'working with a new Mindspore version which breaks backward compatibility. Please report your '
                     'Mindspore version to GitHub issues.')

_norm_signature = inspect.signature(mindspore.ops.norm)
if 'axis' in _norm_signature.parameters and 'p' in _norm_signature.parameters and 'keep_dims' in _norm_signature.parameters:
    def _ms_norm(*args, p=None, axis=None, keep_dims=False, **kwargs):
        return mindspore.ops.norm(*args, p=p, axis=axis, keep_dims=keep_dims, **kwargs)
elif 'dim' in _norm_signature.parameters and 'ord' in _norm_signature.parameters and 'keepdim' in _norm_signature.parameters:
    def _ms_norm(*args, p=None, axis=None, keep_dims=False, **kwargs):
        return mindspore.ops.norm(*args, ord=p, dim=axis, keepdim=keep_dims, **kwargs)
else:
    raise ValueError('Mindspore function mindspore.ops.norm has unsupported signature. It is likely you are '
                     'working with a new Mindspore version which breaks backward compatibility. Please report your '
                     'Mindspore version to GitHub issues.')


#############################################
#     Linear Assignment Problem Solvers     #
#############################################

from pygmtools.numpy_backend import _hung_kernel


def hungarian(s: mindspore.Tensor, n1: mindspore.Tensor = None, n2: mindspore.Tensor = None,
              unmatch1: mindspore.Tensor = None, unmatch2: mindspore.Tensor = None,
              nproc: int = 1) -> mindspore.Tensor:
    """
    mindspore implementation of Hungarian algorithm
    """
    # device = s.device
    batch_num = s.shape[0]

    perm_mat = stop_gradient(s).asnumpy() * -1
    if n1 is not None:
        n1 = n1.asnumpy()
    else:
        n1 = [None] * batch_num
    if n2 is not None:
        n2 = n2.asnumpy()
    else:
        n2 = [None] * batch_num
    if unmatch1 is not None:
        unmatch1 = -unmatch1.asnumpy()
    else:
        unmatch1 = [None] * batch_num
    if unmatch2 is not None:
        unmatch2 = -unmatch2.asnumpy()
    else:
        unmatch2 = [None] * batch_num

    if nproc > 1:
        with Pool(processes=nproc) as pool:
            mapresult = pool.starmap_async(_hung_kernel, zip(perm_mat, n1, n2, unmatch1, unmatch2))
            perm_mat = np.stack(mapresult.get())
    else:
        perm_mat = np.stack(
            [_hung_kernel(perm_mat[b], n1[b], n2[b], unmatch1[b], unmatch2[b]) for b in range(batch_num)])

    perm_mat = mindspore.Tensor(perm_mat)

    return perm_mat


def sinkhorn(s: mindspore.Tensor, nrows: mindspore.Tensor = None, ncols: mindspore.Tensor = None,
             unmatchrows: mindspore.Tensor = None, unmatchcols: mindspore.Tensor = None,
             dummy_row: bool = False, max_iter: int = 10, tau: float = 1.,
             batched_operation: bool = False) -> mindspore.Tensor:
    """
    mindspore implementation of Sinkhorn algorithm
    """
    batch_size = s.shape[0]

    if s.shape[2] >= s.shape[1]:
        transposed = False
    else:
        s = s.swapaxes(1, 2)
        nrows, ncols = ncols, nrows
        unmatchrows, unmatchcols = unmatchcols, unmatchrows
        transposed = True

    if nrows is None:
        nrows = mindspore.Tensor([s.shape[1] for _ in range(batch_size)])
    if ncols is None:
        ncols = mindspore.Tensor([s.shape[2] for _ in range(batch_size)])

    # ensure that in each dimension we have nrow < ncol
    transposed_batch = nrows > ncols
    if transposed_batch.any():
        s_t = s.swapaxes(1, 2)
        s_t = mindspore.ops.concat((
            s_t[:, :s.shape[1], :],
            mindspore.numpy.full((batch_size, s.shape[1], s.shape[2] - s.shape[1]), -float('inf'))),
            axis=2)
        s = mindspore.numpy.where(transposed_batch.view(batch_size, 1, 1), s_t, s)

        new_nrows = mindspore.numpy.where(transposed_batch, ncols, nrows)
        new_ncols = mindspore.numpy.where(transposed_batch, nrows, ncols)
        nrows = new_nrows
        ncols = new_ncols

        if unmatchrows is not None and unmatchcols is not None:
            unmatchrows_pad = mindspore.ops.concat((
                unmatchrows,
                mindspore.numpy.full((batch_size, unmatchcols.shape[1] - unmatchrows.shape[1]),
                                     -float('inf'))),
                axis=1)
            new_unmatchrows = mindspore.numpy.where(transposed_batch.view(batch_size, 1), unmatchcols, unmatchrows_pad)[
                              :,
                              :unmatchrows.shape[1]]
            new_unmatchcols = mindspore.numpy.where(transposed_batch.view(batch_size, 1), unmatchrows_pad, unmatchcols)
            unmatchrows = new_unmatchrows
            unmatchcols = new_unmatchcols

    # operations are performed on log_s
    log_s = s / tau
    if unmatchrows is not None and unmatchcols is not None:
        unmatchrows = unmatchrows / tau
        unmatchcols = unmatchcols / tau

    if dummy_row:
        if not log_s.shape[2] >= log_s.shape[1]:
            raise RuntimeError('Error in Sinkhorn with dummy row')
        dummy_shape = list(log_s.shape)
        dummy_shape[1] = log_s.shape[2] - log_s.shape[1]
        ori_nrows = nrows
        nrows = ncols.copy()
        log_s = mindspore.ops.concat((log_s, mindspore.numpy.full(dummy_shape, -float('inf'), dtype=log_s.dtype)),
                                     axis=1)
        if unmatchrows is not None:
            unmatchrows = mindspore.ops.concat((unmatchrows,
                                                mindspore.numpy.full((dummy_shape[0], dummy_shape[1]),
                                                                     -float('inf'), dtype=log_s.dtype
                                                                     )), axis=1)
        for b in range(batch_size):
            log_s[b, int(ori_nrows[b]):int(nrows[b]), :int(ncols[b])] = -100

    # assign the unmatch weights
    if unmatchrows is not None and unmatchcols is not None:
        new_log_s = mindspore.numpy.full((log_s.shape[0], log_s.shape[1] + 1, log_s.shape[2] + 1),
                                         -float('inf'), dtype=log_s.dtype
                                         )
        new_log_s[:, :-1, :-1] = log_s
        log_s = new_log_s
        for b in range(batch_size):
            r, c = int(nrows[b]), int(ncols[b])
            log_s[b, 0:r, c] = unmatchrows[b, 0:r]
            log_s[b, r, 0:c] = unmatchcols[b, 0:c]
    row_mask = mindspore.numpy.zeros((batch_size, log_s.shape[1], 1), dtype=mindspore.bool_)
    col_mask = mindspore.numpy.zeros((batch_size, 1, log_s.shape[2]), dtype=mindspore.bool_)
    for b in range(batch_size):
        r, c = int(nrows[b]), int(ncols[b])
        row_mask[b, 0:r, 0] = 1
        col_mask[b, 0, 0:c] = 1
    if unmatchrows is not None and unmatchcols is not None:
        ncols += 1
        nrows += 1

    if batched_operation:
        for b in range(batch_size):
            log_s[b, int(nrows[b]):, :] = -float('inf')
            log_s[b, :, int(ncols[b]):] = -float('inf')

        for i in range(max_iter):
            if i % 2 == 0:
                index, m = _ms_max(log_s, axis=2, keep_dims=True)
                log_sum = _ms_logsumexp_keepdim(log_s - m, 2) + m
                log_s = log_s - mindspore.numpy.where(row_mask, log_sum, mindspore.numpy.zeros_like(log_sum))
                if mindspore.ops.isnan(log_s).any():
                    raise RuntimeError(f'NaN encountered in Sinkhorn iter_num={i}/{max_iter}')
            else:
                index, m = _ms_max(log_s, axis=1, keep_dims=True)
                log_sum = _ms_logsumexp_keepdim(log_s - m, 1) + m
                log_s = log_s - mindspore.numpy.where(col_mask, log_sum, mindspore.numpy.zeros_like(log_sum))
                if mindspore.ops.isnan(log_s).any():
                    raise RuntimeError(f'NaN encountered in Sinkhorn iter_num={i}/{max_iter}')

        ret_log_s = log_s
    else:
        ret_log_s = mindspore.numpy.full((batch_size, log_s.shape[1], log_s.shape[2]), -float('inf'), dtype=log_s.dtype)

        for b in range(batch_size):
            row_slice = slice(0, int(nrows[b]))
            col_slice = slice(0, int(ncols[b]))
            log_s_b = log_s[b, row_slice, col_slice]
            row_mask_b = row_mask[b, row_slice, :]
            col_mask_b = col_mask[b, :, col_slice]

            for i in range(max_iter):
                if i % 2 == 0:
                    index, m = _ms_max(log_s_b, axis=1, keep_dims=True)
                    log_sum = _ms_logsumexp_keepdim(log_s_b - m, 1) + m
                    log_s_b = log_s_b - mindspore.numpy.where(row_mask_b, log_sum, mindspore.numpy.zeros_like(log_sum))
                else:
                    index, m = _ms_max(log_s_b, axis=0, keep_dims=True)
                    log_sum = _ms_logsumexp_keepdim(log_s_b - m, 0) + m
                    log_s_b = log_s_b - mindspore.numpy.where(col_mask_b, log_sum, mindspore.numpy.zeros_like(log_sum))

            ret_log_s[b, row_slice, col_slice] = log_s_b

    if unmatchrows is not None and unmatchcols is not None:
        ncols -= 1
        nrows -= 1
        for b in range(batch_size):
            ret_log_s[b, :nrows[b] + 1, ncols[b]] = -float('inf')
            ret_log_s[b, nrows[b], :ncols[b]] = -float('inf')
        ret_log_s = ret_log_s[:, :-1, :-1]

    if dummy_row:
        if dummy_shape[1] > 0:
            ret_log_s = ret_log_s[:, :-dummy_shape[1]]
        for b in range(batch_size):
            ret_log_s[b, ori_nrows[b]:nrows[b], :ncols[b]] = -float('inf')

    if transposed_batch.any():
        s_t = ret_log_s.swapaxes(1, 2)
        s_t = mindspore.ops.concat((
            s_t[:, :ret_log_s.shape[1], :],
            mindspore.numpy.full((batch_size, ret_log_s.shape[1], ret_log_s.shape[2] - ret_log_s.shape[1]),
                                 -float('inf'), )), axis=2)
        ret_log_s = mindspore.numpy.where(transposed_batch.view(batch_size, 1, 1), s_t, ret_log_s)

    if transposed:
        ret_log_s = ret_log_s.swapaxes(1, 2)

    return mindspore.ops.exp(ret_log_s)


#############################################
#    Quadratic Assignment Problem Solvers   #
#############################################


def rrwm(K: mindspore.Tensor, n1: mindspore.Tensor, n2: mindspore.Tensor, n1max, n2max, x0: mindspore.Tensor,
         max_iter: int, sk_iter: int, alpha: float, beta: float) -> mindspore.Tensor:
    """
    mindspore implementation of RRWM algorithm.
    """
    batch_num, n1, n2, n1max, n2max, n1n2, v0 = _check_and_init_gm(K, n1, n2, n1max, n2max, x0)
    # rescale the values in K
    d = K.sum(axis=2, keepdims=True)
    dmax = d.max(axis=1, keepdims=True)
    K = K / (dmax + d.min() * 1e-5)
    v = v0
    for i in range(max_iter):
        # random walk
        v = mindspore.ops.BatchMatMul()(K, v)
        last_v = v
        n = _ms_norm(v, axis=1, p=1, keep_dims=True)
        v = v / n

        # reweighted jump
        s = v.view(batch_num, int(n2max), int(n1max)).swapaxes(1, 2)
        s = beta * s / s.max(axis=1, keepdims=True).max(axis=2, keepdims=True)
        v = alpha * sinkhorn(s, n1, n2, max_iter=sk_iter, batched_operation=True).swapaxes(1, 2).reshape(batch_num, n1n2, 1) + \
            (1 - alpha) * v
        n = _ms_norm(v, axis=1, p=1, keep_dims=True)
        v = mindspore.ops.matmul(v, 1 / n)

        if (v - last_v).sum().sqrt() < 1e-5:
            break

    return v.view(batch_num, int(n2max), int(n1max)).swapaxes(1, 2)


def sm(K: mindspore.Tensor, n1: mindspore.Tensor, n2: mindspore.Tensor, n1max, n2max, x0: mindspore.Tensor,
       max_iter: int) -> mindspore.Tensor:
    """
    mindspore implementation of SM algorithm.
    """
    batch_num, n1, n2, n1max, n2max, n1n2, v0 = _check_and_init_gm(K, n1, n2, n1max, n2max, x0)
    v = vlast = v0
    for i in range(max_iter):
        v = mindspore.ops.BatchMatMul()(K, v)
        n = _ms_norm(v, axis=1, p=2)
        v = mindspore.ops.matmul(v, (1 / n).view(batch_num, 1, 1))
        if (v - vlast).sum().sqrt() < 1e-5:
            break
        vlast = v

    x = v.view(batch_num, int(n2max), int(n1max)).swapaxes(1, 2)
    return x


def ipfp(K: mindspore.Tensor, n1: mindspore.Tensor, n2: mindspore.Tensor, n1max, n2max, x0: mindspore.Tensor,
         max_iter) -> mindspore.Tensor:
    """
    mindspore implementation of IPFP algorithm
    """
    batch_num, n1, n2, n1max, n2max, n1n2, v0 = _check_and_init_gm(K, n1, n2, n1max, n2max, x0)
    v = v0
    last_v = v
    best_v = v
    best_obj = -1

    def comp_obj_score(v1, K, v2):
        return mindspore.ops.BatchMatMul()(mindspore.ops.BatchMatMul()(v1.view(batch_num, 1, -1), K), v2)

    for i in range(max_iter):
        cost = mindspore.ops.BatchMatMul()(K, v).reshape(batch_num, int(n2max), int(n1max)).swapaxes(1, 2)
        binary_sol = hungarian(cost, n1, n2)
        binary_v = binary_sol.swapaxes(1, 2).view(batch_num, -1, 1)
        alpha = comp_obj_score(v, K, binary_v - v)
        beta = comp_obj_score(binary_v - v, K, binary_v - v)
        t0 = - alpha / beta
        v = mindspore.numpy.where(mindspore.ops.logical_or(beta >= 0, t0 >= 1), binary_v, v + t0 * (binary_v - v))
        last_v_obj = comp_obj_score(last_v, K, last_v)

        current_obj = comp_obj_score(binary_v, K, binary_v)
        best_v = mindspore.numpy.where(current_obj > best_obj, binary_v, best_v)
        best_obj = mindspore.numpy.where(current_obj > best_obj, current_obj, best_obj)

        if (_ms_max(mindspore.ops.abs(last_v_obj - current_obj) / last_v_obj)[1] < 1e-3).any():
            break
        last_v = v

    pred_x = best_v.reshape(batch_num, int(n2max), int(n1max)).swapaxes(1, 2)
    return pred_x


def _check_and_init_gm(K, n1, n2, n1max, n2max, x0):
    # get batch number
    batch_num = K.shape[0]
    n1n2 = K.shape[1]

    # get values of n1, n2, n1max, n2max and check
    if n1 is None:
        n1 = mindspore.numpy.full((batch_num,), n1max, dtype=mindspore.numpy.int_)
    if n2 is None:
        n2 = mindspore.numpy.full((batch_num,), n2max, dtype=mindspore.numpy.int_)
    if n1max is None:
        n1max = _ms_max(n1)[1]
    if n2max is None:
        n2max = _ms_max(n2)[1]

    if not n1max * n2max == n1n2:
        raise ValueError('the input size of K does not match with n1max * n2max!')

    # initialize x0 (also v0)
    if x0 is None:
        x0 = mindspore.numpy.zeros((batch_num, int(n1max), int(n2max)), dtype=K.dtype)
        for b in range(batch_num):
            x0[b, 0:n1[b], 0:n2[b]] = mindspore.Tensor(1.) / (n1[b] * n2[b])
    v0 = x0.swapaxes(1, 2).reshape((batch_num, n1n2, 1))

    return batch_num, n1, n2, n1max, n2max, n1n2, v0


def _get_single_pc_opt(X, i, j, Xij=None):
    m, _, n, _ = X.shape
    if Xij is None:
        Xij = X[i, j]
    X_combo = mindspore.ops.matmul(X[i, :], X[:, j])
    return 1 - mindspore.ops.abs(Xij - X_combo).sum() / (2 * n * m)


def _get_batch_pc_opt(X):
    m, _, n, _ = X.shape
    X1 = mindspore.numpy.tile(X.reshape(m, 1, m, n, n), (1, m, 1, 1, 1)).reshape(-1, n, n)
    X2 = mindspore.numpy.tile(X.reshape(1, m, m, n, n), (m, 1, 1, 1, 1)).swapaxes(1, 2).reshape(-1, n, n)
    X_combo = mindspore.ops.matmul(X1, X2).reshape(m, m, m, n, n)
    X_ori = mindspore.numpy.tile(X.reshape(m, m, 1, n, n), (1, 1, m, 1, 1))
    return 1 - mindspore.ops.abs(X_combo - X_ori).sum(axis=(2, 3, 4)) / (2 * n * m)


def cao_solver(K, X, num_graph, num_node, max_iter, lambda_init, lambda_step, lambda_max, iter_boost):
    m, n = num_graph, num_node
    param_lambda = lambda_init
    for iter_idx in range(max_iter):
        if iter_idx >= iter_boost:
            param_lambda = min(param_lambda * lambda_step, lambda_max)
        pair_aff = compute_affinity_score(X.reshape(-1, n, n), K.reshape(-1, n * n, n * n)).reshape(m, m)
        pair_aff = pair_aff - mindspore.ops.eye(m, m, pair_aff.dtype) * pair_aff
        norm = pair_aff.max()
        for i in range(m):
            for j in range(i + 1, m):
                aff_ori = compute_affinity_score(X[i, j].expand_dims(0), K[i, j].expand_dims(0))[0] / norm
                con_ori = _get_single_pc_opt(X, i, j)
                score_ori = aff_ori if iter_idx < iter_boost else aff_ori * (1 - param_lambda) + con_ori * param_lambda
                X_upt = X[i, j]
                for k in range(m):
                    X_combo = mindspore.ops.matmul(X[i, k], X[k, j])
                    aff_combo = compute_affinity_score(X_combo.expand_dims(0), K[i, j].expand_dims(0))[0] / norm
                    con_combo = _get_single_pc_opt(X, i, j, X_combo)
                    score_combo = aff_combo if iter_idx < iter_boost else aff_combo * (1 - param_lambda) + con_combo * param_lambda
                    if score_combo > score_ori:
                        X_upt = X_combo
                        score_ori = score_combo
                X[i, j] = X_upt
                X[j, i] = X_upt.swapaxes(0, 1)
    return X


def cao_fast_solver(K, X, num_graph, num_node, max_iter, lambda_init, lambda_step, lambda_max, iter_boost):
    return cao_solver(K, X, num_graph, num_node, max_iter, lambda_init, lambda_step, lambda_max, iter_boost)


def mgm_floyd_solver(K, X, num_graph, num_node, param_lambda):
    m, n = num_graph, num_node
    for k in range(m):
        pair_aff = compute_affinity_score(X.reshape(-1, n, n), K.reshape(-1, n * n, n * n)).reshape(m, m)
        pair_aff = pair_aff - mindspore.ops.eye(m, m, pair_aff.dtype) * pair_aff
        norm = pair_aff.max()
        for i in range(m):
            for j in range(i + 1, m):
                score_ori = compute_affinity_score(X[i, j].expand_dims(0), K[i, j].expand_dims(0))[0] / norm
                X_combo = mindspore.ops.matmul(X[i, k], X[k, j])
                score_combo = compute_affinity_score(X_combo.expand_dims(0), K[i, j].expand_dims(0))[0] / norm
                if score_combo > score_ori:
                    X[i, j] = X_combo
                    X[j, i] = X_combo.swapaxes(0, 1)
    for k in range(m):
        pair_aff = compute_affinity_score(X.reshape(-1, n, n), K.reshape(-1, n * n, n * n)).reshape(m, m)
        pair_aff = pair_aff - mindspore.ops.eye(m, m, pair_aff.dtype) * pair_aff
        norm = pair_aff.max()
        for i in range(m):
            for j in range(i + 1, m):
                aff_ori = compute_affinity_score(X[i, j].expand_dims(0), K[i, j].expand_dims(0))[0] / norm
                con_ori = _get_single_pc_opt(X, i, j)
                score_ori = aff_ori * (1 - param_lambda) + con_ori * param_lambda
                X_combo = mindspore.ops.matmul(X[i, k], X[k, j])
                aff_combo = compute_affinity_score(X_combo.expand_dims(0), K[i, j].expand_dims(0))[0] / norm
                con_combo = _get_single_pc_opt(X, i, j, X_combo)
                score_combo = aff_combo * (1 - param_lambda) + con_combo * param_lambda
                if score_combo > score_ori:
                    X[i, j] = X_combo
                    X[j, i] = X_combo.swapaxes(0, 1)
    return X


def mgm_floyd_fast_solver(K, X, num_graph, num_node, param_lambda):
    return mgm_floyd_solver(K, X, num_graph, num_node, param_lambda)


def gamgm(A, W, ns, n_univ, U0, init_tau, min_tau, sk_gamma, sk_iter, max_iter, quad_weight,
          converge_thresh, outlier_thresh, bb_smooth, verbose, cluster_M=None, projector='sinkhorn', hung_iter=True):
    num_graphs = A.shape[0]
    if ns is None:
        ns = mindspore.Tensor([A.shape[1]] * num_graphs, dtype=mindspore.int32)
    ns_np = to_numpy(ns).astype('i4')
    n_indices = np.cumsum(ns_np, axis=0)
    supA = np.zeros((n_indices[-1], n_indices[-1]), dtype=np.float32)
    A_np = to_numpy(A)
    for i in range(num_graphs):
        start_n = n_indices[i] - ns_np[i]
        end_n = n_indices[i]
        supA[start_n:end_n, start_n:end_n] = A_np[i, :ns_np[i], :ns_np[i]]
    if isinstance(n_univ, mindspore.Tensor):
        n_univ = int(to_numpy(n_univ).item())
    elif n_univ is None:
        n_univ = int(ns_np.max())
    if U0 is None:
        U0 = mindspore.Tensor(np.full((n_indices[-1], n_univ), 1 / n_univ, dtype=np.float32) + np.random.rand(n_indices[-1], n_univ).astype(np.float32) / 1000)
    if cluster_M is None:
        cluster_M = mindspore.ops.ones((num_graphs, num_graphs), mindspore.float32)
    supW = np.zeros((n_indices[-1], n_indices[-1]), dtype=np.float32)
    W_np = to_numpy(W)
    for i, j in itertools.product(range(num_graphs), repeat=2):
        sx, ex = n_indices[i] - ns_np[i], n_indices[i]
        sy, ey = n_indices[j] - ns_np[j], n_indices[j]
        supW[sx:ex, sy:ey] = W_np[i, j, :ns_np[i], :ns_np[j]]
    U = gamgm_real(mindspore.Tensor(supA), mindspore.Tensor(supW), mindspore.Tensor(ns_np), n_indices, n_univ, num_graphs,
                   U0, init_tau, min_tau, sk_gamma, sk_iter, max_iter, quad_weight,
                   converge_thresh, outlier_thresh, verbose, cluster_M, projector, hung_iter)
    result = pygmtools.utils.MultiMatchingResult(True, 'mindspore')
    for i in range(num_graphs):
        start_n = n_indices[i] - ns_np[i]
        end_n = n_indices[i]
        result[i] = U[start_n:end_n]
    return result


def gamgm_real(supA, supW, ns, n_indices, n_univ, num_graphs, U0, init_tau, min_tau, sk_gamma,
               sk_iter, max_iter, quad_weight, converge_thresh, outlier_thresh, verbose,
               cluster_M, projector, hung_iter):
    U = U0
    sinkhorn_tau = init_tau
    cluster_weight = mindspore.Tensor(np.repeat(np.repeat(to_numpy(cluster_M), to_numpy(ns).astype('i4'), axis=0),
                                                to_numpy(ns).astype('i4'), axis=1))
    while True:
        for _ in range(max_iter):
            UUt = mindspore.ops.matmul(U, U.swapaxes(0, 1))
            lastUUt = UUt
            quad = mindspore.ops.matmul(mindspore.ops.matmul(mindspore.ops.matmul(supA, UUt * cluster_weight), supA), U) * quad_weight * 2
            unary = mindspore.ops.matmul(supW * cluster_weight, U)
            V = (quad + unary) / num_graphs
            if projector == 'hungarian':
                U_list, n_start = [], 0
                for n_end in n_indices:
                    U_list.append(hungarian(V[n_start:n_end].expand_dims(0))[0, :, :n_univ])
                    n_start = n_end
            else:
                V_list, n1, n_start = [], [], 0
                for n_end in n_indices:
                    V_list.append(V[n_start:n_end, :n_univ])
                    n1.append(n_end - n_start)
                    n_start = n_end
                V_batch = build_batch(V_list)
                U_batch = sinkhorn(V_batch, mindspore.Tensor(n1), max_iter=sk_iter, tau=sinkhorn_tau, batched_operation=True, dummy_row=True)
                U_list, n_start = [], 0
                for idx, n_end in enumerate(n_indices):
                    U_list.append(U_batch[idx, :n_end - n_start, :])
                    n_start = n_end
            U = mindspore.ops.concat(U_list, axis=0)
            if UUt.sub(lastUUt).pow(2).sum().sqrt() < converge_thresh:
                break
        if sinkhorn_tau <= min_tau:
            break
        sinkhorn_tau *= sk_gamma
    return U


class PCA_GM_Net:
    def __init__(self, in_channel, hidden_channel, out_channel, num_layers, cross_iter_num=-1):
        self.gnn_layer = num_layers
        self.dict = {}
        for i in range(num_layers):
            if i == 0:
                gnn = Siamese_Gconv(in_channel, hidden_channel)
            elif i < num_layers - 1:
                gnn = Siamese_Gconv(hidden_channel, hidden_channel)
            else:
                gnn = Siamese_Gconv(hidden_channel, out_channel)
                self.dict[f'affinity_{i}'] = WeightedInnerProdAffinity(out_channel)
            self.dict[f'gnn_layer_{i}'] = gnn
            if i == num_layers - 2:
                self.dict[f'cross_graph_{i}'] = Linear(hidden_channel * 2, hidden_channel)
                if cross_iter_num <= 0:
                    self.dict[f'affinity_{i}'] = WeightedInnerProdAffinity(hidden_channel)

    def forward(self, feat1, feat2, A1, A2, n1, n2, cross_iter_num, sk_max_iter, sk_tau):
        sinkhorn_func = functools.partial(sinkhorn, dummy_row=False, max_iter=sk_max_iter, tau=sk_tau, batched_operation=False)
        emb1, emb2 = feat1, feat2
        if cross_iter_num <= 0:
            for i in range(self.gnn_layer):
                emb1, emb2 = self.dict[f'gnn_layer_{i}'].forward([A1, emb1], [A2, emb2])
                if i == self.gnn_layer - 2:
                    s = sinkhorn_func(self.dict[f'affinity_{i}'].forward(emb1, emb2), n1, n2)
                    emb1 = self.dict[f'cross_graph_{i}'].forward(mindspore.ops.concat((emb1, mindspore.ops.matmul(s, emb2)), axis=-1))
                    emb2 = self.dict[f'cross_graph_{i}'].forward(mindspore.ops.concat((emb2, mindspore.ops.matmul(s.swapaxes(1, 2), emb1)), axis=-1))
            s = sinkhorn_func(self.dict[f'affinity_{self.gnn_layer - 1}'].forward(emb1, emb2), n1, n2)
        else:
            for i in range(self.gnn_layer - 1):
                emb1, emb2 = self.dict[f'gnn_layer_{i}'].forward([A1, emb1], [A2, emb2])
            emb1_0, emb2_0 = emb1, emb2
            s = mindspore.ops.zeros((emb1.shape[0], emb1.shape[1], emb2.shape[1]), emb1.dtype)
            for _ in range(cross_iter_num):
                i = self.gnn_layer - 2
                emb1 = self.dict[f'cross_graph_{i}'].forward(mindspore.ops.concat((emb1_0, mindspore.ops.matmul(s, emb2_0)), axis=-1))
                emb2 = self.dict[f'cross_graph_{i}'].forward(mindspore.ops.concat((emb2_0, mindspore.ops.matmul(s.swapaxes(1, 2), emb1_0)), axis=-1))
                i = self.gnn_layer - 1
                emb1, emb2 = self.dict[f'gnn_layer_{i}'].forward([A1, emb1], [A2, emb2])
                s = sinkhorn_func(self.dict[f'affinity_{i}'].forward(emb1, emb2), n1, n2)
        return s


pca_gm_pretrain_path = {
    'voc': (['https://huggingface.co/heatingma/pygmtools/resolve/main/pca_gm_voc_numpy.npy',
             'https://drive.google.com/u/0/uc?export=download&confirm=Z-AR&id=1En_9f5Zi5rSsS-JTIce7B1BV6ijGEAPd',
             'https://www.dropbox.com/s/x79ib1em4cgddqp/pca_gm_voc_numpy.npy?dl=1'], 'd85f97498157d723793b8fc1501841ce'),
    'willow': (['https://huggingface.co/heatingma/pygmtools/resolve/main/pca_gm_willow_numpy.npy',
                'https://drive.google.com/u/0/uc?export=download&confirm=Z-AR&id=1LAnK6ASYu0CO1fEe6WpvMbt5vskuvwLo',
                'https://www.dropbox.com/s/2vo4wpd9467bl5r/pca_gm_willow_numpy.npy?dl=1'], 'c32f7c8a7a6978619b8fdbb6ad5b505f'),
    'voc-all': (['https://huggingface.co/heatingma/pygmtools/resolve/main/pca_gm_voc-all_numpy.npy',
                 'https://drive.google.com/u/0/uc?export=download&confirm=Z-AR&id=1c_aw4wxEBuY7JFC4Rt8rlcise777n189',
                 'https://www.dropbox.com/s/6yunsy3gqxfvdyu/pca_gm_voc-all_numpy.npy?dl=1'], '0e2725b3ac51f87f0303bbcfaae5df80')
}


def pca_gm(feat1, feat2, A1, A2, n1, n2, in_channel, hidden_channel, out_channel, num_layers, sk_max_iter, sk_tau,
           network, pretrain):
    if network is None:
        network = PCA_GM_Net(in_channel, hidden_channel, out_channel, num_layers)
        if pretrain:
            if pretrain not in pca_gm_pretrain_path:
                raise ValueError(f'Unknown pretrain tag. Available tags: {pca_gm_pretrain_path.keys()}')
            filename = pygmtools.utils.download(f'pca_gm_{pretrain}_numpy.npy', *pca_gm_pretrain_path[pretrain])
            params = np.load(filename, allow_pickle=True).item()
            for i in range(network.gnn_layer):
                gnn_layer = network.dict[f'gnn_layer_{i}'].gconv
                gnn_layer.a_fc.weight = mindspore.Tensor(params[f'gnn_layer_{i}.gconv.a_fc.weight'])
                gnn_layer.a_fc.bias = mindspore.Tensor(params[f'gnn_layer_{i}.gconv.a_fc.bias'])
                gnn_layer.u_fc.weight = mindspore.Tensor(params[f'gnn_layer_{i}.gconv.u_fc.weight'])
                gnn_layer.u_fc.bias = mindspore.Tensor(params[f'gnn_layer_{i}.gconv.u_fc.bias'])
                if i == network.gnn_layer - 2:
                    network.dict[f'affinity_{i}'].A = mindspore.Tensor(params[f'affinity_{i}.A'])
                    network.dict[f'cross_graph_{i}'].weight = mindspore.Tensor(params[f'cross_graph_{i}.weight'])
                    network.dict[f'cross_graph_{i}'].bias = mindspore.Tensor(params[f'cross_graph_{i}.bias'])
            network.dict[f'affinity_{network.gnn_layer - 1}'].A = mindspore.Tensor(params[f'affinity_{network.gnn_layer - 1}.A'])
    if feat1 is None:
        return None, network
    if n1 is None:
        n1 = mindspore.Tensor([feat1.shape[1]] * feat1.shape[0], dtype=mindspore.int32)
    if n2 is None:
        n2 = mindspore.Tensor([feat2.shape[1]] * feat2.shape[0], dtype=mindspore.int32)
    return network.forward(feat1, feat2, A1, A2, n1, n2, -1, sk_max_iter, sk_tau), network


def ipca_gm(feat1, feat2, A1, A2, n1, n2, in_channel, hidden_channel, out_channel, num_layers, cross_iter,
            sk_max_iter, sk_tau, network, pretrain):
    if network is None:
        network = PCA_GM_Net(in_channel, hidden_channel, out_channel, num_layers, cross_iter)
        if pretrain:
            filename = pygmtools.utils.download(f'ipca_gm_{pretrain}_numpy.npy',
                                                ['https://huggingface.co/heatingma/pygmtools/resolve/main/ipca_gm_{}_numpy.npy'.format(pretrain)],
                                                None)
            params = np.load(filename, allow_pickle=True).item()
            for i in range(network.gnn_layer - 1):
                gnn_layer = network.dict[f'gnn_layer_{i}'].gconv
                gnn_layer.a_fc.weight = mindspore.Tensor(params[f'gnn_layer_{i}.gconv.a_fc.weight'])
                gnn_layer.a_fc.bias = mindspore.Tensor(params[f'gnn_layer_{i}.gconv.a_fc.bias'])
                gnn_layer.u_fc.weight = mindspore.Tensor(params[f'gnn_layer_{i}.gconv.u_fc.weight'])
                gnn_layer.u_fc.bias = mindspore.Tensor(params[f'gnn_layer_{i}.gconv.u_fc.bias'])
            i = network.gnn_layer - 2
            network.dict[f'cross_graph_{i}'].weight = mindspore.Tensor(params[f'cross_graph_{i}.weight'])
            network.dict[f'cross_graph_{i}'].bias = mindspore.Tensor(params[f'cross_graph_{i}.bias'])
            i = network.gnn_layer - 1
            network.dict[f'affinity_{i}'].A = mindspore.Tensor(params[f'affinity_{i}.A'])
    if feat1 is None:
        return None, network
    if n1 is None:
        n1 = mindspore.Tensor([feat1.shape[1]] * feat1.shape[0], dtype=mindspore.int32)
    if n2 is None:
        n2 = mindspore.Tensor([feat2.shape[1]] * feat2.shape[0], dtype=mindspore.int32)
    return network.forward(feat1, feat2, A1, A2, n1, n2, cross_iter, sk_max_iter, sk_tau), network


class CIE_Net:
    def __init__(self, in_node_channel, in_edge_channel, hidden_channel, out_channel, num_layers):
        self.gnn_layer = num_layers
        self.dict = {}
        for i in range(num_layers):
            if i == 0:
                gnn = Siamese_ChannelIndependentConv(in_node_channel, hidden_channel, in_edge_channel)
            elif i < num_layers - 1:
                gnn = Siamese_ChannelIndependentConv(hidden_channel, hidden_channel, hidden_channel)
            else:
                gnn = Siamese_ChannelIndependentConv(hidden_channel, out_channel, hidden_channel)
                self.dict[f'affinity_{i}'] = WeightedInnerProdAffinity(out_channel)
            self.dict[f'gnn_layer_{i}'] = gnn
            if i == num_layers - 2:
                self.dict[f'cross_graph_{i}'] = Linear(hidden_channel * 2, hidden_channel)
                self.dict[f'affinity_{i}'] = WeightedInnerProdAffinity(hidden_channel)

    def forward(self, feat_node1, feat_node2, A1, A2, feat_edge1, feat_edge2, n1, n2, sk_max_iter, sk_tau):
        sinkhorn_func = functools.partial(sinkhorn, dummy_row=False, max_iter=sk_max_iter, tau=sk_tau, batched_operation=False)
        emb1, emb2, emb_edge1, emb_edge2 = feat_node1, feat_node2, feat_edge1, feat_edge2
        for i in range(self.gnn_layer):
            emb1, emb2, emb_edge1, emb_edge2 = self.dict[f'gnn_layer_{i}'].forward([A1, emb1, emb_edge1], [A2, emb2, emb_edge2])
            if i == self.gnn_layer - 2:
                s = sinkhorn_func(self.dict[f'affinity_{i}'].forward(emb1, emb2), n1, n2)
                emb1 = self.dict[f'cross_graph_{i}'].forward(mindspore.ops.concat((emb1, mindspore.ops.matmul(s, emb2)), axis=-1))
                emb2 = self.dict[f'cross_graph_{i}'].forward(mindspore.ops.concat((emb2, mindspore.ops.matmul(s.swapaxes(1, 2), emb1)), axis=-1))
        return sinkhorn_func(self.dict[f'affinity_{self.gnn_layer - 1}'].forward(emb1, emb2), n1, n2)


def cie(feat_node1, feat_node2, A1, A2, feat_edge1, feat_edge2, n1, n2, in_node_channel, in_edge_channel,
        hidden_channel, out_channel, num_layers, sk_max_iter, sk_tau, network, pretrain):
    if network is None:
        network = CIE_Net(in_node_channel, in_edge_channel, hidden_channel, out_channel, num_layers)
        if pretrain:
            filename = pygmtools.utils.download(f'cie_{pretrain}_numpy.npy',
                                                ['https://huggingface.co/heatingma/pygmtools/resolve/main/cie_{}_numpy.npy'.format(pretrain)],
                                                None)
            params = np.load(filename, allow_pickle=True).item()
            for i in range(network.gnn_layer):
                gnn = network.dict[f'gnn_layer_{i}'].gconv
                gnn.node_fc.weight = mindspore.Tensor(params[f'gnn_layer_{i}.gconv.node_fc.weight'])
                gnn.node_fc.bias = mindspore.Tensor(params[f'gnn_layer_{i}.gconv.node_fc.bias'])
                gnn.node_sfc.weight = mindspore.Tensor(params[f'gnn_layer_{i}.gconv.node_sfc.weight'])
                gnn.node_sfc.bias = mindspore.Tensor(params[f'gnn_layer_{i}.gconv.node_sfc.bias'])
                gnn.edge_fc.weight = mindspore.Tensor(params[f'gnn_layer_{i}.gconv.edge_fc.weight'])
                gnn.edge_fc.bias = mindspore.Tensor(params[f'gnn_layer_{i}.gconv.edge_fc.bias'])
                if i == network.gnn_layer - 2:
                    network.dict[f'affinity_{i}'].A = mindspore.Tensor(params[f'affinity_{i}.A'])
                    network.dict[f'cross_graph_{i}'].weight = mindspore.Tensor(params[f'cross_graph_{i}.weight'])
                    network.dict[f'cross_graph_{i}'].bias = mindspore.Tensor(params[f'cross_graph_{i}.bias'])
            network.dict[f'affinity_{network.gnn_layer - 1}'].A = mindspore.Tensor(params[f'affinity_{network.gnn_layer - 1}.A'])
    if feat_node1 is None:
        return None, network
    if n1 is None:
        n1 = mindspore.Tensor([feat_node1.shape[1]] * feat_node1.shape[0], dtype=mindspore.int32)
    if n2 is None:
        n2 = mindspore.Tensor([feat_node2.shape[1]] * feat_node2.shape[0], dtype=mindspore.int32)
    return network.forward(feat_node1, feat_node2, A1, A2, feat_edge1, feat_edge2, n1, n2, sk_max_iter, sk_tau), network


class NGM_Net:
    def __init__(self, gnn_channels, sk_emb):
        self.gnn_layer = len(gnn_channels)
        self.dict = {}
        for i in range(self.gnn_layer):
            if i == 0:
                gnn = NGMConvLayer(1, 1, gnn_channels[i] + sk_emb, gnn_channels[i], sk_channel=sk_emb)
            else:
                gnn = NGMConvLayer(gnn_channels[i - 1] + sk_emb, gnn_channels[i - 1], gnn_channels[i] + sk_emb,
                                   gnn_channels[i], sk_channel=sk_emb)
            self.dict[f'gnn_layer_{i}'] = gnn
        self.classifier = Linear(gnn_channels[-1] + sk_emb, 1)

    def forward(self, K, n1, n2, n1max, n2max, v0, sk_max_iter, sk_tau):
        sinkhorn_func = functools.partial(sinkhorn, dummy_row=False, max_iter=sk_max_iter, tau=sk_tau, batched_operation=False)
        emb = v0
        A = (K != 0).astype(K.dtype)
        emb_K = mindspore.ops.expand_dims(K, axis=-1)
        for i in range(self.gnn_layer):
            emb_K, emb = self.dict[f'gnn_layer_{i}'].forward(A, emb_K, emb, n1, n2, sk_func=sinkhorn_func)
        v = self.classifier.forward(emb)
        s = v.reshape((v.shape[0], int(n2max), -1)).swapaxes(1, 2)
        return sinkhorn_func(s, n1, n2, dummy_row=True)


def ngm(K, n1, n2, n1max, n2max, x0, gnn_channels, sk_emb, sk_max_iter, sk_tau, network, return_network, pretrain):
    if network is None:
        network = NGM_Net(gnn_channels, sk_emb)
        if pretrain:
            try:
                filename = pygmtools.utils.download(f'ngm_{pretrain}_numpy.npy',
                                                    ['https://huggingface.co/heatingma/pygmtools/resolve/main/ngm_{}_numpy.npy'.format(pretrain)],
                                                    None)
            except Exception:
                filename = os.path.dirname(__file__) + f'/temp/ngm_{pretrain}_numpy.npy'
            params = np.load(filename, allow_pickle=True).item()
            for i in range(network.gnn_layer):
                gnn = network.dict[f'gnn_layer_{i}']
                gnn.classifier.weight = mindspore.Tensor(params[f'gnn_layer_{i}.classifier.weight'])
                gnn.classifier.bias = mindspore.Tensor(params[f'gnn_layer_{i}.classifier.bias'])
                gnn.n_func.getitem(0).weight = mindspore.Tensor(params[f'gnn_layer_{i}.n_func.0.weight'])
                gnn.n_func.getitem(0).bias = mindspore.Tensor(params[f'gnn_layer_{i}.n_func.0.bias'])
                gnn.n_func.getitem(2).weight = mindspore.Tensor(params[f'gnn_layer_{i}.n_func.2.weight'])
                gnn.n_func.getitem(2).bias = mindspore.Tensor(params[f'gnn_layer_{i}.n_func.2.bias'])
                gnn.n_self_func.getitem(0).weight = mindspore.Tensor(params[f'gnn_layer_{i}.n_self_func.0.weight'])
                gnn.n_self_func.getitem(0).bias = mindspore.Tensor(params[f'gnn_layer_{i}.n_self_func.0.bias'])
                gnn.n_self_func.getitem(2).weight = mindspore.Tensor(params[f'gnn_layer_{i}.n_self_func.2.weight'])
                gnn.n_self_func.getitem(2).bias = mindspore.Tensor(params[f'gnn_layer_{i}.n_self_func.2.bias'])
            network.classifier.weight = mindspore.Tensor(params['classifier.weight'])
            network.classifier.bias = mindspore.Tensor(params['classifier.bias'])
    if K is None:
        return None, network
    batch_num, n1, n2, n1max, n2max, n1n2, v0 = _check_and_init_gm(K, n1, n2, n1max, n2max, x0)
    v0 = v0 / v0.mean()
    return network.forward(K, n1, n2, n1max, n2max, v0, sk_max_iter, sk_tau), network


#############################################
#              Utils Functions              #
#############################################

def inner_prod_aff_fn(feat1, feat2):
    """
    mindspore implementation of inner product affinity function
    """
    return mindspore.ops.matmul(feat1, feat2.swapaxes(1, 2))


def gaussian_aff_fn(feat1, feat2, sigma):
    """
    mindspore implementation of Gaussian affinity function
    """
    feat1 = mindspore.ops.expand_dims(feat1, axis=2)
    feat2 = mindspore.ops.expand_dims(feat2, axis=1)
    return mindspore.ops.exp(-((feat1 - feat2) ** 2).sum(axis=-1) / sigma)


def build_batch(input, return_ori_dim=False):
    """
    mindspore implementation of building a batched tensor
    """
    _check_data_type(input[0], 'input', True)
    # device = input[0].device
    it = iter(input)
    t = next(it)
    max_shape = list(t.shape)
    ori_shape = [[_] for _ in max_shape]
    while True:
        try:
            t = next(it)
            for i in range(len(max_shape)):
                max_shape[i] = int(max(max_shape[i], t.shape[i]))
                ori_shape[i].append(t.shape[i])
        except StopIteration:
            break
    max_shape = np.array(max_shape)

    padded_ts = []
    for t in input:
        pad_pattern = np.zeros(2 * len(max_shape), dtype=np.int64)
        pad_pattern[::-2] = max_shape - np.array(t.shape)
        pad_list = list((pad_pattern[2 * i], pad_pattern[2 * i + 1]) for i in range(int(len(pad_pattern) / 2)))
        while len(pad_list) < t.ndim:
            pad_list.append((0, 0))
        pad_list.reverse()
        pad_pattern = tuple(pad_list)
        mindspore_pad = nn.Pad(pad_pattern, mode="CONSTANT")
        padded_ts.append(mindspore_pad(t))

    if return_ori_dim:
        return mindspore.ops.stack(padded_ts, axis=0), tuple(
            [mindspore.Tensor(_, dtype=mindspore.int64) for _ in ori_shape])
    else:
        return mindspore.ops.stack(padded_ts, axis=0)


def dense_to_sparse(dense_adj):
    """
    mindspore implementation of converting a dense adjacency matrix to a sparse matrix
    """
    batch_size = dense_adj.shape[0]
    conn, ori_shape = build_batch([mindspore.ops.nonzero(a) for a in dense_adj], return_ori_dim=True)
    nedges = ori_shape[0]
    edge_weight = build_batch([dense_adj[b][(conn[b, :, 0], conn[b, :, 1])] for b in range(batch_size)])
    return conn, mindspore.ops.expand_dims(edge_weight, axis=-1), nedges


def compute_affinity_score(X, K):
    """
    mindspore implementation of computing affinity score
    """
    b, n, _ = X.shape
    vx = X.swapaxes(1, 2).reshape(b, -1, 1)
    vxt = vx.swapaxes(1, 2)
    return mindspore.ops.matmul(mindspore.ops.matmul(vxt, K), vx).squeeze(-1).squeeze(-1)


def to_numpy(input):
    """
    mindspore function to_numpy
    """
    return stop_gradient(input).asnumpy()


def from_numpy(input, device=None):
    """
    mindspore function from_numpy
    """
    return mindspore.Tensor(input)


def generate_isomorphic_graphs(node_num, graph_num, node_feat_dim):
    """
    mindspore implementation of generate_isomorphic_graphs
    """
    X_gt = mindspore.numpy.zeros((graph_num, node_num, node_num), dtype=mindspore.float32)
    X_gt[0, mindspore.numpy.arange(node_num), mindspore.numpy.arange(node_num)] = 1
    for i in range(1, graph_num):
        X_gt[i, mindspore.numpy.arange(node_num), mindspore.Tensor(np.random.permutation(node_num))] = 1
    joint_X = X_gt.reshape(graph_num * node_num, node_num)
    X_gt = mindspore.ops.matmul(joint_X, joint_X.swapaxes(0, 1))
    X_gt = X_gt.reshape(graph_num, node_num, graph_num, node_num).swapaxes(1, 2)
    A0 = mindspore.Tensor(np.random.rand(node_num, node_num), dtype=mindspore.float32)
    A0 = A0 - mindspore.numpy.diag(mindspore.numpy.diag(A0))
    As = [A0]
    for i in range(1, graph_num):
        As.append(mindspore.ops.matmul(mindspore.ops.matmul(X_gt[i, 0], A0), X_gt[0, i]))
    if node_feat_dim > 0:
        F0 = mindspore.Tensor(np.random.rand(node_num, node_feat_dim), dtype=mindspore.float32)
        Fs = [F0]
        for i in range(1, graph_num):
            Fs.append(mindspore.ops.matmul(X_gt[i, 0], F0))
        return mindspore.ops.stack(As, axis=0), X_gt, mindspore.ops.stack(Fs, axis=0)
    return mindspore.ops.stack(As, axis=0), X_gt


def permutation_loss(pred_dsmat: mindspore.Tensor, gt_perm: mindspore.Tensor, n1: mindspore.Tensor,
                     n2: mindspore.Tensor) -> mindspore.Tensor:
    """
    Pytorch implementation of permutation_loss
    """
    batch_num = pred_dsmat.shape[0]

    pred_dsmat = mindspore.Tensor(pred_dsmat, dtype=mindspore.float32)

    if not mindspore.ops.logical_and(pred_dsmat >= 0, pred_dsmat <= 1).all:
        raise ValueError("pred_dsmat contains invalid numerical entries.")
    if not mindspore.ops.logical_and(gt_perm >= 0, gt_perm <= 1).all:
        raise ValueError("gt_perm contains invalid numerical entries.")

    if n1 is None:
        n1 = mindspore.Tensor([pred_dsmat.shape[1] for _ in range(batch_num)])
    if n2 is None:
        n2 = mindspore.Tensor([pred_dsmat.shape[2] for _ in range(batch_num)])

    loss = mindspore.Tensor(0.)
    n_sum = mindspore.ops.zeros_like(loss)
    for b in range(batch_num):
        batch_slice = [b, slice(n1[b]), slice(n2[b])]
        weight = mindspore.ops.ones_like(pred_dsmat[batch_slice])
        loss += mindspore.ops.BinaryCrossEntropy(reduction='sum')(
            pred_dsmat[batch_slice],
            gt_perm[batch_slice], weight)
        n1_b = mindspore.Tensor(n1[b], dtype=n_sum.dtype)
        n_sum += n1_b

    return loss / n_sum


def _aff_mat_from_node_edge_aff(node_aff: mindspore.Tensor, edge_aff: mindspore.Tensor, connectivity1: mindspore.Tensor,
                                connectivity2: mindspore.Tensor,
                                n1, n2, ne1, ne2):
    """
    mindspore implementation of _aff_mat_from_node_edge_aff
    """
    if edge_aff is not None:
        # device = edge_aff.device
        dtype = edge_aff.dtype
        batch_size = edge_aff.shape[0]
        if n1 is None:
            n1 = mindspore.Tensor([math.sqrt(connectivity1.shape[1])] * batch_size)
        if n2 is None:
            n2 = mindspore.Tensor([math.sqrt(connectivity2.shape[1])] * batch_size)
        if ne1 is None:
            ne1 = [edge_aff.shape[1]] * batch_size
        if ne2 is None:
            ne2 = [edge_aff.shape[2]] * batch_size
    else:
        # device = node_aff.device
        dtype = node_aff.dtype
        batch_size = node_aff.shape[0]
        if n1 is None:
            n1 = [node_aff.shape[1]] * batch_size
        if n2 is None:
            n2 = [node_aff.shape[2]] * batch_size

    n1max = int(max(n1))
    n2max = int(max(n2))
    ks = []
    for b in range(batch_size):
        k = mindspore.numpy.zeros((n2max, n1max, n2max, n1max), dtype=dtype)
        # edge-wise affinity
        if edge_aff is not None:
            conn1 = connectivity1[b][:int(ne1[b])]
            conn2 = connectivity2[b][:int(ne2[b])]
            edge_indices = mindspore.ops.concat(
                [mindspore.ops.repeat_elements(conn1, int(ne2[b]), axis=0),
                 mindspore.numpy.tile(conn2, (int(ne1[b]), 1))],
                axis=1)  # indices: start_g1, end_g1, start_g2, end_g2
            edge_indices = (edge_indices[:, 2], edge_indices[:, 0], edge_indices[:, 3],
                            edge_indices[:, 1])  # indices: start_g2, start_g1, end_g2, end_g1
            k[edge_indices] = edge_aff[b, :int(ne1[b]), :int(ne2[b])].reshape(-1)
        k = k.reshape((n2max * n1max, n2max * n1max))
        # node-wise affinity
        if node_aff is not None:
            k[mindspore.numpy.arange(n2max * n1max), mindspore.numpy.arange(n2max * n1max)] = node_aff[b].transpose(1,
                                                                                                                    0).reshape(
                -1)
            # k_diag = mindspore.numpy.diagonal(k)
            # k_diag[:] = node_aff[b].transpose(0, 1).reshape(-1)
        ks.append(k)

    return mindspore.ops.stack(ks, axis=0)


def _check_data_type(input: mindspore.Tensor, var_name, raise_err):
    """
    mindspore implementation of _check_data_type
    """
    ms_types = [mindspore.Tensor]
    if hasattr(mindspore.common, '_stub_tensor'):  # MS tensor may be automatically transformed to StubTensor
        ms_types += [mindspore.common._stub_tensor.StubTensor]
    is_tensor = any([type(input) is t for t in ms_types])

    if raise_err and not is_tensor:
        raise ValueError(f'Expected MindSpore Tensor{f" for variable {var_name}" if var_name is not None else ""}, '
                         f'but got {type(input)}.')
    return is_tensor


def _check_shape(input, dim_num):
    """
    mindspore implementation of _check_shape
    """
    return len(input.shape) == dim_num


def _get_shape(input):
    """
    mindspore implementation of _get_shape
    """
    return input.shape


def _squeeze(input, dim):
    """
    mindspore implementation of _squeeze
    """
    return mindspore.ops.squeeze(input, axis=dim)


def _unsqueeze(input, dim):
    """
    mindspore implementation of _unsqueeze
    """
    return mindspore.ops.expand_dims(input, axis=dim)


def _transpose(input, dim1, dim2):
    """
    mindspore implementaiton of _transpose
    """
    return input.swapaxes(dim1, dim2)


def _mm(input1, input2):
    """
    mindspore implementation of _mm
    """
    return mindspore.ops.matmul(input1, input2)
