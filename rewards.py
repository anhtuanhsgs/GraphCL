import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import skimage.io as io

def bdr_cnt_mask (bdr, seg, bdr_sum, T, debug=False):
    bdr_cnt = np.array ([0] * ((2**T) + 1))
    bdr_uni = np.unique (bdr, return_counts=True)
    for i in range (len(bdr_uni [0])):
        bdr_cnt [bdr_uni [0][i]] = bdr_uni[1][i]
    _bdr_cnt = bdr_sum - bdr_cnt
    _bdr_cnt [-1] = bdr_cnt [-1] = 0
#     print (bdr_cnt [0], _bdr_cnt [0])

#     plt.imshow (bdr_cnt [seg])
#     plt.show ()
    return (bdr_cnt [seg].astype (np.int32, copy=False), _bdr_cnt [seg].astype (np.int32, copy=False))

def split_reward_s_onlyInr (old_lbl, lbl, gt_lbl, first_step, segs, inrs, bdrs, T, scaler):
    t_spl_rew = np.zeros (lbl.shape, dtype=np.float32) #True split reward
    f_mer_pen = np.zeros (lbl.shape, dtype=np.float32) #False merge penalty

    inr_lbl = np.zeros_like (lbl)
    old_inr_lbl = np.zeros_like (old_lbl)

    for i in np.unique (gt_lbl):
        if i == 0:
            continue
        inr_lbl += lbl * inrs [i]
        old_inr_lbl += old_lbl * inrs [i] 

    for i in np.unique (gt_lbl):
        if i == 0:
            continue

        out1 = (True ^ segs [i])
        out2 = (True ^ bdrs[i])
        # print ("split")
        # fig = plt.figure (figsize=(10,10))
        # fig.add_subplot (1, 3, 1)
        # plt.imshow (gt_lbl, cmap="tab20")
        # fig.add_subplot (1, 3, 2)
        # plt.imshow (bdrs[i], cmap='gray')
        # fig.add_subplot (1, 3, 3)
        # plt.imshow (segs[i], cmap='gray')
        # plt.show ()
        bdr = bdrs [i] * inr_lbl; seg = segs [i] * inr_lbl 
        o_bdr = bdrs[i] * old_inr_lbl; o_seg = segs [i] * old_inr_lbl 
        bdr [(gt_lbl==0)|out2] = (2 ** T); seg [(gt_lbl==0)|out1] = (2 ** T)
        o_bdr [(gt_lbl==0)|out2] = (2 ** T); o_seg [(gt_lbl==0)|out1] = (2 ** T)
        
        bdr_sum = np.count_nonzero (bdrs[i] * gt_lbl) + 1 #Total non background pixels in bdr 
        bdr_cnt, _bdr_cnt = bdr_cnt_mask (bdr, seg, bdr_sum, T) # #of sames, diffs count in each pixel of inner
        o_bdr_cnt, _o_bdr_cnt = bdr_cnt_mask (o_bdr, o_seg, bdr_sum, T)

        t_spl_rew += (_bdr_cnt - _o_bdr_cnt) / bdr_sum
        f_mer_pen += bdr_cnt / (bdr_sum * T)
        
    ret = t_spl_rew - f_mer_pen
    if scaler is not None:
        ret *= scaler
    return ret.astype (np.float32, copy=False)

def split_reward_s (old_lbl, lbl, gt_lbl, first_step, segs, inrs, bdrs, T, scaler):
    t_spl_rew = np.zeros (lbl.shape, dtype=np.float32) #True split reward
    f_mer_pen = np.zeros (lbl.shape, dtype=np.float32) #False merge penalty

    for i in np.unique (gt_lbl):
        if i == 0:
            continue

        out1 = (True ^ segs [i])
        out2 = (True ^ bdrs[i])
        # print ("split")
        # fig = plt.figure (figsize=(10,10))
        # fig.add_subplot (1, 3, 1)
        # plt.imshow (gt_lbl, cmap="tab20")
        # fig.add_subplot (1, 3, 2)
        # plt.imshow (bdrs[i], cmap='gray')
        # fig.add_subplot (1, 3, 3)
        # plt.imshow (segs[i], cmap='gray')
        # plt.show ()
        bdr = bdrs [i] * lbl; seg = segs [i] * lbl 
        o_bdr = bdrs[i] * old_lbl; o_seg = segs [i] * old_lbl 
        bdr [(gt_lbl==0)|out2] = (2 ** T); seg [(gt_lbl==0)|out1] = (2 ** T)
        o_bdr [(gt_lbl==0)|out2] = (2 ** T); o_seg [(gt_lbl==0)|out1] = (2 ** T)
        
        bdr_sum = np.count_nonzero (bdrs[i] * gt_lbl) + 1 #Total non background pixels in bdr 
        bdr_cnt, _bdr_cnt = bdr_cnt_mask (bdr, seg, bdr_sum, T) # #of sames, diffs count in each pixel of inner
        o_bdr_cnt, _o_bdr_cnt = bdr_cnt_mask (o_bdr, o_seg, bdr_sum, T)

        t_spl_rew += (_bdr_cnt - _o_bdr_cnt) / bdr_sum
        f_mer_pen += bdr_cnt / (bdr_sum * T)
        
    ret = t_spl_rew - f_mer_pen
    if scaler is not None:
        ret *= scaler
    return ret.astype (np.float32, copy=False)

def bdr_frac (area, seg, T):
    bdr_uni = np.unique (seg, return_counts=True)
    ret = np.array ([0] * ((2**T) + 1), dtype=np.float32)
    for i in range (len (np.unique (bdr_uni [0]))):
        ret [bdr_uni[0][i]] = 1.0 * bdr_uni[1][i] / area
    ret [2 ** T] = 0
    return ret 

def split_reward_ins (old_lbl, lbl, gt_lbl, first_step, segs, inrs, bdrs, T, scaler):
    t_spl_rew = np.zeros (lbl.shape, dtype=np.float32) # True split reward
    f_mer_pen = np.zeros (lbl.shape, dtype=np.float32) # False merge penalty

    frac_per_ins = {}
    o_frac_per_ins = {}
    for u in np.unique (gt_lbl):
        if u == 0:
            continue
        seg = segs [u] * lbl
        o_seg = segs [u] * old_lbl
        seg [True ^ segs [u]] = (2 ** T)
        o_seg [True ^ segs [u]] = (2 ** T)
        area = np.count_nonzero (segs [u] * gt_lbl)
        frac_per_ins [u] = bdr_frac (area, seg, T)
        o_frac_per_ins [u] = bdr_frac (area, o_seg, T)

    for u in np.unique (gt_lbl):

        # DEBUG = u == 14
        DEBUG = False

        if DEBUG:
            print ("U: ", u)
        if u == 0:
            continue
        bdr = bdrs [u] * gt_lbl;
        neighbor_ids = np.unique (bdr).tolist ()
        seg = segs [u] * lbl
        o_seg = segs [u] * old_lbl

        frac_neighbors = np.array ([0] * ((2**T) + 1), dtype=np.float32)
        o_frac_neighbors = np.array ([0] * ((2**T) + 1), dtype=np.float32)
        _frac_neighbors = np.array ([0] * ((2**T) + 1), dtype=np.float32)
        _o_frac_neighbors = np.array ([0] * ((2**T) + 1), dtype=np.float32)

        if u in neighbor_ids:
            neighbor_ids.remove (u)
        if 0 in neighbor_ids:
            neighbor_ids.remove (0)

        if len (neighbor_ids) == 0:
            t_spl_rew += segs [u]
            continue

        for v in neighbor_ids:
            if (v == 0):
                continue
            frac_neighbors += frac_per_ins [v]
            o_frac_neighbors += o_frac_per_ins [v]
            _frac_neighbors += 1 - frac_per_ins [v]
            _o_frac_neighbors += 1 - o_frac_per_ins [v]

        if DEBUG:
            print ("frac", frac_neighbors)
            print ("o_frac", o_frac_neighbors)
            print ("_frac", _frac_neighbors)
            print ("_o_frac", _o_frac_neighbors)

        num_neighbors = len (neighbor_ids)
        new_spl_frac = (_frac_neighbors [seg] - _o_frac_neighbors [o_seg]) * segs [u]# Wrong here, different values
        new_mer_frac = frac_neighbors [seg] * segs [u]

        t_spl_rew += (new_spl_frac / len (neighbor_ids))
        f_mer_pen += (new_mer_frac / (len (neighbor_ids) * T))

        if DEBUG:
            rows = 4
            columns = 2
            fig = plt.figure(figsize=(8, 8))
            fig.add_subplot(rows, columns, 1)
            plt.imshow (segs [u])
            fig.add_subplot(rows, columns, 2)
            plt.imshow (np.isin (gt_lbl, neighbor_ids))
            fig.add_subplot(rows, columns, 3)
            plt.imshow (old_lbl)
            fig.add_subplot(rows, columns, 4)
            plt.imshow (lbl)
            fig.add_subplot(rows, columns, 5)
            plt.imshow (t_spl_rew)
            fig.add_subplot(rows, columns, 6)
            plt.imshow (f_mer_pen)
            fig.add_subplot(rows, columns, 7)
            plt.imshow (gt_lbl)
            plt.show ()

        # tmp = input ()

        # io.imsave ("lbl.tif", lbl)
        # io.imsave ("gt_lbl.tif", gt_lbl)
        # io.imsave ("t_spl_rew", )
        # tmp = input ()

    ret = t_spl_rew - f_mer_pen

    if scaler is not None:
        ret *= scaler
    return ret.astype (np.float32, copy=False)


def inr_cnt_mask (inr, seg, inr_sum, T, debug=False):
    inr_cnt = np.array ([0] * ((2**T) + 1))
    inr_uni = np.unique (inr, return_counts=True)
    for i in range (len (inr_uni [0])):
        inr_cnt [inr_uni [0][i]] = inr_uni [1][i]

    _inr_cnt = inr_sum - inr_cnt
    _inr_cnt [-1] = inr_cnt [-1] = 0
    
    return (inr_cnt [seg].astype (np.int32, copy=False), _inr_cnt [seg].astype (np.int32, copy=False)) 

def merge_reward_s (old_lbl, lbl, gt_lbl, first_step, segs, inrs, bdrs, T, scaler):
    t_mer_rew = np.zeros (lbl.shape, dtype=np.float32)
    f_spl_pen = np.zeros (lbl.shape, dtype=np.float32)
    for i in np.unique (gt_lbl):
        if i == 0:
            continue
        out0 = (True ^ inrs [i])
        out1 = (True ^ segs [i]) # exclude only segment
        # print ("merge")
        # fig = plt.figure (figsize=(10,10))
        # fig.add_subplot (1, 3, 1)
        # plt.imshow (gt_lbl, cmap="tab20")
        # fig.add_subplot (1, 3, 2)
        # plt.imshow (inrs[i], cmap='gray')
        # fig.add_subplot (1, 3, 3)
        # plt.imshow (segs[i], cmap='gray')
        # plt.show ()
        inr = inrs [i] * lbl; seg = segs [i] * lbl 
        o_inr = inrs[i] * old_lbl; o_seg = segs [i] * old_lbl
        inr [out0] = (2 ** T); seg [(gt_lbl==0)|out1] = (2 ** T)
        o_inr [out0] = (2 ** T); o_seg [(gt_lbl==0)|out1] = (2 ** T)
        
        inr_sum = np.count_nonzero (inrs [i] * gt_lbl) + 1 #Total non background pixels in seg 
        inr_cnt, _inr_cnt = inr_cnt_mask (inr, seg, inr_sum, T)
        o_inr_cnt, _o_inr_cnt = inr_cnt_mask (o_inr, o_seg, inr_sum, T)
   
        t_mer_rew += inr_cnt / (inr_sum * T)
        f_spl_pen += (_inr_cnt - _o_inr_cnt) / inr_sum

    ret = t_mer_rew - f_spl_pen
    if scaler is not None:
        ret *= scaler
    return ret.astype (np.float32, copy=False)

def merge_reward_step (action, gt_lbl, first_step, segs, inrs, bdrs, T, scaler):
    t_mer_rew = np.zeros (action.shape, dtype=np.float32)
    f_spl_pen = np.zeros (action.shape, dtype=np.float32)
    
    for u in np.unique (gt_lbl):

        # DEBUG = u == 14
        DEBUG = False

        if u == 0:
            continue
        seg = action * segs [u]
        area = 1.0 * np.count_nonzero (segs [u] * gt_lbl)
        ones_cnt = 1.0 * np.count_nonzero (seg)
        zeros_cnt = area - ones_cnt
        ones_frac = ones_cnt / area
        zeros_frac = zeros_cnt / area
        t_mer_rew += (1.0 * seg * ones_frac + (1.0 * segs [u] - 1.0 * seg) * zeros_frac) / T
        f_spl_pen += (1.0 - t_mer_rew) * segs [u] / T

        if DEBUG:
            print ("ones_frac", ones_frac, "zeros_frac", zeros_frac)

        if DEBUG:
            rows = 3
            columns = 2
            fig = plt.figure(figsize=(8, 8))
            fig.add_subplot(rows, columns, 1)
            plt.imshow (segs [u])
            fig.add_subplot(rows, columns, 2)
            plt.imshow (action)
            fig.add_subplot(rows, columns, 3)
            plt.imshow (1.0 * seg * ones_frac + (1.0 * segs [u] - 1.0 * seg) * zeros_frac)
            fig.add_subplot(rows, columns, 4)
            plt.imshow ((1.0 - t_mer_rew) * segs [u])
            fig.add_subplot(rows, columns, 5)
            plt.imshow (seg)
            fig.add_subplot(rows, columns, 6)
            plt.imshow (gt_lbl)
            plt.show ()
    
    ret = t_mer_rew - f_spl_pen
    if scaler is not None:
        ret *= scaler

    return ret.astype (np.float32, copy=False) 


def merge_pen_action (action, gt_lbl, first_step, segs, inrs, bdrs, T, scaler):
    t_mer_rew = np.zeros (gt_lbl.shape, dtype=np.float32)
    f_spl_pen = np.zeros (gt_lbl.shape, dtype=np.float32)

    for i in np.unique (gt_lbl):
        if i == 0:
            continue
        out1 = (True ^ segs [i])
        seg = (segs [i] * action).astype (np.int64, copy=False)
        seg [out1] = (2 ** T)

        seg_sum = np.count_nonzero (segs [i] * gt_lbl) + 1 #Total non background pixels in seg 
        seg_cnt, _seg_cnt = inr_cnt_mask (seg, seg, seg_sum, T)

        # t_mer_rew += seg_cnt / (seg_sum * T)
        f_spl_pen += _seg_cnt / (seg_sum * T)

    ret = t_mer_rew - f_spl_pen
    if scaler is not None:
        ret *= scaler
    return ret.astype (np.float32, copy=False)

def split_rew_action (action, gt_lbl, first_step, segs, inrs, bdrs, T, scaler):
    t_spl_rew = np.zeros (lbl.shape, dtype=np.float32) #True split reward
    f_mer_pen = np.zeros (lbl.shape, dtype=np.float32) #False merge penalty

    for i in np.unique (gt_lbl):
        if i == 0:
            continue

        out1 = (True ^ segs [i])
        out2 = (True ^ bdrs[i])
        # print ("split")
        # fig = plt.figure (figsize=(10,10))
        # fig.add_subplot (1, 3, 1)
        # plt.imshow (gt_lbl, cmap="tab20")
        # fig.add_subplot (1, 3, 2)
        # plt.imshow (bdrs[i], cmap='gray')
        # fig.add_subplot (1, 3, 3)
        # plt.imshow (segs[i], cmap='gray')
        # plt.show ()
        bdr = bdrs [i] * action; seg = segs [i] * action 
        bdr [(gt_lbl==0)|out2] = (2 ** T); seg [(gt_lbl==0)|out1] = (2 ** T)
        
        bdr_sum = np.count_nonzero (bdrs[i] * gt_lbl) + 1 #Total non background pixels in bdr 
        bdr_cnt, _bdr_cnt = bdr_cnt_mask (bdr, seg, bdr_sum, T, i==3) # #of sames, diffs count in each pixel of inner

        t_spl_rew += _bdr_cnt / (bdr_sum * T)
        # f_mer_pen += bdr_cnt / (bdr_sum * T)
        
    ret = t_spl_rew - f_mer_pen
    if scaler is not None:
        ret *= scaler
    return ret.astype (np.float32, copy=False)

# def sparse_sampling_weight (gt_lbl, rates=[1, 2, 4]):
#     ret = np.ones (gt_lbl.shape, dtype=np.bool)
#     rates = sorted (rates) [::-1]
#     for rate in rates:
#         sample_lbl = gt_lbl [::rate, ::rate]
#         sample_ret = ret [::rate, ::rate]

#         padded_sample_lbl = 

#         update = (False==sample_ret) &  

# def split_reward_action (action, gt_lbl, first_step, segs, inrs, bdrs, T, scaler):
#     t_mer_rew = np.zeros (gt_lbl.shape, dtype=np.float32)
#     f_spl_pen = np.zeros (gt_lbl.shape, dtype=np.float32)



# def inr_cnt_mask (seg, inr_sum, T, debug=False):
#     inr_cnt = np.array ([0] * ((2**T) + 1))
#     inr_uni = np.unique (seg, return_counts=True)
#     for i in range (len (inr_uni [0])):
#         inr_cnt [inr_uni [0][i]] = inr_uni [1][i]

#     _inr_cnt = inr_sum - inr_cnt
#     _inr_cnt [-1] = inr_cnt [-1] = 0
    
#     return (inr_cnt [seg].astype (np.int32, copy=False), _inr_cnt [seg].astype (np.int32, copy=False)) 

# def merge_reward_s (old_lbl, lbl, gt_lbl, first_step, segs, bdrs, T):
#     t_mer_rew = np.zeros (lbl.shape, dtype=np.float32)
#     f_spl_pen = np.zeros (lbl.shape, dtype=np.float32)
#     for i in np.unique (gt_lbl):
#         if i == 0:
#             continue
#         out1 = (True ^ segs [i]) # exclude only segment
#         seg = segs [i] * lbl 
#         o_seg = segs [i] * old_lbl
#         seg [(gt_lbl==0)|out1] = (2 ** T)
#         o_seg [(gt_lbl==0)|out1] = (2 ** T)
        
#         inr_sum = np.count_nonzero (segs[i] * gt_lbl) + 1 #Total non background pixels in seg 
#         inr_cnt, _inr_cnt = inr_cnt_mask (seg, inr_sum, T)
#         o_inr_cnt, _o_inr_cnt = inr_cnt_mask (o_seg, inr_sum, T)
   
#         t_mer_rew += inr_cnt / (inr_sum * T)
#         f_spl_pen += (_inr_cnt - _o_inr_cnt) / inr_sum

#     ret = t_mer_rew - f_spl_pen
#     return ret.astype (np.float32, copy=False)