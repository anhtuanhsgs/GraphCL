import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

def foreground_reward_g (t_new_lbl, t_lbl, t_gt_lbl, last_step, size, gpu_id):
    lbl = t_lbl
    gt_lbl = t_gt_lbl
    new_lbl = t_new_lbl
    with torch.cuda.device (gpu_id):
        with torch.no_grad ():
            reward = torch.zeros (size, dtype=torch.float32, device='cuda', requires_grad=False)
            foregr_ratio = 0.67
            reward = ((new_lbl != 0) & (lbl == 0) & (gt_lbl != 0)).float () * (1 - foregr_ratio)
            if last_step:
                reward -= ((new_lbl == 0) & (gt_lbl != 0)).float () * (1 - foregr_ratio)
            return reward

def background_reward_g (t_new_lbl, t_lbl, t_gt_lbl, last_step, size, gpu_id):
    lbl = t_lbl
    new_lbl = t_new_lbl
    gt_lbl = t_gt_lbl
    with torch.cuda.device (gpu_id):
        with torch.no_grad ():
            reward = torch.zeros (size, dtype=torch.float32, device='cuda', requires_grad=False)
            foregr_ratio = 0.67
            if last_step:
                reward += ((new_lbl == 0) & (gt_lbl == 0)).float () * foregr_ratio
            reward -= ((new_lbl != 0) & (lbl == 0) & (gt_lbl == 0)).float () * foregr_ratio
            return reward 

def get_I_g (lbl, new_lbl, 
                lbl_cp, new_lbl_cp, 
                gt_lbl, gt_lbl_cp, 
                yr, xr, r, size):

        y_base = r + yr; x_base = r + xr
        I = new_lbl == new_lbl_cp [y_base:y_base+size[0], x_base:x_base+size[1]]
        I_hat = gt_lbl == gt_lbl_cp [y_base:y_base+size[0], x_base:x_base+size[1]]
        I_old = lbl == lbl_cp [y_base:y_base+size[0], x_base:x_base+size[1]]
        return I, I_hat, I_old

def split_reward_g (t_new_lbl, t_lbl, t_gt_lbl, radius, speed, size, density, T, gpu_id, last_step=False, first_step=False):
    lbl = t_lbl
    gt_lbl = t_gt_lbl
    new_lbl = t_new_lbl
    with torch.cuda.device (gpu_id):
        with torch.no_grad ():
            lbl_cp = F.pad (lbl, (radius, radius, radius, radius), 'constant', 0)
            new_lbl_cp = F.pad (new_lbl, (radius, radius, radius, radius), 'constant', 0)
            gt_lbl_cp = F.pad (gt_lbl, (radius, radius, radius, radius), 'constant', 0)
            density_cp = F.pad (density, (radius, radius, radius, radius), 'constant', 0)
            reward = torch.zeros (size, dtype=torch.float32, device="cuda", requires_grad=False)
            r = radius
            I_hat_true_cnt = torch.zeros (size, dtype=torch.float32, device="cuda", requires_grad=False)
            I_hat_false_cnt = torch.zeros (size, dtype=torch.float32, device="cuda", requires_grad=False)
            true_split_reward = torch.zeros (size, dtype=torch.float32, device="cuda", requires_grad=False)
            # false_split_penalty = torch.zeros (size, dtype=torch.float32, device="cuda", requires_grad=False)
            false_merge_penalty = torch.zeros (size, dtype=torch.float32, device="cuda", requires_grad=False)

            for yr in range (-r, r + 1, speed):
                for xr in range (-r, r + 1, speed):
                    if (yr == 0 and xr == 0):
                        continue
                    y_base = r + yr; x_base = r + xr
                    I, I_hat, I_old = get_I_g (lbl, new_lbl, lbl_cp, new_lbl_cp, gt_lbl, gt_lbl_cp, yr, xr, r, size)
                    density_v = density_cp [y_base:y_base+size[0], x_base:x_base+size[1]]
                    gt_lbl_v = gt_lbl_cp [y_base:y_base+size[0], x_base:x_base+size[1]]
                    density_u = density
                    I_hat_true_cnt += density_v * ((I_hat == True) * (gt_lbl != 0) * (gt_lbl_v != 0)).float ()
                    I_hat_false_cnt += density_v * ((I_hat == False) * (gt_lbl != 0) * (gt_lbl_v != 0)).float ()
                    true_split_reward += density_u * density_v * (((I_hat == False) & (I == False) & ((I_old == True) | first_step))  * (gt_lbl != 0) * (gt_lbl_v != 0)).float ()
                    false_merge_penalty += density_u * density_v * (((I_hat == False) & (I == True)) * (gt_lbl_v != 0) * (gt_lbl != 0)).float ()

            reward -= (false_merge_penalty / (I_hat_false_cnt + 1)) / T
            reward += true_split_reward / (I_hat_false_cnt + 1)
            return reward

def merge_reward_g (t_new_lbl, t_lbl, t_gt_lbl, radius, speed, size, density, T, gpu_id, last_step=False, first_step=False):
    lbl = t_lbl
    gt_lbl = t_gt_lbl
    new_lbl = t_new_lbl
    with torch.cuda.device (gpu_id):
        with torch.no_grad ():
            lbl_cp = F.pad (lbl, (radius, radius, radius, radius), 'constant', 0)
            new_lbl_cp = F.pad (new_lbl, (radius, radius, radius, radius), 'constant', 0)
            gt_lbl_cp = F.pad (gt_lbl, (radius, radius, radius, radius), 'constant', 0)
            density_cp = F.pad (density, (radius, radius, radius, radius), 'constant', 0)
            reward = torch.zeros (size, dtype=torch.float32, device="cuda", requires_grad=False)
            r = radius
            I_hat_true_cnt = torch.zeros (size, dtype=torch.float32, device="cuda", requires_grad=False)
            I_hat_false_cnt = torch.zeros (size, dtype=torch.float32, device="cuda", requires_grad=False)
            true_merge_reward = torch.zeros (size, dtype=torch.float32, device="cuda", requires_grad=False)
            # false_merge_penalty = torch.zeros (size, dtype=torch.float32, device="cuda", requires_grad=False)
            false_split_penalty = torch.zeros (size, dtype=torch.float32, device="cuda", requires_grad=False)
            for yr in range (-r, r + 1, speed):
                for xr in range (-r, r + 1, speed):
                    if (yr == 0 and xr == 0):
                        continue
                    y_base = r + yr; x_base = r + xr
                    I, I_hat, I_old = get_I_g (lbl, new_lbl, lbl_cp, new_lbl_cp, gt_lbl, gt_lbl_cp, yr, xr, r, size)
                    density_v = density_cp [y_base:y_base+size[0], x_base:x_base+size[1]]
                    gt_lbl_v = gt_lbl_cp [y_base:y_base+size[0], x_base:x_base+size[1]]
                    density_u = density
                    I_hat_true_cnt += density_v * ((I_hat == True) * (gt_lbl_v != 0) * (gt_lbl != 0)).float ()
                    I_hat_false_cnt += density_v * ((I_hat == False) * (gt_lbl_v != 0) * (gt_lbl != 0)).float ()
                    true_merge_reward += density_u * density_v * (((I_hat == True) & (I == True)) * (gt_lbl_v != 0) * (gt_lbl != 0)).float ()
                    false_split_penalty += density_u * density_v * (((I_hat == True) & (I == False) & ((I_old == True) | first_step)) * (gt_lbl != 0) * (gt_lbl_v != 0)).float ()
                    
            reward -= false_split_penalty / (I_hat_true_cnt + 1)
            reward += (true_merge_reward / (I_hat_true_cnt + 1)) / T
            return reward

def first_step_reward_g (t_new_lbl, t_gt_lbl, size, gpu_id):
    gt_lbl = t_gt_lbl
    new_lbl = t_new_lbl
    with torch.cuda.device (gpu_id):
        with torch.no_grad ():
            reward = torch.zeros (size, dtype=torch.float32, device="cuda", requires_grad=False)
            foregr_ratio = 0.66
            reward += ((new_lbl != 0) & (gt_lbl != 0)).float () * (1.0 - foregr_ratio)
            reward += ((new_lbl == 0) & (gt_lbl == 0)).float () * (foregr_ratio)
            reward -= ((new_lbl == 0) & (gt_lbl != 0)).float () * (1.0 - foregr_ratio)
            reward -= ((new_lbl != 0) & (gt_lbl == 0)).float () * (foregr_ratio)
            return reward



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
        bdr_cnt, _bdr_cnt = bdr_cnt_mask (bdr, seg, bdr_sum, T, i==3) # #of sames, diffs count in each pixel of inner
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
        bdr_cnt, _bdr_cnt = bdr_cnt_mask (bdr, seg, bdr_sum, T, i==3) # #of sames, diffs count in each pixel of inner
        o_bdr_cnt, _o_bdr_cnt = bdr_cnt_mask (o_bdr, o_seg, bdr_sum, T)

        t_spl_rew += (_bdr_cnt - _o_bdr_cnt) / bdr_sum
        f_mer_pen += bdr_cnt / (bdr_sum * T)
        
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