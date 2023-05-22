import torch.nn.functional as F
import math
import torch
from pretraining.utils import master_process
# import wandb
from .pear_loss import Dist_att
import random
import numpy as np
from torch import nn
dist_att = Dist_att()

def data_aug(batch):
    bz, length = batch[1].shape
    ## set prob threshold ##
    if random.random() >= 0.5:
    ## selection postion ##
        # positions = random.sample(range(0,length), 10)
        positions = np.random.rand(bz, length).argpartition(100,axis=1)[:,:100]
    ## selection vocab ##
        # vocab = random.randrange(0, 30522)
        vocab = np.random.rand(bz, 30522).argpartition(100,axis=1)[:,:100]
    ## switch ##
        # batch[1][positions] = torch.from_numpy(vocab).cuda()
        # batch[2][positions] = 1
        for i in range(bz):
            batch[1][i,positions[i,:]]=torch.from_numpy(vocab[i,:]).cuda()
            batch[2][i,positions[i,:]]=1
    return batch

# def hidden_mse_learn():
    
#     return loss
def att_kl(student_atts, student_qkv, teacher_atts, teacher_qkv, layer_selection):
    #TODO: 把这个fp16 32， 正规化，以及看amp方案
    loss_att = 0.
    loss_value = 0.

    batch_size, num_head, length, dk = student_qkv[0][2].shape
    dk_sqrt = math.sqrt(dk)
    layer_selection = [int(item) for item in layer_selection.split(',')]

    new_teacher_atts = [teacher_atts[i] for i in layer_selection]
    if len(layer_selection) == 1: 
        student_atts = [student_atts[-1]]
    #TODO: change to softmax and log 
    for student_att, teacher_att in zip(student_atts, new_teacher_atts):
        # batch_size, head_num, lenght = student_att.shape[0], student_att.shape[1], student_att.shape[2]
        # student_att_logsoft = student_att.log_softmax(dim=3)
        student_att_soft = F.softmax(student_att, dim=-1)
        student_att_logsoft = torch.clamp(student_att_soft, 1e-7, 1).log()
        teacher_att_soft = teacher_att.softmax(dim=3)
        loss_kl_tmp = F.kl_div(student_att_logsoft, teacher_att_soft, reduction='sum')/ (batch_size * num_head * length) #, reduction='batchmean', log_target=True)
        # loss_kl_tmp = F.kl_div(student_att_logsoft.to(torch.float32), teacher_att_soft.to(torch.float32), reduction='sum')/ (batch_size * num_head * length) #, reduction='batchmean', log_target=True)
        # loss_kl_tmp = F.mse_loss(student_att, teacher_att)
        loss_att += loss_kl_tmp
    return loss_att

def att_val_kl(student_atts, student_qkv, teacher_atts, teacher_qkv, layer_selection):
    #TODO: 把这个fp16 32， 正规化，以及看amp方案
    loss_att = 0.
    loss_value = 0.

    batch_size, num_head, length, dk = student_qkv[0][2].shape
    dk_sqrt = math.sqrt(dk)
    layer_selection = [int(item) for item in layer_selection.split(',')]

    new_teacher_atts = [teacher_atts[i] for i in layer_selection]
    if len(layer_selection) == 1: 
        student_atts = [student_atts[-1]]
    #TODO: change to softmax and log 
    for student_att, teacher_att in zip(student_atts, new_teacher_atts):
        # batch_size, head_num, lenght = student_att.shape[0], student_att.shape[1], student_att.shape[2]
        # student_att_logsoft = student_att.log_softmax(dim=3)
        student_att_soft = F.softmax(student_att, dim=-1)
        student_att_logsoft = torch.clamp(student_att_soft, 1e-7, 1).log()
        teacher_att_soft = teacher_att.softmax(dim=3)
        loss_kl_tmp = F.kl_div(student_att_logsoft, teacher_att_soft, reduction='sum')/ (batch_size * num_head * length) #, reduction='batchmean', log_target=True)
        # loss_kl_tmp = F.kl_div(student_att_logsoft.to(torch.float32), teacher_att_soft.to(torch.float32), reduction='sum')/ (batch_size * num_head * length) #, reduction='batchmean', log_target=True)
        # loss_kl_tmp = F.mse_loss(student_att, teacher_att)
        loss_att += loss_kl_tmp

    new_teacher_value = [teacher_qkv[i][2] for i in layer_selection]
    if len(layer_selection) == 1:
        student_vals = [student_qkv[-1][2]]
    else:
        student_vals = [qkv[2] for qkv in student_qkv]
    for student_value, teacher_value in zip(student_vals, new_teacher_value):
        vr_student = torch.bmm(student_value.reshape(-1, length, dk), student_value.reshape(-1, length, dk).transpose(1,2))/dk_sqrt
        vr_student_soft = F.softmax(vr_student, dim=-1)
        vr_student = torch.clamp(vr_student_soft, 1e-7, 1).log()
        # vr_student = F.logsoftmax(vr_student, dim=-1)
        vr_teacher = F.softmax(torch.bmm(teacher_value.reshape(-1, length, dk), teacher_value.reshape(-1, length, dk).transpose(1,2))/dk_sqrt, dim=-1)
        loss_value_tmp = F.kl_div(vr_student, vr_teacher, reduction='sum')/(batch_size * num_head * length)

        # loss_value_tmp = F.kl_div(vr_student.to(torch.float32), vr_teacher.to(torch.float32), reduction='sum')/(batch_size * num_head * length)
        # loss_value_tmp = F.mse_loss(vr_student, vr_teacher)
        loss_value += loss_value_tmp
    # loss  = loss_att + loss_value
    return loss_att, loss_value


def att_mse(student_atts, teacher_atts, layer_selection):
    #TODO: 把这个fp16 32， 正规化，以及看amp方案
    loss_att = 0.
    layer_selection = [int(item) for item in layer_selection.split(',')]

    new_teacher_atts = [teacher_atts[i] for i in layer_selection]
    if len(layer_selection) == 1: 
        student_atts = [student_atts[-1]]
    #TODO: change to softmax and log 
    for student_att, teacher_att in zip(student_atts, new_teacher_atts):
        # batch_size, head_num, lenght = student_att.shape[0], student_att.shape[1], student_att.shape[2]
        # student_att_logsoft = student_att.log_softmax(dim=3)
        # teacher_att_soft = teacher_att.softmax(dim=3)
        # loss_kl_tmp = F.kl_div(student_att_logsoft.to(torch.float32), teacher_att_soft.to(torch.float32), reduction='sum')/ (batch_size * num_head * length) #, reduction='batchmean', log_target=True)
        student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att),
                                    student_att)
        teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att),
                                    teacher_att)
        loss_att_tmp = F.mse_loss(student_att.to(torch.float32), teacher_att.to(torch.float32))
        loss_att += loss_att_tmp
    return loss_att


def minilm_v2(student_atts, student_qkv, teacher_atts, teacher_qkv, layer_selection):
    loss_q = 0.
    loss_k = 0. 
    loss_v = 0.

    bs_t, n_h_t, l_t, dk_t = teacher_qkv[0][2].shape
    dk_sqrt_t = math.sqrt(dk_t)

    layer_selection = [int(item) for item in layer_selection.split(',')]
    new_teacher_q = [teacher_qkv[i][0] for i in layer_selection]
    new_student_q = [student_qkv[-1][0].reshape([bs_t, n_h_t, l_t, -1])]
    dk_s = new_student_q[0].shape[3]
    dk_sqrt_s = math.sqrt(dk_s)
    
    for student_q, teacher_q in zip(new_student_q, new_teacher_q):
        qr_student = F.log_softmax(torch.bmm(student_q.reshape(-1, l_t, dk_s), student_q.reshape(-1, l_t, dk_s).transpose(1,2))/dk_sqrt_s, dim=-1)
        qr_teacher = F.softmax(torch.bmm(teacher_q.reshape(-1, l_t, dk_t), teacher_q.reshape(-1, l_t, dk_t).transpose(1,2))/dk_sqrt_t, dim=-1)
        loss_q_tmp = F.kl_div(qr_student, qr_teacher, reduction='sum')/(bs_t * n_h_t * l_t)
        loss_q += loss_q_tmp
    
    new_teacher_k = [teacher_qkv[i][1].transpose(2, 3) for i in layer_selection]
    new_student_k = [student_qkv[-1][1].transpose(2, 3).reshape([bs_t, n_h_t, l_t, -1])]

    for student_k, teacher_k in zip(new_student_k, new_teacher_k):
        kr_student = F.log_softmax(torch.bmm(student_k.reshape(-1, l_t, dk_s), student_k.reshape(-1, l_t, dk_s).transpose(1,2))/dk_sqrt_s, dim=-1)
        kr_teacher = F.softmax(torch.bmm(teacher_k.reshape(-1, l_t, dk_t), teacher_k.reshape(-1, l_t, dk_t).transpose(1,2))/dk_sqrt_t, dim=-1)
        loss_k_tmp = F.kl_div(kr_student, kr_teacher, reduction='sum')/(bs_t * n_h_t * l_t)
        loss_k += loss_k_tmp
    
    new_teacher_v = [teacher_qkv[i][2] for i in layer_selection]
    new_student_v = [student_qkv[-1][2].reshape([bs_t, n_h_t, l_t, -1])]

    for student_v, teacher_v in zip(new_student_v, new_teacher_v):
        vr_student = F.log_softmax(torch.bmm(student_v.reshape(-1, l_t, dk_s), student_v.reshape(-1, l_t, dk_s).transpose(1,2))/dk_sqrt_s, dim=-1)
        vr_teacher = F.softmax(torch.bmm(teacher_v.reshape(-1, l_t, dk_t), teacher_v.reshape(-1, l_t, dk_t).transpose(1,2))/dk_sqrt_t, dim=-1)
        loss_v_tmp = F.kl_div(vr_student, vr_teacher, reduction='sum')/(bs_t * n_h_t * l_t)
        loss_v += loss_v_tmp

    return  loss_q, loss_k, loss_v


def hidden_mse(hidden_student, hidden_teacher, layer_selection):
    layer_selection = [int(item) for item in layer_selection.split(',')]
    new_teacher_hidden = [hidden_teacher[i] for i in layer_selection]
    if len(layer_selection) == 1:
        new_student_hidden =  [hidden_student[-1]]
    for student_hidd, teacher_hidd in zip(new_student_hidden, new_teacher_hidden):
        loss_tmp = F.mse_loss(student_hidd, teacher_hidd)
        loss_hidden = loss_tmp
    return loss_hidden







def hidden_mse_cls(hidden_student, hidden_teacher, layer_selection):
    layer_selection = [int(item) for item in layer_selection.split(',')]
    new_teacher_hidden = [hidden_teacher[i].transpose(0,1)[:,0] for i in layer_selection]
    hidden_student = [rep.transpose(0,1)[:,0] for rep in hidden_student]
    if len(layer_selection) == 1:
        new_student_hidden =  [hidden_student[-1]]
    for student_hidd, teacher_hidd in zip(new_student_hidden, new_teacher_hidden):
        loss_tmp = F.mse_loss(student_hidd, teacher_hidd)
        loss_hidden = loss_tmp
    return loss_hidden


class HiddenMSECombine(nn.Module):
    def __init__(self, inputSize=9216, outputSize=4608):
        super(HiddenMSECombine, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)
    def forward(self, hidden_student, hidden_teacher, layer_selection):
            # layer_selection = [int(item) for item in layer_selection.split(',')]
        new_student_hidden = torch.cat([t.transpose(0,1) for t in hidden_student], dim=2)
        new_teacher_hidden = torch.cat([t.transpose(0,1) for t in hidden_teacher], dim=2)
        projected_teacher = self.linear(new_teacher_hidden)
        loss_hidden = F.mse_loss(new_student_hidden, projected_teacher)

        return loss_hidden



def att_val_frame(teacher, student, args, batch, global_step, wandb, projector=None, eval=False):
    log = 'eval' if eval else 'train'
    if args.aug and not eval:
        batch = data_aug(batch)
    with torch.no_grad():
        attentions_teacher, qkv_teacher, hidden_teacher, prediction_score_teacher = \
                teacher(batch, output_attentions=True, output_qkv=True, output_loss=False, output_hidden_states=True)
    attentions_st, qkv_st, hidden_student, prediction_score_st = \
            student.forward(batch, output_attentions=True, output_qkv=True, output_loss=False, output_hidden_states=True)


    if args.method == 'att_mse_hidden_mse':
        loss_att= \
            att_mse(attentions_st, attentions_teacher, args.layer_selection)
        loss_hidden = \
            hidden_mse(hidden_student, hidden_teacher, args.layer_selection)
        total_loss = loss_att + loss_hidden
        if master_process(args):
            wandb.log({f"{log}/loss": total_loss}, step=global_step)
            wandb.log({f"{log}/loss_att": loss_att}, step=global_step)
            wandb.log({f"{log}/loss_hidden": loss_hidden}, step=global_step)
    elif args.method == 'hidden_mse_combine':
        loss_hidden = \
            projector.forward(hidden_student, hidden_teacher, args.layer_selection)
        total_loss = loss_hidden
        if master_process(args):
            wandb.log({f"{log}/loss": total_loss}, step=global_step)
            wandb.log({f"{log}/loss_hidden": loss_hidden}, step=global_step)
    elif args.method == 'hidden_cls':
        loss_hidden = \
            hidden_mse_cls(hidden_student, hidden_teacher, args.layer_selection)
        total_loss = loss_hidden
        if master_process(args):
            wandb.log({f"{log}/loss": total_loss}, step=global_step)
            wandb.log({f"{log}/loss_hidden": loss_hidden}, step=global_step)

    elif args.method == 'att_kl_hidden_mse':
        loss_att = \
            att_kl(attentions_st, qkv_st, attentions_teacher, qkv_teacher, args.layer_selection)
        loss_hidden = \
            hidden_mse(hidden_student, hidden_teacher, args.layer_selection)
        total_loss = loss_att + loss_hidden
        if master_process(args):
            wandb.log({f"{log}/loss": total_loss}, step=global_step)
            wandb.log({f"{log}/loss_att": loss_att}, step=global_step)
            wandb.log({f"{log}/loss_hidden": loss_hidden}, step=global_step)
            
    elif args.method == 'att_val_og':
        loss_att, loss_val = \
            att_val_kl(attentions_st, qkv_st, attentions_teacher, qkv_teacher, args.layer_selection)
        total_loss = loss_att + loss_val
        if master_process(args):
            wandb.log({f"{log}/loss": total_loss}, step=global_step)
            wandb.log({f"{log}/loss_att": loss_att}, step=global_step)
            wandb.log({f"{log}/loss_val": loss_val}, step=global_step)

    elif args.method == 'att_kl':
        loss_att = \
            att_kl(attentions_st, qkv_st, attentions_teacher, qkv_teacher, args.layer_selection)
        total_loss = loss_att
        if master_process(args):
            wandb.log({f"{log}/loss": total_loss}, step=global_step)
            wandb.log({f"{log}/loss_att": loss_att}, step=global_step)

    elif args.method == 'minilm_v2':
        loss_q, loss_k, loss_v = \
            minilm_v2(attentions_st, qkv_st, attentions_teacher, qkv_teacher, args.layer_selection)
        total_loss = loss_q + loss_k + loss_v        
        if master_process(args):
            wandb.log({f"{log}/loss": total_loss}, step=global_step)
            wandb.log({f"{log}/loss_q": loss_q}, step=global_step)
            wandb.log({f"{log}/loss_k": loss_k}, step=global_step)
            wandb.log({f"{log}/loss_v": loss_v}, step=global_step)

    elif args.method == 'pear_col':
        inter_token_1, inter_token_2, inter_head, inter_sentence = dist_att.forward(attentions_teacher[-1], attentions_st[-1])
        total_loss = inter_token_1 + inter_token_2 + inter_head + inter_sentence
        if master_process(args):
            wandb.log({f"{log}/loss": total_loss}, step=global_step)
            wandb.log({f"{log}/inter_token_1": inter_token_1}, step=global_step)
            wandb.log({f"{log}/inter_token_2": inter_token_2}, step=global_step)
            wandb.log({f"{log}/inter_head": inter_head}, step=global_step)
            wandb.log({f"{log}/inter_sentence": inter_sentence}, step=global_step)
            
    elif args.method == 'att_mse':
        loss_att= \
            att_mse(attentions_st, attentions_teacher, args.layer_selection)
        total_loss = loss_att
        if master_process(args):
            wandb.log({f"{log}/loss": total_loss}, step=global_step)
            wandb.log({f"{log}/loss_att": loss_att}, step=global_step)

    elif args.method == 'hidden_mse':
        loss_hidden = \
            hidden_mse(hidden_student, hidden_teacher, args.layer_selection)
        total_loss = loss_hidden
        if master_process(args):
            wandb.log({f"{log}/loss": total_loss}, step=global_step)
            wandb.log({f"{log}/loss_hidden": loss_hidden}, step=global_step)
    return total_loss






def twostage(teacher, student,args, batch, time_diff, global_step, wandb, eval=False):
    log = 'eval' if eval else 'train'
    time_proportion = time_diff / args.total_training_time
    if time_proportion < 0.5 :
        mlm_loss , attentions_st, values_st, prediction_score_st = \
            student.forward(batch, output_attentions=True, output_qkv=True, output_loss=True)
        if master_process(args):
            wandb.log({f"{log}/loss": mlm_loss}, step=global_step)
        total_loss = mlm_loss
    else:
        with torch.no_grad():
            attentions_teacher, values_teacher, prediction_score_teacher = \
                teacher(batch, output_attentions=True, output_qkv=True, output_loss=False)
        mlm_loss , attentions_st, values_st, prediction_score_st = \
            student.forward(batch, output_attentions=True, output_qkv=True, output_loss=True)

        loss_att, loss_val = \
                att_val_kl(attentions_st, values_st, attentions_teacher, values_teacher, args.layer_selection)
        total_loss = loss_att + loss_val
        if master_process(args):
            wandb.log({f"{log}/loss": total_loss}, step=global_step)
            wandb.log({f"{log}/loss_att": loss_att}, step=global_step)
            wandb.log({f"{log}/loss_val": loss_val}, step=global_step)
    return total_loss 


