from torch import nn
import torch
import torch.nn.functional as F
# from hflayers import HopfieldPooling, HopfieldLayer, Hopfield
import random
import math




## tiny att only
def att_mse_learn(extra_student, extra_teacher):
    
    attn_student = [attn for attn in extra_student['attn']]
    # len(attn_teacher) = 12
    # attn.shape = [384, l, l] 384=32x12
    attn_teacher = [attn for attn in extra_teacher['attn']]

    teacher_layer_num = len(attn_teacher)
    student_layer_num = len(attn_student)
    # layers_per_block = 4
    layers_per_block = int(teacher_layer_num / student_layer_num)
   
    # attn_teacher[3, 7, 11] 
    new_teacher_atts = [attn_teacher[i * layers_per_block + layers_per_block - 1]
                                for i in range(student_layer_num)]

    loss_att = 0.
    loss_rep = torch.tensor(0.)


    for student_att, teacher_att in zip(attn_student, new_teacher_atts):
        student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to('cuda'),
                                            student_att)
        teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to('cuda'),
                                            teacher_att)
        loss_tmp = F.mse_loss(student_att, teacher_att)
        loss_att += loss_tmp
    

    loss_inter = loss_att
    return loss_inter, loss_rep, loss_att


class Hidden_Mse(nn.Module):

    def __init__(self, inputSize=9984, outputSize=3072):
        super(Hidden_Mse, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)


    def hidden_mse_learn(self, extra_student, extra_teacher, mapping):
        

        attn_student = [attn for attn in extra_student['attn']]
        # len(attn_teacher) = 12
        # attn.shape = [384, l, l] 384=32x12
        attn_teacher = [attn for attn in extra_teacher['attn']]

        teacher_layer_num = len(attn_teacher)
        student_layer_num = len(attn_student)
        # layers_per_block = 4
        layers_per_block = int(teacher_layer_num / student_layer_num)
        # attn_teacher[3, 7, 11] 

        loss_att = torch.tensor(0.)
        loss_rep = 0.


        # len(hidden_student) = 4
        hidden_student = [h for h in extra_student['inner_states']]
        # len(hidden_teacher) = 13
        hidden_teacher = [h for h in extra_teacher['inner_states']]
        
        # hidden_teacher[0,4,8,12]
        if mapping == 'skip':
            new_teacher_reps = [hidden_teacher[i * layers_per_block] for i in range(student_layer_num + 1)]
            for student_rep, teacher_rep in zip(hidden_student, new_teacher_reps):
                loss_tmp = F.mse_loss(student_rep, teacher_rep)
                loss_rep += loss_tmp

        elif mapping == 'last':
            hidden_student = [hidden_student[-1]]
            new_teacher_reps = [hidden_teacher[-1]]
            for student_rep, teacher_rep in zip(hidden_student, new_teacher_reps):
                loss_tmp = F.mse_loss(student_rep, teacher_rep)
                loss_rep += loss_tmp

        elif mapping == 'combine':
            hidden_stat = torch.cat([t.transpose(0, 1) for t in hidden_student], dim=2)
            hidden_state_teacher_pre = torch.cat([t.transpose(0, 1) for t in hidden_teacher], dim=2)

            projected_teacher = self.linear(hidden_state_teacher_pre)

            loss_rep = F.mse_loss(hidden_stat, projected_teacher)
        # elif mapping == 'attention'd:


        loss_inter = loss_rep + loss_att
        return loss_inter, loss_rep, loss_att

class Hidden_Mse_Token(nn.Module):

    def __init__(self, inputSize=9984, outputSize=3072):
        super(Hidden_Mse_Token, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)


    def hidden_mse_learn(self, extra_student, extra_teacher, mapping):
        

        attn_student = [attn for attn in extra_student['attn']]
        # len(attn_teacher) = 12
        # attn.shape = [384, l, l] 384=32x12
        attn_teacher = [attn for attn in extra_teacher['attn']]

        teacher_layer_num = len(attn_teacher)
        student_layer_num = len(attn_student)
        # layers_per_block = 4
        layers_per_block = int(teacher_layer_num / student_layer_num)
        # attn_teacher[3, 7, 11] 

        loss_att = torch.tensor(0.)
        loss_rep = 0.


        # len(hidden_student) = 4
        hidden_student = [h.transpose(0, 1)[:,0] for h in extra_student['inner_states']]
        # len(hidden_teacher) = 13
        hidden_teacher = [h.transpose(0, 1)[:,0] for h in extra_teacher['inner_states']]
        
        # hidden_teacher[0,4,8,12]
        if mapping == 'skip':
            new_teacher_reps = [hidden_teacher[i * layers_per_block] for i in range(student_layer_num + 1)]
            for student_rep, teacher_rep in zip(hidden_student, new_teacher_reps):
                loss_tmp = F.mse_loss(student_rep, teacher_rep)
                loss_rep += loss_tmp

        elif mapping == 'last':
            hidden_student = [hidden_student[-1]]
            new_teacher_reps = [hidden_teacher[-1]]
            for student_rep, teacher_rep in zip(hidden_student, new_teacher_reps):
                loss_tmp = F.mse_loss(student_rep, teacher_rep)
                loss_rep += loss_tmp

        elif mapping == 'combine':
            hidden_stat = torch.cat([t.transpose(0, 1) for t in hidden_student], dim=2)
            hidden_state_teacher_pre = torch.cat([t.transpose(0, 1) for t in hidden_teacher], dim=2)

            projected_teacher = self.linear(hidden_state_teacher_pre)

            loss_rep = F.mse_loss(hidden_stat, projected_teacher)
        # elif mapping == 'attention'd:


        loss_inter = loss_rep + loss_att
        return loss_inter, loss_rep, loss_att



# tiny all
def att_mse_hidden_mse_learn(extra_student, extra_teacher):

    attn_student = [attn for attn in extra_student['attn']]
    # len(attn_teacher) = 12
    # attn.shape = [384, l, l] 384=32x12
    attn_teacher = [attn for attn in extra_teacher['attn']]

    teacher_layer_num = len(attn_teacher)
    student_layer_num = len(attn_student)
    # layers_per_block = 4
    layers_per_block = int(teacher_layer_num / student_layer_num)
    # attn_teacher[3, 7, 11] 
    new_teacher_atts = [attn_teacher[i * layers_per_block + layers_per_block - 1]
                                for i in range(student_layer_num)]

    loss_att = 0.
    loss_rep = 0.


    for student_att, teacher_att in zip(attn_student, new_teacher_atts):
        student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to('cuda'),
                                            student_att)
        teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to('cuda'),
                                            teacher_att)
        loss_tmp = F.mse_loss(student_att, teacher_att)
        loss_att += loss_tmp

    # len(hidden_student) = 4
    hidden_student = [h for h in extra_student['inner_states']]
    # len(hidden_teacher) = 13
    hidden_teacher = [h for h in extra_teacher['inner_states']]
    # hidden_teacher[0,4,8,12]
    new_teacher_reps = [hidden_teacher[i * layers_per_block] for i in range(student_layer_num + 1)]

    for student_rep, teacher_rep in zip(hidden_student, new_teacher_reps):
        loss_tmp = F.mse_loss(student_rep, teacher_rep)
        loss_rep += loss_tmp
    



    loss_inter = loss_rep + loss_att
    return loss_inter, loss_rep, loss_att



# mobile att
def att_kl_learn(extra_student, extra_teacher, kl_weight):
    attn_student = [attn for attn in extra_student['attn']]
            # len(attn_teacher) = 12
    attn_teacher = [attn for attn in extra_teacher['attn']]

    teacher_layer_num = len(attn_teacher)
    student_layer_num = len(attn_student)
    # layers_per_block = 4
    layers_per_block = int(teacher_layer_num / student_layer_num)
    # attn_teacher[3, 7, 11] 
    new_teacher_atts = [attn_teacher[i * layers_per_block + layers_per_block - 1]
                                for i in range(student_layer_num)]

    # new_teacher_atts = []
    # for i in [0,3,5,7]:   
    #     new_teacher_atts.append(attn_teacher[i])

    loss_att = 0.
    loss_rep = torch.tensor(0.)
    loss_mean = 0.
    loss_var = 0.
    for student_att, teacher_att in zip(attn_student, new_teacher_atts):
        student_att_soft = F.softmax(student_att, dim=-1)
        student_att_logsoft = torch.clamp(student_att_soft, 1e-7, 1).log()
        # student_att = F.log_softmax(student_att, dim=-1)
        teacher_att = F.softmax(teacher_att, dim=-1)
        # batch_size, head_num, lenght = student_att.shape[0], student_att.shape[1], student_att.shape[2]
        loss_att_tmp = F.kl_div(student_att_logsoft, teacher_att, reduction='sum')/ (student_att.shape[0] * student_att.shape[1]) 
        # loss_att_tmp = F.kl_div(student_att, teacher_att, reduction='sum')/ (student_att.shape[0] * student_att.shape[1]) #, reduction='batchmean', log_target=True)
        loss_att += loss_att_tmp
    
    #  # len(hidden_student) = 4
    # hidden_student = [h for h in extra_student['inner_states']]
    # # len(hidden_teacher) = 13
    # hidden_teacher = [h for h in extra_teacher['inner_states']]
    # # hidden_teacher[0,4,8,12]
    # new_teacher_reps = [hidden_teacher[i * layers_per_block] for i in range(student_layer_num + 1)]

    # for student_rep, teacher_rep in zip(hidden_student, new_teacher_reps):
    #     loss_tmp = F.mse_loss(student_rep, teacher_rep)
    #     loss_rep += loss_tmp

    #     student_mean = torch.mean(student_rep, dim=-1)
    #     teacher_mean = torch.mean(teacher_rep, dim=-1)
    #     student_var = torch.var(student_rep, dim=-1)
    #     teacher_var = torch.var(teacher_rep, dim=-1)

    #     loss_mean_tmp = F.mse_loss(student_mean, teacher_mean)
    #     loss_var_tmp = F.mse_loss(student_var, teacher_var)

    #     loss_mean += loss_mean_tmp
    #     loss_var += loss_var_tmp


    # loss_inter = loss_att  #loss_rep + kl_weight * loss_kl #+ loss_mean + loss_var
    return   loss_att, loss_rep, loss_att #loss_inter, loss_rep, loss_kl#, loss_mean, loss_var


# mobile all
def att_kl_hiden_mse_learn(extra_student, extra_teacher, kl_weight):
    attn_student = [attn for attn in extra_student['attn']]
            # len(attn_teacher) = 12
    attn_teacher = [attn for attn in extra_teacher['attn']]

    teacher_layer_num = len(attn_teacher)
    student_layer_num = len(attn_student)
    # layers_per_block = 4
    layers_per_block = int(teacher_layer_num / student_layer_num)
    # attn_teacher[3, 7, 11] 
    new_teacher_atts = [attn_teacher[i * layers_per_block + layers_per_block - 1]
                                for i in range(student_layer_num)]

    # new_teacher_atts = []
    # for i in [0,3,5,7]:   
    #     new_teacher_atts.append(attn_teacher[i])

    loss_att = 0.
    loss_rep = 0.
    loss_mean = 0.
    loss_var = 0.
    for student_att, teacher_att in zip(attn_student, new_teacher_atts):
        student_att = F.log_softmax(student_att, dim=-1)
        teacher_att = F.softmax(teacher_att, dim=-1)
        # batch_size, head_num, lenght = student_att.shape[0], student_att.shape[1], student_att.shape[2]
        loss_kl_tmp = F.kl_div(student_att, teacher_att, reduction='sum')/ (student_att.shape[0] * student_att.shape[1]) #, reduction='batchmean', log_target=True)
        loss_att += loss_kl_tmp
    
     # len(hidden_student) = 4
    hidden_student = [h for h in extra_student['inner_states']]
    # len(hidden_teacher) = 13
    hidden_teacher = [h for h in extra_teacher['inner_states']]
    # hidden_teacher[0,4,8,12]
    new_teacher_reps = [hidden_teacher[i * layers_per_block] for i in range(student_layer_num + 1)]

    for student_rep, teacher_rep in zip(hidden_student, new_teacher_reps):
        loss_tmp = F.mse_loss(student_rep, teacher_rep)
        loss_rep += loss_tmp

        # student_mean = torch.mean(student_rep, dim=-1)
        # teacher_mean = torch.mean(teacher_rep, dim=-1)
        # student_var = torch.var(student_rep, dim=-1)
        # teacher_var = torch.var(teacher_rep, dim=-1)

        # loss_mean_tmp = F.mse_loss(student_mean, teacher_mean)
        # loss_var_tmp = F.mse_loss(student_var, teacher_var)

        # loss_mean += loss_mean_tmp
        # loss_var += loss_var_tmp


    loss_inter =  loss_rep + loss_att #+ loss_mean + loss_var
    return   loss_inter, loss_rep, loss_att #loss_inter, loss_rep, loss_kl#, loss_mean, loss_var


# miniml
def minilm_feature_learn(extra_student, extra_teacher, kl_weight, layer_selection):
    
    attn_student = [attn[0] for attn in extra_student['attn']]
            # len(attn_teacher) = 12
    attn_teacher = [attn[0] for attn in extra_teacher['attn']]

    batch_head_size, length, dk = extra_student['attn'][0][1].shape
    dk_sqrt = math.sqrt(dk)

    teacher_layer_num = len(attn_teacher)
    student_layer_num = len(attn_student)
    # layers_per_block = 4
    layers_per_block = int(teacher_layer_num / student_layer_num)

    # 能整除的情况下 
    # layer_selection = [int(item) for item in layer_selection.split(',')]
    # if len(layer_selection) == 1: 
    #     attn_student = [attn_student[-1]]

    new_teacher_atts = [attn_teacher[i * layers_per_block + layers_per_block - 1]
                                for i in range(student_layer_num)]
    # new_teacher_atts = [attn_teacher[i] for i in layer_selection]
    # new_teacher_atts = []
    # for i in [0,3,5,7]:   
    #     new_teacher_atts.append(attn_teacher[i])

    loss_att = 0.
    loss_value = 0.
    # loss_rep = 0.
    # loss_mean = 0.
    # loss_var = 0.
    for student_att, teacher_att in zip(attn_student, new_teacher_atts):
        student_att = F.log_softmax(student_att, dim=-1)
        teacher_att = F.softmax(teacher_att, dim=-1)
        # batch_size, head_num, lenght = student_att.shape[0], student_att.shape[1], student_att.shape[2]
        loss_kl_tmp = F.kl_div(student_att, teacher_att, reduction='sum')/ (batch_head_size * length) #, reduction='batchmean', log_target=True)
        loss_att += loss_kl_tmp

    
    value_student = [attn[1] for attn in extra_student['attn']]
    value_teacher = [attn[1] for attn in extra_teacher['attn']]
    if len(layer_selection) == 1: 
        value_student = [value_student[-1]]

    # new_teacher_atts = [attn_teacher[i * layers_per_block + layers_per_block - 1]
    #                             for i in range(student_layer_num)]
    # new_teacher_value = [value_teacher[i] for i in layer_selection]
    
    new_teacher_value = [value_teacher[i * layers_per_block + layers_per_block - 1]
                            for i in range(student_layer_num)]


    for student_value, teacher_value in zip(value_student, new_teacher_value):
        vr_student = F.log_softmax(torch.bmm(student_value, student_value.transpose(1,2))/dk_sqrt, dim=-1)
        vr_teacher = F.softmax(torch.bmm(teacher_value, teacher_value.transpose(1,2))/dk_sqrt, dim=-1)

        loss_value_tmp = F.kl_div(vr_student, vr_teacher, reduction='sum')/(batch_head_size * length)
        loss_value += loss_value_tmp

    
    loss  = loss_att + loss_value

    return loss, loss_att, loss_value


def minilm_feature_learn_single_layer(extra_student, extra_teacher, kl_weight):
    attn_student = [attn[0] for attn in extra_student['attn']]
            # len(attn_teacher) = 12
    attn_teacher = [attn[0] for attn in extra_teacher['attn']]

    batch_head_size, length, dk = extra_student['attn'][0][1].shape
    dk_sqrt = math.sqrt(dk)

    teacher_layer_num = len(attn_teacher)
    student_layer_num = len(attn_student)
    # layers_per_block = 4
    layers_per_block = int(teacher_layer_num / student_layer_num)

    # 能整除的情况下 

    new_teacher_atts = [attn_teacher[i * layers_per_block + layers_per_block - 1]
                                for i in range(student_layer_num)]

    # new_teacher_atts = []
    # for i in [0,3,5,7]:   
    #     new_teacher_atts.append(attn_teacher[i])

    loss_att = 0.
    loss_value = 0.
    # loss_rep = 0.
    # loss_mean = 0.
    # loss_var = 0.
    # for student_att, teacher_att in zip(attn_student, new_teacher_atts):
    student_att = F.log_softmax(attn_student[-1], dim=-1)
    teacher_att = F.softmax(attn_teacher[-1], dim=-1)
    # batch_size, head_num, lenght = student_att.shape[0], student_att.shape[1], student_att.shape[2]
    loss_att = F.kl_div(student_att, teacher_att, reduction='sum')/ (batch_head_size * length) #, reduction='batchmean', log_target=True)
     

    
    value_student = [attn[1] for attn in extra_student['attn']]
    value_teacher = [attn[1] for attn in extra_teacher['attn']]

    
    new_teacher_value = [value_teacher[i * layers_per_block + layers_per_block - 1]
                            for i in range(student_layer_num)]


    student_value = value_student[-1]
    teacher_value = value_teacher[-1]
    vr_student = F.log_softmax(torch.bmm(student_value, student_value.transpose(1,2))/dk_sqrt, dim=-1)
    vr_teacher = F.softmax(torch.bmm(teacher_value, teacher_value.transpose(1,2))/dk_sqrt, dim=-1)

    loss_value = F.kl_div(vr_student, vr_teacher, reduction='sum')/(batch_head_size * length)

    
    loss  = loss_att + loss_value

    return loss, loss_att, loss_value



def minilm_feature_learn_random(extra_student, extra_teacher, kl_weight):
    attn_student = [attn[0] for attn in extra_student['attn']]
            # len(attn_teacher) = 12
    attn_teacher = [attn[0] for attn in extra_teacher['attn']]

    batch_head_size, length, dk = extra_student['attn'][0][1].shape
    dk_sqrt = math.sqrt(dk)

    teacher_layer_num = len(attn_teacher)
    student_layer_num = len(attn_student)
    # layers_per_block = 4
    layers_per_block = int(teacher_layer_num / student_layer_num)


    new_teacher_atts = [attn_teacher[i * layers_per_block + layers_per_block - 1]
                                for i in range(student_layer_num)]

    # new_teacher_atts = []
    # for i in [0,3,5,7]:   
    #     new_teacher_atts.append(attn_teacher[i])

    loss_att = 0.
    loss_value = 0.
    # loss_rep = 0.
    # loss_mean = 0.
    # loss_var = 0.
    for idx, student_att in enumerate(attn_student):
        idx_t = random.sample(range(2 * idx, 2 * idx + 2), 1)[0]
        teacher_att = attn_teacher[idx_t]
        student_att = F.log_softmax(student_att, dim=-1)
        teacher_att = F.softmax(teacher_att, dim=-1)
        # batch_size, head_num, lenght = student_att.shape[0], student_att.shape[1], student_att.shape[2]
        loss_kl_tmp = F.kl_div(student_att, teacher_att, reduction='sum')/ (batch_head_size * length) #, reduction='batchmean', log_target=True)
        loss_att += loss_kl_tmp

     

    
    value_student = [attn[1] for attn in extra_student['attn']]
    value_teacher = [attn[1] for attn in extra_teacher['attn']]

    
    new_teacher_value = [value_teacher[i * layers_per_block + layers_per_block - 1]
                            for i in range(student_layer_num)]


    for idx, student_value in enumerate(value_student):
        idx_t = random.sample(range(2 * idx, 2 * idx + 2), 1)[0]
        teacher_value = value_teacher[idx_t]

        vr_student = F.log_softmax(torch.bmm(student_value, student_value.transpose(1,2))/dk_sqrt, dim=-1)
        vr_teacher = F.softmax(torch.bmm(teacher_value, teacher_value.transpose(1,2))/dk_sqrt, dim=-1)

        loss_value_tmp = F.kl_div(vr_student, vr_teacher, reduction='sum')/(batch_head_size * length)
        loss_value += loss_value_tmp

    
    loss  = loss_att + loss_value

    return loss, loss_att, loss_value








def key_query_val(extra_student, extra_teacher):

    query_student = [attn[0] for attn in extra_student['attn']]
    key_student = [attn[1] for attn in extra_student['attn']]
    value_student = [attn[2] for attn in extra_student['attn']]

    query_teacher = [attn[0] for attn in extra_teacher['attn']]
    key_teacher = [attn[1] for attn in extra_teacher['attn']]
    value_teacher = [attn[2] for attn in extra_teacher['attn']]

    teacher_layer_num = len(query_teacher)
    student_layer_num = len(query_student)
    layers_per_block = int(teacher_layer_num / student_layer_num)

    new_teacher_value = [value_teacher[i * layers_per_block + layers_per_block - 1]
                            for i in range(student_layer_num)]
    new_teacher_key = [key_teacher[i * layers_per_block + layers_per_block - 1]
                            for i in range(student_layer_num)]
    new_teacher_query = [query_teacher[i * layers_per_block + layers_per_block - 1]
                            for i in range(student_layer_num)]

    batch_head_size, length, dk = extra_student['attn'][0][0].shape
    dk_sqrt = math.sqrt(dk)



    loss_value = 0.
    loss_key = 0.
    loss_query = 0.

    for student_value, teacher_value in zip(value_student, new_teacher_value):
        vr_student = F.log_softmax(torch.bmm(student_value, student_value.transpose(1,2))/dk_sqrt, dim=-1)
        vr_teacher = F.softmax(torch.bmm(teacher_value, teacher_value.transpose(1,2))/dk_sqrt, dim=-1)

        loss_value_tmp = F.kl_div(vr_student, vr_teacher, reduction='sum')/(batch_head_size * length)
        loss_value += loss_value_tmp

    for student_key, teacher_key in zip(key_student, new_teacher_key):
        kr_student = F.log_softmax(torch.bmm(student_key, student_key.transpose(1,2))/dk_sqrt, dim=-1)
        kr_teacher = F.softmax(torch.bmm(teacher_key, teacher_key.transpose(1,2))/dk_sqrt, dim=-1)

        loss_key_tmp = F.kl_div(kr_student, kr_teacher, reduction='sum')/(batch_head_size * length)
        loss_key += loss_key_tmp



    for student_query, teacher_query in zip(query_student, new_teacher_query):
        qr_student = F.log_softmax(torch.bmm(student_query, student_query.transpose(1,2))/dk_sqrt, dim=-1)
        qr_teacher = F.softmax(torch.bmm(teacher_query, teacher_query.transpose(1,2))/dk_sqrt, dim=-1)

        loss_query_tmp = F.kl_div(qr_student, qr_teacher, reduction='sum')/(batch_head_size * length)
        loss_query += loss_query_tmp



    loss = loss_value + loss_key + loss_query
    return loss