# TODO: Move Tiny, Mobile, Crd method to here
from torch import nn
import torch
import torch.nn.functional as F
# from hflayers import HopfieldPooling, HopfieldLayer, Hopfield
import random


def tiny_feature_learn(extra_student, extra_teacher):
    

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
# class TinyFeatureLearn(nn.Module):

# TODO: Create MobileBERT method
def mobile_feature_learn(extra_student, extra_teacher, kl_weight):
    attn_student = [attn for attn in extra_student['attn']]
            # len(attn_teacher) = 12
    attn_teacher = [attn for attn in extra_teacher['attn']]

    teacher_layer_num = len(attn_teacher)
    student_layer_num = len(attn_student)
    # layers_per_block = 4
    layers_per_block = int(teacher_layer_num / student_layer_num)

    new_teacher_atts = [attn_teacher[i * layers_per_block + layers_per_block - 1]
                                for i in range(student_layer_num)]

    # new_teacher_atts = []
    # for i in [0,2,4,6,8,9,10,11]:   
    #     new_teacher_atts.append(attn_teacher[i])

    loss_kl = 0.
    loss_rep = 0.
    loss_mean = 0.
    loss_var = 0.
    for student_att, teacher_att in zip(attn_student, new_teacher_atts):
        student_att = F.log_softmax(student_att, dim=-1)
        teacher_att = F.softmax(teacher_att, dim=-1)
        # batch_size, head_num, lenght = student_att.shape[0], student_att.shape[1], student_att.shape[2]
        loss_kl_tmp = F.kl_div(student_att, teacher_att, reduction='sum')/ (student_att.shape[0] * student_att.shape[1]) #, reduction='batchmean', log_target=True)
        loss_kl += loss_kl_tmp
    
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


    # loss_inter = (1 - kl_weight) * loss_rep + kl_weight * loss_kl #+ loss_mean + loss_var
    return   loss_kl #loss_inter, loss_rep, loss_kl#, loss_mean, loss_var

# TODO: develop hopfiel method


class HopfieldFeatureLearn(nn.Module):

    def  __init__(self):
        super(HopfieldFeatureLearn, self).__init__()

        # self.hopfield_1 = Hopfield( # scaling=beta,
        #                         # do not project layer input
        #                         state_pattern_as_static=True,
        #                         stored_pattern_as_static=True,
        #                         pattern_projection_as_static=True,

        #                         # do not pre-process layer input
        #                         normalize_stored_pattern=False,
        #                         normalize_stored_pattern_affine=False,
        #                         normalize_state_pattern=False,
        #                         normalize_state_pattern_affine=False,
        #                         normalize_pattern_projection=False,
        #                         normalize_pattern_projection_affine=False,
        #                         # do not post-process layer output
        #                         disable_out_projection=True)
        # self.hopfield_2 = Hopfield( # scaling=beta,
        #                         # do not project layer input
        #                         state_pattern_as_static=True,
        #                         stored_pattern_as_static=True,
        #                         pattern_projection_as_static=True,

        #                         # do not pre-process layer input
        #                         normalize_stored_pattern=False,
        #                         normalize_stored_pattern_affine=False,
        #                         normalize_state_pattern=False,
        #                         normalize_state_pattern_affine=False,
        #                         normalize_pattern_projection=False,
        #                         normalize_pattern_projection_affine=False,

        #                         # do not post-process layer output
        #                         disable_out_projection=True)
        # self.hopfield_3 = Hopfield( # scaling=beta,
        #                         # do not project layer input
        #                         state_pattern_as_static=True,
        #                         stored_pattern_as_static=True,
        #                         pattern_projection_as_static=True,

        #                         # do not pre-process layer input
        #                         normalize_stored_pattern=False,
        #                         normalize_stored_pattern_affine=False,
        #                         normalize_state_pattern=False,
        #                         normalize_state_pattern_affine=False,
        #                         normalize_pattern_projection=False,
        #                         normalize_pattern_projection_affine=False,

        #                         # do not post-process layer output
        # #                         disable_out_projection=True)
        self.hopfield_pooling_1 = HopfieldPooling(
                                input_size=768,
                                quantity=4,
                                # do not project layer input
                                state_pattern_as_static=True,
                                stored_pattern_as_static=True, 
                                pattern_projection_as_static=True,

                                # do not pre-process layer input
                                normalize_stored_pattern=False,
                                normalize_stored_pattern_affine=False,
                                normalize_state_pattern=False,
                                normalize_state_pattern_affine=False,
                                normalize_pattern_projection=False,
                                normalize_pattern_projection_affine=False,

                                # do not post-process layer output
                                disable_out_projection=True
                                )

        # self.hopfield_pooling_2 = HopfieldPooling(
        #                         input_size=512,
        #                         quantity=512,
        #                         # do not project layer input
        #                         state_pattern_as_static=True,
        #                         stored_pattern_as_static=True, 
        #                         pattern_projection_as_static=True,

        #                         # do not pre-process layer input
        #                         normalize_stored_pattern=False,
        #                         normalize_stored_pattern_affine=False,
        #                         normalize_state_pattern=False,
        #                         normalize_state_pattern_affine=False,
        #                         normalize_pattern_projection=False,
        #                         normalize_pattern_projection_affine=False,

        #                         # do not post-process layer output
        #                         disable_out_projection=True
        #                         )
        # self.hopfield_pooling_3 = HopfieldPooling(
        #                         input_size=512,
        #                         quantity=512,
        #                         # do not project layer input
        #                         state_pattern_as_static=True,
        #                         stored_pattern_as_static=True, 
        #                         pattern_projection_as_static=True,

        #                         # do not pre-process layer input
        #                         normalize_stored_pattern=False,
        #                         normalize_stored_pattern_affine=False,
        #                         normalize_state_pattern=False,
        #                         normalize_state_pattern_affine=False,
        #                         normalize_pattern_projection=False,
        #                         normalize_pattern_projection_affine=False,

        #                         # do not post-process layer output
        #                         disable_out_projection=True
        #                         )
        self.ce_loss = nn.CrossEntropyLoss()
    def hopfield_feature_learn_nopooling(self, extra_student, extra_teacher, head_num):
        # attn_student = [attn for attn in extra_student['attn']]
                # len(attn_teacher) = 12
        # attn_teacher = [attn for attn in extra_teacher['attn']]
        # TODO: collect all layers' attention and make the dimensions right
        lenth = extra_student['attn'][0].shape[-1]
        batch_size = int(extra_student['attn'][0].shape[0]/head_num)

        attn_student_1 = F.softmax(extra_student['attn'][0].reshape(batch_size, head_num, -1))
        attn_student_2 = F.softmax(extra_student['attn'][1].reshape(batch_size, head_num, -1))
        attn_student_3 = F.softmax(extra_student['attn'][2].reshape(batch_size, head_num, -1))
        attn_teacher_1 = torch.cat([F.softmax(attn.reshape(batch_size, head_num, -1)) for attn in extra_teacher['attn'][:4]], dim=1)
        attn_teacher_2 = torch.cat([F.softmax(attn.reshape(batch_size, head_num, -1)) for attn in extra_teacher['attn'][4:8]], dim=1)
        attn_teacher_3 = torch.cat([F.softmax(attn.reshape(batch_size, head_num, -1)) for attn in extra_teacher['attn'][8:12]], dim=1)
        
        

        # TODO: construct hopfield layer for retrieval
        attn_teacher_1_retrieved = self.hopfield_1((attn_teacher_1, attn_student_1, attn_teacher_1))
        attn_teacher_2_retrieved = self.hopfield_2((attn_teacher_2, attn_student_2, attn_teacher_2))
        attn_teacher_3_retrieved = self.hopfield_3((attn_teacher_3, attn_student_3, attn_teacher_3))

        # loss
        loss1 = F.kl_div(F.log_softmax(extra_student['attn'][0].reshape(batch_size, head_num, lenth, lenth)), attn_teacher_1_retrieved.reshape(batch_size, head_num, lenth, lenth), reduction='sum')/(batch_size * head_num * lenth)
        loss2 = F.kl_div(F.log_softmax(extra_student['attn'][1].reshape(batch_size, head_num, lenth, lenth)), attn_teacher_2_retrieved.reshape(batch_size, head_num, lenth, lenth), reduction='sum')/(batch_size * head_num * lenth)
        loss3 = F.kl_div(F.log_softmax(extra_student['attn'][2].reshape(batch_size, head_num, lenth, lenth)), attn_teacher_3_retrieved.reshape(batch_size, head_num, lenth, lenth), reduction='sum')/(batch_size * head_num * lenth)


        loss_inter = loss1 + loss2 + loss3




        # TODO: compare student with retrieved att for MSE losss

        return loss_inter





    def hopfield_feature_learn_pooling(self, extra_student, extra_teacher, head_num):
        
        

        batch_size = extra_student['attn'][0][0].shape[0]
        length = extra_student['attn'][0][0].shape[-1]

        p1d = (0, 512-length)
        key_padding_mask = extra_student['attn'][0][1]
        key_padding_mask_pad = F.pad(key_padding_mask, p1d, 'constant', 1)

        stach_head = head_num * length
        attn_student_1 = extra_student['attn'][0][0]
        attn_student_2 = extra_student['attn'][1][0]
        attn_student_3 = extra_student['attn'][2][0]
        attn_teacher_1 = torch.cat([attn[0].reshape(batch_size, stach_head, length) for attn in extra_teacher['attn'][:4]], dim=1)
        attn_teacher_2 = torch.cat([attn[0].reshape(batch_size, stach_head, length) for attn in extra_teacher['attn'][4:8]], dim=1)
        attn_teacher_3 = torch.cat([attn[0].reshape(batch_size, stach_head, length) for attn in extra_teacher['attn'][8:12]], dim=1)

        attn_student_list = [attn_student_1, attn_student_2, attn_student_3]
        attn_teacher_list = [attn_teacher_1, attn_teacher_2, attn_teacher_3]
        hopfield_layer_list = [self.hopfield_pooling_1, self.hopfield_pooling_2, self.hopfield_pooling_3]

        

        loss_inter = 0.

        dim = 12 *4 *length

        p2d = (0, 512-length, 0, 512-length)
        for attn_std, attn_thr, hopfield_pooling in zip(attn_student_list, attn_teacher_list, hopfield_layer_list):
            attn_std = torch.where(attn_std <= -1e2, torch.zeros_like(attn_std).to('cuda'),
                                            attn_std)
            attn_thr = torch.where(attn_thr <= -1e2, torch.zeros_like(attn_thr).to('cuda'),
                                            attn_thr)
            attn_std = F.pad(attn_std, p2d, 'constant', 0)
            attn_thr = F.pad(attn_thr, p1d, 'constant', 0)



            #attn mask
            c = torch.zeros((batch_size,512,dim), dtype=torch.bool)
            c[key_padding_mask_pad,:]=1
            attn_mask_hopfield = c      



            attn_thr_pooled = hopfield_pooling(attn_thr, None, attn_mask_hopfield.to('cuda'))       
            loss_tmp =  F.mse_loss(attn_std, attn_thr_pooled.nan_to_num(), reduction='sum')/(batch_size * length * length)         
            loss_inter += loss_tmp
    

        return loss_inter

    def hopfield_feature_learn_dim_pooling(self, extra_student, extra_teacher):
        # attn_student = [attn for attn in extra_student['attn']]
        # # len(attn_teacher) = 12
        # # attn.shape = [384, l, l] 384=32x12
        # attn_teacher = [attn for attn in extra_teacher['attn']]

        # teacher_layer_num = len(attn_teacher)
        # student_layer_num = len(attn_student)
        # # layers_per_block = 4
        # layers_per_block = int(teacher_layer_num / student_layer_num)
        # attn_teacher[3, 7, 11] 
        # new_teacher_atts = [attn_teacher[i * layers_per_block + layers_per_block - 1]
        #                             for i in range(student_layer_num)]
        
        # hidden_stat_pre = torch.cat([h. for h in extra_student['inner_states']])
        # # len(hidden_teacher) = 13
        # hidden_state_teacher_pre = torch.cat([h for h in extra_teacher['inner_states']])
        # hidden_teacher[0,4,8,12]
        # new_teacher_reps = [hidden_teacher[i * layers_per_block] for i in range(student_layer_num + 1)]


        
        hidden_stat_pre = torch.cat([t.transpose(0, 1) for t in extra_student['inner_states'][1:]], dim=2).transpose(1, 2)
        hidden_state_teacher_pre = torch.cat([t.transpose(0, 1) for t in extra_teacher['inner_states'][1:]], dim=2).transpose(1, 2)
        p1d = (0, 512 - hidden_stat_pre.shape[2])
        hidden_stat_pad = F.pad(hidden_stat_pre, p1d, "constant", 0)
        
        p2d = (0, 512 - hidden_state_teacher_pre.shape[2])
        hidden_state_teacher_pad = F.pad(hidden_state_teacher_pre, p2d, "constant", 0)
        
        dim_t = hidden_state_teacher_pad.shape[1]
        hidden_state_teacher_1 = hidden_state_teacher_pad[:, :3072,:]
        hidden_state_teacher_2 = hidden_state_teacher_pad[:,3072: 6144,:]
        hidden_state_teacher_3 = hidden_state_teacher_pad[:,6144: ,:]

        hidden_state_teache_pooled_1 = self.hopfield_pooling_1(hidden_state_teacher_1)
        hidden_state_teache_pooled_2 = self.hopfield_pooling_2(hidden_state_teacher_2)
        hidden_state_teache_pooled_3 = self.hopfield_pooling_3(hidden_state_teacher_3)

        
        loss_feature = F.mse_loss(hidden_stat_pad[:,:768,:], hidden_state_teache_pooled_1) +\
                        F.mse_loss(hidden_stat_pad[:,768:1536,:], hidden_state_teache_pooled_2) +\
                            F.mse_loss(hidden_stat_pad[:,1536:,:], hidden_state_teache_pooled_3)
        
        
        
        return loss_feature

    def hopfield_sentence_pooling(self, extra_student, extra_teacher):
        attn_student = [attn for attn in extra_student['attn']]
            # len(attn_teacher) = 12
        attn_teacher = [attn for attn in extra_teacher['attn']]

        teacher_layer_num = len(attn_teacher)
        student_layer_num = len(attn_student)
        # layers_per_block = 4
        layers_per_block = int(teacher_layer_num / student_layer_num)
        
        
        # len(hidden_student) = 4
        hidden_student = [h for h in extra_student['inner_states']]
        # len(hidden_teacher) = 13
        hidden_teacher = [h for h in extra_teacher['inner_states']]
        # hidden_teacher[0,4,8,12]
        new_teacher_reps = [hidden_teacher[i * layers_per_block] for i in range(student_layer_num + 1)]


        loss_rep = 0.
        for student_rep, teacher_rep in zip(hidden_student, new_teacher_reps):
            # student_rep = student_rep.transpose(0, 1)[:, 0]
            # teacher_rep = teacher_rep.transpose(0, 1)[:, 0]

            


            # norm_student = F.normalize(student_rep, dim=1)
            # norm_teacher = F.normalize(teacher_rep, dim=1)



            loss_tmp = F.mse_loss(norm_student, norm_teacher, reduction='sum')

            loss_rep += loss_tmp

        loss_inter = loss_rep

        return loss_inter

    def hopfield_layer_association(self, extra_student, extra_teacher):
        #TODO: construct the layer features
        # which form?  Try cls one 
        attn_student = [attn for attn in extra_student['attn']]
            # len(attn_teacher) = 12
        attn_teacher = [attn for attn in extra_teacher['attn']]

        teacher_layer_num = len(attn_teacher)
        student_layer_num = len(attn_student)
        # layers_per_block = 4
        layers_per_block = int(teacher_layer_num / student_layer_num)
        
        
        # len(hidden_student) = 4
        hidden_student = torch.cat([h[0:1,:,:] for h in extra_student['inner_states']]).transpose(0,1)
        # len(hidden_teacher) = 13
        hidden_teacher = torch.cat([h[0:1,:,:] for h in extra_teacher['inner_states']]).transpose(0,1).transpose(1,2)
        # hidden_teacher[0,4,8,12]
        # new_teacher_reps = [hidden_teacher[i * layers_per_block] for i in range(student_layer_num + 1)]

        product = torch.bmm(hidden_student, hidden_teacher)

        #  teacher [bsz, 12, 768 ], student [bsz, 3, 768]

        target = torch.tensor([0,4,8,12]).to('cuda')
        bsz = product.shape[0]

        loss = 0.
        for i in range(product.shape[1]):
            layer = i * 4
            target = torch.tensor([layer]).repeat(bsz).to('cuda')
            loss_tmp = self.ce_loss(product[:,i,:], target)
            loss += loss_tmp
        # retrieved = self.hopfield((teacher, student,))

        

        #TODO: feed to hopfield layer and get ass matrix

        #TODO: CE loss


        return loss

    def tiny_with_association(self, extra_student, extra_teacher):

        attn_student = [attn for attn in extra_student['attn']]
            # len(attn_teacher) = 12
        attn_teacher = [attn for attn in extra_teacher['attn']]

        teacher_layer_num = len(attn_teacher)
        student_layer_num = len(attn_student)
        # layers_per_block = 4
        layers_per_block = int(teacher_layer_num / student_layer_num)

        new_teacher_atts = [attn_teacher[i * layers_per_block + layers_per_block - 1]
                                for i in range(student_layer_num)]

        loss_kl = 0.
        for student_att, teacher_att in zip(attn_student, new_teacher_atts):
            student_att = F.log_softmax(student_att, dim=-1)
            teacher_att = F.softmax(teacher_att, dim=-1)
            # batch_size, head_num, lenght = student_att.shape[0], student_att.shape[1], student_att.shape[2]
            loss_kl_tmp = F.kl_div(student_att, teacher_att, reduction='sum')/ (student_att.shape[0] * student_att.shape[1]) #, reduction='batchmean', log_target=True)
            loss_kl += loss_kl_tmp


        
         # len(hidden_student) = 4
        hidden_student = [h.transpose(0,1) for h in extra_student['inner_states']]
        # len(hidden_teacher) = 13
        hidden_teacher = [h.transpose(0,1) for h in extra_teacher['inner_states']]
        # hidden_teacher[0,4,8,12]
        new_teacher_reps = [hidden_teacher[i * layers_per_block] for i in range(student_layer_num + 1)]
        
        hidden_teacher_mem = torch.cat(hidden_teacher, dim=1)

        loss_rep = 0.
        for student_rep, teacher_rep in zip(hidden_student, new_teacher_reps):
            retrieved_teacher_rep = self.hopfield_1((hidden_teacher_mem, student_rep, hidden_teacher_mem))
            loss_tmp = F.mse_loss(retrieved_teacher_rep, teacher_rep)
            loss_rep += loss_tmp
        rep_weight = 0.8
        
        loss = rep_weight*loss_rep + (1-rep_weight)*loss_kl
        return loss, loss_rep, loss_kl



    def hopfield_cls_pooling(self,  extra_student, extra_teacher):
        hidden_student = [h for h in extra_student['inner_states']]
        # len(hidden_teacher) = 13
        hidden_teacher = [h for h in extra_teacher['inner_states']]

        hidden_state_student = torch.cat([t.transpose(0, 1)[:, 0:1] for t in hidden_student], dim=1)

        hidden_state_teacher = torch.cat([t.transpose(0, 1)[:, 0:1] for t in hidden_teacher], dim=1)


        pooled_feature = self.hopfield_pooling_1(hidden_state_teacher)
        
        
        norm_student = F.normalize(hidden_state_student, dim=1)
        norm_teacher = F.normalize(pooled_feature, dim=1)



        loss = F.mse_loss(norm_student, norm_teacher)

        return loss
    def hopfield_cls_nopool(self, extra_student, extra_teacher):
        hidden_student = [h for h in extra_student['inner_states']]
        # len(hidden_teacher) = 13
        hidden_teacher = [h for h in extra_teacher['inner_states']]

        hidden_state_student = torch.cat([t.transpose(0, 1)[:, 0:1] for t in hidden_student], dim=1)

        hidden_state_teacher = torch.cat([t.transpose(0, 1)[:, 0:1] for t in hidden_teacher], dim=1)


        pooled_feature = self.hopfield_1((hidden_state_teacher, hidden_state_student, hidden_state_teacher))
        
        
        norm_student = F.normalize(hidden_state_student, dim=1)
        norm_teacher = F.normalize(pooled_feature, dim=1)
        


        loss = F.mse_loss(norm_student, norm_teacher)
        return loss
    def hopfield_mean_pooling(self, extra_student, extra_teacher):
        hidden_student = [h for h in extra_student['inner_states']]
        # len(hidden_teacher) = 13
        hidden_teacher = [h for h in extra_teacher['inner_states']]

        hidden_state_student = torch.cat([t.transpose(0, 1).mean(1, keepdim=True) for t in hidden_student], dim=1)

        hidden_state_teacher = torch.cat([t.transpose(0, 1).mean(1, keepdim=True) for t in hidden_teacher], dim=1)


        pooled_feature = self.hopfield_1((hidden_state_teacher, hidden_state_student, hidden_state_teacher))
        
        
        norm_student = F.normalize(hidden_state_student, dim=1)
        norm_teacher = F.normalize(pooled_feature, dim=1)
        


        loss = F.mse_loss(norm_student, norm_teacher)
        return loss




def pkd_feature_learn(extra_student, extra_teacher ):
    attn_student = [attn for attn in extra_student['attn']]
            # len(attn_teacher) = 12
    attn_teacher = [attn for attn in extra_teacher['attn']]

    teacher_layer_num = len(attn_teacher)
    student_layer_num = len(attn_student)
    # layers_per_block = 4
    layers_per_block = int(teacher_layer_num / student_layer_num)
    
    
      # len(hidden_student) = 4
    hidden_student = [h for h in extra_student['inner_states']]
    # len(hidden_teacher) = 13
    hidden_teacher = [h for h in extra_teacher['inner_states']]
    # hidden_teacher[0,4,8,12]
    new_teacher_reps = [hidden_teacher[i * layers_per_block] for i in range(student_layer_num + 1)]


    loss_rep = 0.
    for student_rep, teacher_rep in zip(hidden_student, new_teacher_reps):
        student_rep = student_rep.transpose(0, 1)[:, 0]
        teacher_rep = teacher_rep.transpose(0, 1)[:, 0]
        norm_student = F.normalize(student_rep, dim=1)
        norm_teacher = F.normalize(teacher_rep, dim=1)



        loss_tmp = F.mse_loss(norm_student, norm_teacher, reduction='sum')

        loss_rep += loss_tmp

    loss_inter = loss_rep

    return loss_inter


def attn_rand(extra_student, extra_teacher):
    attn_student = [attn for attn in extra_student['attn']]
            # len(attn_teacher) = 12
    attn_teacher = [attn for attn in extra_teacher['attn']]

    sample_size = len(attn_student)

    attn_teacher_rand = [attn_teacher[i] for i in sorted(random.sample(range(len(attn_teacher)), sample_size))]
    
    loss_kl = 0.

    for student_att, teacher_att in zip(attn_student,attn_teacher_rand):
        student_att = F.log_softmax(student_att, dim=-1)
        teacher_att = F.softmax(teacher_att, dim=-1)
        # batch_size, head_num, lenght = student_att.shape[0], student_att.shape[1], student_att.shape[2]
        loss_kl_tmp = F.kl_div(student_att, teacher_att, reduction='sum')/ (student_att.shape[0] * student_att.shape[1]) #, reduction='batchmean', log_target=True)
        loss_kl += loss_kl_tmp
    return loss_kl

def alp_kd(extra_student, extra_teacher):
    hidden_student = [h for h in extra_student['inner_states']]
        # len(hidden_teacher) = 13
    hidden_teacher = [h for h in extra_teacher['inner_states']]

    hidden_state_student = torch.cat([t.transpose(0, 1)[:, 0:1] for t in hidden_student], dim=1)

    hidden_state_teacher = torch.cat([t.transpose(0, 1)[:, 0:1] for t in hidden_teacher], dim=1)

    product = torch.bmm(hidden_state_student, hidden_state_teacher.transpose(1,2))
    product_softmax = F.softmax(product, dim=-1)
    associated = torch.bmm(product_softmax, hidden_state_teacher)
    loss = F.mse_loss(hidden_state_student, associated)

    
    return loss



class Test(nn.Module):
    def  __init__(self):
        super(Test, self).__init__()    
        self.embed_s = nn.Linear(in_features=768, out_features=512)
        self.embed_t = nn.Linear(in_features=768, out_features=512)

    def test(self, extra_student, extra_teacher ):
        attn_student = [attn for attn in extra_student['attn']]
                # len(attn_teacher) = 12
        attn_teacher = [attn for attn in extra_teacher['attn']]

        teacher_layer_num = len(attn_teacher)
        student_layer_num = len(attn_student)
        # layers_per_block = 4
        layers_per_block = int(teacher_layer_num / student_layer_num)
        
        
        # len(hidden_student) = 4
        hidden_student = [h for h in extra_student['inner_states']]
        # len(hidden_teacher) = 13
        hidden_teacher = [h for h in extra_teacher['inner_states']]
        # hidden_teacher[0,4,8,12]
        new_teacher_reps = [hidden_teacher[i * layers_per_block] for i in range(student_layer_num + 1)]


        loss_rep = 0.
        for student_rep, teacher_rep in zip(hidden_student, new_teacher_reps):
            student_rep = student_rep.transpose(0, 1)[:, 0]
            teacher_rep = teacher_rep.transpose(0, 1)[:, 0]
            norm_student = F.normalize(student_rep, dim=1)
            norm_teacher = F.normalize(teacher_rep, dim=1)

            student = self.embed_s(norm_student)
            teacher = self.embed_t(norm_teacher)


            loss_tmp = F.mse_loss(student, teacher, reduction='sum')

            loss_rep += loss_tmp

        loss_inter = loss_rep

        return loss_inter




# class 