# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import argparse

import torch
import numpy as np
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.models.roberta.model import RobertaModel
from crd.criterion import CRDLoss
from methods.method_comparison import att_mse_learn, att_mse_hidden_mse_learn, \
    att_kl_learn, att_kl_hiden_mse_learn, minilm_feature_learn, minilm_feature_learn_single_layer, minilm_feature_learn_random, \
        key_query_val, Hidden_Mse, Hidden_Mse_Token
from methods.augmentation import data_aug

@register_criterion('sentence_prediction_test')
class SentencePredictionCriterionTest(FairseqCriterion):

    def __init__(self, task, classification_head_name, regression_target):
        super().__init__(task)
        self.classification_head_name = classification_head_name
        self.regression_target = regression_target

        self.teacher_model = RobertaModel.from_pretrained(model_name_or_path=task.args.teacher_model_checkpoint,
                                                          checkpoint_file=task.args.teacher_model_pt,
                                                          data_name_or_path=task.args.data_name_or_path).model
        
        
        self.has_ta = False
        # if task.args.ta1_model_checkpoint is not None:
        #     self.ta1_model = RobertaModel.from_pretrained(model_name_or_path=task.args.ta1_model_checkpoint,
        #                                                   checkpoint_file=task.args.ta1_model_pt,
        #                                                   data_name_or_path=task.args.data_name_or_path).model
        #     self.has_ta = True

        
        ##freeze teacher model anyway
        # for param in self.teacher_model.parameters():
        #     param.requires_grad = False
        if self.teacher_model is None:
            print('teacher model not initialized')

        self.print_teacher_loss = False
        self.use_mse = False

        self.T = task.args.temperature
        self.beta = task.args.kd_weight
        if task.args.use_mse:
            self.kd_loss_func = F.mse_loss
        else:
            self.kd_loss_func = torch.nn.KLDivLoss(reduction='sum')

        self.num_samples = -1
        self.pred_distill = task.args.pred_distill
        self.all_in_one = task.args.all_in_one
        self.feature_learn = task.args.feature_learn
        self.kl_weight = task.args.kl_weight
        self.inter_weight = task.args.inter_weight
        self.aug = task.args.aug
        self.layer_selection = task.args.layer_selection
        self.mapping = task.args.mapping
        self.hidden = Hidden_Mse()
        self.hidden_token = Hidden_Mse_Token()

        
        if  self.feature_learn == 'crd' and 'train' in task.datasets.keys():
            self.crd_weight = task.args.crd_weight
            # self.crd_weight > 0.0:
            self.num_samples = -1
            self.add_train_cls_label(task)

            self.nce_k = self.task.args.nce_k
            opt = argparse.Namespace(
                s_dim=self.task.args.s_dim_feat,
                t_dim=self.task.args.t_dim_feat,
                feat_dim=self.task.args.crd_feat_dim,
                nce_k=self.nce_k,
                nce_m=0.5,
                nce_t=0.07,
                n_data=self.num_samples
            )
            self.crd_criterion = CRDLoss(opt)
            self.replace = self.nce_k >= min([len(n) for n in self.cls_negative])

    def add_train_cls_label(self, task):
        try:
            dataset = task.datasets['train']
            label = np.array([t['target'].item() for t in dataset])
            num_classes = len(np.unique(label))
            self.num_samples = len(label)
            self.cls_positive = [[] for _ in range(num_classes)]
            for i in range(self.num_samples):
                self.cls_positive[label[i]].append(i)

            self.cls_negative = [[] for i in range(num_classes)]
            for i in range(num_classes):
                for j in range(num_classes):
                    if j == i:
                        continue
                    self.cls_negative[i].extend(self.cls_positive[j])

            self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
            self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]

            self.cls_positive = np.asarray(self.cls_positive, dtype=object)
            self.cls_negative = np.asarray(self.cls_negative, dtype=object)
        except KeyError:
            raise ValueError('dataset does not have training')

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--classification-head-name',
                            default='sentence_classification_head',
                            help='name of the classification head to use')
        # fmt: on

    def sample_neg_idx(self, pos_idx, target):
        neg_idx = np.array([np.random.choice(self.cls_negative[t.item()], self.nce_k, replace=self.replace) for t in target])
        sample_idx = torch.cat([torch.unsqueeze(pos_idx, 1), torch.from_numpy(neg_idx).to(target.device)], dim=1)
        return sample_idx

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert (
            hasattr(model, 'classification_heads')
            and self.classification_head_name in model.classification_heads
        ), 'model must provide sentence classification head for --criterion=sentence_prediction'

        sample['net_input']['return_all_hiddens'] = True
        loss_att = None
        loss_rep = None
        loss_inter = None
        loss_val = None
        if self.aug and model.training:
            sample['net_input'] = data_aug(sample['net_input'])
        # 拆成两个
        if self.feature_learn == 'att_mse_learn':
            logits, extra = model(
                **sample['net_input'],
                features_only=True,
                classification_head_name=self.classification_head_name,
                attn_form='attn_weights_before_softmax'
            )
            with torch.no_grad():
                logits_teacher, extra_teacher = self.teacher_model(
                    **sample['net_input'],
                    features_only=True,
                    classification_head_name=self.classification_head_name,
                    attn_form='attn_weights_before_softmax'
                )
            
            loss_inter, loss_rep, loss_att = att_mse_learn(extra, extra_teacher)

        if self.feature_learn == 'hidden_mse_learn':
            logits, extra = model(
                **sample['net_input'],
                features_only=True,
                classification_head_name=self.classification_head_name,
                attn_form='attn_weights_before_softmax'
            )
            with torch.no_grad():
                logits_teacher, extra_teacher = self.teacher_model(
                    **sample['net_input'],
                    features_only=True,
                    classification_head_name=self.classification_head_name,
                    attn_form='attn_weights_before_softmax'
                )
            
            loss_inter, loss_rep, loss_att = self.hidden.hidden_mse_learn(extra, extra_teacher, self.mapping)

        if self.feature_learn == 'hidden_mse_token':
            logits, extra = model(
                **sample['net_input'],
                features_only=True,
                classification_head_name=self.classification_head_name,
                attn_form='attn_weights_before_softmax'
            )
            with torch.no_grad():
                logits_teacher, extra_teacher = self.teacher_model(
                    **sample['net_input'],
                    features_only=True,
                    classification_head_name=self.classification_head_name,
                    attn_form='attn_weights_before_softmax'
                )
            
            loss_inter, loss_rep, loss_att = self.hidden_token.hidden_mse_learn(extra, extra_teacher, self.mapping)


        if self.feature_learn == 'att_mse_hidden_mse_learn':
            logits, extra = model(
                **sample['net_input'],
                features_only=True,
                classification_head_name=self.classification_head_name,
                attn_form='attn_weights_before_softmax'
            )
            with torch.no_grad():
                logits_teacher, extra_teacher = self.teacher_model(
                    **sample['net_input'],
                    features_only=True,
                    classification_head_name=self.classification_head_name,
                    attn_form='attn_weights_before_softmax'
                )
            
            loss_inter, loss_rep, loss_att = att_mse_hidden_mse_learn(extra, extra_teacher)


        





        # 拆成两个
        if self.feature_learn == 'att_kl_learn':
            logits, extra = model(
                **sample['net_input'],
                features_only=True,
                classification_head_name=self.classification_head_name,
                attn_form='attn_weights_before_softmax'
            )
            with torch.no_grad():
                logits_teacher, extra_teacher = self.teacher_model(
                    **sample['net_input'],
                    features_only=True,
                    classification_head_name=self.classification_head_name,
                    attn_form='attn_weights_before_softmax'
                )
            # loss_inter, loss_rep, loss_att, loss_mean, loss_var = mobile_feature_learn(extra, extra_teacher, self.kl_weight)
            loss_inter, loss_rep, loss_att = att_kl_learn(extra, extra_teacher, self.kl_weight) #, self.layer_selection)


        if self.feature_learn == 'att_kl_hiden_mse_learn':
            logits, extra = model(
                **sample['net_input'],
                features_only=True,
                classification_head_name=self.classification_head_name,
                attn_form='attn_weights_before_softmax'
            )
            with torch.no_grad():
                logits_teacher, extra_teacher = self.teacher_model(
                    **sample['net_input'],
                    features_only=True,
                    classification_head_name=self.classification_head_name,
                    attn_form='attn_weights_before_softmax'
                )
            # loss_inter, loss_rep, loss_att, loss_mean, loss_var = mobile_feature_learn(extra, extra_teacher, self.kl_weight)
            loss_inter, loss_rep, loss_att = att_kl_hiden_mse_learn(extra, extra_teacher, self.kl_weight)


        if self.feature_learn == 'crd':
            logits, extra = model(
                **sample['net_input'],
                features_only=True,
                classification_head_name=self.classification_head_name,
                attn_form='attn_weights_before_softmax'
                # attn_form='attn_weights_before_softmax_mean_key_pad_mask'
            )
            with torch.no_grad():
                logits_teacher, extra_teacher = self.teacher_model(
                    **sample['net_input'],
                    features_only=True,
                    classification_head_name=self.classification_head_name,
                    attn_form='attn_weights_before_softmax'
                    # attn_form='attn_weights_before_softmax_key_pad_mask'
                )
            attn_student = [attn for attn in extra['attn']]
            # len(attn_teacher) = 12
            # attn.shape = [384, l, l] 384=32x12
            attn_teacher = [attn for attn in extra_teacher['attn']]

            teacher_layer_num = len(attn_teacher)
            student_layer_num = len(attn_student)
            # layers_per_block = 4
            layers_per_block = int(teacher_layer_num / student_layer_num)
            hidden_student = [h for h in extra['inner_states']]
            hidden_teacher = [h for h in extra_teacher['inner_states']]

            # new_teacher_reps = [hidden_teacher[i * layers_per_block] for i in range(student_layer_num + 1)]

            hidden_stat = torch.cat([t.transpose(0, 1)[:, 0] for t in hidden_student], dim=1)
            # # 暂时把student也隔两层一抽
            hidden_state_teacher = torch.cat([t.transpose(0, 1)[:, 0] for t in hidden_teacher], dim=1)
           
            targets = model.get_targets(sample, [logits]).view(-1)

            idx = sample['id']
            neg_idx = self.sample_neg_idx(idx, targets)
            loss_inter = self.crd_criterion(hidden_stat, hidden_state_teacher, idx, neg_idx)


        # keep
        if self.feature_learn == 'minilm':
            logits, extra = model(
                **sample['net_input'],
                features_only=True,
                classification_head_name=self.classification_head_name,
                attn_form='weights_value_before_softmax'
            )
            with torch.no_grad():
                logits_teacher, extra_teacher = self.teacher_model(
                    **sample['net_input'],
                    features_only=True,
                    classification_head_name=self.classification_head_name,
                    attn_form='weights_value_before_softmax'
                )
            loss_inter, loss_rep, loss_att= minilm_feature_learn(extra, extra_teacher, self.kl_weight, self.layer_selection)
            # loss_inter, loss_att, loss_val= minilm_feature_learn_single_layer(extra, extra_teacher, self.kl_weight)

        if self.feature_learn == 'dense-ta-minilm':
            logits, extra = model(
                **sample['net_input'],
                features_only=True,
                classification_head_name=self.classification_head_name,
                attn_form='weights_value_before_softmax'
            )
            with torch.no_grad():
                logits_teacher, extra_teacher = self.teacher_model(
                    **sample['net_input'],
                    features_only=True,
                    classification_head_name=self.classification_head_name,
                    attn_form='weights_value_before_softmax'
                )
                
                logits_ta1, extra_ta1 = self.ta1_model(
                    **sample['net_input'],
                    features_only=True,
                    classification_head_name=self.classification_head_name,
                    attn_form='weights_value_before_softmax'
                )
            # loss_inter, loss_rep, loss_att, loss_mean, loss_var = mobile_feature_learn(extra, extra_teacher, self.kl_weight)
            loss_inter, loss_att, loss_val= minilm_feature_learn(extra, extra_teacher, self.kl_weight)
            loss_inter_ta, loss_att_ta, loss_val_ta= minilm_feature_learn(extra, extra_ta1, self.kl_weight)
            loss_inter += loss_inter_ta
            loss_att += loss_att_ta
            loss_val += loss_val_ta

        if self.feature_learn == 'minilm_random':
            logits, extra = model(
                **sample['net_input'],
                features_only=True,
                classification_head_name=self.classification_head_name,
                attn_form='weights_value_before_softmax'
            )
            with torch.no_grad():
                logits_teacher, extra_teacher = self.teacher_model(
                    **sample['net_input'],
                    features_only=True,
                    classification_head_name=self.classification_head_name,
                    attn_form='weights_value_before_softmax'
                )
            # loss_inter, loss_rep, loss_att, loss_mean, loss_var = mobile_feature_learn(extra, extra_teacher, self.kl_weight)
            loss_inter, loss_att, loss_val= minilm_feature_learn_random(extra, extra_teacher, self.kl_weight)

        if self.feature_learn == 'qkv':
            logits, extra = model(
                **sample['net_input'],
                features_only=True,
                classification_head_name=self.classification_head_name,
                attn_form='q_k_v'
            )
            with torch.no_grad():
                logits_teacher, extra_teacher = self.teacher_model(
                    **sample['net_input'],
                    features_only=True,
                    classification_head_name=self.classification_head_name,
                    attn_form='q_k_v'
                )
            # loss_inter, loss_rep, loss_att, loss_mean, loss_var = mobile_feature_learn(extra, extra_teacher, self.kl_weight)
            loss_inter = key_query_val(extra, extra_teacher)


        if self.pred_distill and not self.feature_learn:
            logits, extra = model(
                **sample['net_input'],
                features_only=True,
                classification_head_name=self.classification_head_name,
                attn_form='attn_weights_before_softmax'
            )

            with torch.no_grad():
                logits_teacher, extra_teacher = self.teacher_model(
                    **sample['net_input'],
                    features_only=True,
                    classification_head_name=self.classification_head_name,
                    attn_form='attn_weights_before_softmax'
                )
            
            if self.has_ta == True:
                with torch.no_grad():
                    logits_ta1, extra_ta1 = self.ta1_model(
                    **sample['net_input'],
                    features_only=True,
                    classification_head_name=self.classification_head_name,
                    attn_form='attn_weights_before_softmax'
                )

        targets = model.get_targets(sample, [logits]).view(-1)
        sample_size = targets.numel()
        
        
        if self.pred_distill:

            if not self.regression_target:
                lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
                loss_ce = F.nll_loss(lprobs, targets, reduction='sum')

                loss_kd_teacher = self.kd_loss_func(F.log_softmax(logits/self.T, dim=-1),
                                            F.softmax(logits_teacher/self.T, dim=-1)) * self.T ** 2
                loss_kd = loss_kd_teacher
                if self.has_ta:
                    loss_kd_ta1 = self.kd_loss_func(F.log_softmax(logits/self.T, dim=-1),
                                                F.softmax(logits_ta1/self.T, dim=-1)) * self.T ** 2
                    loss_kd = loss_kd_teacher + loss_kd_ta1

                _, preds = torch.max(lprobs, dim=1)
            else:
                logits = logits.view(-1).float()
                targets = targets.float()
                loss_ce = F.mse_loss(logits, targets, reduction='sum')
                loss_kd_teacher = F.mse_loss(logits, logits_teacher, reduction='sum')
                loss_kd = loss_kd_teacher
                if self.has_ta:
                    loss_kd_ta1 = F.mse_loss(logits, logits_ta1, reduction='sum')
                    loss_kd = loss_kd_teacher + loss_kd_ta1
                preds = logits


        if self.feature_learn and self.pred_distill:
            loss = (1-self.beta) * loss_ce + self.beta * loss_kd + self.inter_weight * loss_inter
        if self.pred_distill and not self.feature_learn:
            loss = (1-self.beta) * loss_ce + self.beta * loss_kd
        if  not self.pred_distill and self.feature_learn:
            loss  =  loss_inter

        logging_output = {
            # 'loss': loss.data,
            'loss': loss.data,
            # 'ce_loss': loss_ce.data,
            # 'kd_loss': loss_kd.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample_size,
            'sample_size': sample_size,
            # 'preds': preds,
            # 'labels': targets
        }

        # if self.feature_learn == 'tiny':
        #     logging_output['loss_att'] = loss_att.data
        #     logging_output['loss_rep'] = loss_rep.data

        # if self.feature_learn == 'mobile':
        #     logging_output['loss_att'] = loss_att.data
        #     logging_output['loss_rep'] = loss_rep.data
        #     # logging_output['loss_mean'] = loss_mean.data
        #     # logging_output['loss_var'] = loss_var.data
        if loss_inter is not None:
            logging_output['loss_inter'] = loss_inter.data
        if loss_att is not None:
            logging_output['loss_att'] = loss_att.data
        if loss_rep is not None:
            logging_output['loss_rep'] = loss_rep.data
        if loss_val is not None:
            logging_output['loss_val'] = loss_val.data
        if self.pred_distill:
            logging_output['loss_ce'] = loss_ce.data
            logging_output['loss_kd'] = loss_kd.data
            logging_output['preds'] = preds
            logging_output['labels'] = targets

        if not self.regression_target:
            preds = logits.argmax(dim=1)
            logging_output['ncorrect'] = (preds == targets).sum()

        if self.print_teacher_loss:
            with torch.no_grad():
                lprobs = F.log_softmax(logits_teacher, dim=-1, dtype=torch.float32)
                loss_ce_teacher = F.nll_loss(lprobs, targets, reduction='sum')
            logging_output['ce_loss_teacher'] = loss_ce_teacher
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)

        if 'loss_ce' in logging_outputs[0]:
            loss_ce_sum = sum(log.get('loss_ce', 0) for log in logging_outputs)
            loss_kd_sum = sum(log.get('loss_kd', 0) for log in logging_outputs)
            metrics.log_scalar('loss_ce', loss_ce_sum / sample_size / math.log(2), sample_size, round=3)
            metrics.log_scalar('loss_kd', loss_kd_sum / sample_size / math.log(2), sample_size, round=3)
        if 'loss_att' in logging_outputs[0]:
            loss_attn_sum = sum(log.get('loss_att', 0) for log in logging_outputs)
            metrics.log_scalar('loss_att', loss_attn_sum / sample_size / math.log(2), sample_size, round=3)

        if 'loss_rep' in logging_outputs[0]:
            loss_reps_sum = sum(log.get('loss_rep', 0) for log in logging_outputs)
            metrics.log_scalar('loss_rep', loss_reps_sum / sample_size / math.log(2), sample_size, round=3)

        if 'loss_mean' in logging_outputs[0]:
            loss_mean_sum = sum(log.get('loss_mean', 0) for log in logging_outputs)
            metrics.log_scalar('loss_mean', loss_mean_sum / sample_size / math.log(2), sample_size, round=3)
            loss_var_sum = sum(log.get('loss_var', 0) for log in logging_outputs)
            metrics.log_scalar('loss_var', loss_var_sum / sample_size / math.log(2), sample_size, round=3)

        if 'loss_inter' in logging_outputs[0]:
            loss_inter_sum = sum(log.get('loss_inter', 0) for log in logging_outputs)
            metrics.log_scalar('loss_inter', loss_inter_sum / sample_size / math.log(2), sample_size, round=3)

        if 'loss_val' in logging_outputs[0]:
            loss_val_sum = sum(log.get('loss_val', 0) for log in logging_outputs)
            metrics.log_scalar('loss_val', loss_val_sum / sample_size / math.log(2), sample_size, round=3)
            
        if 'ce_loss_teacher' in logging_outputs[0]:
            loss_ce_teacher = sum(log.get('ce_loss_teacher', 0) for log in logging_outputs)
            metrics.log_scalar('loss_ce_teacher', loss_ce_teacher / sample_size / math.log(2), sample_size, round=3)

        if sample_size != ntokens:
            metrics.log_scalar('nll_loss', loss_sum / ntokens / math.log(2), ntokens, round=3)

        if len(logging_outputs) > 0 and 'ncorrect' in logging_outputs[0]:
            ncorrect = sum(log.get('ncorrect', 0) for log in logging_outputs)
            metrics.log_scalar('accuracy', 100.0 * ncorrect / nsentences, nsentences, round=1)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
