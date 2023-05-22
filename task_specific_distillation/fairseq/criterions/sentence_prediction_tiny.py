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
from methods.feature_learn import tiny_feature_learn, mobile_feature_learn, \
    HopfieldFeatureLearn, pkd_feature_learn, Test, attn_rand, alp_kd

@register_criterion('sentence_prediction_tiny')
class SentencePredictionCriterionTiny(FairseqCriterion):

    def __init__(self, task, classification_head_name, regression_target):
        super().__init__(task)
        self.classification_head_name = classification_head_name
        self.regression_target = regression_target

        self.teacher_model = RobertaModel.from_pretrained(model_name_or_path=task.args.teacher_model_checkpoint,
                                                          checkpoint_file=task.args.teacher_model_pt,
                                                          data_name_or_path=task.args.data_name_or_path).model
        ##freeze teacher model anyway
        for param in self.teacher_model.parameters():
            param.requires_grad = False
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
        if self.feature_learn == 'test':
            self.test_learn = Test()
        if self.feature_learn is not None:
            if 'hopfield' in self.feature_learn:
                self.hopfield_feature_learn = HopfieldFeatureLearn()
        

        
        if  self.feature_learn == 'crd':
            self.crd_weight = task.args.crd_weight
            # self.crd_weight > 0.0:
            # self.num_samples = -1
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

        if self.feature_learn == 'tiny':
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
            
            loss_inter, loss_rep, loss_att = tiny_feature_learn(extra, extra_teacher)

            
        if self.feature_learn == 'mobile':
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
            loss_inter= mobile_feature_learn(extra, extra_teacher, self.kl_weight)

        if self.feature_learn == 'hopfield':
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
            loss_inter =  self.hopfield_feature_learn.hopfield_feature_learn_nopooling(extra, extra_teacher, self.task.args.encoder_attention_heads)
            # loss_inter = self.hopfield_feature_learn.hopfield_feature_learn_pooling(extra, extra_teacher, self.task.args.encoder_attention_heads)

        if self.feature_learn == 'hopfield_attn_pool':
            logits, extra = model(
                **sample['net_input'],
                features_only=True,
                classification_head_name=self.classification_head_name,
                # attn_form='attn_weights_before_softmax'
                attn_form='attn_weights_before_softmax_mean_key_pad_mask'
            )
            with torch.no_grad():
                logits_teacher, extra_teacher = self.teacher_model(
                    **sample['net_input'],
                    features_only=True,
                    classification_head_name=self.classification_head_name,
                    # attn_form='attn_weights_before_softmax'
                    attn_form='attn_weights_before_softmax_key_pad_mask'
                )
            loss_inter = self.hopfield_feature_learn.hopfield_feature_learn_pooling(extra, extra_teacher, self.task.args.encoder_attention_heads)


        if self.feature_learn == 'pkd':
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
            loss_inter = pkd_feature_learn(extra, extra_teacher)


        if  self.feature_learn == 'crd':
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



        if self.feature_learn == 'hopfield_retrive_tiny':
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
            loss_inter, loss_rep, loss_att =  self.hopfield_feature_learn.tiny_with_association(extra, extra_teacher)

        if self.feature_learn == 'test':
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
            loss_inter =  self.test_learn.test(extra, extra_teacher)

        if self.feature_learn == 'hopfield_cls_pool':
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
            loss_inter = self.hopfield_feature_learn.hopfield_cls_pooling(extra, extra_teacher)

        if self.feature_learn == 'hopfield_cls_nopool':
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
            loss_inter = self.hopfield_feature_learn.hopfield_cls_nopool(extra, extra_teacher)

        if self.feature_learn == 'hopfield_mean_pool':
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
            loss_inter = self.hopfield_feature_learn.hopfield_mean_pooling(extra, extra_teacher)

        if self.feature_learn == 'attn_rand':
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
            loss_inter = attn_rand(extra, extra_teacher)

        if self.feature_learn == 'alp-kd':
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
            loss_inter = alp_kd(extra, extra_teacher)




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
        targets = model.get_targets(sample, [logits]).view(-1)
        sample_size = targets.numel()
        
        
        if self.pred_distill:

            if not self.regression_target:
                lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
                loss_ce = F.nll_loss(lprobs, targets, reduction='sum')

                loss_kd = self.kd_loss_func(F.log_softmax(logits/self.T, dim=-1),
                                            F.softmax(logits_teacher/self.T, dim=-1)) * self.T ** 2

                _, preds = torch.max(lprobs, dim=1)
            else:
                logits = logits.view(-1).float()
                targets = targets.float()
                loss_ce = F.mse_loss(logits, targets, reduction='sum')
                loss_kd = F.mse_loss(logits, logits_teacher, reduction='sum')
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
