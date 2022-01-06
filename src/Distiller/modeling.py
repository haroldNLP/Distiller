import torch
from transformers.activations import ACT2FN, get_activation
from transformers.modeling_outputs import QuestionAnsweringModelOutput,SequenceClassifierOutput
import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss, MSELoss


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class ElectraClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = get_activation("gelu")(x)  # although BERT uses tanh here, it seems Electra authors used gelu here
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


def cross_entropy(input, target):
    logsoftmax = nn.LogSoftmax(dim=-1)
    return torch.mean(torch.sum(- target * logsoftmax(input), 1))

class SequenceClassificationModel(torch.nn.Module):
    def __init__(self, model, config):
        super().__init__()
        self.num_labels = config.num_labels
        self.model = model
        self.config = config
        self.model_type = config.model_type
        if 'electra' in self.model_type:
            self.classifier = ElectraClassificationHead(config)
        elif 'bert' in self.model_type:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
        else:
            raise NotImplementedError
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        mixup_labels=None,
        mixup_value=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if 'electra' in self.model_type:
            sequence_output = outputs[0]
        elif 'bert' in self.model_type:
            pooled_output = outputs[1]
            sequence_output = self.dropout(pooled_output)
        else:
            raise NotImplementedError
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                if mixup_labels is not None:
                    labels = mixup_value * labels + (1 - mixup_value) * mixup_labels
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                if mixup_labels is not None:
                    labels = nn.functional.one_hot(labels,
                                                            num_classes=self.num_labels)
                    mixup_labels = nn.functional.one_hot(mixup_labels,
                                                                  num_classes=self.num_labels)
                    labels = mixup_value * labels + (1 - mixup_value) * mixup_labels
                    loss = cross_entropy(logits, labels)
                else:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                # loss_fct = CrossEntropyLoss()
                # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class QuestionAnsweringModel(nn.Module):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, model, config):
        super().__init__()
        self.num_labels = config.num_labels
        self.config = config
        self.model = model
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        mixup_start_positions=None,
        mixup_end_positions=None,
        mixup_value=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        # if start_positions is not None and end_positions is not None:
        #     # If we are on multi-GPU, split add a dimension
        #     if len(start_positions.size()) > 1:
        #         start_positions = start_positions.squeeze(-1)
        #     if len(end_positions.size()) > 1:
        #         end_positions = end_positions.squeeze(-1)
        #     # sometimes the start/end positions are outside our model inputs, we ignore these terms
        #     ignored_index = start_logits.size(1)
        #     start_positions.clamp_(0, ignored_index)
        #     end_positions.clamp_(0, ignored_index)
        #
        #     loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
        #     start_loss = loss_fct(start_logits, start_positions)
        #     end_loss = loss_fct(end_logits, end_positions)
        #     total_loss = (start_loss + end_loss) / 2
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            if mixup_start_positions is not None and end_positions is not None:
                # If we are on multi-GPU, split add a dimension
                if len(mixup_start_positions.size()) > 1:
                    mixup_start_positions = mixup_start_positions.squeeze(-1)
                if len(mixup_end_positions.size()) > 1:
                    mixup_end_positions = mixup_end_positions.squeeze(-1)
                mixup_start_positions.clamp_(0, ignored_index)
                mixup_end_positions.clamp_(0, ignored_index)
                start_positions = nn.functional.one_hot(start_positions,
                                                        num_classes=self.config.max_position_embeddings)
                end_positions = nn.functional.one_hot(end_positions, num_classes=self.config.max_position_embeddings)
                mixup_start_positions = nn.functional.one_hot(mixup_start_positions, num_classes=self.config.max_position_embeddings)
                mixup_end_positions = nn.functional.one_hot(mixup_end_positions, num_classes=self.config.max_position_embeddings)
                start_positions = mixup_value*start_positions + (1-mixup_value)*mixup_start_positions
                end_positions = mixup_value * end_positions + (1 - mixup_value) * mixup_end_positions
                loss_fct = cross_entropy
            else:
                loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )