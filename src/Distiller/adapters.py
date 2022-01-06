def BertForQAAdaptor(batch, model_outputs, no_mask=False, no_logits=False):
    dict_obj = {'hidden':  model_outputs.hidden_states, 'attention': model_outputs.attentions,"loss":model_outputs.loss}
    if no_mask is False:
        dict_obj['inputs_mask'] = batch['attention_mask']
    if no_logits is False:
        dict_obj['logits'] = (model_outputs.start_logits,model_outputs.end_logits)

    return dict_obj


def BertForGLUEAdptor(batch, model_outputs, no_mask=False, no_logits=False):
    dict_obj = {'hidden': model_outputs.hidden_states, 'attention': model_outputs.attentions,
                "loss": model_outputs.loss}
    if no_mask is False:
        dict_obj['inputs_mask'] = batch['attention_mask']
    if no_logits is False:
        dict_obj['logits'] = (model_outputs.logits)
    return dict_obj