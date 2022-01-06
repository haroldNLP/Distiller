import logging
import collections
import math
import numpy as np
import six
from logging import handlers
from scipy.special import logsumexp
import os
import torch
import boto3
import re
import json
import string


class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }#日志级别关系映射

    def __init__(self,filename,level='info',when='D',backCount=3,fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)#设置日志格式
        self.logger.setLevel(self.level_relations.get(level))#设置日志级别
        sh = logging.StreamHandler()#往屏幕上输出
        sh.setFormatter(format_str) #设置屏幕上显示的格式
        th = handlers.TimedRotatingFileHandler(filename=filename,when=when,backupCount=backCount,encoding='utf-8')#往文件里写入#指定间隔时间自动生成文件的处理器
        #实例化TimedRotatingFileHandler
        #interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位，单位有以下几种：
        # S 秒
        # M 分
        # H 小时、
        # D 天、
        # W 每星期（interval==0时代表星期一）
        # midnight 每天凌晨
        th.setFormatter(format_str)#设置文件里写入的格式
        self.logger.addHandler(sh) #把对象加到logger里
        self.logger.addHandler(th)

logger = Logger("all.log",level="debug").logger


def cal_layer_mapping(args, t_config, s_config):
    """
    This function is used to calculate layer mapping of different mapping strategy
    Strategy can be EMD, last or skip
    """
    matches = []
    t_num_layers = t_config.num_hidden_layers
    s_num_layers = s_config.num_hidden_layers
    k = t_num_layers/s_num_layers
    if args.intermediate_strategy and args.intermediate_strategy.lower() == "emd":
        if args.intermediate_loss_type in ["cos", "pkd","mi"]:
            loss_type = args.intermediate_loss_type
        elif args.intermediate_loss_type in ["ce", "mse"]:
            loss_type = "hidden_" + args.intermediate_loss_type
        else:
            raise NotImplementedError
        matches = {'layer_num_S':s_config.num_hidden_layers+1, 'layer_num_T':t_config.num_hidden_layers+1,  #number of hidden_states + embedding_layer
                                          'feature':'hidden','loss':loss_type,
                                          'weight': args.inter_loss_weight,'proj':['linear',s_config.hidden_size,t_config.hidden_size] if s_config.hidden_size<t_config.hidden_size and args.intermediate_loss_type != "mi" else None}
    else:
        for feature in args.intermediate_features:
            if args.intermediate_loss_type in ["cos", "pkd","mi","nce","nwj","tuba"]:
                loss_type = args.intermediate_loss_type
            elif args.intermediate_loss_type in ["ce", "mse"]:
                loss_type = feature+"_"+args.intermediate_loss_type
            else:
                raise NotImplementedError
            if args.intermediate_strategy == "skip":
                if feature == "hidden":
                    for i in range(s_num_layers+1):
                        matches.append({'layer_T': int(i*k),'layer_S':i, 'feature':feature, 'loss':loss_type, 'weight':args.inter_loss_weight,'proj':['linear',s_config.hidden_size,t_config.hidden_size] if s_config.hidden_size<t_config.hidden_size and args.intermediate_loss_type != "mi" else None})
                elif feature == "attention":
                    for i in range(s_num_layers):
                        matches.append({'layer_T': int((i+1)*k-1), 'layer_S': i, 'feature':feature, 'loss':loss_type, 'weight':args.inter_loss_weight})
                else:
                    continue
            elif args.intermediate_strategy == "last":
                if feature == "hidden":
                    for i in range(s_num_layers+1):
                        matches.append({'layer_T': int(t_num_layers-s_num_layers+i), 'layer_S': i, 'feature':feature, 'loss':loss_type, 'weight':args.inter_loss_weight,'proj':['linear',s_config.hidden_size,t_config.hidden_size] if s_config.hidden_size<t_config.hidden_size and args.intermediate_loss_type != "mi" else None})
                elif feature == "attention":
                    for i in range(s_num_layers):
                        matches.append({"layer_T": int(t_num_layers-s_num_layers+i),"layer_S":i, 'feature':feature, 'loss':loss_type, 'weight':args.inter_loss_weight})
                else:
                    continue
            else:
                pass
    return matches



def write_predictions_squad(tokenizer, all_examples, all_features, all_results, n_best_size,
                             max_answer_length, do_lower_case, output_prediction_file,
                             output_nbest_file, output_null_log_odds_file, version_2_with_negative,
                             null_score_diff_threshold, write_prediction):
    """Write final predictions to the json file and log-odds of null if needed."""

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction", ["feature_index", "start_index", "end_index", "start_logit", "end_logit"]
    )

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if version_2_with_negative:
                feature_null_score = result.start_logits[0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index],
                        )
                    )
        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit,
                )
            )
        prelim_predictions = sorted(prelim_predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"]
        )

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index: (pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start: (orig_doc_end + 1)]

                tok_text = tokenizer.convert_tokens_to_string(tok_tokens)

                # tok_text = " ".join(tok_tokens)
                #
                # # De-tokenize WordPieces that have been split off.
                # tok_text = tok_text.replace(" ##", "")
                # tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, tokenizer)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(_NbestPrediction(text=final_text, start_logit=pred.start_logit, end_logit=pred.end_logit))
        # if we didn't include the empty option in the n-best, include it
        if version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(_NbestPrediction(text="", start_logit=null_start_logit, end_logit=null_end_logit))

            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest) == 1:
                nbest.insert(0, _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(_NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1, "No valid predictions"

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1, "No valid predictions"

        if not version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]["text"]
        else:
            # predict "" iff the null score - the score of best non-null > threshold
            score_diff = score_null - best_non_null_entry.start_logit - (best_non_null_entry.end_logit)
            scores_diff_json[example.qas_id] = score_diff
            if score_diff > null_score_diff_threshold:
                all_predictions[example.qas_id] = ""
            else:
                all_predictions[example.qas_id] = best_non_null_entry.text
        all_nbest_json[example.qas_id] = nbest_json
    if write_prediction:
        if output_prediction_file:
            logger.info(f"Writing predictions to: {output_prediction_file}")
            with open(output_prediction_file, "w") as writer:
                writer.write(json.dumps(all_predictions, indent=4) + "\n")

        if output_nbest_file:
            logger.info(f"Writing nbest to: {output_nbest_file}")
            with open(output_nbest_file, "w") as writer:
                writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

        if output_null_log_odds_file and version_2_with_negative:
            logger.info(f"Writing null_log_odds to: {output_null_log_odds_file}")
            with open(output_null_log_odds_file, "w") as writer:
                writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def log_softmax1d(scores):
    if not scores:
        return []
    x = np.array(scores)
    z = logsumexp(x)
    return x - z


def log_sigmoid(score):
    return math.log(1 / (1 + math.exp(-score)))


def _get_best_indexes(logits, n_best_size, offset=0):
    """Get the n-best logits from a list."""
    sorted_indices = np.argsort(logits)[::-1] + offset
    return list(sorted_indices[:n_best_size])


def get_final_text(pred_text, orig_text, tokenizer, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.

    tok_text = " ".join(tokenizer.tokenize(orig_text))
    tok_text = tok_text.replace(" ##", "")
    tok_text = tok_text.replace("##", "")
    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                        orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_raw_scores(examples, preds):
    """
    Computes the exact and f1 scores from the examples and the model predictions
    """
    exact_scores = {}
    f1_scores = {}

    for example in examples:
        qas_id = example.qas_id
        gold_answers = [answer["text"] for answer in example.answers if normalize_answer(answer["text"])]

        if not gold_answers:
            # For unanswerable questions, only correct answer is empty string
            gold_answers = [""]

        if qas_id not in preds:
            print("Missing prediction for %s" % qas_id)
            continue

        prediction = preds[qas_id]
        exact_scores[qas_id] = max(compute_exact(a, prediction) for a in gold_answers)
        f1_scores[qas_id] = max(compute_f1(a, prediction) for a in gold_answers)

    return exact_scores, f1_scores


def find_all_best_thresh_v2(main_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans):
    best_exact, exact_thresh, has_ans_exact = find_best_thresh_v2(preds, exact_raw, na_probs, qid_to_has_ans)
    best_f1, f1_thresh, has_ans_f1 = find_best_thresh_v2(preds, f1_raw, na_probs, qid_to_has_ans)
    main_eval["best_exact"] = best_exact
    main_eval["best_exact_thresh"] = exact_thresh
    main_eval["best_f1"] = best_f1
    main_eval["best_f1_thresh"] = f1_thresh
    main_eval["has_ans_exact"] = has_ans_exact
    main_eval["has_ans_f1"] = has_ans_f1


def find_best_thresh_v2(preds, scores, na_probs, qid_to_has_ans):
    num_no_ans = sum(1 for k in qid_to_has_ans if not qid_to_has_ans[k])
    cur_score = num_no_ans
    best_score = cur_score
    best_thresh = 0.0
    qid_list = sorted(na_probs, key=lambda k: na_probs[k])
    for i, qid in enumerate(qid_list):
        if qid not in scores:
            continue
        if qid_to_has_ans[qid]:
            diff = scores[qid]
        else:
            if preds[qid]:
                diff = -1
            else:
                diff = 0
        cur_score += diff
        if cur_score > best_score:
            best_score = cur_score
            best_thresh = na_probs[qid]

    has_ans_score, has_ans_cnt = 0, 0
    for qid in qid_list:
        if not qid_to_has_ans[qid]:
            continue
        has_ans_cnt += 1

        if qid not in scores:
            continue
        has_ans_score += scores[qid]

    return 100.0 * best_score / len(scores), best_thresh, 1.0 * has_ans_score / has_ans_cnt


def make_qid_to_has_ans(dataset):
    qid_to_has_ans = {}
    for article in dataset:
        for p in article["paragraphs"]:
            for qa in p["qas"]:
                qid_to_has_ans[qa["id"]] = bool(qa["answers"])
    return qid_to_has_ans


def apply_no_ans_threshold(scores, na_probs, qid_to_has_ans, na_prob_thresh):
    new_scores = {}
    for qid, s in scores.items():
        pred_na = na_probs[qid] > na_prob_thresh
        if pred_na:
            new_scores[qid] = float(not qid_to_has_ans[qid])
        else:
            new_scores[qid] = s
    return new_scores


def make_eval_dict(exact_scores, f1_scores, qid_list=None):
    if not qid_list:
        total = len(exact_scores)
        return collections.OrderedDict(
            [
                ("exact", 100.0 * sum(exact_scores.values()) / total),
                ("f1", 100.0 * sum(f1_scores.values()) / total),
                ("total", total),
            ]
        )
    else:
        total = len(qid_list)
        return collections.OrderedDict(
            [
                ("exact", 100.0 * sum(exact_scores[k] for k in qid_list) / total),
                ("f1", 100.0 * sum(f1_scores[k] for k in qid_list) / total),
                ("total", total),
            ]
        )


def merge_eval(main_eval, new_eval, prefix):
    for k in new_eval:
        main_eval["%s_%s" % (prefix, k)] = new_eval[k]




def cal_params(vocab_size=30522, token_type=2, max_seqlen=512, num_layer=12, hidden_size=768, intermediate_size=3072):
    embedding_params = (vocab_size + max_seqlen + token_type + 2) * hidden_size
    layer_params = 4 * hidden_size * (hidden_size + 1) + 2 * hidden_size + (
                intermediate_size + 1) * hidden_size + intermediate_size * (hidden_size + 1) + 2 * hidden_size
    pooler_params = (hidden_size + 1) * hidden_size
    return embedding_params + num_layer * layer_params + pooler_params



def uploadDirectory(path, bucketname="haoyu-nlp"):
    session = boto3.Session(aws_access_key_id="AKIA237V3YQYPZMNHRNV",
                            aws_secret_access_key="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")
    s3 = session.resource('s3').Bucket(bucketname)
    for root, dirs, files in os.walk(path):
        for file in files:
            s3.upload_file(os.path.join(root, file), "experiments/"+os.path.join(root, file))

def mlp(in_dim, hidden_size, out_dim):
    return torch.nn.Sequential(
        torch.nn.Linear(in_dim, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, out_dim),
    )


def mean_tensor(t,length,out_dim):
    return t.unsqueeze(1).view(-1,length,out_dim).mean(dim=1)


class mlp_critic(torch.nn.Module):
    def __init__(self, t_dim, s_dim=None, length=None, hidden_size=64, out_dim=32):
        super(mlp_critic, self).__init__()
        if length:
            self.length = length
        else:
            self.length = None
        self.out_dim = out_dim
        self._t = mlp(t_dim, hidden_size, out_dim)
        if s_dim:
            self._s = mlp(s_dim, hidden_size, out_dim)


    def forward(self, x=None, y=None, mask_S=None, mask_T=None):
        if x==None:
            if self.length:
                return mean_tensor(self._t(y),self.length,self.out_dim)
            else:
                return self._t(y)
        else:
            if self.length:
                return torch.matmul(mean_tensor(self._s(x),self.length,self.out_dim), mean_tensor(self._t(y),self.length,self.out_dim).T)
            else:
                return torch.matmul(self._s(x), self._t(y).T)


class PositionalEncoding(torch.nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class transformer_encoder(torch.nn.Module):
    
    def __init__(self,d_model=512, length=128, nhead=8, hidden_size=2048, num_layers=3, dropout=0.0, out_dim=32):
        super(transformer_encoder, self).__init__()
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=hidden_size, dropout=dropout)
        encoder_norm = torch.nn.LayerNorm(d_model)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=encoder_norm)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.decoder = torch.nn.Linear(d_model, out_dim)
        self.d_model = d_model
    def forward(self, x, mask=None):
        padding_mask = mask.masked_fill(mask == 0, True).masked_fill(mask == 1, False).to(torch.bool) if mask is not None else None
        pos_opt = self.pos_encoder(x)
        encoder_opt = self.encoder(pos_opt.transpose(0,1), src_key_padding_mask=padding_mask) * math.sqrt(self.d_model)
        opt = self.decoder(encoder_opt)
        return opt


class transformer_critic(torch.nn.Module):
    def __init__(self, t_dim, s_dim=None, hidden_size=512, out_dim=32, nhead=8,
                 num_layers=2, dropout=0.1, length=128):
        super(transformer_critic, self).__init__()
        self._t = transformer_encoder(t_dim, length, nhead, hidden_size, num_layers, dropout=dropout, out_dim=out_dim)
        if s_dim:
            self._s = transformer_encoder(s_dim, length, nhead, hidden_size, num_layers, dropout=dropout, out_dim=out_dim)

    def forward(self, x=None, y=None, mask_S=None, mask_T=None):
        if x==None:
            return self._t(y, mask_T).mean(1)
        else:
            s_opt = torch.nn.functional.normalize(self._s(x, mask_S))
            t_opt = torch.nn.functional.normalize(self._t(y, mask_T))
            return torch.matmul(s_opt.view(s_opt.shape[0], -1), t_opt.view(t_opt.shape[0], -1).T)



class lstm_critic(torch.nn.Module):
    def __init__(self, t_dim, s_dim=None, hidden_size=512, out_dim=32, num_layers=1, bidirectional=True, dropout=0.1):
        super(lstm_critic, self).__init__()
        self._t = torch.nn.LSTM(t_dim, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout, proj_size=out_dim)
        if s_dim:
            self._s = torch.nn.LSTM(s_dim, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout, proj_size=out_dim)

    def forward(self, x=None, y=None, mask_S=None, mask_T=None):

        if x==None:
            output, (h_n, c_n) = self._t(y)
            return torch.sum(h_n, 0)
        else:
            _, (h_n_t, c_n_t) = self._t(y)
            _, (h_n_s, c_n_s) = self._s(x)
            return torch.matmul(torch.sum(h_n_s,0), torch.sum(h_n_t, 0).T)

class Critic(torch.nn.Module):
    def __init__(self, type, t_dim, s_dim=None, hidden_size=512, out_dim=32, length=None, num_layers=1,
                 bidirectional=True, dropout=0.1, nhead=8):
        super(Critic, self).__init__()
        self.type = type
        if type == 'mlp':
            self.critic = mlp_critic(t_dim, s_dim, None, hidden_size, out_dim)
        elif type == 'lstm':
            self.critic = lstm_critic(t_dim, s_dim, hidden_size, out_dim, num_layers, bidirectional, dropout)
        elif type == 'transformer':
            self.critic = transformer_critic(t_dim, s_dim, hidden_size, out_dim, nhead, num_layers, dropout=dropout, length=length)
        else:
            raise NotImplementedError

    def forward(self, x=None, y=None, mask_S=None, mask_T=None):
        return self.critic(x, y, mask_S, mask_T)

def glue_criterion(task_name):
    return {'cola':['mcc'],
            'sst-2':['acc'],
            'mrpc':['f1','acc_and_f1','acc'],
            'stsb':['corr','spearmanr','pearson'],
            'qqp':['f1','acc_and_f1','acc'],
            'mnli':['m_mm_acc','mnli-mm/acc','mnli/acc'],
            'qnli':['acc'],
            'rte':['acc'],
            'wnli':['acc'],
            'kaggle':['acc'],
            'fake': ["roc_auc"],
            'cloth': ['r2'],
            'boolq': ['acc'],
            'squad':['exact','f1'],
            'boolq': ['acc'],
            'squad2':['exact','f1']}[task_name]
