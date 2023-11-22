import numpy as np

import torch
import random
from tqdm import tqdm
from rectorch.metrics import Metrics


def ranking_meansure_testset(pred_scores, ground_truth, k, test_item):
    # test_item (list)
    # user_num * item_num (score)
    # user_num * item_num (1/0)

    # user_num * test_item_num (score)
    # user_num * test_item_num (1/0)
    pred_scores = pred_scores[:, test_item]
    ground_truth = ground_truth[:, test_item]

    # user_num
    ndcg_list = Metrics.ndcg_at_k(pred_scores, ground_truth, k).tolist()
    recall_list = Metrics.recall_at_k(pred_scores, ground_truth, k).tolist()
    mrr_list = Metrics.mrr_at_k(pred_scores, ground_truth, k).tolist()

    return np.mean(ndcg_list), np.mean(recall_list), np.mean(mrr_list)



def ranking_meansure_degree_testset(pred_scores, ground_truth, k, itemDegrees, seperate_rate, test_item):
    sorted_item_degrees = sorted(itemDegrees.items(), key=lambda x: x[1])
    item_list_sorted, _ = zip(*sorted_item_degrees)
    body_length = int(len(item_list_sorted) * (1-seperate_rate))
    tail_length = int(len(item_list_sorted) * seperate_rate)
    head_length = int(len(item_list_sorted) * seperate_rate)

    head_item = list(set(item_list_sorted[-head_length:]).intersection(set(test_item)))
    tail_item = list(set(item_list_sorted[:tail_length]).intersection(set(test_item)))
    body_item = list(set(item_list_sorted[:body_length]).intersection(set(test_item)))


    head_ndcg_list = np.nan_to_num(Metrics.ndcg_at_k(pred_scores[:, head_item], ground_truth[:, head_item], k)).tolist()
    head_recall_list = np.nan_to_num(Metrics.recall_at_k(pred_scores[:, head_item], ground_truth[:, head_item], k)).tolist()

    tail_ndcg_list = np.nan_to_num(Metrics.ndcg_at_k(pred_scores[:, tail_item], ground_truth[:, tail_item], k)).tolist()
    tail_recall_list = np.nan_to_num(Metrics.recall_at_k(pred_scores[:, tail_item], ground_truth[:, tail_item], k)).tolist()

    body_ndcg_list = np.nan_to_num(Metrics.ndcg_at_k(pred_scores[:, body_item], ground_truth[:, body_item], k)).tolist()
    body_recall_list = np.nan_to_num(Metrics.recall_at_k(pred_scores[:, body_item], ground_truth[:, body_item], k)).tolist()


    return np.mean(head_ndcg_list), np.mean(head_recall_list), np.mean(tail_ndcg_list), np.mean(tail_recall_list), np.mean(body_ndcg_list), np.mean(body_recall_list)
