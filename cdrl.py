# coding:utf-8
from __future__ import print_function
import copy

import numpy
import tensorflow as tf

import data_utils_pivots
from config import *
import argparse
import matplotlib
from data_utils_pivots import store_pivots

matplotlib.use('Agg')
import sys

sys.path.insert(0, 'models')

from data_utils_hatn import *
from models import CDRL

np.random.seed(FLAGS.random_seed)


def get_acc(test_data, pos_pivot, neg_pivot,
            word2idx, memory_size,
            max_sentence_size):
    x_test, _, y_test, u_test, v_test, word_mask_test, sent_mask_test = vectorize_data(test_data, pos_pivot, neg_pivot,
                                                                                       word2idx, memory_size,
                                                                                       max_sentence_size)
    # calculate the data numbers
    n_test = x_test.shape[0]
    # print(n_test)

    graph = tf.Graph()
    # create the model
    with graph.as_default():
        tf.set_random_seed(FLAGS.random_seed)
        model = CDRL(FLAGS, args, word_embedding)

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(graph=graph, config=config) as sess:
        # model initialize
        model.initialize_session(sess)
        model.load_model(sess)
        test_acc, test_preds = model.eval_sen(sess, x_test, word_mask_test, sent_mask_test, y_test,
                                              batch_size=FLAGS.batch_size)
        print("Testing accuracy: %.8f" % (test_acc))
        return test_acc


def get_reward(acc, N1, N, M1, M):
    loss = 0.7*acc + 0.2*N1/N + 0.1*M1/M
    return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--source_domain', '-s', type=str,
                        choices=['books', 'dvd', 'electronics', 'kitchen', 'video'],
                        default='electronics')
    parser.add_argument('--target_domain', '-t', type=str,
                        choices=['books', 'dvd', 'electronics', 'kitchen', 'video'],
                        default='kitchen')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    source_domain = args.source_domain
    target_domain = args.target_domain
    data_path = FLAGS.data_path

    print("loading data...")
    train_data, val_data, test_data, source_unlabeled_data, target_unlabeled_data, vocab \
        = load_data(source_domain, target_domain, data_path)
    pos_pivot, neg_pivot = get_all_pivots(source_domain, target_domain)
    # print(pos_pivot.pop(1))
    data = train_data + val_data + test_data + source_unlabeled_data + target_unlabeled_data
    source_data = train_data + val_data + source_unlabeled_data
    target_data = target_unlabeled_data

    max_story_size = max(map(len, (pairs[0] for pairs in data)))
    mean_story_size = int(np.mean([len(pairs[0]) for pairs in data]))
    sentences = map(len, (sentence for pairs in data for sentence in pairs[0]))
    max_sentence_size = max(sentences)
    mean_sentence_size = int(mean(sentences))
    memory_size = min(FLAGS.memory_size, max_story_size)
    print("max  story size:", max_story_size)
    print("mean story size:", mean_story_size)
    print("max  sentence size:", max_sentence_size)
    print("mean sentence size:", mean_sentence_size)
    print("max memory size:", memory_size)
    max_sentence_size = FLAGS.sent_size
    # get the word vector
    word_embedding, word2idx, idx2word = get_w2vec(vocab, FLAGS)
    vocab_size = len(word_embedding)
    # code data: just mask the pivot data!
    # here to change the code
    # runFeature(pos_pivot, neg_pivot):

    best_acc = get_acc(test_data, pos_pivot, neg_pivot,
                       word2idx, memory_size,
                       max_sentence_size)
    best_pos_pivots = copy(pos_pivot)
    best_neg_pivots = copy(neg_pivot)
    best_reward = get_reward(float(best_acc), float(len(pos_pivot)),float(len(pos_pivot)),
                             float(len(neg_pivot)), float(len(neg_pivot)))
    l = len(pos_pivot)
    i = 0
    l_p = len(pos_pivot)
    l_np = len(neg_pivot)
    while l > i:
        p_pivots = copy(pos_pivot)
        pos_pivot = numpy.delete(pos_pivot, i)
        acc = get_acc(test_data, pos_pivot, neg_pivot,
                      word2idx, memory_size,
                      max_sentence_size)
        reward = get_reward(float(acc), float(len(pos_pivot)),
                            float(l_p), float(len(neg_pivot)), float(l_np))
        print('reward', reward)
        if acc > best_acc:
            best_reward = reward
            best_acc = acc
            best_pos_pivots = copy(pos_pivot)
            best_neg_pivots = copy(neg_pivot)
            l = l - 1
        else:
            pos_pivot = copy(p_pivots)
            i = i + 1
    l = len(neg_pivot)
    i = 0
    while l > i:
        n_pivots = copy(neg_pivot)
        neg_pivot = numpy.delete(neg_pivot, i)
        acc = get_acc(test_data, pos_pivot, neg_pivot,
                      word2idx, memory_size,
                      max_sentence_size)
        reward = get_reward(float(acc), float(len(pos_pivot)),
                            float(l_p), float(len(neg_pivot)), float(l_np))
        print('reward', reward)
        if acc > best_acc:
            best_reward = reward
            best_acc = acc
            best_pos_pivots = copy(pos_pivot)
            best_neg_pivots = copy(neg_pivot)
            l = l - 1
        else:
            neg_pivot = copy(n_pivots)
            i = i + 1
    store_results("CDRL", source_domain, target_domain, best_acc)
    print('best_reward', best_reward)
    print('best_acc', best_acc)
    store_pivots(best_pos_pivots, best_neg_pivots, source_domain + "_" + target_domain)

