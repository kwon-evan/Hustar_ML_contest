from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
from tqdm import tqdm, trange
import multiprocessing

from . import (tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index)

logger = logging.getLogger(__name__)

# remove comments, tokenize code and extract dataflow
def extract_dataflow(code, parser):
    # obtain dataflow
    try:
        tree = parser[0].parse(bytes(code, 'utf8'))
        root_node = tree.root_node
        tokens_index = tree_to_token_index(root_node)
        code = code.split('\n')
        code_tokens = [index_to_code_token(x, code) for x in tokens_index]
        index_to_code = {}
        for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
            index_to_code[index] = (idx, code)
        try:
            DFG, _ = parser[1](root_node, index_to_code, {})
        except:
            DFG = []
        DFG = sorted(DFG, key=lambda x: x[1])
        indexs = set()
        for d in DFG:
            if len(d[-1]) != 0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG = []
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg = new_DFG
    except:
        dfg = []
    return code_tokens, dfg


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 input_tokens_1,
                 input_ids_1,
                 position_idx_1,
                 dfg_to_code_1,
                 dfg_to_dfg_1,
                 input_tokens_2,
                 input_ids_2,
                 position_idx_2,
                 dfg_to_code_2,
                 dfg_to_dfg_2,
                 label
                 ):
        # The first code function
        self.input_tokens_1 = input_tokens_1
        self.input_ids_1 = input_ids_1
        self.position_idx_1 = position_idx_1
        self.dfg_to_code_1 = dfg_to_code_1
        self.dfg_to_dfg_1 = dfg_to_dfg_1

        # The second code function
        self.input_tokens_2 = input_tokens_2
        self.input_ids_2 = input_ids_2
        self.position_idx_2 = position_idx_2
        self.dfg_to_code_2 = dfg_to_code_2
        self.dfg_to_dfg_2 = dfg_to_dfg_2

        # label
        self.label = label


def convert_examples_to_features(item):
    # source
    code1, code2, label, tokenizer, parser, code_length, data_flow_length, cache = item

    for code in [code1, code2]:
        if code not in cache:

            # extract data flow
            code_tokens, dfg = extract_dataflow(code, parser)
            code_tokens = [tokenizer.tokenize('@ ' + x)[1:] if idx != 0 else tokenizer.tokenize(x) for idx, x in
                           enumerate(code_tokens)]
            ori2cur_pos = {}
            ori2cur_pos[-1] = (0, 0)
            for i in range(len(code_tokens)):
                ori2cur_pos[i] = (ori2cur_pos[i - 1][1], ori2cur_pos[i - 1][1] + len(code_tokens[i]))
            code_tokens = [y for x in code_tokens for y in x]

            # truncating
            code_tokens = code_tokens[
                          :code_length + data_flow_length - 3 - min(len(dfg), data_flow_length)][
                          :512 - 3]
            source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
            source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
            position_idx = [i + tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]
            dfg = dfg[:code_length + data_flow_length - len(source_tokens)]
            source_tokens += [x[0] for x in dfg]
            position_idx += [0 for x in dfg]
            source_ids += [tokenizer.unk_token_id for x in dfg]
            padding_length = code_length + data_flow_length - len(source_ids)
            position_idx += [tokenizer.pad_token_id] * padding_length
            source_ids += [tokenizer.pad_token_id] * padding_length

            # reindex
            reverse_index = {}
            for idx, x in enumerate(dfg):
                reverse_index[x[1]] = idx
            for idx, x in enumerate(dfg):
                dfg[idx] = x[:-1] + ([reverse_index[i] for i in x[-1] if i in reverse_index],)
            dfg_to_dfg = [x[-1] for x in dfg]
            dfg_to_code = [ori2cur_pos[x[1]] for x in dfg]
            length = len([tokenizer.cls_token])
            dfg_to_code = [(x[0] + length, x[1] + length) for x in dfg_to_code]
            cache[code] = source_tokens, source_ids, position_idx, dfg_to_code, dfg_to_dfg

    source_tokens_1, source_ids_1, position_idx_1, dfg_to_code_1, dfg_to_dfg_1 = cache[code]
    source_tokens_2, source_ids_2, position_idx_2, dfg_to_code_2, dfg_to_dfg_2 = cache[code]
    return InputFeatures(source_tokens_1, source_ids_1, position_idx_1, dfg_to_code_1, dfg_to_dfg_1,
                         source_tokens_2, source_ids_2, position_idx_2, dfg_to_code_2, dfg_to_dfg_2,
                         label)


class TextDataset(Dataset):
    def __init__(self, tokenizer, parser, code_length=512, data_flow_length=256, file_path=None):
        self.examples = []
        self.code_length = code_length
        self.data_flow_length = data_flow_length

        # load index
        logger.info("Creating features from index file at %s ", file_path)
        csv_data = pd.read_csv(file_path)

        # load code function according to index
        data = []
        cache = {}
        for idx, (code1, code2, similar) in csv_data.iterrows():
            code1 = code1.strip()
            code2 = code2.strip()
            label = similar
            data.append((code1, code2, label, tokenizer, parser, code_length, data_flow_length, cache))

        # convert example to input features
        self.examples = [convert_examples_to_features(x) for x in tqdm(data, total=len(data))]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        # calculate graph-guided masked function
        attn_mask_1 = np.zeros((self.code_length + self.data_flow_length,
                                self.code_length + self.data_flow_length), dtype=bool)
        # calculate begin index of node and max length of input
        node_index = sum([i > 1 for i in self.examples[item].position_idx_1])
        max_length = sum([i != 1 for i in self.examples[item].position_idx_1])
        # sequence can attend to sequence
        attn_mask_1[:node_index, :node_index] = True
        # special tokens attend to all tokens
        for idx, i in enumerate(self.examples[item].input_ids_1):
            if i in [0, 2]:
                attn_mask_1[idx, :max_length] = True
        # nodes attend to code tokens that are identified from
        for idx, (a, b) in enumerate(self.examples[item].dfg_to_code_1):
            if a < node_index and b < node_index:
                attn_mask_1[idx + node_index, a:b] = True
                attn_mask_1[a:b, idx + node_index] = True
        # nodes attend to adjacent nodes
        for idx, nodes in enumerate(self.examples[item].dfg_to_dfg_1):
            for a in nodes:
                if a + node_index < len(self.examples[item].position_idx_1):
                    attn_mask_1[idx + node_index, a + node_index] = True

                    # calculate graph-guided masked function
        attn_mask_2 = np.zeros((self.code_length + self.data_flow_length,
                                self.code_length + self.data_flow_length), dtype=bool)
        # calculate begin index of node and max length of input
        node_index = sum([i > 1 for i in self.examples[item].position_idx_2])
        max_length = sum([i != 1 for i in self.examples[item].position_idx_2])
        # sequence can attend to sequence
        attn_mask_2[:node_index, :node_index] = True
        # special tokens attend to all tokens
        for idx, i in enumerate(self.examples[item].input_ids_2):
            if i in [0, 2]:
                attn_mask_2[idx, :max_length] = True
        # nodes attend to code tokens that are identified from
        for idx, (a, b) in enumerate(self.examples[item].dfg_to_code_2):
            if a < node_index and b < node_index:
                attn_mask_2[idx + node_index, a:b] = True
                attn_mask_2[a:b, idx + node_index] = True
        # nodes attend to adjacent nodes
        for idx, nodes in enumerate(self.examples[item].dfg_to_dfg_2):
            for a in nodes:
                if a + node_index < len(self.examples[item].position_idx_2):
                    attn_mask_2[idx + node_index, a + node_index] = True

        return (torch.tensor(self.examples[item].input_ids_1),
                torch.tensor(self.examples[item].position_idx_1),
                torch.tensor(attn_mask_1),
                torch.tensor(self.examples[item].input_ids_2),
                torch.tensor(self.examples[item].position_idx_2),
                torch.tensor(attn_mask_2),
                torch.tensor(self.examples[item].label))
