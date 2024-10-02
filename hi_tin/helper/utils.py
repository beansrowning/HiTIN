#!/usr/bin/env python

import codecs
import torch
from hi_tin.models.structure_model.tree import Tree


def load_checkpoint(model_file, model, config, optimizer=None):
    """
    load models
    :param model_file: Str, file path
    :param model: Computational Graph
    :param config: helper.configure, Configure object
    :param optimizer: optimizer, torch.Adam
    :return: best_performance -> [Float, Float], config -> Configure
    """
    checkpoint_model = torch.load(model_file)
    config.train.start_epoch = checkpoint_model['epoch'] + 1
    best_performance = checkpoint_model['best_performance']
    model.load_state_dict(checkpoint_model['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_model['optimizer'])
    return best_performance, config


def save_checkpoint(state, model_file):
    """
    :param state: Dict, e.g. {'state_dict': state,
                              'optimizer': optimizer,
                              'best_performance': [Float, Float],
                              'epoch': int}
    :param model_file: Str, file path
    :return:
    """
    torch.save(state, model_file)


def get_hierarchy_relations(hierar_taxonomy, label_map, root=None, fortree=False):
    """
    get parent-children relationships from given hierar_taxonomy
    parent_label \t child_label_0 \t child_label_1 \n
    :param hierar_taxonomy: Str, file path of hierarchy taxonomy
    :param label_map: Dict, label to id
    :param root: Str, root tag
    :param fortree: Boolean, True : return label_tree -> List
    :return: label_tree -> List[Tree], hierar_relation -> Dict{parent_id: List[child_id]}
    """
    label_tree = dict()
    label_tree[0] = root
    hierar_relations = {}
    with codecs.open(hierar_taxonomy, "r", "utf8") as f:
        relation_data = f.readlines()
    for relation in relation_data:
        line_split = relation.rstrip().split('\t')
        parent_label, children_label = line_split[0], line_split[1:]
        assert parent_label not in hierar_relations.keys()

        # Add idx to hierar relations dictionary
        if parent_label == "Root":
            parent_label_id = -1
        else:
            parent_label_id = label_map[parent_label]
        child_label_ids = [label_map[lab] for lab in children_label]
        hierar_relations[parent_label_id] = child_label_ids

        if fortree:
            try:
                parent_tree = label_tree[parent_label_id + 1]
            except KeyError:
                # Parent not observed yet
                label_tree[parent_label_id + 1] = Tree(parent_label_id)
                parent_tree = label_tree[parent_label_id + 1]

            for child in child_label_ids:
                try:
                    child_tree = label_tree[child + 1]
                except KeyError:
                    # child not observed yet
                    label_tree[child + 1] = Tree(child)
                    child_tree = label_tree[child + 1]

                parent_tree.add_child(child_tree)
    if fortree:
        return hierar_relations, label_tree
    else:
        return hierar_relations
