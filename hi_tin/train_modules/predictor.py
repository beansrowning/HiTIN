import os
from itertools import chain
from typing import Union

import numpy as np
import numpy.ma as ma
import torch
import transformers
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import roc_curve

from hi_tin import Configure, HiAGM, Vocab, data_loaders
from hi_tin.models.structure_model.tree import Tree

from ..helper.utils import get_checkpoint_name


class Predictor(object):
    def __init__(self, config: Configure, checkpoint_type: str = "macro"):
        """
        Similar to Trainer, but used exclusively for returning predictions from a saved model
        :param config: Configuration object
        :param checkpoint_type: prefix of checkpoint file to use (usually "macro" or "micro")
        """
        self.config = config
        self.device = config.train.device_setting.device
        self.flat_threshold = config.eval.threshold
        self.top_k = config.eval.top_k

        # Build vocab (or load in pre-saved vocab)
        self.vocab = Vocab(config, min_freq=5, max_size=50000)

        if config.text_encoder.type == "bert":
            self.tokenizer = transformers.BertTokenizer.from_pretrained(
                config.text_encoder.bert_model_dir
            )
        else:
            self.tokenizer = None

        # Load in data
        self.data_loaders = {
            k: v
            for k, v in zip(
                ["train", "validation", "test"],
                data_loaders(config, self.vocab, tokenizer=self.tokenizer),
            )
        }

        # Load in latest checkpoint data
        latest_checkpoint = self._get_model_checkpoint(
            config.train.checkpoint.dir, checkpoint_type
        )
        print(f"Loading pre-trained model weights from {latest_checkpoint}...")
        self.model_chkpt = torch.load(latest_checkpoint)

        # Init model
        self.model = HiAGM(
            config, self.vocab, model_type=config.model.type, model_mode="TEST"
        )
        self.model.load_state_dict(self.model_chkpt["state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def _get_model_checkpoint(self, ckpt_dir: os.PathLike, type: str = "macro") -> os.PathLike:
        """
        Dumb helper to return the path of the latest model checkpoint file
        :param ckpt_dir: The root checkpoint folder, usually defined in the config file
        :param type: either 'macro' or 'micro', specifying type of checkpoint if there's more than one
        """
        model_name = get_checkpoint_name(self.config)

        if not os.path.isdir(ckpt_dir):
            return None
        else:
            file_list = [
                file.path
                for file in os.scandir(ckpt_dir)
                if file.name.startswith(f"best_{type}_{model_name}")
            ]

        return file_list[0]

    @staticmethod
    def _optimize_threshold(y_true: ArrayLike, y_pred: ArrayLike) -> float:
        """
        Return optimal decision threshold based on Youden's J
        """
        fpr, tpr, thres = roc_curve(y_true, y_pred)
        optim_idx = np.argmax(tpr - fpr)

        return thres[optim_idx]

    def predict(
        self,
        dataset: str = "test",
        optimized_thresholds: bool = True,
        label_output: str = "flat",
        back_transform: bool = False,
        return_input: bool = False,
    ) -> (
        Union[tuple[list[list[int]], list[list[int]]], tuple[list[str], list[list[str]], list[list[str]]]]
    ):
        """
        Run prediction on the specified dataset (almost assuredly "test")
        using computed label thresholds on the validation set
        :param dataset: one of "test", "train", "validation"
        :param optimized_thresholds: Use label-optimized cutpoints (True), or a flat cutpoint when determining label assignments (False) (only observed if label_output == "flat")
        :param label_output: How to reconstruct the label(s), "flat" which returns top-k in a hierarchy naive way, "bottom_up" which re-constructs the hierarchy from the argmaxed leafnode prediction, or "top_down" which performs a masked argmax at each level of the hierarchy
        :param back_transform: Should integer labels be back-transformed to their plain-text descriptions?
        :param return input: Should input (X) be returned as plain-text as the first item?
        :return:
            A tuple: y_pred, y_true, both list[list[int]] containing the predicted and true label values for each sample in `dataset` if return_input is False
            or
            A tuple: input_text, y_pred, y_true, all list[str] with y_pred and y_true back-transformed from their integer encoding
        """

        output_opts = ["bottom_up", "top_down", "flat"]

        input_text, pred_proba, y_true = self.run(dataset, return_input=True)

        y_pred = []

        # Either length 1 or m (number of labels)
        if optimized_thresholds and label_output == "flat":
            # We only need this in the circumstance that
            # we're preforming "flat" label predictions
            self.compute_label_thresholds()
            thresh = self.lab_thresholds
        else:
            thresh = [self.flat_threshold] * pred_proba.shape[1]

        # Take top-k labels by probability
        for samp in pred_proba:
            samp_labels = []
            if label_output == "bottom_up":
                samp_labels = self._recompose_label_bottom_up(samp)
            elif label_output == "top_down":
                samp_labels = self._recompose_label_top_down(samp)
            elif label_output == "flat":
                # take top-k (which may or may not be in the correct hierarchy)
                samp_masked = ma.masked_where(samp <= thresh, samp)
                samp_labels = np.argsort(-samp_masked)[: self.top_k].tolist()
            else:
                raise ValueError(f"Output format must be one of: {str(output_opts)}, recieved {label_output}")

            y_pred.append(samp_labels)
        if back_transform:
            # back-xform integers to text
            y_pred = [
                [self.vocab.i2v["doc_label_list"][i] for i in sample]
                for sample in y_pred
            ]
            y_true = [
                [self.vocab.i2v["doc_label_list"][i] for i in sample]
                for sample in y_true
            ]

            if label_output != "flat":
                y_pred = ["/".join(labs) for labs in y_pred]
                y_true = ["/".join(labs) for labs in y_true]
        # Return along with input text
        if return_input:
            (input_text, y_pred, y_true)
        else:
            return (y_pred, y_true)

    def _recompose_label_top_down(self, sample_probs: np.array) -> list[int]:
        """
        Recompose full hierarchical label from the highest level of the hierarchy to the lowest
        conditioning the prediction at each lower level of the hierarhy on the previous level to
        ensure total label is valid
        """
        label_dict = self.model.structure_encoder.hierarchical_label_dict
        out = []
        # -1 is root node
        parent = -1

        while True:
            try:
                children = label_dict[parent]
            except:
                break
            # Mask array to only possible decendants of parent label and argmax
            samp_masked = ma.masked_array(
                sample_probs,
                mask=[0 if i in children else 1 for i in range(sample_probs.shape[0])],
            )
            child = samp_masked.argmax()
            out.append(child)
            # Child becomes new parent
            parent = child

        return out

    def _recompose_label_bottom_up(self, samp: np.array) -> list[int]:
        """
        Recompose full hierarchical label from the lowest level of the hierarchy to the highest
        ensuring that final label consists of only descendants of predicted leaf node (and thus, is valid)
        """
        # Identify leaf nodes
        parents = [
            elem for elem in self.model.structure_encoder.hierarchical_label_dict.keys()
        ]
        children = [
            [elem for elem in elems]
            for elems in self.model.structure_encoder.hierarchical_label_dict.values()
        ]

        leafs = list(set(chain(*children)).difference(set(parents)))

        # Mask to only the leaf nodes and argmax to begin
        samp_masked = ma.masked_array(
            samp, mask=[0 if i in leafs else 1 for i in range(samp.shape[0])]
        )
        label_idx = samp_masked.argmax()

        label_tree = self.model.structure_encoder.label_trees

        full_label = []
        full_label.append(label_idx)

        while True:
            try:
                lab_tree = label_tree[full_label[-1] + 1]
                parent = lab_tree.parent.idx
            except:
                break
            # If we're at the top level, stop traversing
            if parent == -1:
                break
            full_label.append(parent)
        # We re-built from bottom up, so we have to reverse to get in the right order
        full_label.reverse()

        return full_label

    def compute_label_thresholds(self):
        """
        Using validation set, optimize decision thresholds for each label and store
        for later use on testing data
        (assuming multi-label scenario where predicted probabilites for each label can be interpreted in the binary sense)
        """
        y_pred, y_true = self.run(dataset="validation")

        cut_points = []
        # Loop col-wise over each label prob
        for i, pred_proba in enumerate(y_pred.T):
            # Determine which samples had the label
            true = [1 if i in x else 0 for x in y_true]
            thresh = self._optimize_threshold(y_true=true, y_pred=pred_proba)
            cut_points.append(thresh)

        self.lab_thresholds = cut_points
        self.lab_thresholds_dict = {
            self.vocab.i2v["doc_label_list"][i]: v for i, v in enumerate(cut_points)
        }

    def run(
        self, dataset: str = "train", return_input: bool = False
    ) -> Union[tuple[NDArray, list[list[int]]], tuple[list[str], NDArray, list[list[int]]]]:
        """
        Returns predictions from the model using the specified dataset
        """
        predict_probs = []
        y_true = []
        x = []
        # Evaluate test data batch by batch
        with torch.no_grad():
            for batch in self.data_loaders[dataset]:
                logits = self.model(batch)
                predict_results = torch.sigmoid(logits).cpu().tolist()
                predict_probs.extend(predict_results)
                y_true.extend(batch["label_list"])
                if return_input:
                    x.extend(batch["input_text"])

        pred_prob_mat = np.array(predict_probs)

        if return_input:
            return (x, pred_prob_mat, y_true)
        return (pred_prob_mat, y_true)
