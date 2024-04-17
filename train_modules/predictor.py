import os

import numpy as np
import torch
import transformers
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import roc_curve

from hi_tin import Configure, HiAGM, Vocab, data_loaders


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
            config, self.vocab, model_type=config.model.type, model_mode="EVAL"
        )
        self.model.load_state_dict(self.model_chkpt["state_dict"])
        self.model.to(self.device)

    @staticmethod
    def _get_model_checkpoint(ckpt_dir: os.PathLike, type: str = "macro") -> os.PathLike:
        """
        Dumb helper to return the path of the latest model checkpoint file
        :param ckpt_dir: The root checkpoint folder, usually defined in the config file
        :param type: either 'macro' or 'micro', specifying type of checkpoint if there's more than one
        """
        if not os.path.isdir(ckpt_dir):
            return None
        else:
            dir_list = [dir.path for dir in os.scandir(ckpt_dir) if dir.is_dir()]
            dir_list.sort(key=lambda fn: os.path.getatime(fn))

            file_list = [
                file.path
                for file in os.scandir(dir_list[0])
                if file.name.startswith(f"best_{type}")
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
        self, dataset: str = "test", optimized_thresholds: bool = True
    ) -> tuple[list[list[int]], list[list[int]]]:
        """
        Run prediction on the specified dataset (almost assuredly "test")
        using computed label thresholds on the validation set
        :param dataset: one of "test", "train", "validation"
        :param optimized_thresholds: Use label-optimized cutpoints (True), or a flat cutpoint when determining label assignments (False)
        :return: A tuple: y_pred, y_true, both list[list[int]] containing the predicted and true label values for each sample in `dataset`
        """
        pred_proba, y_true = self.run(dataset)

        y_pred = []

        # Either length 1 or m (number of labels)
        if optimized_thresholds:
            self.compute_label_thresholds()
            thresh = self.lab_thresholds
        else:
            thresh = self.flat_threshold

        for samp in pred_proba:
            y_pred.append(np.where(samp > thresh)[0].tolist())

        return (y_pred, y_true)

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
            self.vocab.i2v["doc_label"][i]: v for i, v in enumerate(cut_points)
        }

    def run(self, dataset: str = "train") -> tuple[NDArray, list[list[int]]]:
        """
        Returns predictions from the model using the specified dataset
        """
        predict_probs = []
        y_true = []

        # Evaluate test data batch by batch
        for batch in self.data_loaders[dataset]:
            logits = self.model(batch)
            predict_results = torch.sigmoid(logits).cpu().tolist()
            predict_probs.extend(predict_results)
            y_true.extend(batch["label_list"])

        pred_prob_mat = np.array(predict_probs)
        return (pred_prob_mat, y_true)
