# Copyright (c) 2018-present, Ahmed H. Al-Ghidani.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

__author__ = "Ahmed H. Al-Ghidani"
__copyright__ = "Copyright 2018, The nlpeus Project, https://github.com/AhmedHani/nlpeus"
__license__ = "GPL"
__maintainer__ = "Ahmed H. Al-Ghidani"
__email__ = "ahmed.hani.ibrahim@gmail.com"

from common.evaluator import SupervisedEvaluator


class SupervisedTrainer(object):

    def __init__(self, model, classes):
        self.model = model

        self.class2index = classes[0]
        self.index2class = classes[1]

        self.batches_confusion_matrix = []
        self.complete_conf_matrix = None

    def fit_batch(self, x_train, y_train):
        prediction, hidden = self.model(x_train)
        gradient = self.model.calculate_gradient(prediction, y_train)
        self.model.optimize()

        return gradient

    def fit(self, x_train, y_train):
        return self.fit_batch(x_train, y_train)

    def eval_batch(self, x_valid, y_valid):
        predictions = self.model.predict_classes(x_valid)

        conf_matrix = SupervisedEvaluator.get_confusion_matrix(predictions, y_valid, self.index2class)

        if self.complete_conf_matrix is None:
            self.complete_conf_matrix = conf_matrix
        else:
            self.complete_conf_matrix += conf_matrix

        return conf_matrix

    def eval(self, x_valid, y_valid):
        return self.eval_batch(x_valid, y_valid)

    def show_evaluation(self, precision_recall_fscore=True, conf_matrix=True, accuracy=True, stdout='stdout', pickle_path=None):
        SupervisedEvaluator.evaluate_batches(self.complete_conf_matrix,
                                                precision_recall_fscore,
                                                conf_matrix,
                                                accuracy,
                                                self.index2class,
                                                stdout,
                                                pickle_path)

    def test(self, x_test):
        predictions = self.model.predict_classes(x_test)

        return [self.index2class[index] for index in predictions]

    def predict_classes(self, x_sample):
        predictions = self.model.predict_classes(x_sample)

        return predictions

    def predict_probs(self, x_sample):
        return self.model.predict_probs(x_sample)

    def save(self, path):
        return self.model.save_weights(path)

    def load(self, path):
        return self.model.load_weights(path)
    
    def model_args(self):
        return self.model.args()
    
    def model_class(self):
        return self.model.__class__