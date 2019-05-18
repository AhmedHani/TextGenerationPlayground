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

import numpy as np
import pickle as pkl
from terminaltables import AsciiTable


class SupervisedEvaluator:

    @staticmethod
    def evaluate_batches(complete_matrix,
                         precision_recall_fscore,
                         conf_matrix,
                         accuracy,
                         index2class,
                         stdout,
                         pickle_path):
        results = {}

        if stdout == "stdout":
            headings = ['class index', 'class name']

            table_data = [headings]

            for item in index2class.items():
                table_data.append(['class ' + str(item[1]), str(item[0])])

            print(AsciiTable(table_data).table)
            print('\n')

            if conf_matrix:
                print('confusion matrix\n')

                cm_headings = ['--'] + ['class ' + str(i) for i in range(len(index2class))]
                table_data = [cm_headings]

                for i in range(0, len(index2class)):
                    table_data.append(['class ' + str(i)] + [str(int(val)) for val in complete_matrix[i]])

                print(AsciiTable(table_data).table)
                print('\n')

            if accuracy:
                tp = np.diag(complete_matrix)
                total_tp = np.sum(tp)
                all_sum = np.sum(complete_matrix)

                print('accuracy: {}'.format(round(float(total_tp) / float(all_sum), 3)))
                results['accuracy'] = round(float(total_tp) / float(all_sum), 3)

            if precision_recall_fscore:
                tp = np.diag(complete_matrix)
                fp = np.sum(complete_matrix, axis=1) - tp
                fn = np.sum(complete_matrix, axis=0) - tp

                tn = []
                for i in range(len(index2class)):
                    temp = np.delete(complete_matrix, i, 0)
                    temp = np.delete(temp, i, 1)
                    tn.append(sum(sum(temp)))

                tn = np.asarray(tn)

                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                fscore = 2.0 * ((precision * recall) / (precision + recall))

                cm_headings = ['--'] + ['class ' + str(i) for i in range(len(index2class))]
                table_data = [cm_headings]

                table_data.append(['precision'] + [str(val) for val in list(precision.tolist())])
                table_data.append(['recall'] + [str(val) for val in list(recall.tolist())])
                table_data.append(['fscore'] + [str(val) for val in list(fscore.tolist())])

                results['average_precision'] = sum(list(precision.tolist())) / len(list(precision.tolist()))
                results['average_recall'] = sum(list(recall.tolist())) / len(list(recall.tolist()))
                results['average_fscore'] = sum(list(fscore.tolist())) / len(list(fscore.tolist()))

                print(AsciiTable(table_data).table)
                print("\n\n")
                print("average fscore: {}\n\n".format(sum(list(fscore.tolist())) / len(list(fscore.tolist()))))
        else:
            import codecs
            writer = codecs.open(stdout, 'w', encoding='utf-8')

            headings = ['class index', 'class name']

            table_data = [headings]

            for item in index2class.items():
                table_data.append(['class ' + str(item[1]), str(item[0])])

            writer.write(AsciiTable(table_data).table)
            writer.write('\n\n')

            if conf_matrix:
                writer.write('confusion matrix\n')

                cm_headings = ['--'] + ['class ' + str(i) for i in range(len(index2class))]
                table_data = [cm_headings]

                for i in range(0, len(index2class)):
                    table_data.append(['class ' + str(i)] + [str(int(val)) for val in complete_matrix[i]])

                writer.write(AsciiTable(table_data).table)
                writer.write("\n\n")

            if accuracy:
                tp = np.diag(complete_matrix)
                total_tp = np.sum(tp)
                all_sum = np.sum(complete_matrix)

                writer.write('accuracy: {}'.format(round(float(total_tp) / float(all_sum), 3)))
                writer.write('\n\n')

                results['accuracy'] = round(float(total_tp) / float(all_sum), 3)

            if precision_recall_fscore:
                tp = np.diag(complete_matrix)
                fp = np.sum(complete_matrix, axis=1) - tp
                fn = np.sum(complete_matrix, axis=0) - tp

                tn = []
                for i in range(len(index2class)):
                    temp = np.delete(complete_matrix, i, 0)
                    temp = np.delete(temp, i, 1)
                    tn.append(sum(sum(temp)))

                tn = np.asarray(tn)

                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                fscore = 2.0 * ((precision * recall) / (precision + recall))

                cm_headings = ['--'] + ['class ' + str(i) for i in range(len(index2class))]
                table_data = [cm_headings]

                table_data.append(['precision'] + [str(val) for val in list(precision.tolist())])
                table_data.append(['recall'] + [str(val) for val in list(recall.tolist())])
                table_data.append(['fscore'] + [str(val) for val in list(fscore.tolist())])

                results['average_precision'] = sum(list(precision.tolist())) / len(list(precision.tolist()))
                results['average_recall'] = sum(list(recall.tolist())) / len(list(recall.tolist()))
                results['average_fscore'] = sum(list(fscore.tolist())) / len(list(fscore.tolist()))

                writer.write(AsciiTable(table_data).table)
                writer.write("\n\n")
                writer.write("average fscore: {}\n\n".format(sum(list(fscore.tolist())) / len(list(fscore.tolist()))))

                writer.flush()
                writer.close()

        with open(pickle_path, 'wb') as writer:
            pkl.dump(results, writer)

    @staticmethod
    def get_confusion_matrix(prediction, target, classes):
        prediction = np.asarray(prediction)
        target = np.asarray(target)

        conf_matrix = np.zeros((len(classes), len(classes)))

        for i in range(0, prediction.shape[0]):
            predicted_label = prediction[i]
            actual_label = target[i]

            conf_matrix[int(predicted_label), int(actual_label)] += 1

        return conf_matrix
