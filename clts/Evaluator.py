import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

def to_array(X, n=2):
    return np.array([np.eye(n)[x] for x in X])


class Evaluator:
    def __init__(self, aspect2ind, metric='acc', logger=None):
        self.aspect2ind = aspect2ind
        self.ind2aspect = {aspect2ind[a]: a for a in aspect2ind}
        self.logger = logger
        self.metric = metric

    def printstr(self, s):
        if self.logger:
            self.logger.info(s)
        else:
            print(s)

    def evaluate(self, df, pred_col='pred_id', true_col='label', descr='Teacher'):
        total = df.shape[0]

        if len(set(df[true_col])) == 1:
            self.printstr("Teacher predictions:\n{}".format(df[pred_col].value_counts()))
            self.printstr("Cannot evaluate {} on unlabeled data...".format(descr))
            return {
                'acc': -1,
                'ignore': -1,
                'ignore_percent': -1
            }

        # ignore -1
        ignore = df[df[pred_col] == -1].shape[0]
        ignore_percent = ignore / float(total)
        self.printstr("{} ignored {}/{} samples ({:.1f}%)".format(descr, ignore, total, 100 * ignore_percent))


        test = df[df[pred_col] != -1]
        test['pred'] = test[pred_col].map(lambda x: self.ind2aspect[x])
        test['true'] = test[true_col]

        if self.metric == 'acc':
            acc = 100 * accuracy_score(y_true=test['label'], y_pred=test['pred'])
        elif self.metric == 'f1':
            acc = 100 * f1_score(y_true=test['label'], y_pred=test['pred'], average='macro')

        adjusted_acc = acc * (total - ignore) / total  # adjusting accuracy to include non-assignments
        self.printstr("{} {}: {:.2f}%".format(descr, self.metric, adjusted_acc))
        # self.printstr("{} Confusion matrix:\n{}\n".format(descr, confusion_matrix(y_true=test['true'], y_pred=test['pred'])))
        # self.printstr("\n{}".format(classification_report(y_true=test['true'], y_pred=test['pred'])))
        res = {
            'acc': adjusted_acc,
            'ignore': ignore,
            'ignore_percent': ignore_percent,
        }
        return res