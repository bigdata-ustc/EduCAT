import numpy as np

from CAT.strategy.abstract_strategy import AbstractStrategy
from CAT.model import AbstractModel
from CAT.dataset import AdapTestDataset


class KLIStrategy(AbstractStrategy):

    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return 'Kullback-Leibler Information Strategy'

    def adaptest_select(self, model: AbstractModel, adaptest_data: AdapTestDataset):
        assert hasattr(model, 'get_kli'), \
            'the models must implement get_kli method'
        assert hasattr(model, 'get_pred'), \
            'the models must implement get_pred method for accelerating'
        pred_all = model.get_pred(adaptest_data)
        selection = {}
        n = len(adaptest_data.tested[0])
        for sid in range(adaptest_data.num_students):
            theta = model.get_theta(sid)
            untested_questions = np.array(list(adaptest_data.untested[sid]))
            untested_kli = [model.get_kli(sid, qid, n, pred_all) for qid in untested_questions]
            j = np.argmax(untested_kli)
            selection[sid] = untested_questions[j]
        return selection

class MKLIStrategy(KLIStrategy):

    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return 'Multivariate Kullback-Leibler Information Strategy'