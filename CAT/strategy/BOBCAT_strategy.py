import numpy as np
from scipy.optimize import minimize
from CAT.strategy.abstract_strategy import AbstractStrategy
from CAT.model import AbstractModel
from CAT.dataset import AdapTestDataset

class BOBCAT(AbstractStrategy):

    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return 'BOBCAT'
    def adaptest_select(self, model: AbstractModel, adaptest_data: AdapTestDataset,S_set):
        assert hasattr(model, 'get_kli'), \
            'the models must implement get_kli method'
        assert hasattr(model, 'get_pred'), \
            'the models must implement get_pred method for accelerating'
        selection = {}
        for sid in range(adaptest_data.num_students):
            untested_questions = np.array(list(adaptest_data.untested[sid]))
            j = model.bobcat_policy(S_set[sid],untested_questions)
            selection[sid] = j
        return selection