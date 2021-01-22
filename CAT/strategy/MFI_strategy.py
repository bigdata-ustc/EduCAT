import numpy as np
import torch

from CAT.strategy.abstract_strategy import AbstractStrategy
from CAT.model import AbstractModel
from CAT.dataset import AdapTestDataset


class MFIStrategy(AbstractStrategy):
    """
    Maximum Fisher Information Strategy
    D-opt Strategy when using MIRT(num_dim != 1)
    """

    def __init__(self):
        super().__init__()
        self.I = None

    @property
    def name(self):
        return 'Maximum Fisher Information Strategy'

    def adaptest_select(self, model: AbstractModel, adaptest_data: AdapTestDataset):
        assert hasattr(model, 'get_fisher'), \
            'the models must implement get_fisher method'
        if self.I is None:
            self.I = [np.zeros((model.model.num_dim, model.model.num_dim))] * adaptest_data.num_students    
        selection = {}
        n = len(adaptest_data.tested[0])
        for sid in range(adaptest_data.num_students):
            untested_questions = np.array(list(adaptest_data.untested[sid]))
            untested_dets = []
            untested_fisher = []
            for qid in untested_questions:
                fisher_info = model.get_fisher(sid, qid)
                untested_fisher.append(fisher_info)
                untested_dets.append(np.linalg.det(self.I[sid] + fisher_info))
            j = np.argmax(untested_dets)
            selection[sid] = untested_questions[j]
            self.I[sid] += untested_fisher[j]
        return selection