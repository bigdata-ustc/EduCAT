import numpy as np

from CAT.strategy.abstract_strategy import AbstractStrategy
from CAT.model import AbstractModel
from CAT.dataset import AdapTestDataset


class MAATStrategy(AbstractStrategy):

    def __init__(self, n_candidates=10):
        super().__init__()
        self.n_candidates = n_candidates

    @property
    def name(self):
        return 'Model Agnostic Adaptive Testing'

    def _compute_coverage_gain(self, sid, qid, adaptest_data: AdapTestDataset):
        concept_cnt = {}
        for q in adaptest_data.data[sid]:
            for c in adaptest_data.concept_map[q]:
                concept_cnt[c] = 0
        for q in list(adaptest_data.tested[sid]) + [qid]:
            for c in adaptest_data.concept_map[q]:
                concept_cnt[c] += 1
        return (sum(cnt / (cnt + 1) for c, cnt in concept_cnt.items())
                / sum(1 for c in concept_cnt))

    def adaptest_select(self, model: AbstractModel, adaptest_data: AdapTestDataset):
        assert hasattr(model, 'expected_model_change'), \
            'the models must implement expected_model_change method'
        pred_all = model.get_pred(adaptest_data)
        selection = {}
        for sid in range(adaptest_data.num_students):
            untested_questions = np.array(list(adaptest_data.untested[sid]))
            emc_arr = [model.expected_model_change(sid, qid, adaptest_data, pred_all) for qid in untested_questions]
            candidates = untested_questions[np.argsort(emc_arr)[::-1][:self.n_candidates]]
            selection[sid] = max(candidates, key=lambda qid: self._compute_coverage_gain(sid, qid, adaptest_data))
        return selection