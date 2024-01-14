from math import exp as exp
from sklearn.metrics import roc_auc_score
from CAT.dataset import AdapTestDataset
from sklearn.metrics import accuracy_score
from collections import namedtuple
from scipy.optimize import minimize
from CAT.strategy.abstract_strategy import AbstractStrategy
from CAT.model.NCAT import NCATModel

class NCATs(AbstractStrategy):

    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return 'NCAT'

    def adaptest_select(self,  adaptest_data: AdapTestDataset,concept_map,config,test_length):
        used_actions = []
        for sid in range(adaptest_data.num_students):
            NCATdata = adaptest_data
            model = NCATModel(NCATdata,concept_map,config,test_length)
            THRESHOLD = config['THRESHOLD']
            model.ncat_policy(sid,THRESHOLD,used_actions)

        return used_actions
