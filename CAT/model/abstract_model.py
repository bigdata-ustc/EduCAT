from abc import ABC, abstractmethod
try:
    # for python module
    from ..dataset import AdapTestDataset, TrainDataset, _Dataset
except (ImportError, SystemError):  # pragma: no cover
    # for python script
    from dataset import AdapTestDataset, TrainDataset, _Dataset


class AbstractModel(ABC):

    @property
    @abstractmethod
    def name(self):
        raise NotImplementedError

    @abstractmethod
    def adaptest_update(self, adaptest_data: AdapTestDataset):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, adaptest_data: AdapTestDataset):
        raise NotImplementedError

    @abstractmethod
    def init_model(self, data: _Dataset):
        raise NotImplementedError

    @abstractmethod
    def train(self, train_data: TrainDataset):
        raise NotImplementedError

    @abstractmethod
    def adaptest_save(self, path):
        raise NotImplementedError

    @abstractmethod
    def adaptest_load(self, path):
        raise NotImplementedError