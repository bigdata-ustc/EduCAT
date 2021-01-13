from abc import ABC, abstractmethod


class AbstractStrategy(ABC):

    @property
    @abstractmethod
    def name(self):
        """ the name of the strategy
        Returns:
            name: str
        """
        raise NotImplementedError

    @abstractmethod
    def adaptest_select(self, model, adaptest_data):
        """
        Args:
            model: AbstractModel
            adaptest_data: AdapTestDataset
        Returns:
            selected_questions: dict, {student_idx: question_idx}
        """
        raise NotImplementedError