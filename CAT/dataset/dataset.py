from collections import defaultdict, deque


class Dataset(object):

    def __init__(self, data, concept_map,
                 num_students, num_questions, num_concepts):
        """
        Args:
            data: list, [(sid, qid, score)]
            concept_map: dict, concept map {qid: cid}
            num_students: int, total student number
            num_questions: int, total question number
            num_concepts: int, total concept number
        """
        self._raw_data = data
        self._concept_map = concept_map
        self.n_students = num_students
        self.n_questions = num_questions
        self.n_concepts = num_concepts
        
        # reorganize datasets
        self._data = {}
        for sid, qid, correct in data:
            self._data.setdefault(sid, {})
            self._data[sid].setdefault(qid, {})
            self._data[sid][qid] = correct

        student_ids = set(x[0] for x in data)
        question_ids = set(x[1] for x in data)
        concept_ids = set(sum(concept_map.values(), []))

        assert max(student_ids) < num_students, \
            'Require student ids renumbered'
        assert max(question_ids) < num_questions, \
            'Require student ids renumbered'
        assert max(concept_ids) < num_concepts, \
            'Require student ids renumbered'

    @property
    def num_students(self):
        return self.n_students

    @property
    def num_questions(self):
        return self.n_questions

    @property
    def num_concepts(self):
        return self.n_concepts

    @property
    def raw_data(self):
        return self._raw_data

    @property
    def data(self):
        return self._data

    @property
    def concept_map(self):
        return self._concept_map