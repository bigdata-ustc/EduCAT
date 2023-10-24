import torch
import logging
import numpy as np
import torch.nn as nn
import torch.utils.data as data
from sklearn.metrics import roc_auc_score,accuracy_score
import math
from CAT.model.abstract_model import AbstractModel
from CAT.dataset import AdapTestDataset, TrainDataset, Dataset


class NCD(nn.Module):
    '''
    NeuralCDM
    '''
    def __init__(self, student_n, exer_n, knowledge_n, prednet_len1=128, prednet_len2=64):
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = prednet_len1, prednet_len2  # changeable

        super(NCD, self).__init__()

        # network structure
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_discrimination = nn.Embedding(self.exer_n, 1)
        self.prednet_full1 = nn.Linear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = nn.Linear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = nn.Linear(self.prednet_len2, 1)

        # initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, exer_id, kn_emb):
        '''
        :param stu_id: LongTensor
        :param exer_id: LongTensor
        :param kn_emb: FloatTensor, the knowledge relevancy vectors
        :return: FloatTensor, the probabilities of answering correctly
        '''
        # before prednet
        stu_emb = torch.sigmoid(self.student_emb(stu_id))
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10
        # prednet
        input_x = e_discrimination * (stu_emb - k_difficulty) * kn_emb
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output = torch.sigmoid(self.prednet_full3(input_x))

        return output

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)

    def get_knowledge_status(self, stu_id):
        stat_emb = torch.sigmoid(self.student_emb(stu_id))
        return stat_emb.data

    def get_exer_params(self, exer_id):
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10
        return k_difficulty.data, e_discrimination.data


class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)


class NCDModel(AbstractModel):

    def __init__(self, **config):
        super().__init__()
        self.config = config
        self.model = None

    @property
    def name(self):
        return 'Neural Cognitive Diagnosis'

    def init_model(self, data: Dataset):
        self.model = NCD(data.num_students, data.num_questions, data.num_concepts, self.config['prednet_len1'], self.config['prednet_len2'])
    
    def train(self, train_data: TrainDataset):
        lr = self.config['learning_rate']
        batch_size = self.config['batch_size']
        epochs = self.config['num_epochs']
        device = self.config['device']
        self.model.to(device)
        logging.info('train on {}'.format(device))

        train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for ep in range(1, epochs + 1):
            loss = 0.0
            log_step = 1
            for cnt, (student_ids, question_ids, concepts_emb, labels) in enumerate(train_loader):
                student_ids = student_ids.to(device)
                question_ids = question_ids.to(device)
                concepts_emb = concepts_emb.to(device)
                labels = labels.to(device)
                pred = self.model(student_ids, question_ids, concepts_emb)
                bz_loss = self._loss_function(pred, labels)
                optimizer.zero_grad()
                bz_loss.backward()
                optimizer.step()
                self.model.apply_clipper()
                loss += bz_loss.data.float()
                if cnt % log_step == 0:
                    logging.info('Epoch [{}] Batch [{}]: loss={:.5f}'.format(ep, cnt, loss / cnt))
    
    def _loss_function(self, pred, real):
        pred_0 = torch.ones(pred.size()).to(self.config['device']) - pred
        output = torch.cat((pred_0, pred), 1)
        criteria = nn.NLLLoss()
        return criteria(torch.log(output), real)
    
    def adaptest_save(self, path):
        """
        Save the model. Do not save the parameters for students.
        """
        model_dict = self.model.state_dict()
        model_dict = {k:v for k,v in model_dict.items() if 'student' not in k}
        torch.save(model_dict, path)
    
    def adaptest_load(self, path):
        """
        Reload the saved model
        """
        self.model.load_state_dict(torch.load(path), strict=False)
        self.model.to(self.config['device'])
    
    def adaptest_update(self, adaptest_data: AdapTestDataset):
        lr = self.config['learning_rate']
        batch_size = self.config['batch_size']
        epochs = self.config['num_epochs']
        device = self.config['device']
        optimizer = torch.optim.Adam(self.model.student_emb.parameters(), lr=lr)

        tested_dataset = adaptest_data.get_tested_dataset(last=True)
        dataloader = torch.utils.data.DataLoader(tested_dataset, batch_size=batch_size, shuffle=True)

        for ep in range(1, epochs + 1):
            loss = 0.0
            log_steps = 100
            for cnt, (student_ids, question_ids, concepts_emb, labels) in enumerate(dataloader):
                student_ids = student_ids.to(device)
                question_ids = question_ids.to(device)
                labels = labels.to(device)
                concepts_emb = concepts_emb.to(device)
                pred = self.model(student_ids, question_ids, concepts_emb)
                bz_loss = self._loss_function(pred, labels)
                optimizer.zero_grad()
                bz_loss.backward()
                optimizer.step()
                self.model.apply_clipper()
                loss += bz_loss.data.float()
                # if cnt % log_steps == 0:
                    # print('Epoch [{}] Batch [{}]: loss={:.3f}'.format(ep, cnt, loss / cnt))

    def evaluate(self, adaptest_data: AdapTestDataset):
        data = adaptest_data.data
        concept_map = adaptest_data.concept_map
        device = self.config['device']

        real = []
        pred = []
        with torch.no_grad():
            self.model.eval()
            for sid in data:
                student_ids = [sid] * len(data[sid])
                question_ids = list(data[sid].keys())
                concepts_embs = []
                for qid in question_ids:
                    concepts = concept_map[qid]
                    concepts_emb = [0.] * adaptest_data.num_concepts
                    for concept in concepts:
                        concepts_emb[concept] = 1.0
                    concepts_embs.append(concepts_emb)
                real += [data[sid][qid] for qid in question_ids]
                student_ids = torch.LongTensor(student_ids).to(device)
                question_ids = torch.LongTensor(question_ids).to(device)
                concepts_embs = torch.Tensor(concepts_embs).to(device)
                output = self.model(student_ids, question_ids, concepts_embs).view(-1)
                pred += output.tolist()
            self.model.train()

        coverages = []
        for sid in data:
            all_concepts = set()
            tested_concepts = set()
            for qid in data[sid]:
                all_concepts.update(set(concept_map[qid]))
            for qid in adaptest_data.tested[sid]:
                tested_concepts.update(set(concept_map[qid]))
            coverage = len(tested_concepts) / len(all_concepts)
            coverages.append(coverage)
        cov = sum(coverages) / len(coverages)

        real = np.array(real)
        real = np.where(real < 0.5, 0.0, 1.0)
        pred = np.array(pred)
        auc = roc_auc_score(real, pred)
        
        # Calculate accuracy
        threshold = 0.5  # You may adjust the threshold based on your use case
        binary_pred = (pred >= threshold).astype(int)
        acc = accuracy_score(real, binary_pred)

        return {
            'auc': auc,
            'cov': cov,
            'acc': acc
        }
    
    def get_pred(self, adaptest_data: AdapTestDataset):
        data = adaptest_data.data
        concept_map = adaptest_data.concept_map
        device = self.config['device']

        pred_all = {}
        with torch.no_grad():
            self.model.eval()
            for sid in data:
                pred_all[sid] = {}
                student_ids = [sid] * len(data[sid])
                question_ids = list(data[sid].keys())
                concepts_embs = []
                for qid in question_ids:
                    concepts = concept_map[qid]
                    concepts_emb = [0.] * adaptest_data.num_concepts
                    for concept in concepts:
                        concepts_emb[concept] = 1.0
                    concepts_embs.append(concepts_emb)
                student_ids = torch.LongTensor(student_ids).to(device)
                question_ids = torch.LongTensor(question_ids).to(device)
                concepts_embs = torch.Tensor(concepts_embs).to(device)
                output = self.model(student_ids, question_ids, concepts_embs).view(-1).tolist()
                for i, qid in enumerate(list(data[sid].keys())):
                    pred_all[sid][qid] = output[i]
            self.model.train()
        return pred_all

    def expected_model_change(self, sid: int, qid: int, adaptest_data: AdapTestDataset, pred_all: dict):
        """ get expected model change
        Args:
            student_id: int, student id
            question_id: int, question id
        Returns:
            float, expected model change
        """
        epochs = self.config['num_epochs']
        lr = self.config['learning_rate']
        device = self.config['device']
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for name, param in self.model.named_parameters():
            if 'student' not in name:
                param.requires_grad = False

        original_weights = self.model.student_emb.weight.data.clone()

        student_id = torch.LongTensor([sid]).to(device)
        question_id = torch.LongTensor([qid]).to(device)
        concepts = adaptest_data.concept_map[qid]
        concepts_emb = [0.] * adaptest_data.num_concepts
        for concept in concepts:
            concepts_emb[concept] = 1.0
        concepts_emb = torch.Tensor([concepts_emb]).to(device)
        correct = torch.LongTensor([1]).to(device)
        wrong = torch.LongTensor([0]).to(device)

        for ep in range(epochs):
            optimizer.zero_grad()
            pred = self.model(student_id, question_id, concepts_emb)
            loss = self._loss_function(pred, correct)
            loss.backward()
            optimizer.step()

        pos_weights = self.model.student_emb.weight.data.clone()
        self.model.student_emb.weight.data.copy_(original_weights)

        for ep in range(epochs):
            optimizer.zero_grad()
            pred = self.model(student_id, question_id, concepts_emb)
            loss = self._loss_function(pred, wrong)
            loss.backward()
            optimizer.step()

        neg_weights = self.model.student_emb.weight.data.clone()
        self.model.student_emb.weight.data.copy_(original_weights)

        for param in self.model.parameters():
            param.requires_grad = True

        # pred = self.model(student_id, question_id, concepts_emb).item()
        pred = pred_all[sid][qid]
        return pred * torch.norm(pos_weights - original_weights).item() + \
               (1 - pred) * torch.norm(neg_weights - original_weights).item()
               
    def get_BE_weights(self, pred_all):
        """
        Returns:
            predictions, dict[sid][qid]
        """
        d = 100
        Pre_true={}
        Pre_false={}
        for qid, pred in pred_all.items():
            Pre_true[qid] = pred
            Pre_false[qid] = 1 - pred
        w_ij_matrix={}
        for i ,_ in pred_all.items():
            w_ij_matrix[i] = {}
            for j,_ in pred_all.items(): 
                w_ij_matrix[i][j] = 0
        for i,_ in pred_all.items():
            for j,_ in pred_all.items():
                criterion_true_1 = nn.BCELoss()  # Binary Cross-Entropy Loss for loss(predict_true, 1)
                criterion_false_1 = nn.BCELoss()  # Binary Cross-Entropy Loss for loss(predict_false, 1)
                criterion_true_0 = nn.BCELoss()  # Binary Cross-Entropy Loss for loss(predict_true, 0)
                criterion_false_0 = nn.BCELoss()  # Binary Cross-Entropy Loss for loss(predict_false, 0)
                tensor_11=torch.tensor(Pre_true[i],requires_grad=True)
                tensor_12=torch.tensor(Pre_true[j],requires_grad=True)
                loss_true_1 = criterion_true_1(tensor_11, torch.tensor(1.0))
                loss_false_1 = criterion_false_1(tensor_11, torch.tensor(0.0))
                loss_true_0 = criterion_true_0(tensor_12, torch.tensor(1.0))
                loss_false_0 = criterion_false_0(tensor_12, torch.tensor(0.0))
                loss_true_1.backward()
                grad_true_1 = tensor_11.grad.clone()
                tensor_11.grad.zero_()
                loss_false_1.backward()
                grad_false_1 = tensor_11.grad.clone()
                tensor_11.grad.zero_()
                loss_true_0.backward()
                grad_true_0 = tensor_12.grad.clone()
                tensor_12.grad.zero_()
                loss_false_0.backward()
                grad_false_0 = tensor_12.grad.clone()
                tensor_12.grad.zero_()
                diff_norm_00 = math.fabs(grad_true_1 - grad_true_0)
                diff_norm_01 = math.fabs(grad_true_1 - grad_false_0)
                diff_norm_10 = math.fabs(grad_false_1 - grad_true_0)
                diff_norm_11 = math.fabs(grad_false_1 - grad_false_0)
                Expect = Pre_false[i]*Pre_false[j]*diff_norm_00 + Pre_false[i]*Pre_true[j]*diff_norm_01 +Pre_true[i]*Pre_false[j]*diff_norm_10 + Pre_true[i]*Pre_true[j]*diff_norm_11
                w_ij_matrix[i][j] = d - Expect
        return w_ij_matrix

    def F_s_func(self,S_set,w_ij_matrix):
        res = 0.0
        for w_i in w_ij_matrix:
            if(w_i not in S_set):
                mx = float('-inf')
                for j in S_set:
                    if w_ij_matrix[w_i][j] > mx:
                        mx = w_ij_matrix[w_i][j]
                res +=mx
                
        return res

    def delta_q_S_t(self, question_id, pred_all,S_set,sampled_elements):
        """ get BECAT Questions weights delta
        Args:
            student_id: int, student id
            question_id: int, question id
        Returns:
            v: float, Each weight information
        """     
        
        Sp_set = list(S_set)
        b_array = np.array(Sp_set)
        sampled_elements = np.concatenate((sampled_elements, b_array), axis=0)
        if question_id not in sampled_elements:
            sampled_elements = np.append(sampled_elements, question_id)
        sampled_dict = {key: value for key, value in pred_all.items() if key in sampled_elements}
        
        w_ij_matrix = self.get_BE_weights(sampled_dict)
        
        F_s = self.F_s_func(Sp_set,w_ij_matrix)
        
        Sp_set.append(question_id)
        F_sp =self.F_s_func(Sp_set,w_ij_matrix)
        return F_sp - F_s