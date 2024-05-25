
import vegas
import logging
import torch
import torch.nn as nn
import numpy as np
import math
import torch.utils.data as data
from math import exp as exp
from sklearn.metrics import roc_auc_score
from scipy import integrate
from CAT.model.abstract_model import AbstractModel
from CAT.dataset import AdapTestDataset, TrainDataset, Dataset
from sklearn.metrics import accuracy_score
from collections import namedtuple
from .utils import StraightThrough
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class IRT(nn.Module):
    def __init__(self, num_students, num_questions, num_dim):
        # num_dim: IRT if num_dim == 1 else MIRT
        super().__init__()
        self.num_dim = num_dim
        self.num_students = num_students
        self.num_questions = num_questions
        self.theta = nn.Embedding(self.num_students, self.num_dim)
        self.alpha = nn.Embedding(self.num_questions, self.num_dim)
        self.beta = nn.Embedding(self.num_questions, 1)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, student_ids, question_ids):
        theta = self.theta(student_ids)
        alpha = self.alpha(question_ids)
        beta = self.beta(question_ids)
        pred = (alpha * theta).sum(dim=1, keepdim=True) + beta
        pred = torch.sigmoid(pred)
        return pred

class IRTModel(AbstractModel):

    def __init__(self, **config):
        super().__init__()
        self.config = config
        self.model = None

    @property
    def name(self):
        return 'Item Response Theory'

    def init_model(self, data: Dataset):
        policy_lr=0.0005
        self.model = IRT(data.num_students, data.num_questions, self.config['num_dim'])
        self.policy = StraightThrough(data.num_questions, data.num_questions,policy_lr, self.config)
        self.n_q = data.num_questions

    def train(self, train_data: TrainDataset, log_step=1):
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
            for cnt, (student_ids, question_ids, _, labels) in enumerate(train_loader):
                student_ids = student_ids.to(device)
                question_ids = question_ids.to(device)
                labels = labels.to(device).float()
                pred = self.model(student_ids, question_ids).view(-1)
                bz_loss = self._loss_function(pred, labels)
                optimizer.zero_grad()
                bz_loss.backward()
                optimizer.step()
                loss += bz_loss.data.float()
                if cnt % log_step == 0:
                    logging.info('Epoch [{}] Batch [{}]: loss={:.5f}'.format(ep, cnt, loss / cnt))

    def adaptest_save(self, path):
        """
        Save the model. Only save the parameters of questions(alpha, beta)
        """
        model_dict = self.model.state_dict()
        model_dict = {k:v for k,v in model_dict.items() if 'alpha' in k or 'beta' in k}
        torch.save(model_dict, path)

    def adaptest_load(self, path):
        """
        Reload the saved model
        """
        if self.config['policy'] =='bobcat':
            self.policy.policy.load_state_dict(torch.load(self.config['policy_path']),strict=False)
        self.model.load_state_dict(torch.load(path), strict=False)
        self.model.to(self.config['device'])

    def adaptest_update(self, adaptest_data: AdapTestDataset,sid=None):
        """
        Update CDM with tested data
        """
        lr = self.config['learning_rate']
        batch_size = self.config['batch_size']
        epochs = self.config['num_epochs']
        device = self.config['device']
        optimizer = torch.optim.Adam(self.model.theta.parameters(), lr=lr)

        tested_dataset = adaptest_data.get_tested_dataset(last=True,ssid=sid)
        dataloader = torch.utils.data.DataLoader(tested_dataset, batch_size=batch_size, shuffle=True)
        for ep in range(1, epochs + 1):
            loss = 0.0
            log_steps = 100
            for cnt, (student_ids, question_ids, _, labels) in enumerate(dataloader):
                student_ids = student_ids.to(device)
                question_ids = question_ids.to(device)
                labels = labels.to(device).float()
                pred = self.model(student_ids, question_ids).view(-1)
                bz_loss = self._loss_function(pred, labels)
                optimizer.zero_grad()
                bz_loss.backward()
                optimizer.step()
                loss += bz_loss.data.float()
                # if cnt % log_steps == 0:
                    # print('Epoch [{}] Batch [{}]: loss={:.3f}'.format(ep, cnt, loss / cnt))
        return loss
    def one_student_update(self, adaptest_data: AdapTestDataset):
        lr = self.config['learning_rate']
        batch_size = self.config['batch_size']
        epochs = self.config['num_epochs']
        device = self.config['device']
        optimizer = torch.optim.Adam(self.model.theta.parameters(), lr=lr)
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
                real += [data[sid][qid] for qid in question_ids]
                student_ids = torch.LongTensor(student_ids).to(device)
                question_ids = torch.LongTensor(question_ids).to(device)
                output = self.model(student_ids, question_ids).view(-1)
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
        """
        Returns:
            predictions, dict[sid][qid]
        """
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
                student_ids = torch.LongTensor(student_ids).to(device)
                question_ids = torch.LongTensor(question_ids).to(device)
                output = self.model(student_ids, question_ids).view(-1).tolist()
                for i, qid in enumerate(list(data[sid].keys())):
                    pred_all[sid][qid] = output[i]
            self.model.train()

        return pred_all

    def _loss_function(self, pred, real):
        return -(real * torch.log(0.0001 + pred) + (1 - real) * torch.log(1.0001 - pred)).mean()
    
    def get_alpha(self, question_id):
        """ get alpha of one question
        Args:
            question_id: int, question id
        Returns:
            alpha of the given question, shape (num_dim, )
        """
        return self.model.alpha.weight.data.cpu().numpy()[question_id]
    
    def get_beta(self, question_id):
        """ get beta of one question
        Args:
            question_id: int, question id
        Returns:
            beta of the given question, shape (1, )
        """
        return self.model.beta.weight.data.cpu().numpy()[question_id]
    
    def get_theta(self, student_id):
        """ get theta of one student
        Args:
            student_id: int, student id
        Returns:
            theta of the given student, shape (num_dim, )
        """
        return self.model.theta.weight.data.cpu().numpy()[student_id]

    def get_kli(self, student_id, question_id, n, pred_all):
        """ get KL information
        Args:
            student_id: int, student id
            question_id: int, question id
            n: int, the number of iteration
        Returns:
            v: float, KL information
        """
        if n == 0:
            return np.inf
        device = self.config['device']
        dim = self.model.num_dim
        sid = torch.LongTensor([student_id]).to(device)
        qid = torch.LongTensor([question_id]).to(device)
        theta = self.get_theta(sid) # (num_dim, )
        alpha = self.get_alpha(qid) # (num_dim, )
        beta = self.get_beta(qid)[0] # float value
        pred_estimate = pred_all[student_id][question_id]
        def kli(x):
            """ The formula of KL information. Used for integral.
            Args:
                x: theta of student sid
            """
            if type(x) == float:
                x = np.array([x])
            pred = np.matmul(alpha.T, x) + beta
            pred = 1 / (1 + np.exp(-pred))
            q_estimate = 1  - pred_estimate
            q = 1 - pred
            return pred_estimate * np.log(pred_estimate / pred) + \
                    q_estimate * np.log((q_estimate / q))
        c = 3
        boundaries = [[theta[i] - c / np.sqrt(n), theta[i] + c / np.sqrt(n)] for i in range(dim)]
        if len(boundaries) == 1:
            # KLI
            v, err = integrate.quad(kli, boundaries[0][0], boundaries[0][1])
            return v
        # MKLI
        integ = vegas.Integrator(boundaries)
        result = integ(kli, nitn=10, neval=1000)
        return result.mean

    def get_fisher(self, student_id, question_id, pred_all):
        """ get Fisher information
        Args:
            student_id: int, student id
            question_id: int, question id
        Returns:
            fisher_info: matrix(num_dim * num_dim), Fisher information
        """
        device = self.config['device']
        qid = torch.LongTensor([question_id]).to(device)
        alpha = self.model.alpha(qid).clone().detach().cpu()
        pred = pred_all[student_id][question_id]
        q = 1 - pred
        fisher_info = (q*pred*(alpha * alpha.T)).numpy()
        return fisher_info
    
    def bce_loss_derivative(self,pred, target):
        """ get bce_loss_derivative
        Args:
            pred: float,
            target: int,
        Returns:
            the derivative of bce_loss
        """
        derivative = (pred - target) / (pred * (1 - pred))
        return derivative
    
    def get_BE_weights(self, pred_all):
        """ get BE matrix
        Args:
            pred_all: dict, the questions you want to sample and their probability
        Returns:
            the BE matrix weights
        """
        d = 100
        Pre_true={}
        Pre_false={}
        Der={}
        for qid, pred in pred_all.items():
            Pre_true[qid] = pred
            Pre_false[qid] = 1 - pred
            Der[qid] =pred*(1-pred)*self.get_alpha(qid)
        w_ij_matrix={}
        for i ,_ in pred_all.items():
            w_ij_matrix[i] = {}
            for j,_ in pred_all.items(): 
                w_ij_matrix[i][j] = 0
        for i,_ in pred_all.items():
            for j,_ in pred_all.items():
                gradients_theta1 = self.bce_loss_derivative(Pre_true[i],1.0) * Der[i]
                gradients_theta2 = self.bce_loss_derivative(Pre_true[i],0.0) * Der[i]
                gradients_theta3 = self.bce_loss_derivative(Pre_true[j],1.0) * Der[j]
                gradients_theta4 = self.bce_loss_derivative(Pre_true[j],0.0) * Der[j]
                diff_norm_00 = math.fabs(gradients_theta1 - gradients_theta3)
                diff_norm_01 = math.fabs(gradients_theta1 - gradients_theta4)
                diff_norm_10 = math.fabs(gradients_theta2 - gradients_theta3)
                diff_norm_11 = math.fabs(gradients_theta2 - gradients_theta4)
                Expect = Pre_false[i]*Pre_false[j]*diff_norm_00 + Pre_false[i]*Pre_true[j]*diff_norm_01 +Pre_true[i]*Pre_false[j]*diff_norm_10 + Pre_true[i]*Pre_true[j]*diff_norm_11
                w_ij_matrix[i][j] = d - Expect
        return w_ij_matrix

    def F_s_func(self,S_set,w_ij_matrix):
        """ get F_s of the questions have been chosen
        Args:
            S_set:list , the questions have been chosen
            w_ij_matrix: dict, the weight matrix
        Returns:
            the F_s of the chosen questions
        """
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
            question_id: int, question id
            pred_all:dict, the untest questions and their probability
            S_set:dict, chosen questions
            sampled_elements:nparray, sampled set from untest questions
        Returns:
            delta_q: float, delta_q of questions id
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
            if 'theta' not in name:
                param.requires_grad = False

        original_weights = self.model.theta.weight.data.clone()

        student_id = torch.LongTensor([sid]).to(device)
        question_id = torch.LongTensor([qid]).to(device)
        correct = torch.LongTensor([1]).to(device).float()
        wrong = torch.LongTensor([0]).to(device).float()

        for ep in range(epochs):
            optimizer.zero_grad()
            pred = self.model(student_id, question_id)
            loss = self._loss_function(pred, correct)
            loss.backward()
            optimizer.step()

        pos_weights = self.model.theta.weight.data.clone()
        self.model.theta.weight.data.copy_(original_weights)

        for ep in range(epochs):
            optimizer.zero_grad()
            pred = self.model(student_id, question_id)
            loss = self._loss_function(pred, wrong)
            loss.backward()
            optimizer.step()

        neg_weights = self.model.theta.weight.data.clone()
        self.model.theta.weight.data.copy_(original_weights)

        for param in self.model.parameters():
            param.requires_grad = True

        pred = pred_all[sid][qid]
        return pred * torch.norm(pos_weights - original_weights).item() + \
               (1 - pred) * torch.norm(neg_weights - original_weights).item()
    
    def bobcat_policy(self,S_set,untested_questions):
        """ get expected model change
        Args:
            S_set:list , the questions have been chosen
            untested_questions: dict, untested_questions
        Returns:
            float, expected model change
        """
        device = self.config['device']
        action_mask = [0.0] * self.n_q
        train_mask=[-0.0]*self.n_q
        for index in untested_questions:
            action_mask[index] = 1.0
        for state in S_set:
            keys = list(state.keys())
            key = keys[0]
            values = list(state.values())
            val = values[0]
            train_mask[key] = (float(val)-0.5)*2 
        action_mask = torch.tensor(action_mask).to(device)
        train_mask = torch.tensor(train_mask).to(device)
        action = self.policy.policy(train_mask, action_mask)
        return action.item()


