import math
from collections import Counter
from math import exp as exp
from sklearn.metrics import roc_auc_score,accuracy_score
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy as cp
import numpy as np
from scipy.optimize import minimize
from CAT.dataset import AdapTestDataset,Dataset
from CAT.model.IRT import IRT,IRTModel
import os

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

def tensor_to_numpy(tensor):

    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().detach().numpy()
    else:
        return tensor
    
def decay_function1(x):
    x = 50+x
    return max(2.0/(1+np.power(x,0.2)),0.001)

def decay_function(x,config):
    start = config['start']
    end = config['end']
    START = decay_function1(start)
    END = decay_function1(end)
    x = max(min(end,x),start)
    return (decay_function1(x)-END)/(START-END+0.0000001)

class NCAT(nn.Module):
    def __init__(self, n_question, d_model=10, n_blocks=1,kq_same=True, dropout=0.0, policy_fc_dim=512, n_heads=1, d_ff=2048,  l2=1e-5, separate_qa=None, pad=0):
        super().__init__()
        """
        Input:
            d_model: question emb and dimension of attention block 
            final_fc_dim: dimension of final fully connected net before prediction
            n_heads: number of heads in multi-headed attention
            d_ff : dimension for fully conntected net inside the basic block
        """
        self.device = torch.device('cpu')
        self.pad = pad
        self.n_question = n_question
        self.dropout = dropout
        self.kq_same = kq_same
        self.l2 = l2
        self.separate_qa = separate_qa
        embed_l = d_model
        self.q_embed_0 = nn.Embedding(self.n_question, embed_l) # 两个通道是否用相同embedding
        self.q_embed_1 = nn.Embedding(self.n_question, embed_l)
        self.contradiction = MultiHeadedAttention_con(n_heads, d_model)
        self.con_ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        c = cp.deepcopy
        attn = MultiHeadedAttention(n_heads, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.self_atten_0 = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), n_blocks)
        self.self_atten_1 = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), n_blocks)
        
        self.policy_layer = nn.Sequential(
            nn.Linear(d_model * 4, policy_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(policy_fc_dim, n_question)
        )
        for name, param in self.named_parameters():
            # print(name)
            if param.dim() > 1:
                nn.init.xavier_normal_(param)

    def forward(self, p_0_rec, p_1_rec, p_0_target, p_1_target):
        # embedding layer
        bs = len(p_0_rec)
        item_emb_0 = self.q_embed_0(p_0_rec) # C
        item_emb_1 = self.q_embed_1(p_1_rec) # K
        src_mask_0 = mask(p_0_rec, p_0_target + 1).unsqueeze(-2)
        src_mask_1 = mask(p_1_rec, p_1_target + 1).unsqueeze(-2)
        item_per_0 = self.self_atten_0(item_emb_0, src_mask_0) # bs len emb_dim
        # print(item_per_0.shape)
        item_per_1 = self.self_atten_1(item_emb_1, src_mask_1)
        # contradiction learning
        input_01, input_10 = self.contradiction(item_emb_0, item_emb_1, item_per_1, item_per_0)
        # print('???', input_01.shape)
        input_01, input_10 = input_01.mean(-2), input_10.mean(-2)
        
        # cat 
        input_0 = item_per_0[torch.arange(bs), p_0_target]
        input_1 = item_per_1[torch.arange(bs), p_1_target] # bs dim
        input_emb = torch.cat([input_0, input_1, input_01, input_10], dim=-1)
        # policy layer
        output_value = self.policy_layer(input_emb) # bs n_item

        return output_value

    def predict(self, data):
        self.eval()
        with torch.no_grad():
            p_0_rec, p_1_rec, p_0_target, p_1_target = \
                                    data['p_0_rec'], data['p_1_rec'],data['p_0_t'],data['p_1_t']
            
            p_0_rec, p_1_rec, p_0_target, p_1_target = \
                                    torch.LongTensor(p_0_rec).to(self.device), \
                                    torch.LongTensor(p_1_rec).to(self.device), torch.LongTensor(p_0_target).to(self.device), \
                                    torch.LongTensor(p_1_target).to(self.device)
            policy = self.forward(p_0_rec, p_1_rec, p_0_target, p_1_target)
            policy = tensor_to_numpy(policy)
        return policy
    
    def optimize_model(self, data, lr):
        self.train()
        p_0_rec, p_1_rec, p_0_target, p_1_target, target, goal = \
                                data['p_0_rec'], data['p_1_rec'],data['p_0_t'],data['p_1_t'], data['iid'], data['goal']
        
        p_0_rec, p_1_rec, p_0_target, p_1_target, target, goal = \
                                torch.LongTensor(p_0_rec).to(self.device), \
                                torch.LongTensor(p_1_rec).to(self.device), torch.LongTensor(p_0_target).to(self.device), \
                                torch.LongTensor(p_1_target).to(self.device), torch.LongTensor(target).to(self.device), \
                                torch.FloatTensor(goal).to(self.device)
        op = optim.Adam(self.parameters(), lr=lr)
        op.zero_grad()
        policy = self.forward(p_0_rec, p_1_rec, p_0_target, p_1_target)
        pre_value = policy[torch.arange(len(p_0_rec)), target]
        loss_func = torch.nn.MSELoss(reduction='mean')  
        loss = loss_func(pre_value, goal)
        loss.backward()
        op.step()
        # loss = tensor_to_numpy(loss)
        return tensor_to_numpy(loss)

    @classmethod
    def create_model(cls, config):
        model = cls(config.item_num, config.latent_factor, config.num_blocks,
                 True, config.dropout_rate, policy_fc_dim=512, n_heads=config.num_heads, d_ff=2048,  l2=1e-5, separate_qa=None, pad=0)
        return model.to(torch.device('cpu'))


def mask(src, s_len):
    if type(src) == torch.Tensor:
        mask = torch.zeros_like(src)
        for i in range(len(src)):
            mask[i, :s_len[i]] = 1

    return mask

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([cp.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    output = torch.matmul(p_attn, value)
    return scores, output, p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch. 
        _, x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class MultiHeadedAttention_con(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention_con, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 5)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value1, value2, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key))]

        value1 = value1.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        value2 = value2.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        # 2) Apply attention on all the projected vectors in batch. 
        
        # print(query.shape, key.shape, value1.shape)
        scores, x1, attn_s = attention(query, key, value1, mask=mask, 
                                 dropout=self.dropout)

        x2 = torch.matmul(attn_s.transpose(-1, -2), value2)
        
        # 3) "Concat" using a view and apply a final linear. 
        x1 = x1.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        
        x2 = x2.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x1), self.linears[-1](x2), 

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
    
class env:
    def __init__(self,data,concept_map,config,T):
        self.config = config
        self.T = T
        self.CDM = 'IRT'
        self.rates = {}
        self.users = {}
        self.utypes = {}
        #self.args = args
        self.device = torch.device('cpu')
        self.rates, self._item_num, self.know_map = self.load_data(data,concept_map)
        self.tsdata=data
        self.setup_train_test()
        self.sup_rates, self.query_rates = self.split_data(ratio=0.5)
        pth_path='../ckpt/irt.pt'
        name = 'IRT'
        self.model, self.dataset = self.load_CDM(name,data,pth_path,config)
        #print(self.model)
    
    def split_data(self, ratio=0.5):
        sup_rates, query_rates = {}, {}
        for u in self.rates:
            all_items = list(self.rates[u].keys())
            np.random.shuffle(all_items)
            sup_rates[u] = {it: self.rates[u][it] for it in all_items[:int(ratio*len(all_items))]}
            query_rates[u] = {it: self.rates[u][it] for it in all_items[int(ratio*len(all_items)):]}
        return sup_rates, query_rates

    def re_split_data(self, ratio=0.5):
        self.sup_rates, self.query_rates = self.split_data(ratio)

    @property
    def candidate_items(self):
        return set(self.sup_rates[self.state[0][0]].keys())

    @property
    def user_num(self):
        return len(self.rates) + 1

    @property
    def item_num(self):
        return self._item_num + 1

    @property
    def utype_num(self):
        return len(self.utypes) + 1

    def setup_train_test(self):
        users = list(range(1, self.user_num))
        np.random.shuffle(users)
        self.training, self.validation, self.evaluation = np.split(np.asarray(users), [int(.8 * self.user_num - 1),
                                                                                       int(.9 * self.user_num - 1)])

    def reset(self):
        self.reset_with_users(np.random.choice(self.training))

    def reset_with_users(self, uid):
        self.state = [(uid,1), []]
        self.short = {}
        return self.state
    
    def load_data(self, ncatdata,concept):
        return ncatdata.data, ncatdata.num_questions, concept
    
    def load_CDM(self,name,data,pth_path,config):
        if name == 'IRT':
            model = IRTModel(**config)
            model.init_model(data)
            model.adaptest_load(pth_path)
            #model.to(self.config['device'])
        return model ,data.data
    
    def step(self, action,sid):
        assert action in self.sup_rates[self.state[0][0]] and action not in self.short
        reward, acc, auc, rate = self.reward(action,sid)
        if len(self.state[1]) < self.T - 1:
            done = False
        else:
            done = True
        self.short[action] = 1
        t = self.state[1] + [[action, reward, done]]
        info = {"ACC": acc,
                "AUC": auc,
                "rate":rate}
        self.state[1].append([action, reward, done, info])
        return self.state, reward, done, info

    def reward(self, action,sid):
        items = [state[0] for state in self.state[1]] + [action]
        
        correct = [self.rates[self.state[0][0]][it] for it in items]
        self.tsdata.apply_selection(sid, action)
        loss = self.model.adaptest_update(self.tsdata,sid)
        result=self.model.evaluate(self.tsdata)
        auc = result['auc']
        acc = result['acc']
        return -loss, acc, auc, correct[-1]

    def precision(self, episode):
        return sum([i[1] for i in episode])


    def recall(self, episode, uid):
        return sum([i[1] for i in episode]) / len(self.rates[uid])

    def step_policy(self,policy):
        policy = policy[:self.args.T]
        rewards = []
        for action in policy:
            if action in self.rates[self.state[0][0]]:
                rewards.append(self.rates[self.state[0][0]][action])
            else:
                rewards.append(0)
        t = [[a,rewards[i],False] for i,a in enumerate(policy)]
        info = {"precision": self.precision(t),
                "recall": self.recall(t, self.state[0][0])}
        self.state[1].extend(t)
        return self.state,rewards,True,info

    def ndcg(self, episode, uid):
        if len(self.rates[uid]) > len(episode):
            return self.dcg_at_k(list(map(lambda x: x[1], episode)),
                                 len(episode),
                                 method=1) / self.dcg_at_k(sorted(list(self.rates[uid].values()),reverse=True),
                                                           len(episode),
                                                           method=1)
        else:
            return self.dcg_at_k(list(map(lambda x: x[1], episode)),
                                 len(episode),
                                 method=1) / self.dcg_at_k(
                list(self.rates[uid].values()) + [0] * (len(episode) - len(self.rates[uid])),
                len(episode), method=1)

    def dcg_at_k(self, r, k, method=1):
        r = np.asfarray(r)[:k]
        if r.size:
            if method == 0:
                return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
            elif method == 1:
                return np.sum(r / np.log2(np.arange(2, r.size + 2)))
            else:
                raise ValueError('method must be 0 or 1.')

    def alpha_dcg(self, item_list, k=10, alpha=0.5, *args):
        items = []
        G = []
        for i, item in enumerate(item_list[:k]):
            items += item
            G.append(sum(map(lambda x: math.pow(alpha, x - 1), dict(Counter(items)).values())) / math.log(i + 2, 2))
        return sum(G)
    
class NCATModel():

    def __init__(self, NCATdata,concept_map,config,test_length):
        super().__init__()
        self.config = config
        self.model = None
        self.env = env(data=NCATdata,concept_map=concept_map,config=config,T=test_length)
        self.memory = []
        self.item_num =self.env.item_num
        self.user_num = self.env.user_num
        self.device = config['device']
        self.fa = NCAT(n_question=NCATdata.num_questions+1).to(self.device)
        self.memory_size = 50000
        self.tau = 0

    def ncat_policy(self,sid,THRESHOLD,used_actions,type,epoch):
        actions = {}
        rwds = 0
        done = False
        state = self.env.reset_with_users(sid)
        while not done:
            data = {"uid": [state[0][1]]}
            for i in range(2):
                p_r, pnt = self.convert_item_seq2matrix([[0]+[item[0] for item in state[1] if item[3]["rate"] == i]])
                data["p_"+str(i)+"_rec"] = p_r
                data["p_"+str(i)+"_t"] = pnt
            data["uid"] = torch.tensor(data["uid"], device=self.device)
            policy = self.fa.predict(data)[0]
            if type == "training":
                if np.random.random()<5*THRESHOLD/(THRESHOLD+self.tau): policy = np.random.uniform(0,1,(self.item_num,))
            for item in actions: policy[item] = -np.inf
            for item in range(self.item_num):
                if item not in self.env.candidate_items:
                    policy[item] = -np.inf
            action = np.argmax(policy[1:]) + 1
            s_pre = cp.deepcopy(state)
            state_next, rwd, done, info = self.env.step(action,sid)
            if type == "training":
                self.memory.append([s_pre,action,rwd,done,cp.deepcopy(state_next)])

            actions[action] = 1
            rwds += rwd
            state = state_next
        used_actions.extend(list(actions.keys()))
        if type == "training":
            if len(self.memory) >= self.config['batch_size']:
                self.memory = self.memory[-self.memory_size:]
                batch = [self.memory[item] for item in np.random.choice(range(len(self.memory)),(self.args.batch,))]
                data = self.convert_batch2dict(batch,epoch)
                loss = self.fa.optimize_model(data, self.args.learning_rate)
                self.tau += 5
        return 
    
    def convert_batch2dict(self, batch, epoch):
        uids = []
        pos_recs = {i:[] for i in range(2)}
        next_pos = {i:[] for i in range(2)}
        iids = []
        goals = []
        dones = []
        for item in batch:
            uids.append(item[0][0][1])
            ep = item[0][1] # 
            for xxx in range(2):
                pos_recs[xxx].append([0] + [j[0] for j in ep if j[3]["rate"]==xxx])
            iids.append(item[1]) # action  
            goals.append(item[2])
            if item[3]:dones.append(0.0)
            else:dones.append(1.0)
            ep = item[4][1] 
            for xxx in range(2):
                next_pos[xxx].append([0] + [j[0] for j in ep if j[3]["rate"] == xxx])
        data = {"uid":uids}
        for xxx in range(2):
            p_r, pnt = self.convert_item_seq2matrix(next_pos[xxx])
            data["p_" + str(xxx) + "_rec"] = p_r
            data["p_" + str(xxx) + "_t"] = pnt
        value = self.fa.predict(data)
        value[:,0] = -500
        goals = np.max(value,axis=-1)*np.asarray(dones)*min(self.args.gamma,decay_function(max(self.config['end']-epoch,0)+1),self.config) + goals
        data = {"uid":uids,"iid":iids,"goal":goals}
        for i in range(2):
            p_r, pnt = self.convert_item_seq2matrix(pos_recs[i])
            data["p_" + str(i) + "_rec"] = p_r
            data["p_" + str(i) + "_t"] = pnt
        return data

    def convert_item_seq2matrix(self, item_seq):
        max_length = max([len(item) for item in item_seq])
        matrix = np.zeros((len(item_seq), max_length), dtype=np.int32)
        for x, xx in enumerate(item_seq):
            for y, yy in enumerate(xx):
                matrix[x, y] = yy
        target_index = [len(i) - 1 for i in item_seq]
        return matrix, target_index