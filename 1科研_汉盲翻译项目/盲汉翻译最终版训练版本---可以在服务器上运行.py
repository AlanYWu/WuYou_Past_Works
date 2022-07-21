import torch
import torch.nn as nn
# 构造RNN网络，x的维度5，隐层的维度10,网络的层数2
rnn_seq = nn.RNN(5, 10,2)  
# 构造一个输入序列，句长为 6，batch 是 3， 每个单词使用长度是 5的向量表示
x = torch.randn(6, 3, 5)
#out,ht = rnn_seq(x,h0) 
out,ht = rnn_seq(x) #h0可以指定或者不指定
# 输入维度 50，隐层100维，两层
lstm_seq = nn.LSTM(50, 100, num_layers=2)
# 输入序列seq= 10，batch =3，输入维度=50
lstm_input = torch.randn(10, 3, 50)
out, (h, c) = lstm_seq(lstm_input) # 使用默认的全 0 隐藏状态

class RNN(torch.nn.Module):
    def __init__(self,input_size, hidden_size, num_layers):
        super(RNN,self).__init__()
        self.input_size = input_size
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.lstm = torch.nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,batch_first=True)
    
    def forward(self,input):
        # input应该为(batch_size,seq_len,input_szie)
        self.hidden = self.initHidden(input.size(0))
        out,self.hidden = self.lstm(input,self.hidden)#lstm->self.lstm
        return out,self.hidden
    
    def initHidden(self,batch_size):
        if self.lstm.bidirectional:
            return (torch.rand(self.num_layers*2,batch_size,self.hidden_size),torch.rand(self.num_layers*2,batch_size,self.hidden_size))
        else:
            return (torch.rand(self.num_layers,batch_size,self.hidden_size),torch.rand(self.num_layers,batch_size,self.hidden_size))

input_size = 12
hidden_size = 10
num_layers = 3
batch_size = 2
model = RNN(input_size,hidden_size,num_layers)
# input (seq_len, batch, input_size) 包含特征的输入序列，如果设置了batch_first，则batch为第一维
input = torch.rand(2,4,12)    #生成一个用随机数初始化的矩阵
model(input)
import torch.nn.functional as F
class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # LSTM以word_embeddings作为输入, 输出维度为 hidden_dim 的隐藏状态值
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # 线性层将隐藏状态空间映射到标注空间
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # 一开始并没有隐藏状态所以我们要先初始化一个
        # 关于维度为什么这么设计请参考Pytoch相关文档
        # 各个维度的含义是 (num_layers*num_directions, batch_size, hidden_dim)
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

EMBEDDING_DIM = 6
HIDDEN_DIM = 6
#Making training data
training_data = []
t = open('./对应拼音标调.txt',"r")
f = open("./训练集.txt", "r")
#计数 数一共有多少行
lines = f.readlines()
tags = t.readlines()
# counter = len(lines)
# counter = 0
# for i in line:   
#     counter = counter + 1

for index, line in enumerate(lines):
    line = lines[index].rstrip().split()
    tag = tags[index].rstrip().split()
    training_data.append((line, tag))
    # f = open("./训练集.txt", "r",encoding="ANSI")
    # line = f.readline().split()
    # index = t.readline().split()
    # for i in range(len(line)):
    #     temp=[]
    #     temp.append(line[i])
    #     temp.append(tag[i])
    #     temp = tuple(temp)
    #     training_data.append(temp)
'''
training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
'''

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

word_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
# print(word_to_ix)
# tag_to_ix = {"DET": 0, "NN": 1, "V": 2}
tag_to_ix = {"0":0, "1":1, "2":2, "3":3, "4":4, ",":5, ".":6}

# 实际中通常使用更大的维度如32维, 64维.
# 这里我们使用小的维度, 为了方便查看训练过程中权重的变化.
EMBEDDING_DIM = 6
HIDDEN_DIM = 6
# training_data为[(data,tag),…]
from tqdm import tqdm
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 查看训练前的分数
# 注意: 输出的 i,j 元素的值表示单词 i 的 j 标签的得分
# 这里我们不需要训练不需要求导所以使用torch.no_grad()
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    # print(training_data[0][0], word_to_ix)
    
    tag_scores = model(inputs)
    #print(tag_scores)
    targets = prepare_sequence(training_data[0][1], tag_to_ix)
    #print((torch.argmax(tag_scores, dim=-1) == targets).sum().item() / len(targets))


for epoch in range(8):  # 实际情况下你不会训练300个周期, 此例中我们只是随便设了一个值
    print(epoch)
    training_acuracy = []
    validation_acuracy = []
    for sentence, tags in tqdm(training_data[:5000]):
        # 第一步: 请记住Pytorch会累加梯度.
        # 我们需要在训练每个实例前清空梯度
        model.zero_grad()

        # 此外还需要清空 LSTM 的隐状态,
        # 将其从上个实例的历史中分离出来.
        model.hidden = model.init_hidden()

        # 准备网络输入, 将其变为词索引的 Tensor 类型数据
        
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)
        # 第三步: 前向传播.
        tag_scores = model(sentence_in)

        # 第四步: 计算损失和梯度值, 通过调用 optimizer.step() 来更新梯度
        loss = loss_function(tag_scores, targets)
        a = (torch.argmax(tag_scores, dim=-1) == targets).sum().item() / len(targets)
        training_acuracy.append(a)
        loss.backward()
        optimizer.step()
    #Validation
    for sentence, tags in tqdm(training_data[5000:6000]):
        model.zero_grad()
        model.hidden = model.init_hidden()

        # 准备网络输入, 将其变为词索引的 Tensor 类型数据
        
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)
        # 第三步: 前向传播.
        tag_scores = model(sentence_in)
        #计算损失
        a = (torch.argmax(tag_scores, dim=-1) == targets).sum().item() / len(targets)
        validation_acuracy.append(a)

        
    #LSTM 可以做padding
    print(loss)
    #print(torch.argmax(tag_scores, dim=-1))
    print("training_acuracy",sum(training_acuracy)/len(training_acuracy))
    print("validation_acuracy:",sum(validation_acuracy)/len(validation_acuracy))

# 查看训练后的得分
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)

    # 句子是 "the dog ate the apple", i,j 表示对于单词 i, 标签 j 的得分.
    # 我们采用得分最高的标签作为预测的标签. 从下面的输出我们可以看到, 预测得
    # 到的结果是0 1 2 0 1. 因为 索引是从0开始的, 因此第一个值0表示第一行的
    # 最大值, 第二个值1表示第二行的最大值, 以此类推. 所以最后的结果是 DET
    # NOUN VERB DET NOUN, 整个序列都是正确的!
    print(tag_scores)
