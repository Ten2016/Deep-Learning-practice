# 导入相应的包
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
torch.manual_seed(1)
 
# 准备数据的阶段
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)
  
with open("/home/lstm_train.txt", encoding='utf8') as f:
    train_data = []
    word = []
    label = []
    data = f.readline().strip()
    while data:
        data = data.strip()
        SP = data.split(' ')
        if len(SP) == 2:
            word.append(SP[0])
            label.append(SP[1])
        else:
            if len(word) == 100 and 'I-PRO' in label:
                train_data.append((word, label))
            word = []
            label = []
        data = f.readline()
 
word_to_ix = {}
for sent, _ in train_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
 
tag_to_ix = {"O": 0, "I-PRO": 1}
for i in range(len(train_data)):
    train_data[i] = ([word_to_ix[t] for t in train_data[i][0]], [tag_to_ix[t] for t in train_data[i][1]])
 
# 词向量的维度
EMBEDDING_DIM = 128
 
# 隐藏层的单元数
HIDDEN_DIM = 128
 
# 批大小
batch_size = 10
class LSTMTagger(nn.Module):
 
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, batch_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
 
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
 
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
 
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        # input_tensor = embeds.view(self.batch_size, len(sentence) // self.batch_size, -1)
        lstm_out, _ = self.lstm(embeds)
        tag_space = self.hidden2tag(lstm_out)
        scores = F.log_softmax(tag_space, dim=2)
        return scores
 
    def predict(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_space = self.hidden2tag(lstm_out)
        scores = F.log_softmax(tag_space, dim=2)
        return scores
 
loss_function = nn.NLLLoss()
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix), batch_size)
optimizer = optim.SGD(model.parameters(), lr=0.1)
 
data_set_word = []
data_set_label = []
for data_tuple in train_data:
    data_set_word.append(data_tuple[0])
    data_set_label.append(data_tuple[1])
torch_dataset = Data.TensorDataset(torch.tensor(data_set_word, dtype=torch.long), torch.tensor(data_set_label, dtype=torch.long))
# 把 dataset 放入 DataLoader
loader = Data.DataLoader(
    dataset=torch_dataset,  # torch TensorDataset format
    batch_size=batch_size,  # mini batch size
    shuffle=True,  #
    num_workers=2,  # 多线程来读数据
)
 
# 训练过程
for epoch in range(200):
    for step, (batch_x, batch_y) in enumerate(loader):
        # 梯度清零
        model.zero_grad()
        tag_scores = model(batch_x)
 
        # 计算损失
        tag_scores = tag_scores.view(-1, tag_scores.shape[2])
        batch_y = batch_y.view(batch_y.shape[0]*batch_y.shape[1])
        loss = loss_function(tag_scores, batch_y)
        print(loss)
        # 后向传播
        loss.backward()
 
        # 更新参数
        optimizer.step()
 
# 测试过程
with torch.no_grad():
    inputs = torch.tensor([data_set_word[0]], dtype=torch.long)
    print(inputs)
    tag_scores = model.predict(inputs)
    print(tag_scores.shape)
    print(torch.argmax(tag_scores, dim=2))
