import torch
from d2l import  torch as d2l
from torch import nn



class Seq2SeqEncoder(d2l.Encoder):
    def __init__(self,vocab_size,embed_size,num_hiddens,num_layers,
                 dropout=0,** kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size,embed_size)
        self.rnn=nn.GRU(embed_size,num_hiddens,num_layers,dropout=dropout)

    def forward(self, x,**kwargs):
        X=self.embedding(x)
        #[timestep, batch, embed_size]
        X=X.permute(1,0,2)
        output,state=self.rnn(X)
        return output,state

class Seq2SeqDecoder(d2l.Decoder):
    def __init__(self,vocab_size,embed_size,num_hiddens,num_layers,
                 dropout=0,** kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size,embed_size)
        self.rnn=nn.GRU(embed_size+num_hiddens,num_hiddens,num_layers,dropout=dropout)
        # decoder有输出层
        self.dense = nn.Linear(num_hiddens,vocab_size)

    def init_state(self, enc_outputs, *args):
        # 拿到state
        return enc_outputs[1]

    def forward(self, x,state,**kwargs):
        X=self.embedding(x).permute(1,0,2)
        context=state[-1].repeat(X.shape[0],1,1)
        X_and_context=torch.cat((X,context),dim=2)
        output,state=self.rnn(X_and_context,state)
        output=self.dense(output).permute(1,0,2)
        return output,state
