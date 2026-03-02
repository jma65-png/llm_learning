import numpy as np

def relu(x):
    return np.maximum(x,0)

def softmax(x):
    e_x = np.exp(x - np.max(x,axis=-1,keepdims=True))
    return e_x / e_x.sum(axis=-1,keepdims=True)

class MaskedAttention:
    def __init__(self,d_model=128):
        self.d_model=d_model
        self.Wq=np.random.randn(d_model,d_model)*0.01
        self.Wk=np.random.randn(d_model,d_model)*0.01
        self.Wv=np.random.randn(d_model,d_model)*0.01

    def forward(self,x):
        Q=np.dot(x,self.Wq)
        K=np.dot(x,self.Wk)
        V=np.dot(x,self.Wv)

        scores=np.dot(Q.K.T)/np.sqrt(self.d_model)

        mask=np.triu(np.ones(scores.shape),k=1)*-1e9
        scores+=mask

        weights=softmax(scores)
        return np.dot(weights,V)