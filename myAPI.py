import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from pickle import dump, load

class MLP(Chain):
    def __init__(self, inNum, hiddenLayers, outNum, actiFunc, outActiFunc, dropOutRatio, isTraining, bias='rand'):
        '''
        inNum       : int, number of input nodes
        hiddenLayers: list of ints, number of nodes of each hidden layers
        outNum      : int, number of input nodes
        actiFunc    : activation function for hidden layers
        outActiFunc : activation function for output layer
        dropOutRatio: float, should be in [0,1)
        isTraining  : is training
        bias        : 'rand' for using np.random.randn, a float value for contant initialization
        '''
        # ref: http://wazalabo.com/chainer-stacked-auto-encoder.html
        super(MLP, self).__init__()
        self.actiFunc = actiFunc
        self.outActiFunc = outActiFunc
        self.dropoutRatio = dropOutRatio
        self.isTraining = isTraining
        self.layers = [inNum] + hiddenLayers + [outNum]
        for i in range(len(self.layers)-1): # 5 layers (3 hidden) case: 4 links, link0~link3
            if(bias=='rand'):
                self.add_link('link{}'.format(i), L.Linear(self.layers[i], self.layers[i+1], initial_bias=np.random.randn(self.layers[i+1])))
            else:
                self.add_link('link{}'.format(i), L.Linear(self.layers[i], self.layers[i+1], bias=float(bias)))
    def __call__(self, x, keepHiddenIdx=-1, earlyStop=False):
        '''
        x            : the input
        keepHiddenIdx: keep the output of a specified hidden layer, zero-based
        earlyStop    : set to True if the layers after the "keepHiddenIdx"-th  have not to be computed
        '''
        h = x
        # non-output layers
        for i in range(len(self.layers)-2): # 5 layers (3 hidden) case: compute link0~link2
            y = self.__getitem__("link{}".format(i))(h)
            y = self.actiFunc(y)
            h = F.dropout(y, ratio=self.dropoutRatio, train=self.isTraining)
            if(i==keepHiddenIdx):
                self.hiddenOut = np.copy(cuda.to_cpu(h.data)) # In case the model is on GPU
                if(earlyStop):
                    return self.hiddenOut
        # output layer
        i = len(self.layers)-2
        y = self.__getitem__("link{}".format(i))(h)
        h = self.outActiFunc(y)
        return h
# End class

class CNNMLP(Chain):
    def __init__(self, cnnParam, hiddenLayers, outNum, actiFunc, outActiFunc, dropOutRatio, isTraining):
        '''
        cnnParam format: [
                input_dim_i, input_dim_j,
                [conv_1_in_channel_num, conv_1_out_channel_num, filter_size_1, pool_size_1, pad_size_1],
                [conv_2_in_channel_num, conv_2_out_channel_num, filter_size_2, pool_size_2, pad_size_2],
                ...
            ]
            - conv_1_in_channel_num: only int(1) is implemented in this class
            - conv_(i+1)_in_channel_num should be equal to conv_(i)_out_channel_num
            - filter_size, pool_size, pad_size: int of tuple of two ints. "pad_size" is optional
        '''
        super(CNNMLP, self).__init__()
        self.p = []
        self.actiFunc = actiFunc
        self.outActiFunc = outActiFunc
        self.dropoutRatio = dropOutRatio
        self.isTraining = isTraining
        # Utility functions
        def getPadding(param):
            if(len(param)>=5):
                return cnnParam[i][4] if type(cnnParam[i][4]) is tuple and len(cnnParam[i][4])==2 else (cnnParam[i][4], cnnParam[i][4])
            return (0,0)
        # Create links
        ii = cnnParam[0]
        jj = cnnParam[1]
        for i in range(2, len(cnnParam)):
            n = cnnParam[i][2] if type(cnnParam[i][2]) is tuple and len(cnnParam[i][2])==2 else (cnnParam[i][2], cnnParam[i][2])
            p = cnnParam[i][3] if type(cnnParam[i][3]) is tuple and len(cnnParam[i][3])==2 else (cnnParam[i][3], cnnParam[i][3])
            d = getPadding(cnnParam)
            self.add_link('conv{}'.format(i-2), L.Convolution2D(cnnParam[i][0], cnnParam[i][1], n, pad=d))
            self.p.append(p)
            ii = np.ceil(1.0*(ii+d[0]*2-n[0]+1)/p[0])
            jj = np.ceil(1.0*(jj+d[1]*2-n[1]+1)/p[1])
        inNum = int(ii * jj * cnnParam[-1][1])
        self.layers = [inNum] + hiddenLayers + [outNum]
        for i in range(len(self.layers)-1): # 5 layers (3 hidden) case: 4 links, link0~link3
            self.add_link('link{}'.format(i), L.Linear(self.layers[i], self.layers[i+1], initial_bias=np.random.randn(self.layers[i+1])))
    def __call__(self, x):
        h = x
        for i in range(len(self.p)):
            h = self.__getitem__("conv{}".format(i))(h)
            h = F.relu(h)
            h = F.max_pooling_2d(h, self.p[i])
        for i in range(len(self.layers)-2): # 5 layers (3 hidden) case: compute link0~link2
            y = self.__getitem__("link{}".format(i))(h)
            y = self.actiFunc(y)
            h = F.dropout(y, ratio=self.dropoutRatio, train=self.isTraining)
        # output layer
        i = len(self.layers)-2
        y = self.__getitem__("link{}".format(i))(h)
        h = self.outActiFunc(y)
        return h
# End class

def setActiFunc(funcName):
    if  (funcName=='sigm')     : return F.sigmoid
    elif(funcName=='relu')     : return F.relu
    elif(funcName=='softmax')  : return F.softmax
    elif(funcName=='softplus') : return F.softplus
    elif(funcName=='elu')      : return F.elu
    elif(funcName=='leakyrelu'): return F.leaky_relu
    elif(funcName=='cliprelu') : return F.clipped_relu
    elif(funcName=='tanh')     : return F.tanh
    elif(funcName=='hardsigm') : return F.hard_sigmoid
    elif(funcName=='identity') : return F.identity
    else: raise ValueError('Unknown activation function: ' + funcName)

def setCostFunc(funcName, useClassType):
    '''
    useClassType:
    - 'chain' if the above classes (MLP, CNNMLP) are used directly
    - 'claif' if L.Classifier(model, cost_func) is used as your model
    '''
    if(useClassType=='claif'):
        if(funcName=='sigm'):
            return F.sigmoid_cross_entropy
        elif(funcName=='softmax'):
            return F.softmax_cross_entropy
    elif(useClassType=='chain'):
        if(funcName=='mse'):
            return F.mean_squared_error
    raise ValueError('Unknown cost function: ' + funcName)

def setOptimizer(model, method, params):
    learningRate = params['learningRate'] if(params.has_key('learningRate')) else 0.001
    alpha        = params['alpha']        if(params.has_key('alpha'))        else 0.001
    if(method=='adam'):
        optimizer = optimizers.Adam(alpha=alpha)
    elif(method=='smorms3'):
        optimizer = optimizers.SMORMS3(lr=learningRate)
    elif(method=='rmsprop'):
        optimizer = optimizers.RMSprop(lr=learningRate)
    elif(method=='sgd'):
        optimizer = optimizers.SGD(lr=learningRate)
    elif(method=='momentum'):
        optimizer = optimizers.MomentumSGD(lr=learningRate)
    elif(method=='adagrad'):
        optimizer = optimizers.AdaGrad(lr=learningRate)
    elif(method=='adadelta'):
        optimizer = optimizers.AdaDelta()
    optimizer.setup(model)
    return optimizer
