import myAPI
import chainer.links as L

actiFunc    = myAPI.setActiFunc('relu')
outActiFunc = myAPI.setActiFunc('identity')
costFunc    = myAPI.setCostFunc('softmax_cross_entropy', 'claif')
model = myAPI.MLP(100, [256 256 256], 10, actiFunc, outActiFunc, dropOutRatio, True, bias=0.1)
model = L.Classifier(model, costFunc)
optimizer = myAPI.setOptimizer(model, method, optmParams)
