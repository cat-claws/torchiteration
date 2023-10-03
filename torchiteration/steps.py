import torch
import torch.nn as nn
from torch.nn import functional as F

def classification_step(net, batch, batch_idx, **kw):
	inputs, labels = batch
	inputs, labels = inputs.to(kw['device']), labels.to(kw['device'])
	scores = net(inputs)
	loss = F.cross_entropy(scores, labels, reduction = 'sum')

	max_scores, max_labels = scores.max(1)
	correct = (max_labels == labels).sum()
	return {'loss':loss, 'correct':correct}

def attacked_classification_step(net, batch, batch_idx, **kw):
	inputs, labels = batch
	inputs, labels = inputs.to(kw['device']), labels.to(kw['device'])
	inputs_ = kw['atk'](inputs, labels)

	scores = net(inputs_)
	loss = F.cross_entropy(scores, labels, reduction = 'sum')

	max_scores, max_labels = scores.max(1)
	correct = (max_labels == labels).sum()
	return {'loss':loss, 'correct':correct}

def predict_classification_step(net, batch, batch_idx, **kw):
	inputs, _ = batch
	inputs = inputs.to(kw['device'])
	scores = net(inputs)

	max_scores, max_labels = scores.max(1)
	return {'predictions':max_labels}
