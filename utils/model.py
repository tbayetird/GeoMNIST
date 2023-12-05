import torch
from torch import nn
from collections import OrderedDict

class ResNet101Model():

	def __init__(self):
		self.model = self._load_model()

	def _load_model(self):
		model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', weights='ResNet101_Weights.DEFAULT')
		for param in model.parameters():
			param.requires_grad = False
		classifier = nn.Sequential(OrderedDict([
		('fc1', nn.Linear(2048, 512)),
		('fc2', nn.Linear(512, 10))
		]))
		model.fc = classifier
		return model
