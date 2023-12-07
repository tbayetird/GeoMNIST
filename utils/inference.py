import torch
import sys,os
import numpy as np
from torch.utils.data import DataLoader

local_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(local_dir)
from model import ResNet101Model
from dataset import GeoMNIST

# csv_geomnist_a = "C:\\Users\\Theophile Bayet\\workspace\\THESIS\\GDS\\perfAPI\\GeoMNIST\\data\\GEOMNIST_A.csv"
#
#
# class Config():
#
# 	def __init__(self):
# 		self.CSV_DATASET=csv_geomnist_a
# 		self.MODEL_PATH = "C:\\Users\\Theophile Bayet\\workspace\\THESIS\\GDS\\perfAPI\\GeoMNIST\\data\\model_test.pth"


class Evaluator():

	def __init__(self, config):
		## Load model
		self.model = ResNet101Model().model
		self.model.load_state_dict(torch.load(config.MODEL_PATH))
		self.model.eval()
		self.device = torch.device('cpu')

		## Load data
		self.dataset=GeoMNIST(config.CSV_DATASET,split='test')
		self.data_loader = DataLoader(self.dataset,
									  batch_size=config.BATCH_SIZE,
									  shuffle=False)

	"""
	Inference of the model from the evaluator on a given data
	"""
	def inference_on_data(self,data):
		output = self.model(data.to(torch.float).to(self.device).unsqueeze(0))
		output_list = output[0].tolist()
		predicted_label = output_list.index(max(output_list))
		self.visualise_results(predicted_label, data)

	"""
	Visualise the results from an inference
	"""
	def visualise_results(self, result, data):
		print(data)
		print("Predicted label : {}".format(result))


	"""
	Evaluate a dataset with the model
	Metric will be accuracy for now and may be changed later.
	"""
	def evaluate(self):
		running_metric=0
		for inputs,labels in self.data_loader:
			inputs = inputs.to(self.device)
			labels = labels.to(self.device)
			outputs = self.model(inputs.to(torch.float))
			_, preds = torch.max(outputs, 1)
			#todo : maybe create a class for computing metrics
			running_metric += self._compute_acc(labels,preds)
		output_metric = running_metric/len(self.dataset)
		return output_metric

	"""
	Evaluate a dataset with the model, by data.
	Results are reported in a csv with the predicted output added to each row
	"""
	def evaluate_by_data(self):
		by_data_data_loader = DataLoader(self.dataset,
								 batch_size=1,
								 shuffle=False)
		running_metric=0
		outputs_list = []
		for inputs,labels in by_data_data_loader:
			inputs = inputs.to(self.device)
			labels = labels.to(self.device)
			outputs = self.model(inputs.to(torch.float))
			_, preds = torch.max(outputs, 1)
			#todo : maybe create a class for computing metrics
			running_metric += self._compute_acc(labels,preds)
			outputs_list.append(preds.tolist()[0])
		output_metric = running_metric/len(self.dataset)
		return output_metric,outputs_list


	"""
	Compute accuracy between label and prediction in single label setup
	"""
	def _compute_acc(self,label,pred):
		result = np.sum([1 if a==b else 0 for a,b in zip(label.tolist(),pred.tolist())])
		return result


# config = Config()
# evaluator = Evaluator(config)
#
# ## ------------------------------------- ##
# data = evaluator.dataset[0]
# label = data[1]
# evaluator.inference_on_data(data[0])
# print("Expected label : {}".format(label))
