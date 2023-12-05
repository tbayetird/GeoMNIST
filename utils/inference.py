from model import ResNet101Model
from dataset import GeoMNIST
import torch

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
	"""
	def evaluate(self):
		pass

	"""
	Write the output to the desired format
	todo : explain format
	"""
	def write_outputs(self):
		pass

# config = Config()
# evaluator = Evaluator(config)
#
# ## ------------------------------------- ##
# data = evaluator.dataset[0]
# label = data[1]
# evaluator.inference_on_data(data[0])
# print("Expected label : {}".format(label))
