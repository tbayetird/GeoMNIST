import time
import copy
import torch
import sys,os
# import torch.nn
from torch.utils.data import DataLoader
local_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(local_dir)
from dataset import GeoMNIST
from model import ResNet101Model
# from pytorch_utils.engine import train_one_epoch, evaluate


class Config():

	def __init__(self,epochs=1,batch_size=64,csv_dataset=os.path.join(os.path.dirname(local_dir),"data\\GEOMNIST_A.csv"),it_verb=150,model_output_path = "model_test.pth"):
		self.EPOCHS=epochs
		self.BATCH_SIZE=batch_size
		self.CSV_DATASET=csv_dataset
		self.IT_VERB = it_verb
		self.MODEL_OUTPUT_PATH = model_output_path


class Trainer():

	def __init__(self, config):
		## Collect config
		self.csv_dataset = config.CSV_DATASET
		self.num_epochs = config.EPOCHS
		self.batch_size=config.BATCH_SIZE
		self.iteration_verbose=config.IT_VERB
		self.output_path = config.MODEL_OUTPUT_PATH

		## Load necessities
		self._load_model()
		self._load_data_loaders()

		## Add training functions
		self.criterion = torch.nn.CrossEntropyLoss()
		params = [p for p in self.resnet.model.parameters() if p.requires_grad]
		self.optimizer = torch.optim.SGD(
				params,
				lr=0.005,
				momentum=0.9,
				weight_decay=0.0005
				)
		self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
				self.optimizer,
				step_size=3,
				gamma=0.1
				)


	"""
	Loads the model and associated things
	"""
	def _load_model(self):
		self.resnet = ResNet101Model()
		self.device = torch.device('cpu')
		self.resnet.model.to(self.device)


	"""
	Loads the data loaders and associated things
	"""
	def _load_data_loaders(self):
		train_dataset = GeoMNIST(self.csv_dataset,'train')
		val_dataset = GeoMNIST(self.csv_dataset,'val')
		self.dataloaders={
					'train' : DataLoader(train_dataset,
										batch_size=self.batch_size,
										shuffle=False,
										),
					'val' : DataLoader(val_dataset,
										batch_size=self.batch_size,
										shuffle=False,
										)
					}

	"""
	Trains the model according to config parameters
	returns the model as well as best accuracy
	"""
	def train_model(self, is_inception=False):
		since = time.time()
		model = self.resnet.model
		val_acc_history = []

		best_model_wts = copy.deepcopy(model.state_dict())
		best_acc = 0.0
		for epoch in range(self.num_epochs):
			print('Epoch {}/{}'.format(epoch, self.num_epochs))
			print('-' * 10)

			# Each epoch has a training and validation phase
			for phase in ['train', 'val']:
				if phase == 'train':
	 				model.train()  # Set model to training mode
				else:
					model.eval()   # Set model to evaluate mode

				running_loss = 0.0
				running_corrects = 0
				count =0
				batch_number=0
				# Iterate over data.
				for inputs, labels in self.dataloaders[phase]:
					inputs = inputs.to(self.device)
					labels = labels.to(self.device)

					# zero the parameter gradients
					self.optimizer.zero_grad()

					# forward
					# track history if only in train
					with torch.set_grad_enabled(phase == 'train'):
						# Get model outputs and calculate loss
						# Special case for inception because in training it has an auxiliary output. In train
						# mode we calculate the loss by summing the final output and the auxiliary output
						# but in testing we only consider the final output.
						if is_inception and phase == 'train':
							# From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
							outputs, aux_outputs = model(inputs)
							loss1 = self.criterion(outputs, labels)
							loss2 = self.criterion(aux_outputs, labels)
							loss = loss1 + 0.4*loss2
						else:
							outputs = model(inputs.to(torch.float))
							loss = self.criterion(outputs, labels)

						_, preds = torch.max(outputs, 1)

						# backward + optimize only if in training phase
						if phase == 'train':
							loss.backward()
							self.optimizer.step()

					# statistics
					running_loss += loss.item() * inputs.size(0)
					running_corrects += torch.sum(preds == labels.data)
					count+=1
					batch_number+=1
					if count==self.iteration_verbose:
						print('-- In epoch {}, batch {} -- {} Running Loss: {:.4f} Running Acc: {:.4f}'.format(epoch, batch_number, phase, running_loss/batch_number, running_corrects/batch_number))
						count=0


				epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
				epoch_acc = running_corrects.double() / len(self.dataloaders[phase].dataset)

				print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

				# deep copy the model
				if phase == 'val' and epoch_acc > best_acc:
					best_acc = epoch_acc
					best_model_wts = copy.deepcopy(model.state_dict())
				if phase == 'val':
					val_acc_history.append(epoch_acc)

			print()
		time_elapsed = time.time() - since
		print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
		print('Best val Acc: {:4f}'.format(best_acc))
		# load best model weights
		model.load_state_dict(best_model_wts)
		return model, val_acc_history

	def run_trainer(self):
		best_model,best_acc = self.train_model()
		torch.save(best_model.state_dict(),self.output_path)
#
# config = Config()
# trainer = Trainer(config)
# trainer.run_trainer()
# torch.save(model.state_dict(), "model_test.pth")
