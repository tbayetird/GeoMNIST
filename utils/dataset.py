import os
import torch
import numpy as np
import pandas as pd
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import DataLoader,Dataset



class GeoMNIST(Dataset):
	""" GEO-MNIST dataset generated from csv """

	def __init__(self,csv_file,split):
		"""
		Arguments:
			csv_file (string) : Path to the csv file with data info
		"""
		self.split = split
		self.data_csv = pd.read_csv(csv_file)
		self.data_csv = self.data_csv.loc[self.data_csv['SPLIT']==split]
		self.data_train = datasets.MNIST(
			root="data",
			train=True,
			download=True,
		)
		self.data_test = datasets.MNIST(
			root="data",
			train=False,
			download=True,
		)

	def __len__(self):
		return len(self.data_csv)

	def _str2rgb(self,rgb_string):
		"""
		rgb_string is a string with format '[a,b,c]' with a, b  and c integers
		between 0 and 255.
		Returns the [a,b,c] in a list format
		"""
		assert type(rgb_string) is str, "Wrong input type, should be str, found {}".format(type(rgb_string))
		assert rgb_string[0]=='[', "Input string should start with '['"
		assert rgb_string[-1]==']', "Input string should end with ']'"

		rgb_list = [int(x) for x in rgb_string[1:-1].split(',')]
		return torch.IntTensor(rgb_list)


	def __getitem__(self,idx):
		if self.split=='train':
			img, target = self.data_train[idx]
			alloted_idx = idx
		if self.split=='val':
			img, target = self.data_train[50000+idx]
		if self.split=='test':
			img, target = self.data_test[idx]

		# if idx >=60000:
		# 	img, target = self.data_test[idx-60000]
		# else :
		# 	img, target = self.data_train[idx]

		# print(img)
		data_row = self.data_csv.iloc[idx]
		bg_color = self._str2rgb(data_row['BG_COLOR'])
		dg_color = self._str2rgb(data_row['DG_COLOR'])
		rot_angle = int(data_row['ROT_ANGLE'])

		# transformations
		trans = transforms.Compose([transforms.ToTensor(),transforms.Lambda(lambda x: x.repeat(3,1,1))])
		img = trans(img)

		# Rotates the image
		img = transforms.functional.rotate(img,angle=rot_angle)

		# Switching background and digit color
		for i in range(28):
			for j in range(28):
				elem = img[:,i,j]
				if torch.equal(elem,torch.IntTensor([0.,0.,0.])):
					img[:,i,j]=bg_color.type(torch.IntTensor)
				else :
					scale = float(elem[0])
					new_color = dg_color.clone().detach()
					new_bg = bg_color.clone().detach()
					new_color.apply_(lambda x : x*scale).type(torch.IntTensor)
					new_bg.apply_(lambda x : x*(1-scale)).type(torch.IntTensor)

					img[:,i,j]=new_color+new_bg

		return img.type(torch.IntTensor),target

class GeoMNIST_float(Dataset):
	""" GEO-MNIST dataset generated from csv """

	def __init__(self,csv_file,split):
		"""
		Arguments:
			csv_file (string) : Path to the csv file with data info
		"""
		self.split = split
		self.data_csv = pd.read_csv(csv_file)
		self.data_csv = self.data_csv.loc[self.data_csv['SPLIT']==split]
		self.data_train = datasets.MNIST(
			root="data",
			train=True,
			download=True,
		)
		self.data_test = datasets.MNIST(
			root="data",
			train=False,
			download=True,
		)

	def __len__(self):
		return len(self.data_csv)

	def _str2rgb(self,rgb_string):
		"""
		rgb_string is a string with format '[a,b,c]' with a, b  and c integers
		between 0 and 255.
		Returns the [a,b,c] in a list format
		"""
		assert type(rgb_string) is str, "Wrong input type, should be str, found {}".format(type(rgb_string))
		assert rgb_string[0]=='[', "Input string should start with '['"
		assert rgb_string[-1]==']', "Input string should end with ']'"

		rgb_list = [float(int(x)/255) for x in rgb_string[1:-1].split(',')]
		return torch.FloatTensor(rgb_list)


	def __getitem__(self,idx):
		if self.split=='train':
			img, target = self.data_train[idx]
			alloted_idx = idx
		if self.split=='val':
			img, target = self.data_train[50000+idx]
		if self.split=='test':
			img, target = self.data_test[idx]

		# if idx >=60000:
		# 	img, target = self.data_test[idx-60000]
		# else :
		# 	img, target = self.data_train[idx]

		# print(img)
		data_row = self.data_csv.iloc[idx]
		bg_color = self._str2rgb(data_row['BG_COLOR'])
		dg_color = self._str2rgb(data_row['DG_COLOR'])
		rot_angle = int(data_row['ROT_ANGLE'])

		# transformations
		trans = transforms.Compose([transforms.ToTensor(),transforms.Lambda(lambda x: x.repeat(3,1,1))])
		img = trans(img)

		# Rotates the image
		img = transforms.functional.rotate(img,angle=rot_angle)

		# Switching background and digit color
		for i in range(28):
			for j in range(28):
				elem = img[:,i,j]
				if torch.equal(elem,torch.FloatTensor([0.,0.,0.])):
					img[:,i,j]=bg_color.type(torch.FloatTensor)
				else :
					scale = float(elem[0])
					new_color = dg_color.clone().detach()
					new_bg = bg_color.clone().detach()
					new_color.apply_(lambda x : x*scale).type(torch.FloatTensor)
					new_bg.apply_(lambda x : x*(1-scale)).type(torch.FloatTensor)

					img[:,i,j]=new_color+new_bg

		return img.type(torch.FloatTensor),target

#
# # Create data loaders.
# data=GeoMNIST_float("C:\\Users\\Theophile Bayet\\workspace\\THESIS\\GDS\\perfAPI\\GeoMNIST\\data\\GEOMNIST_A.csv",split='train')
# batch_size = 2
# train_dataloader = DataLoader(data, batch_size=batch_size)
#
# # print(len(data))
# # # # data[1]
# # # # data[1]
# # #
# # # test_dataloader = DataLoader(data, batch_size=batch_size,split='test')
# #
# for X, y in train_dataloader:
# 	# print(f"Shape of X [N, C, H, W]: {X.shape}")
# 	# print(f"Shape of y: {y.shape} {y.dtype}")
# 	figure = plt.figure()
# 	img = plt.imshow(np.array(X[0].permute(1,2,0)))
# 	plt.show()
# 	figure = plt.figure()
# 	img = plt.imshow(np.array(X[1].permute(1,2,0)))
# 	plt.show()
#
# 	break
