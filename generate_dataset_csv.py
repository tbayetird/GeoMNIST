from torchvision import datasets
import pandas as pd
import numpy as np
import argparse
import random
import sys,os

local_dir = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("-csvl", "--csv-latlon-path", default = os.path.join(local_dir,"data\\ISO_3166_World_lat_lon.csv"),
	help="path to the csv file")
ap.add_argument("-csvt", "--csv-temp-path", default =  os.path.join(local_dir,"data\\ISO_3166_World_avg_T.csv"),
	help="path to the csv file")
ap.add_argument("-csvp", "--csv-pib_path", default =  os.path.join(local_dir,"data\\ISO_3166_World_GDP_PPP.csv"),
	help="path to the csv file")
ap.add_argument("-csvd", "--csv-demo_path", default =  os.path.join(local_dir,"data\\ISO_3166_World_demo.csv"),
	help="path to the csv file")
ap.add_argument("-csvr", "--csv-rep_path", default =  os.path.join(local_dir,"data\\ISO_3166_World_data_rep.csv"),
	help="path to the csv file")
ap.add_argument("-outp", "--output_path", default =  os.path.join(local_dir,"data\\"),
	help="path to the csv file")
args = vars(ap.parse_args())
output_path = args['output_path']

######
#
# This script aims at generating csv containing keys to replicable GEO-MNIST datasets.
# It will attribute a country and transformation attribute to each file of the MNIST dataset
# It will use the repartition files in inputs to chose a repartition of countries amongst data.
# We will keep in the csv information about data ID, data label, associated country, associated backgroung color, associated digit color, associated rotation angle.
# Digit and background color will be kept in [R,G,B] formatin the csv.
#
#####

class GEOMNIST_CSV_GENERATOR():

	def __init__(self,args):
		self.demographic_repartition_data =self._prep_data_demo(args['csv_demo_path'])
		self.imagenet_repartition_data = self._prep_data_rep(args['csv_rep_path'])
		self.equity_repartition_data = self._prep_data_eq()
		# print(self.demographic_repartition_data)
		# print(self.demographic_repartition_data['DATA'])
		self.world_lat_lon = pd.read_csv(args['csv_latlon_path'])
		self.world_temp = pd.read_csv(args['csv_temp_path'])
		self.world_temp['temp']=self.world_temp['temp'].apply(float)
		self.world_pib = pd.read_csv(args['csv_pib_path'])

		self.max_pib, self.min_pib = self._get_max_min_pib()

		self.MNIST_train = datasets.MNIST(
            root="data",
            train=True,
            download=True,
        )
		self.MNIST_test = datasets.MNIST(
            root="data",
            train=False,
            download=True,
        )

	def _get_max_min_pib(self):
		# in the case we don't have data, just use a random value that is neither min nor max
		df = self.world_pib.apply(lambda x : float(2000) if x.dropna()[-1]=='NY.GDP.PCAP.PP.CD' else float(x.dropna()[-1]), axis=1)
		return np.max(df), np.min(df)

	def _prep_data_eq(self):
		eq_df = self.demographic_repartition_data[['Country']]
		eq_df['DATA']=1/len(eq_df)
		return eq_df

	def _prep_data_rep(self,csv_path):
		'''
		Get some data on data repartition in visual datasets
		'''
		df = pd.read_csv(csv_path)
		total_rep = np.sum(df['Data rep'])
		total_no_rep = len(df[df['Data rep']==0])
		leftover_rep = (100-total_rep)/total_no_rep
		df['DATA']=df['Data rep'].apply(lambda x : leftover_rep if x==0 else x)
		print(df)
		return df.sort_values(by=['DATA'])


	def _prep_data_demo(self,csv_path):
		'''
		Get some data on demography
		'''
		df = pd.read_csv(csv_path)
		# print(df)
		total_pop = np.sum(df['DATA'])
		df['DATA']=df['DATA'].apply(lambda x : x/total_pop)
		return df.sort_values(by=['DATA'])

	"""
	Returns the digit color attributed to a country in the [R,G,B] format.
	"""
	def get_digit_color(self,country):
		data_row= self.world_temp[self.world_temp['Country']==country]
		if data_row.empty:
			return [0,127,0]
		if len(data_row)!=1:
			data_row=data_row.iloc[0]
			print("Patching due to {} having multiple rows in digit color".format(data_row['Country']))
		T_max = np.max(self.world_temp['temp'])
		T_min = np.min(self.world_temp['temp'])
		R=int(((data_row['temp']-T_min)/(T_max-T_min))*255)
		G=127
		B=int(((data_row['temp']-T_min)/(T_max-T_min))*255)
		return [R,G,B]

	"""
	Returns the background color attributed to a country in the [R,G,B] format.
	"""
	def get_background_color(self,country):
		data_row= self.world_lat_lon[self.world_lat_lon['Country']==country]
		if data_row.empty :
			return [127,127,127]
		if len(data_row)!=1:
			data_row=data_row.iloc[0]
			print("Patching due to {} having multiple rows in background color".format(data_row['Country']))
		R=int(((data_row['Longitude']+180)/360)*255)
		G=127
		B=int(((data_row['Latitude']+90)/180)*255)
		return [R,G,B]

	"""
	Returns the rotation angle associated to a data
	"""
	def get_rotation_angle(self,country):
		data_row= self.world_pib[self.world_pib['Country']==country]
		data_row = data_row.dropna(axis=1)
		if data_row.empty:
			return random.randint(0,90)
		data_pib = data_row[data_row.columns[-1]]
		if data_pib.values[0] == 'NY.GDP.PCAP.PP.CD':
			return random.randint(0,90)
		theta_max = int((1 - (data_pib-self.min_pib)/(self.max_pib-self.min_pib))*90)
		# we've got maximum rotation angle for the country ,
		# now we get the associated rotation angle for the data
		theta = random.randint(0,theta_max)
		return theta

	"""
	Depending on the selected dataset, generate a country according to wanted
	repartition properties
	"""
	def get_country_table(self,dataset_type="A",n=60000):
		if dataset_type=="A":
			data_rep = self.equity_repartition_data
			data_rep['data_count']=data_rep['DATA'].apply(lambda x : int(np.floor(x*n)))
		if dataset_type=="B":
			data_rep = self.demographic_repartition_data
			data_rep['data_count']=data_rep['DATA'].apply(lambda x : int(np.ceil(x*n)))
		if dataset_type=="C":
			data_rep = self.imagenet_repartition_data
			data_rep['data_count']=data_rep['DATA'].apply(lambda x : int(np.ceil(x*n/100)))
		country_list = []
		for country in list(data_rep['Country'].unique()):
			country_count = data_rep[data_rep['Country']==country]['data_count']
			print(country,country_count)
			# print(country_list)
			country_list += [country]*country_count.values[0]
		if len(country_list)<n:
			country_list+=list(data_rep['Country'].unique())
		return country_list

	"""
	Generate the metadata that will be associated to the MNIST datas

	"""
	def _generate_data(self,output_dic,dataset,dataset_type="A",n=60000,split="train"):
		if split=="train":
			country_table_train = self.get_country_table(dataset_type=dataset_type,n=n-10000)
			country_table_val = self.get_country_table(dataset_type=dataset_type,n=10000)
			country_tables = [country_table_train,country_table_val]
		else :
			country_tables=[self.get_country_table(dataset_type=dataset_type,n=n)]

		country_table = country_tables[0]
		switch=False
		print("Country table of length {} with split {}".format(len(country_table),split))
		for i,elem in enumerate(dataset):
			## switch splits when needed
			if split=="val" and switch ==False:
				country_table=country_tables[1]
				switch = True
				print("Country table of length {} with split {}".format(len(country_table),split))
			## determine what country will be associated to the data
			if split=="train" and i>50000:
				split="val"

			label = elem[1]
			if split =="val":
				country=country_table[i-50000]
			else:
				country = country_table[i]

			bg_color = self.get_background_color(country)
			dg_color = self.get_digit_color(country)
			rot_angle = self.get_rotation_angle(country)
			## fill the dic
			output_dic['ID'].append(i)
			output_dic['LABEL'].append(label)
			output_dic['SPLIT'].append(split)
			output_dic['COUNTRY'].append(country)
			output_dic['BG_COLOR'].append(bg_color)
			output_dic['DG_COLOR'].append(dg_color)
			output_dic['ROT_ANGLE'].append(rot_angle)


	"""
	Generates the csv for a replicable GEO-MNIST-A dataset.
	This dataset has the same amount of data for each country in the world (equity)
	The output csv should have the following keys : ID, LABEL, COUNTRY, BG_COLOR, DG_COLOR, ROT_ANGLE

	"""
	def generate_GEOMNIST_A_csv(self,directory):
		output_dic ={"ID":[], "LABEL":[],"SPLIT":[], "COUNTRY":[], "BG_COLOR":[], "DG_COLOR":[], "ROT_ANGLE":[]}
		self._generate_data(output_dic,self.MNIST_train)
		self._generate_data(output_dic,self.MNIST_test,dataset_type="A",n=10000,split="test")
		output_df = pd.DataFrame(output_dic)
		output_df.to_csv(os.path.join(directory,"GEOMNIST_A.csv"),index=False)
		return output_df

	"""
	Generates the csv for a replicable GEO-MNIST-B dataset.
	This dataset has the same repartition as the population around the world
	The output csv should have the following keys : ID, LABEL, COUNTRY, BG_COLOR, DG_COLOR, ROT_ANGLE

	"""
	def generate_GEOMNIST_B_csv(self,directory):
		output_dic ={"ID":[], "LABEL":[], "SPLIT":[], "COUNTRY":[], "BG_COLOR":[], "DG_COLOR":[], "ROT_ANGLE":[]}
		self._generate_data(output_dic,self.MNIST_train,dataset_type="B",n=60000)
		self._generate_data(output_dic,self.MNIST_test,dataset_type="B",n=10000,split="test")
		output_df = pd.DataFrame(output_dic)
		output_df.to_csv(os.path.join(directory,"GEOMNIST_B.csv"),index=False)
		return output_df

	"""
	Generates the csv for a replicable GEO-MNIST-C dataset.
	This dataset has the same repartition as the imagenet dataset
	The output csv should have the following keys : ID, LABEL, COUNTRY, BG_COLOR, DG_COLOR, ROT_ANGLE

	"""
	def generate_GEOMNIST_C_csv(self,directory):
		output_dic ={"ID":[], "LABEL":[], "SPLIT":[], "COUNTRY":[], "BG_COLOR":[], "DG_COLOR":[], "ROT_ANGLE":[]}
		self._generate_data(output_dic,self.MNIST_train,dataset_type="C",n=60000)
		self._generate_data(output_dic,self.MNIST_test,dataset_type="C",n=10000,split="test")
		output_df = pd.DataFrame(output_dic)
		output_df.to_csv(os.path.join(directory,"GEOMNIST_C.csv"),index=False)
		return output_df


generator = GEOMNIST_CSV_GENERATOR(args)
generator.generate_GEOMNIST_A_csv(output_path)
generator.generate_GEOMNIST_B_csv(output_path)
generator.generate_GEOMNIST_C_csv(output_path)
