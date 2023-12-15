import io
import time
import copy
import torch
import sys,os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# import torch.nn
from torch.utils.data import DataLoader
local_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(local_dir)
from utils.inference import Evaluator

class Config():

	def __init__(self,csv_dataset,model_path,batch_size=64):
		self.CSV_DATASET=csv_dataset
		self.MODEL_PATH = model_path
		self.BATCH_SIZE = batch_size


### Some useful variables
csv_geomnist_a=os.path.join(local_dir,"data\\GEOMNIST_A.csv")
csv_geomnist_b=os.path.join(local_dir,"data\\GEOMNIST_B.csv")
csv_geomnist_c=os.path.join(local_dir,"data\\GEOMNIST_C.csv")
csv_datasets={"A":csv_geomnist_a, "B": csv_geomnist_b, "C":csv_geomnist_c}

model_a_path = os.path.join(local_dir,"outputs\\model_expe_1_A.pth")
model_b_path = os.path.join(local_dir,"outputs\\model_expe_1_B.pth")
model_c_path = os.path.join(local_dir,"outputs\\model_expe_1_C.pth")
models={"A":model_a_path, "B": model_b_path, "C":model_c_path}

### Some useful test variables
csv_test = os.path.join(local_dir,"data\\GEOMNIST_A_mini.csv")
model_test = os.path.join(local_dir,"data\\model_test.pth")
by_class_output_path = os.path.join(local_dir,"outputs\\GEOMNIST_A_mini_by_class_results.csv")
conf_matrix_output_path = by_class_output_path.replace('results.csv','conf_matrix.png')
output_test = csv_test.replace('data\\','outputs\\').replace('.csv','_report.csv')

 
csv_test_paths={"A":csv_test, "B": csv_test, "C":csv_test}
model_test_paths={"A":model_test, "B": model_test, "C":model_test}

"""
Generates the report that adds PREDICTION and METRIC columns with valid values
"""
def generate_report(csv_path,model_path,output_path):
	conf = Config(csv_path,model_path)
	evaluator = Evaluator(conf)
	res,preds = evaluator.evaluate_by_data()

	df = pd.read_csv(csv_path)
	df = df.loc[df['SPLIT']=='test']
	df.insert(len(df.columns),"PREDICTION",preds)
	df = df.apply(generate_metric,axis=1)
	mean_metric = np.sum(df['METRIC'].tolist())/len(df)
	print("mean accuracy : {}".format(mean_metric))
	df.to_csv(output_path)
	return df,mean_metric


"""
Compute accuracy between label and prediction in single label setup
"""
#should come from a utils/metric ?
def _compute_acc(label,pred):
	result = 1 if label==pred else 0
	return result


"""
Supposed to be called as a .apply function for a DataFrame
Takes a row and returns the same row with the added PREDICTION and METRIC
columns filled.
"""
# should come from a utils/metrics ?
def generate_metric(row):
	label = int(row['LABEL'])
	pred = int(row['PREDICTION'])
	metric = _compute_acc(label,pred)
	row['METRIC']=metric
	return row


"""
Generates the results by class : metric by class.
"""
def generate_by_class_result(df_report,output_path):
	output_dic = {}
	labels = df_report['LABEL'].unique()
	labels.sort()
	for label in labels:
		df_class = df_report.loc[df_report['LABEL']==label]
		df_class.apply(generate_metric,axis=1)
		class_acc = np.sum(df_class['METRIC'].tolist())/len(df_class)
		output_dic[label]=[class_acc]
	output_df = pd.DataFrame(output_dic)
	output_df.to_csv(output_path,index=False)
	return output_df

"""
Generates the confusion matrix for the by class results.
"""
def generate_confusion_matrix(df_report,conf_matrix_output_path):
	cm = confusion_matrix(df_report['LABEL'].tolist(),df_report['PREDICTION'].tolist())
	display = ConfusionMatrixDisplay(cm)
	display.plot()
	plt.savefig(conf_matrix_output_path)
	# plt.show()
	return cm

"""
Generates all the reports needed for a dataset and model combo
"""
def full_report_single_csv(csv_path,model_path,output_path):
	by_class_output_path = output_path.replace('report','by_class_result')
	conf_matrix_output_path = by_class_output_path.replace('result.csv','conf_matrix.png')
	df_report,mean_metric = generate_report(csv_path,model_path,output_path)
	by_class_df = generate_by_class_result(df_report,by_class_output_path)
	cm = generate_confusion_matrix(df_report,conf_matrix_output_path)
	return df_report,mean_metric

"""
Generates all the reports needed for expe 1
"""
def expe_1_full_report(csv_datasets,models):
	df_expe_1 = pd.DataFrame({"A":[],"B":[],"C":[]})
	k=0
	for i,dataset_index in enumerate(['A','B','C']):
		csv_path = csv_datasets[dataset_index]
		for j,model_index in enumerate(['A','B','C']):
			model_path = models[model_index]
			output_path = model_a_path = os.path.join(local_dir,"expe_1\\GEOMNIST_{}_model_{}_report.csv".format(dataset_index,model_index))
			df_report,mean_metric = full_report_single_csv(csv_path,model_path,output_path)
			# df_expe_1.loc[model_index,dataset_index]=k
			# k+=1
	df_expe_1.to_csv(os.path.join(local_dir,"expe_1\\expe_metric_report.csv"))

# full_report_single_csv(csv_test,model_test,output_test)
expe_1_full_report(csv_test_paths,model_test_paths)
