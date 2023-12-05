import time
import copy
import torch
import sys,os
# import torch.nn
from torch.utils.data import DataLoader
local_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(local_dir)
from utils.train import Config, Trainer


### Some useful variables
csv_geomnist_a=os.path.join(local_dir,"\\data\\GEOMNIST_A.csv")
csv_geomnist_b=os.path.join(local_dir,"\\data\\GEOMNIST_B.csv")
csv_geomnist_c=os.path.join(local_dir,"\\data\\GEOMNIST_C.csv")
configs = []
##### Design here the configurations needed for the different experiments #####
##### We only handle the training stuff here, the evaluation will take place elsewhere #####

##################
## Experiment 1 ##
##################
"""
Comparaison croisée d'un algo entraîné sur les différents jeux de données.
=> Entraînement du Resnet sur les trois jeux de données (3 configurations)
-> Choix à réaliser au niveau des epochs, batch size, ...
"""
epochs_expe_1=100
batvch_size_expe_1=64
model_output_path_expe_1 =os.path.join(local_dir,'outputs','model_expe_1_X.pth')

config_1_a = Config(epochs=epochs_expe_1,
					batch_size=batvch_size_expe_1,
					csv_dataset=csv_geomnist_a,
					model_output_path=model_output_path_expe_1.replace('_X,_A')
					)

config_1_b = Config(epochs=epochs_expe_1,
					batch_size=batvch_size_expe_1,
					csv_dataset=csv_geomnist_b,
					model_output_path=model_output_path_expe_1.replace('_X,_B')
					)

config_1_c = Config(epochs=epochs_expe_1,
					batch_size=batvch_size_expe_1,
					csv_dataset=csv_geomnist_c,
					model_output_path=model_output_path_expe_1.replace('_X,_C')
					)

configs.append(config_1_a,config_1_b,config_1_c)


##################
## Experiment 2 ##
##################
"""
Entraînement sur GEOMNIST-C et test sur des données issues d'Afrique de l'Ouest
=> pas d'entraînement requis, on reprend le modèle de config_1_c
"""

##################
## Experiment 3 ##
##################
"""
Caractérisation du biais géographique de GEOMNIST-C vers GEOMNIST-A
=> Pas d'entraînement requis, on reprend le modèle de config_1_c
"""


for config in configs:
	trainer = Trainer(config)
	trainer.run_trainer()
