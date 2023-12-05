import os
import pickle
import argparse
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

### PARAMS ###
local_dir = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("-m49p", "--m49-path", default = os.path.join(local_dir,"data\\UNSD_M49.csv"),
	help="path to the M49 UNSD file")
ap.add_argument("-csvl", "--csv-latlon-path", default = os.path.join(local_dir,"data\\ISO_3166_World_lat_lon.csv"),
	help="path to the csv file")
ap.add_argument("-csvt", "--csv-temp-path", default = os.path.join(local_dir,"data\\ISO_3166_World_avg_T.csv"),
	help="path to the csv file")
ap.add_argument("-csvp", "--csv-pib_path", default = os.path.join(local_dir,"data\\ISO_3166_World_GDP_PPP.csv"),
	help="path to the csv file")
ap.add_argument("-csvd", "--csv-demo_path", default = os.path.join(local_dir,"data\\ISO_3166_World_demo.csv"),
	help="path to the csv file")
ap.add_argument("-csvr", "--csv-rep_path", default = os.path.join(local_dir,"data\\ISO_3166_World_data_rep.csv"),
	help="path to the csv file")
ap.add_argument("-csvg", "--csv-geo_path", default = os.path.join(local_dir,"data\\GEOMNIST_A.csv"),
	help="path to the csv file")
args = vars(ap.parse_args())
m49_path = args['m49_path']
csv_lat_lon_path = args['csv_latlon_path']
csv_temp_path = args['csv_temp_path']
csv_pib_path = args['csv_pib_path']
csv_demo_path = args['csv_demo_path']
csv_rep_path = args['csv_rep_path']
csv_geomnist = args['csv_geo_path']
# geoyfcc_path = args['geoyfcc_path']


def extract_country_name(path):
	return path.split('\\')[-1].replace('_',' ')

def generate_discrete_color_map(data_df):
    '''
    We need a discrete map that links a country to a color based on RGB columns
    '''
    color_discrete_map={}
    for _,row in data_df.iterrows():
        color_discrete_map[row['Country']]='rgb({},{},{})'.format(int(row['R']),int(row['G']),int(row['B']))
    return color_discrete_map

def get_last_pib(row):
	return row.dropna()[-1]

def prep_data_check_csv(csv_path):
	'''
	Get some data on csv for GEOMNIST dataset
	'''
	csv_df = pd.read_csv(csv_path)
	df_dic = {'Location':[],'DATA':[],'SPLIT':[]}
	for country in csv_df['COUNTRY'].unique():
		# A country could appear as often as three times, so I should handle that
		# and create three dataframes and output them.
		country_df = csv_df.loc[csv_df['COUNTRY']==country]
		country_df_train = country_df.loc[country_df['SPLIT']=='train']
		country_df_test = country_df.loc[country_df['SPLIT']=='test']
		country_df_val = country_df.loc[country_df['SPLIT']=='val']
		df_dic['Location']+=[country,country,country]
		df_dic['DATA']+=[len(country_df_train),len(country_df_test),len(country_df_val)]
		df_dic['SPLIT']+=['train','test','val']
		# df_dic['DATA'].append(len(csv_df[csv_df['COUNTRY']==country]))
		# df_dic['SPLIT'].append(csv_df[csv_df['COUNTRY']==country]['SPLIT'].values[0])
	return pd.DataFrame(df_dic)

def prep_data(csv_path):
	'''
	Get some data
	'''
	df = pd.read_csv(csv_path)
	# df['Location']=df['DIRECTORY'].apply(from_dsd_to_m49_name)
	df['Location']=df['DIRECTORY'].apply(extract_country_name)
	df['Metric_value']=df['METRIC']
	return df

def prep_data_rep(csv_path):
	'''
	Get some data on data repartition in visual datasets
	'''
	df = pd.read_csv(csv_path)
	total_rep = np.sum(df['Data rep'])
	total_no_rep = len(df[df['Data rep']==0])
	leftover_rep = (100-total_rep)/total_no_rep
	df['Location']=df['Country']
	df['DATA']=df['Data rep'].apply(lambda x : leftover_rep if x==0 else x)
	# print(df)
	return df


def prep_data_demo(csv_path):
	'''
	Get some data on demography
	'''
	df = pd.read_csv(csv_path)
	alt_df = df[['Region, subregion, country or area *','ISO3 Alpha-code','Type','Year','Total Population, as of 1 July (thousands)']]
	alt_df = alt_df[alt_df['Type']=='Country/Area']
	alt_df = alt_df[alt_df['Year']==2021.0]

	out_df = alt_df[['Year']]
	out_df['Country']=alt_df['Region, subregion, country or area *']
	out_df['CODE']=alt_df['ISO3 Alpha-code']
	out_df['DATA']=alt_df['Total Population, as of 1 July (thousands)']
	return out_df

def prep_data_lat_lon(csv_path):
    '''
    Get some data on latitude and longitude
    '''
    df = pd.read_csv(csv_path)
    df['Location']=df['Country']
    df['R']=((df['Longitude']+180)/360)*255
    df['R']=df['R'].apply(int)
    df['G']=127
    df['B']=((df['Latitude']+90)/180)*255
    df['B']=df['B'].apply(int)
    return df

def prep_data_pib(csv_path):
	'''
	Get some data on pib
	'''
	df = pd.read_csv(csv_path)
	df['DATA']=df.apply(get_last_pib,axis=1)
	df['Location']=df['Country']
	return df

def prep_data_temp(csv_path):
    '''
    Get some data on world temp
    '''
    df = pd.read_csv(csv_path)
    df['Location']=df['Country']
    df['temp']=df['temp'].apply(float)
    T_max = np.max(df['temp'])
    T_min = np.min(df['temp'])
    # print(type(df['temp'][0]))
    # print(type(T_min))
    df['R']=((df['temp']-T_min)/(T_max-T_min))*255
    df['R']=df['R'].apply(int)
    df['G']=127
    df['B']=((df['temp'].apply(int)-T_min)/(T_max-T_min))*255
    df['B']=df['B'].apply(int)
    return df


def from_data_to_visu_format(path,data):
	'''
	from the data format to a panda dataframe with adequate format for visualisation
	parameters :
	- path : path to the M49 norms excel
	- data : data to be converted, as a dict ; should have the "values" and "labels" keys.
	'''
	df = pd.read_csv(path)
	df = df.loc[:,['Sub-region Name','Intermediate Region Name','Country or Area','ISO-alpha3 Code']]
	# output = {'CODE':[],'DATA' : []}
	output = {'CODE':[],'Country':[]}
	if 'DATA' in data :
		output['DATA']=[]

	for i,elem in enumerate(df['Sub-region Name']):
		for j,label in enumerate(data['Location']):
			if elem == "South-eastern Asia":
				elem = "South eastern Asia"
			if elem == label :
				# print(elem)
				output['CODE'].append(df['ISO-alpha3 Code'].iloc[i])
				output['Country'].append(data['Location'].iloc[j])
				# output['DATA'].append(list(data["Metric_value"])[j])
	for i,elem in enumerate(df['Country or Area']):
		if elem=="Côte d’Ivoire":
			elem = "Côte d'Ivoire"
		for j,label in enumerate(data['Location']):
			if label=="Côte d’Ivoire":
				label = "Côte d'Ivoire"
			# handling differences in naming ...
			# if label=="Côte d’Ivoire":
			# 	label = "Cote d'Ivoire"
			# if elem == "Viet Nam":
			# 	elem = "Vietnam"
			# if elem == "Republic of Korea":
			# 	elem = "South Korea"
			# if label == "Türkiye":
			# 	label = "Turkey"
			# if "Ivoire" in elem and "Ivoire" in label:
			# 	print(elem)
			# 	print(label)
			if (elem == label):
				# print(label)
				output['CODE'].append(df['ISO-alpha3 Code'].iloc[i])
				output['Country'].append(data['Location'].iloc[j])
				if 'DATA' in data :
					output['DATA'].append(data['DATA'].iloc[j])

				# output['DATA'].append(list(data["Metric_value"])[j])
	for i,elem in enumerate(df['Intermediate Region Name']):
		for j,label in enumerate(data['Location']):
			if elem == label :
				# print(elem)
				output['CODE'].append(df['ISO-alpha3 Code'].iloc[i])
				output['Country'].append(data['Location'].iloc[j])
				# output['DATA'].append(list(data["Metric_value"])[j])
				# print(label , df['ISO-alpha3 Code'][i],list(data["values"])[j])
	return pd.DataFrame(output)



def world_map_visu(df,title="title dummy"):
	fig = go.Figure(data=go.Choropleth(
	    locations = df['CODE'],
	    z = df['DATA'],
	    # text = df['COUNTRY'],
	    colorscale = 'Greys',
	    autocolorscale=False,
	    reversescale=False,
	    marker_line_color='darkgray',
	    marker_line_width=0.5,
	    # colorbar_tickprefix = '$',
	    colorbar_title = '',
		colorbar_orientation="v",
		colorbar_tickfont_size=25,
	))
	fig.update_layout(
		title_text=title,
	    geo=dict(
	        showframe=True,
	        showcoastlines=True,
	        projection_type='equirectangular'
	    ),
		font=dict(size=25),
	)

	fig.show()
	# fig.write_image(file=img_path,format='png',engine='orca')

def world_map_visu_discrete_map(df,discrete_map,title=""):
    fig = px.choropleth(df, color_discrete_map=discrete_map,
                    locations="CODE",
                    color="Country",
                    projection="mercator",
                )
    fig.update_layout(
        title_text=title,
        geo=dict(
            showframe=True,
            showcoastlines=True,
            projection_type='equirectangular'
            ),
        font=dict(size=25),
        )
    fig.show()
	# fig.write_image(file=img_path,format='png',engine='orca')



def visualize_world_map(csv_data,m49_path,title=""):
	# csv_data = prep_data(csv_path)
	df = from_data_to_visu_format(m49_path,csv_data)
	world_map_visu(df,title)

def visualize_world_map_custom_color(csv_data,m49_path,title=""):
	color_discrete_map_lat_lon = generate_discrete_color_map(csv_data)
	csv_data=pd.merge(csv_data, from_data_to_visu_format(m49_path,csv_data), left_on = 'Country', right_on = 'Country', how = 'inner')
	world_map_visu_discrete_map(csv_data,color_discrete_map_lat_lon,title)

def check_dataset(dataset_df,m49_path,title=""):
	dataset_train_df = dataset_df[dataset_df['SPLIT']=='train']
	dataset_test_df = dataset_df[dataset_df['SPLIT']=='test']
	dataset_val_df = dataset_df[dataset_df['SPLIT']=='val']
	# print(dataset_test_df)
	# print(dataset_val_df)
	# print(dataset_val_df['Location'].iloc[6])
	visualize_world_map(dataset_train_df,m49_path,title+" Train")
	visualize_world_map(dataset_test_df,m49_path,title+" Test")
	visualize_world_map(dataset_val_df,m49_path,title+" Val")

# visualize_world_map(csv_path,m49_path,"")

### for PIB and demo data, almost everything is already done, just prep data and visualize
# world_map_visu(prep_data_pib(csv_pib_path),"PIB par pays")
# world_map_visu(pd.read_csv(csv_demo_path),"Demographie par pays")
# visualize_world_map_custom_color(prep_data_lat_lon(csv_lat_lon_path),m49_path,"Courleur du pays en fonction de la localité")
# visualize_world_map_custom_color(prep_data_temp(csv_temp_path),m49_path,"Courleur du pays en fonction de la température")
# visualize_world_map(prep_data_rep(csv_rep_path),m49_path,"Repartition d'Imagenet par pays")
# visualize_world_map(prep_data_check_csv(csv_geomnist),m49_path,"Check csv ")
check_dataset(prep_data_check_csv(csv_geomnist),m49_path,"Check dataset A ")
check_dataset(prep_data_check_csv(csv_geomnist.replace('_A.csv','_B.csv')),m49_path,"Check dataset B ")
check_dataset(prep_data_check_csv(csv_geomnist.replace('_A.csv','_C.csv')),m49_path,"Check dataset C ")
