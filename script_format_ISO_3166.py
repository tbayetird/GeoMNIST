import pandas as pd
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-m49p", "--m49-path", default = os.path.join(local_dir,"data\\UNSD_M49.csv"),
	help="path to the M49 UNSD file")
ap.add_argument("-csvl", "--csv-latlon-path", default = os.path.join(local_dir,"data\\World_lat_lon.csv"),
	help="path to the csv file")
ap.add_argument("-csvt", "--csv-temp-path", default = os.path.join(local_dir,"data\\World_avg_T.csv"),
	help="path to the csv file")
ap.add_argument("-csvp", "--csv-pib_path", default = os.path.join(local_dir,"data\\World_GDP_PPP.csv"),
	help="path to the csv file")
ap.add_argument("-csvd", "--csv-demo_path", default = os.path.join(local_dir,"data\\World_demo.xlsx"),
	help="path to the csv file")
ap.add_argument("-csvr", "--csv-rep_path", default = os.path.join(local_dir,"data\\World_data_rep.csv"),
	help="path to the csv file")
args = vars(ap.parse_args())
m49_path = args['m49_path']
csv_lat_lon_path = args['csv_latlon_path']
csv_temp_path = args['csv_temp_path']
csv_pib_path = args['csv_pib_path']
csv_demo_path = args['csv_demo_path']
csv_rep_path = args['csv_rep_path']
csv_paths=[csv_lat_lon_path,csv_temp_path,csv_pib_path,csv_rep_path,csv_demo_path]

m49_data = pd.read_csv(m49_path)

def country_name_ISO(country_name):
	# print(country_name)
	if country_name not in m49_data['Country or Area'].unique():
		# print("Country not found in data csv : {}".format(country_name))
		old_country_name = country_name
		if country_name == "Saint Martin (French part)":
			country_name = "Saint Martin (French Part)"
		if country_name == "Micronesia (Fed. States of)":
			country_name = "Micronesia (Federated States of)"
		if country_name == "Federated States of Micronesia":
			country_name = "Micronesia (Federated States of)"
		if country_name == "Bonaire, Sint Eustatius and Saba":
			country_name = "Bonaire"
		# doesn't exists in UN M49 norm cause it's not recognized.
		if country_name == "Kosovo (under UNSC res. 1244)":
			country_name = "to_delete"
		if country_name == "United Kingdom":
			country_name = "United Kingdom of Great Britain and Northern Ireland"
		if country_name == "Dem. People's Republic of Korea":
			country_name = "Democratic People's Republic of Korea"
		# doesn't exists in UN M49 norm cause it's not recognized.
		if country_name == "China, Taiwan Province of China":
			country_name = "to_delete"
		if country_name == "Côte d'Ivoire":
			country_name = "Côte d’Ivoire"
		if country_name == "Turkey":
			country_name = "Türkiye"
		if country_name == "TFYR Macedonia":
			country_name = "North Macedonia"
		if country_name == "Swaziland":
			country_name = "Eswatini"
		if country_name == "Faeroe Islands":
			country_name = "Faroe Islands"
		if country_name == "Republic of the Congo":
			country_name = "Congo"
		if country_name == "Channel Islands":
			country_name = "Guernsey"
		# not enough precision to understand what it is
		if country_name == "Caribbean Netherlands":
			country_name = "to_delete"
		if country_name == "Bolivia":
			country_name = "Bolivia (Plurinational State of)"
		if country_name == "United States":
			country_name = "United States of America"
		if country_name == "Tanzania":
			country_name = "United Republic of Tanzania"
		if country_name == "Syria":
			country_name = "Syrian Arab Republic"
		if country_name == "Russia":
			country_name = "Russian Federation"
		if country_name == "Moldovia":
			country_name = "Republic of Moldova"
		if country_name == "Iran":
			country_name = "Iran (Islamic Republic of)"
		if country_name == "Hon Kon":
			country_name = "China, Hong Kong SAR"
		if country_name == "Czech Republic":
			country_name = "Czechia"
		if country_name == "Cape Verde":
			country_name = "Cabo Verde"
		if country_name == "Brunei":
			country_name = "Brunei Darussalam"
		if country_name == "Venezuela":
			country_name = "Venezuela (Bolivarian Republic of)"
		if country_name == "Moldova":
			country_name = "Republic of Moldova"
		if country_name == "Hong Kon":
			country_name = "China, Hong Kong SAR"

		if country_name == "Africa Eastern and Southern":
			country_name = "to_delete"
		if country_name == "Africa Western and Central":
			country_name = "to_delete"
		if country_name == "Arab World":
			country_name = "to_delete"
		if country_name == "Central Europe":
			country_name = "to_delete"
		if country_name == "Caribbean small states":
			country_name = "to_delete"
		if country_name == "East Asia & Pacific (excluding high income)":
			country_name = "to_delete"
		if country_name == "Early-demographic dividend":
			country_name = "to_delete"
		if country_name == "East Asia & Pacific":
			country_name = "to_delete"
		if country_name == "Europe & Central Asia (excluding high income)":
			country_name = "to_delete"
		if country_name == "Europe & Central Asia":
			country_name = "to_delete"
		if country_name == "Euro area":
			country_name = "to_delete"
		if country_name == "European Union":
			country_name = "to_delete"
		if country_name == "Fragile and conflict affected situations":
			country_name = "to_delete"
		if country_name == "High income":
			country_name = "to_delete"
		if country_name == "Heavily indebted poor countries (HIPC)":
			country_name = "to_delete"
		if country_name == "IBRD only":
			country_name = "to_delete"
		if country_name == "IDA & IBRD total":
			country_name = "to_delete"
		if country_name == "IDA total":
			country_name = "to_delete"
		if country_name == "IDA blend":
			country_name = "to_delete"
		if country_name == "IDA only":
			country_name = "to_delete"
		if country_name == "Not classified":
			country_name = "to_delete"
		if country_name == "Latin America & Caribbean (excluding high income)":
			country_name = "to_delete"
		if country_name == "Lao PDR":
			country_name = "to_delete"
		if country_name == "Latin America & Caribbean":
			country_name = "to_delete"
		if country_name == "Least developed countries: UN classification":
			country_name = "to_delete"
		if country_name == "Low income":
			country_name = "to_delete"
		if country_name == "Lower middle income":
			country_name = "to_delete"
		if country_name == "Low & middle income":
			country_name = "to_delete"
		if country_name == "Late-demographic dividend":
			country_name = "to_delete"
		if country_name == "Middle East & North Africa":
			country_name = "to_delete"
		if country_name == "Middle income":
			country_name = "to_delete"
		if country_name == "Middle East & North Africa (excluding high income)":
			country_name = "to_delete"
		if country_name == "North America":
			country_name = "to_delete"
		if country_name == "OECD members":
			country_name = "to_delete"
		if country_name == "Other small states":
			country_name = "to_delete"
		if country_name == "Pre-demographic dividend":
			country_name = "to_delete"
		if country_name == "West Bank and Gaza":
			country_name = "to_delete"
		if country_name == "Pacific island small states":
			country_name = "to_delete"
		if country_name == "Post-demographic dividend":
			country_name = "to_delete"
		if country_name == "South Asia":
			country_name = "to_delete"
		if country_name == "Sub-Saharan Africa (excluding high income)":
			country_name = "to_delete"
		if country_name == "Sub-Saharan Africa":
			country_name = "to_delete"
		if country_name == "Small states":
			country_name = "to_delete"
		if country_name == "East Asia & Pacific (IDA & IBRD countries)":
			country_name = "to_delete"
		if country_name == "Europe & Central Asia (IDA & IBRD countries)":
			country_name = "to_delete"
		if country_name == "Latin America & the Caribbean (IDA & IBRD countries)":
			country_name = "to_delete"
		if country_name == "Middle East & North Africa (IDA & IBRD countries)":
			country_name = "to_delete"
		if country_name == "South Asia (IDA & IBRD)":
			country_name = "to_delete"
		if country_name == "Sub-Saharan Africa (IDA & IBRD countries)":
			country_name = "to_delete"
		if country_name == "Upper middle income":
			country_name = "to_delete"
		if country_name == "World":
			country_name = "to_delete"
		if country_name == "Kosovo":
			country_name = "to_delete"
		if country_name == "Central Europe and the Baltics":
			country_name = "to_delete"
		if country_name == "":
			country_name = "to_delete"

		if country_name == "Bahamas, The":
			country_name = "Bahamas"
		if country_name == "Cote d'Ivoire":
			country_name = "Côte d’Ivoire"
		if country_name == "Congo, Dem. Rep.":
			country_name = "Democratic Republic of the Congo"
		if country_name == "Congo, Rep.":
			country_name = "Congo"
		if country_name == "Curacao":
			country_name = "Curaçao"
		if country_name == "Egypt, Arab Rep.":
			country_name = "Egypt"
		if country_name == "Micronesia, Fed. Sts.":
			country_name = "Micronesia (Federated States of)"
		if country_name == "Gambia, The":
			country_name = "Gambia"
		if country_name == "Hong Kong SAR, China":
			country_name = "China, Hong Kong SAR"
		if country_name == "Iran, Islamic Rep.":
			country_name = "Iran (Islamic Republic of)"
		if country_name == "Kyrgyz Republic":
			country_name = "Kyrgyzstan"
		if country_name == "St. Kitts and Nevis":
			country_name = "Saint Kitts and Nevis"
		if country_name == "Korea, Rep.":
			country_name = "Republic of Korea"
		if country_name == "St. Lucia":
			country_name = "Saint Lucia"
		if country_name == "Macao SAR, China":
			country_name = "China, Macao SAR"
		if country_name == "St. Martin (French part)":
			country_name = "Saint Martin (French Part)"
		if country_name == "Korea, Dem. People's Rep.":
			country_name = "Democratic People's Republic of Korea"
		if country_name == "Slovak Republic":
			country_name = "Slovakia"
		if country_name == "Turkiye":
			country_name = "Türkiye"
		if country_name == "St. Vincent and the Grenadines":
			country_name = "Saint Vincent and the Grenadines"
		if country_name == "Venezuela, RB":
			country_name = "Venezuela (Bolivarian Republic of)"
		if country_name == "Virgin Islands (U.S.)":
			country_name = "United States Virgin Islands"
		if country_name == "Yemen, Rep.":
			country_name = "Yemen"
		if country_name == "":
			country_name = ""



		if country_name==old_country_name:
			print("{} not modified".format(country_name))
	return country_name


def prep_data_demo(csv_path):
	'''
	Get some data on demography
	'''
	df = pd.read_excel(csv_path,engine="openpyxl",sheet_name="Estimates")
	alt_df = df[['Region, subregion, country or area *','ISO3 Alpha-code','Type','Year','Total Population, as of 1 July (thousands)']]
	alt_df = alt_df[alt_df['Type']=='Country/Area']
	alt_df = alt_df[alt_df['Year']==2021.0]

	out_df = alt_df[['Year']]
	out_df['Country']=alt_df['Region, subregion, country or area *']
	out_df['CODE']=alt_df['ISO3 Alpha-code']
	out_df['DATA']=alt_df['Total Population, as of 1 July (thousands)']
	return out_df

def format_ISO_3166(csv_path):
	if csv_path.endswith('.csv'):
		df = pd.read_csv(csv_path)
	if csv_path.endswith('.xlsx'):
		df = prep_data_demo(csv_path)

	if 'Country' in df.columns :
		country_column = 'Country'
	if 'Country or Area' in df.columns :
		country_column = 'Country or Area'
	if 'Region, subregion, country or area *' in df.columns :
		country_column= 'Region, subregion, country or area *'

	df[country_column]=df[country_column].apply(lambda x : country_name_ISO(x))
	df = df.drop(df[df[country_column]=="to_delete"].index)
	# print(df[country_column].iloc[0])
	# print(iso_data[iso_data['Country']=='Burundi']['ISO 3166-1 alpha-3'].iloc[0])
	df['ISO-3166-1 alpha-3']=df[country_column].apply(lambda x : m49_data[m49_data['Country or Area']==x]['ISO-alpha3 Code'].values[0])
	df.to_csv(csv_path.replace('World','ISO_3166_World').replace('xlsx','csv'),index=False)

# format_ISO_3166(csv_demo_path)
# format_ISO_3166(csv_lat_lon_path)
for csv_path in csv_paths:
	print("File : {}".format(csv_path))
	format_ISO_3166(csv_path)
