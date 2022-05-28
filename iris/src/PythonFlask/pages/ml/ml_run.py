import numpy as np 
import pandas as pd
from tensorflow.keras import utils
from tensorflow.keras.models import  load_model
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pickle import load

def OHE_function(data):
  label_encoder = LabelEncoder()
  rezult = label_encoder.fit_transform(data)
  rezult = utils.to_categorical(rezult)
  return (rezult)

def days_ammount_function(data):
  start_time = data[0:data.find('--')]
  end_time = data[data.find('--')+2::]
  if len(data)== 22:
    start_time = datetime.strptime(start_time,'%Y-%m-%d')
    end_time = datetime.strptime(end_time,'%Y-%m-%d')
  else:
    start_time = datetime.strptime(start_time,'%Y-%m')
    end_time = datetime.strptime(end_time,'%Y-%m') + relativedelta(months=1)
  return (end_time - start_time).days

def OHE_from_labels_function(data,labels):
  rezult = []
  if isinstance(data,str) != 1:
    for i in range(len(data)):
      index = labels.index(data[i])
      rezult.append(utils.to_categorical(index,len(labels))) 
  else:
    index = labels.index(data)
    rezult.append(utils.to_categorical(index,len(labels)))
  return (rezult)

def index_from_labels_function(data,labels):
  if isinstance(data,str) != 1:
    for i in range(len(data)):
      index = labels.index(data[i])
  else:
    index = labels.index(data)
  return (np.array(index))

def observedPropertyDeterminandCode_function(data):
  labels = ['CAS_14797-65-0', 'EEA_3164-07-6', 'CAS_14797-55-8',
       'EEA_3151-01-7', 'CAS_7723-14-0', 'EEA_3164-08-7', 'EEA_3131-01-9',
       'EEA_3161-03-3', 'EEA_3152-01-0', 'EEA_3163-01-7', 'EEA_3133-01-5',
       'CAS_14265-44-2', 'EEA_3161-01-1', 'EEA_3161-05-5',
       'CAS_14798-03-9', 'EEA_3164-01-0', 'EEA_3142-01-6',
       'EEA_31615-01-7', 'EEA_3121-01-5', 'EEA_3161-02-2',
       'EEA_3133-03-7', 'EEA_3133-06-0', 'CAS_15307-86-5',
       'EEA_31613-01-1']
  return (OHE_from_labels_function(data,labels))

def waterBodyIdentifier_function(data):
  labels = list(np.load('waterBodyIdentifier.npy', allow_pickle= True))
  return (index_from_labels_function(data,labels))

def Country_function(data):
  labels = ['France', 'Spain', 'United Kingdom', 'Lithuania', 'Portugal',
       'Austria', 'Bulgaria', 'Germany', 'Finland', 'Czech Republic',
       'Belgium', 'Denmark', 'None', 'Italy', 'Ireland', 'Sweden',
       'Romania', 'Serbia', 'Slovakia', 'Netherlands', 'Poland',
       'Luxembourg', 'Switzerland', 'Latvia', 'Norway', 'Croatia',
       'Belarus', 'Russia']
  return (OHE_from_labels_function(data,labels))

def getter(df):
  df_copy = df.reset_index(drop=True)
  x_data = []
  y_data = []
  for i in range(len(df_copy)):
    observedPropertyDeterminandCode = observedPropertyDeterminandCode_function(df_copy['observedPropertyDeterminandCode'][i])[0]
    phenomenonTimeReferenceYear = np.array(df_copy['phenomenonTimeReferenceYear'][i] - 2006)
    parameterSamplingPeriod = np.array(days_ammount_function(df_copy['parameterSamplingPeriod'][i]))
    resultMeanValue = np.array(df_copy['resultMeanValue'][i])
    waterBodyIdentifier = waterBodyIdentifier_function(df_copy['waterBodyIdentifier'][i])
    Country = Country_function(df_copy['Country'][i])[0]


    PopulationDensity = np.array(df_copy['PopulationDensity'][i])
    TerraMarineProtected_2016_2018 = np.array(df_copy['TerraMarineProtected_2016_2018'][i])
    TouristMean_1990_2020 = np.array(df_copy['TouristMean_1990_2020'][i])
    VenueCount = np.array(df_copy['VenueCount'][i])
    netMigration_2011_2018 = np.array(df_copy['netMigration_2011_2018'][i])
    droughts_floods_temperature = np.array(df_copy['droughts_floods_temperature'][i])
    literacyRate_2010_2018 = np.array(df_copy['literacyRate_2010_2018'][i])
    combustibleRenewables_2009_2014 = np.array(df_copy['combustibleRenewables_2009_2014'][i])
    gdp = np.array(df_copy['gdp'][i])
    composition_food_organic_waste_percent = np.array(df_copy['composition_food_organic_waste_percent'][i]/100)
    composition_glass_percent = np.array(df_copy['composition_glass_percent'][i]/100)
    composition_metal_percent = np.array(df_copy['composition_metal_percent'][i]/100)
    composition_other_percent = np.array(df_copy['composition_other_percent'][i]/100)
    composition_paper_cardboard_percent = np.array(df_copy['composition_paper_cardboard_percent'][i]/100)
    composition_plastic_percent = np.array(df_copy['composition_plastic_percent'][i]/100)
    composition_rubber_leather_percent = np.array(df_copy['composition_rubber_leather_percent'][i]/100)
    composition_wood_percent = np.array(df_copy['composition_wood_percent'][i]/100)
    composition_yard_garden_green_waste_percent = np.array(df_copy['composition_yard_garden_green_waste_percent'][i]/100)
    waste_treatment_recycling_percent = np.array(df_copy['waste_treatment_recycling_percent'][i]/100)


    x_data.append(np.hstack([observedPropertyDeterminandCode,
                        phenomenonTimeReferenceYear,
                        parameterSamplingPeriod,
                        waterBodyIdentifier,
                        Country,
                        PopulationDensity,
                        TerraMarineProtected_2016_2018,
                        TouristMean_1990_2020,
                        VenueCount,
                        netMigration_2011_2018,
                        droughts_floods_temperature,
                        literacyRate_2010_2018,
                        combustibleRenewables_2009_2014,
                        gdp,
                        composition_food_organic_waste_percent,
                        composition_glass_percent,
                        composition_metal_percent,
                        composition_other_percent,
                        composition_paper_cardboard_percent,
                        composition_plastic_percent,
                        composition_rubber_leather_percent,
                        composition_wood_percent,
                        composition_yard_garden_green_waste_percent,
                        waste_treatment_recycling_percent]))
    y_data.append(np.array(resultMeanValue))
  x_data = np.array(x_data)
  y_data = np.array(y_data)
  y_data = np.expand_dims(y_data, axis=1)
  return(x_data, y_data)

############################################## нужно заполнить данными.
# использую датафрейм потому что функцию парсинга данных под этот формат писал
# в папке рядом со скриптом 3 файла, которые сохраняет скрипт обучения и использует этот, если что-то будет переноситься - то вместе с ними.
def run_ml_model(observedPropertyDeterminandCode,
                          phenomenonTimeReferenceYear,
                          parameterSamplingPeriod,
                          waterBodyIdentifier,
                          Country,
                          PopulationDensity,
                          TerraMarineProtected_2016_2018,
                          TouristMean_1990_2020,
                          VenueCount,
                          netMigration_2011_2018,
                          droughts_floods_temperature,
                          literacyRate_2010_2018,
                          combustibleRenewables_2009_2014,
                          gdp,
                          composition_food_organic_waste_percent,
                          composition_glass_percent,
                          composition_metal_percent,
                          composition_other_percent,
                          composition_paper_cardboard_percent,
                          composition_plastic_percent,
                          composition_rubber_leather_percent,
                          composition_wood_percent,
                          composition_yard_garden_green_waste_percent,
                          waste_treatment_recycling_percent):
  input_df = pd.DataFrame([[observedPropertyDeterminandCode,
                          phenomenonTimeReferenceYear,
                          parameterSamplingPeriod,
                          waterBodyIdentifier,
                          Country,
                          PopulationDensity,
                          TerraMarineProtected_2016_2018,
                          TouristMean_1990_2020,
                          VenueCount,
                          netMigration_2011_2018,
                          droughts_floods_temperature,
                          literacyRate_2010_2018,
                          combustibleRenewables_2009_2014,
                          gdp,
                          composition_food_organic_waste_percent,
                          composition_glass_percent,
                          composition_metal_percent,
                          composition_other_percent,
                          composition_paper_cardboard_percent,
                          composition_plastic_percent,
                          composition_rubber_leather_percent,
                          composition_wood_percent,
                          composition_yard_garden_green_waste_percent,
                          waste_treatment_recycling_percent,0]],
                          columns = ["observedPropertyDeterminandCode",
                          "phenomenonTimeReferenceYear",
                          "parameterSamplingPeriod",
                          "waterBodyIdentifier",
                          "Country",
                          "PopulationDensity",
                          "TerraMarineProtected_2016_2018",
                          "TouristMean_1990_2020",
                          "VenueCount",
                          "netMigration_2011_2018",
                          "droughts_floods_temperature",
                          "literacyRate_2010_2018",
                          "combustibleRenewables_2009_2014",
                          "gdp",
                          "composition_food_organic_waste_percent",
                          "composition_glass_percent",
                          "composition_metal_percent",
                          "composition_other_percent",
                          "composition_paper_cardboard_percent",
                          "composition_plastic_percent",
                          "composition_rubber_leather_percent",
                          "composition_wood_percent",
                          "composition_yard_garden_green_waste_percent",
                          "waste_treatment_recycling_percent",
                          "resultMeanValue"])
  ############################################################
  # rewers substance name and WaterBody name to code
  WaterBody_label = pd.read_csv('WaterBody_label.csv')
  observedPropertyDeterminandCode_label = pd.read_csv('observedPropertyDeterminandCode_label.csv')

  input_df['waterBodyIdentifier'] = WaterBody_label.loc[WaterBody_label['WaterBody_label'] == input_df['waterBodyIdentifier'].values[0]]['waterBodyIdentifier'].values[0]
  input_df['observedPropertyDeterminandCode'] = observedPropertyDeterminandCode_label.loc[observedPropertyDeterminandCode_label['observedPropertyDeterminandCode_label'] == input_df['observedPropertyDeterminandCode'].values[0]]['observedPropertyDeterminandCode'].values[0]

  input_data, y_data = getter(input_df)
  water_model= load_model('water_model.h5')
  pred = water_model.predict(input_data)
  y_scaler = load(open('y_scaler.pkl', 'rb')) 
  pred = y_scaler.inverse_transform(pred)

  resultUom = pd.read_csv('resultUom.csv')
  return("{'result': '"+ str(round(pred[0][0],2)) + ' ' +  resultUom.loc[resultUom['observedPropertyDeterminandCode'] == input_df['observedPropertyDeterminandCode'].values[0]]['resultUom'].values[0] + '\'}')# возвращаемое значение

def test():
  WaterBody_label = pd.read_csv('WaterBody_label.csv')
  return WaterBody_label.to_json()