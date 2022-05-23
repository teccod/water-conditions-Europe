# Commented out IPython magic to ensure Python compatibility.
import numpy as np 
import pandas as pd
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import LabelEncoder
from datetime import date, datetime
import time
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import train_test_split
import jaydebeapi

pd.set_option('display.max_columns', 30)

url = "jdbc:IRIS://localhost:1972/IRISAPP"
driver = "com.intersystems.jdbc.IRISDriver"
user = "SuperUser"
password = "SYS"
jarfile = "./intersystems-jdbc-3.1.0.jar"

conn = jaydebeapi.connect(driver, url, [user, password], jarfile)
curs = conn.cursor()

dataTable = 'dc_data_teccod.waterPollution'
df = pd.read_sql("select * from %s" % dataTable, conn)

df = df.dropna(axis='rows')

# drop unnecessary data
df = df.loc[df['parameterWaterBodyCategory'] == 'RW']
df = df.loc[df['procedureAnalysedFraction'] == 'total']
df = df.loc[df['procedureAnalysedMedia'] == 'water']
df = df.loc[df['phenomenonTimeReferenceYear'] >2005]
df = df.loc[df['phenomenonTimeReferenceYear'] < 2017]
df = df.reset_index(drop=True)

mask = []
mask_condition = df['observedPropertyDeterminandCode'].value_counts() > 100
for i in range(len(df)):
   mask.append(mask_condition[df['observedPropertyDeterminandCode'][i]])
df = df.loc[mask]
df = df.reset_index(drop=True)


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
  rezult = []
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
  labels = list(df['waterBodyIdentifier'].unique())
  return (index_from_labels_function(data,labels))

def Country_function(data):
  labels = ['France', 'Spain', 'United Kingdom', 'Lithuania', 'Portugal',
       'Austria', 'Bulgaria', 'Germany', 'Finland', 'Czech Republic',
       'Belgium', 'Denmark', 'None', 'Italy', 'Ireland', 'Sweden',
       'Romania', 'Serbia', 'Slovakia', 'Netherlands', 'Poland',
       'Luxembourg', 'Switzerland', 'Latvia', 'Norway', 'Croatia',
       'Belarus', 'Russia']
  return (OHE_from_labels_function(data,labels))

x_data = []
y_data = []
for i in range(len(df)):
  observedPropertyDeterminandCode = observedPropertyDeterminandCode_function(df['observedPropertyDeterminandCode'][i])[0]
  phenomenonTimeReferenceYear = np.array(df['phenomenonTimeReferenceYear'][i] - 2006)
  parameterSamplingPeriod = np.array(days_ammount_function(df['parameterSamplingPeriod'][i]))
  resultMeanValue = np.array(df['resultMeanValue'][i])
  waterBodyIdentifier = waterBodyIdentifier_function(df['waterBodyIdentifier'][i])
  Country = Country_function(df['Country'][i])[0]


  PopulationDensity = np.array(df['PopulationDensity'][i])
  TerraMarineProtected_2016_2018 = np.array(df['TerraMarineProtected_2016_2018'][i])
  TouristMean_1990_2020 = np.array(df['TouristMean_1990_2020'][i])
  VenueCount = np.array(df['VenueCount'][i])
  netMigration_2011_2018 = np.array(df['netMigration_2011_2018'][i])
  droughts_floods_temperature = np.array(df['droughts_floods_temperature'][i])
  literacyRate_2010_2018 = np.array(df['literacyRate_2010_2018'][i])
  combustibleRenewables_2009_2014 = np.array(df['combustibleRenewables_2009_2014'][i])
  gdp = np.array(df['gdp'][i])
  composition_food_organic_waste_percent = np.array(df['composition_food_organic_waste_percent'][i]/100)
  composition_glass_percent = np.array(df['composition_glass_percent'][i]/100)
  composition_metal_percent = np.array(df['composition_metal_percent'][i]/100)
  composition_other_percent = np.array(df['composition_other_percent'][i]/100)
  composition_paper_cardboard_percent = np.array(df['composition_paper_cardboard_percent'][i]/100)
  composition_plastic_percent = np.array(df['composition_plastic_percent'][i]/100)
  composition_rubber_leather_percent = np.array(df['composition_rubber_leather_percent'][i]/100)
  composition_wood_percent = np.array(df['composition_wood_percent'][i]/100)
  composition_yard_garden_green_waste_percent = np.array(df['composition_yard_garden_green_waste_percent'][i]/100)
  waste_treatment_recycling_percent = np.array(df['waste_treatment_recycling_percent'][i]/100)


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

print(x_data.shape)
print(y_data.shape)

y_scaler = StandardScaler()
y_data = y_scaler.fit_transform(y_data)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1)

water_model = Sequential()
water_model.add(BatchNormalization(input_dim=x_data.shape[1]))
water_model.add(Dense(1024, activation='relu'))
water_model.add(Dropout(0.35))
water_model.add(BatchNormalization())
water_model.add(Dense(512, activation='tanh'))
water_model.add(Dropout(0.3))
water_model.add(BatchNormalization())
water_model.add(Dense(256, activation='linear'))
water_model.add(Dropout(0.2))
water_model.add(Dense(64, activation='relu'))
water_model.add(Dropout(0.1))
water_model.add(Dense(1, activation='linear'))

water_model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse', metrics=['mae'])

history = water_model.fit(x_train, 
                    y_train, 
                    epochs=200, 
                    batch_size=64,
                    validation_split=0.1, 
                    verbose=1)

water_model.compile(optimizer=Adam(learning_rate=1e-5), loss='mse', metrics=['mae'])

history = water_model.fit(x_train, 
                    y_train, 
                    epochs=60, 
                    batch_size=128,
                    validation_split=0.1, 
                    verbose=1)

water_model.save('model_19_615.h5')

water_model.load_model('model2.h5')

def eval_net(model, x_train, y_train, y_scaler = None, n = 10, limit = 1000.):
  
    pred = model.predict(x_train)            
    if y_scaler:                              
        pred = y_scaler.inverse_transform(pred)
        for i in range(len(pred)):
          if (pred[i]<0):
            pred[i] = 0
        y_train = y_scaler.inverse_transform(y_train)
    print('MSE', mean_absolute_error(pred, y_train), '\n')

    for i in range(n):
        print('Real: {:6.2f}  Pred: {:6.2f}  dif: {:6.2f}'.format(y_train[i,0],pred[i,0],abs(y_train[i,0] - pred[i,0])))

eval_net(water_model, x_test, y_test, y_scaler)