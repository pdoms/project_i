
from matplotlib import colors
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys




sys.path.append("..")
from tools.linear_regression.linear_regression import LinReg, get_column, scale_for_print

###load dataset 

# source: https://www.kaggle.com/nehalbirla/vehicle-dataset-from-cardekho?select=Car+details+v3.csv
dir_path = os.path.dirname(os.path.realpath(__file__))
source = os.path.join(dir_path, 'data', 'Cars', 'car_details.csv')
df = pd.read_csv(source)
pd.set_option("display.max_rows", None, "display.max_columns", None)

###remove units indicators and change fields' dtype to float64
removable_units = [('mileage', ['kmpl', 'km/kg']), ('engine', 'CC'), ('max_power', 'bhp') ]

def replace_convert(value, unit='', *args, **kwargs):
    if type(value) == str:
        if unit in value:
            value = value.replace(unit, '').strip()
            try:
                return float(value)
            except:
                return 0.0
        else:
            return float(value)
    else:
        return value        

def mileage(value):
    # 1 liter mean(LPG, gas) = approx. 730gr - thus, e.g., 730gr * 1.37 = 1,000.1
    if type(value) == str:
        if 'kmpl' in value:
            return float(value.replace('kmpl', '').strip())
        elif 'km/kg' in value:
            return float(value.replace('km/kg', '').strip()) * 1.37     
    else:
        return value

df['engine'] = df['engine'].apply(replace_convert, unit='CC')
df['max_power'] = df['max_power'].apply(replace_convert, unit='bhp')
df['mileage'] = df['mileage'].apply(mileage)


# util function that was run to determine levels of some variables
counts = {}
def count(value, counts):
    if type(value) == str:
        value = value.replace(' ', '_')
    if value in counts.keys():
        counts[value] += 1
    else: 
        counts[value] = 1
# results



#fuel = {'Diesel': 4402, 'Petrol': 3631, 'LPG': 38, 'CNG': 57}
#seller_type = {'Individual': 6766, 'Dealer': 1126, 'Trustmark_Dealer': 236}
#transmission = {'Manual': 7078, 'Automatic': 1050}
#owner = {'First_Owner': 5289, 'Second_Owner': 2105, 'Third_Owner': 555, 'Fourth_&_Above_Owner': 174, 'Test_Drive_Car': 5}

### convet categoricals to numericals - first attempt: simple number coding

def numerical_fuel(value):
    if value == 'Diesel':
        return 0
    elif value == 'Petrol':
        return 1
    elif value == 'LPG' or value == 'CNG': 
        return 2
    else:
        return 2

def nummerical_seller(value):
    if value == 'Dealer' or value == 'Trustmark Dealer':
        return 1
    elif value == 'Individual':
        return 0
    else: 
        return 1

def numerical_trans(value):
    if value == 'Manual':
        return 0
    elif value == 'Automatic':
        return 1
    else:
        return 1
        

def numerical_owner(value):
    if value == 'First Owner':
        return 0
    elif value == 'Second Owner':
        return 1
    elif value == 'Third Owner':
        return 2
    elif value == 'Fourth & Above Owner':
        return 3
    elif value == 'Test_Drive_Car':
        return 4
    else:
        return 4


categorical = [('fuel', numerical_fuel), ('seller_type', nummerical_seller), ('transmission', numerical_trans), ('owner', numerical_owner)]
for col, func in categorical:
    df[col] = df[col].apply(func)

### convert seling price
    # no explanation in the dataset, but since the source is CarDheko, based in India and prices are indicated in LAKH, it is assumed
    # that the prices in the dataset are INR, because the mean is at 638000, which would equal 63,800,000,000 INR converted from LAKH
    # 100,000 INR equals 1342.73 USD (mid-market exchange rate at 10:45 UTC, Nov 3, 2021) will result in using a conversion rate of 74.5 (INR to USD)
    # example selling_price: 290000 INR;  3,892.62 USD

def convert_to_USD(value):
    return value/74.5
df['selling_price'] = df['selling_price'].apply(convert_to_USD)





### transform selling price to python list to use as target variable/y
target = df['selling_price'].tolist()

### drop unwanted columms: name is irrelevant, torque seems irrelevant as max_power is there (at early state), selling_price is used as target
df.drop(labels=['selling_price', 'name', 'torque'], axis=1, inplace=True)

for col in df.columns:
    df[col].fillna(value=df[col].mean(), inplace=True)

### create python list of x variables

x_vars = df.values.tolist()

#prepare column names
tar_label = ['selling_price']
cols = [*tar_label, *df.columns]

### get feel for data by looking at the individual graphs in relation to target













