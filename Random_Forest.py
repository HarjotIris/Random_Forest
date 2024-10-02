# Learning Random Forests
import pandas as pd
# load data
iowa_file_path = r"C:\Users\HP\train.csv"
home_data = pd.read_csv(iowa_file_path)
# filtering rows with missing values
#filtered_home_data = home_data.dropna(axis=0)

# create target object
y = home_data.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF',
             'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
x = home_data[features]

from sklearn.model_selection import train_test_split
train_x, val_x, train_y, val_y = train_test_split(x, y, random_state=1)

# Building a Random Forest Model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

rf_model = RandomForestRegressor(random_state=0)
rf_model.fit(train_x, train_y)
preds_val = rf_model.predict(val_x)
print("Validation MAE for random forest model : {}".format(
    mean_absolute_error(val_y, preds_val)))