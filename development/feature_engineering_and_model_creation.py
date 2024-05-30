import pandas as pd
import numpy as np
import datetime
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score
from joblib import dump

# Import raw data from Excel-file
raw_df = pd.read_excel("all_cars_data.xlsx") 

# Drop all rows with NA-values
cars_df = raw_df[raw_df != '-']
cars_df = cars_df.dropna().reset_index(drop = True)
cars_df = cars_df.drop_duplicates()

###############################################################################
############################# FEATURE ENGINEERING #############################
###############################################################################

#_______________________________ BRAND COLUMN _________________________________

# Count number of each brand
cars_df['brand'].value_counts()

# One-hot-encoding the brand column
cars_df = pd.get_dummies(cars_df, columns = ['brand'])

# Remove column to avoid multicollinearity
cars_df = cars_df.drop(['brand_VW'], axis=1)
#_______________________________ PRICE COLUMN _________________________________

# Remove non-numeric characters from the price column
cars_df['price'] = cars_df['price'].str.replace(r'\D', '', regex=True)

# Convert the price column to integer
cars_df['price'] = pd.to_numeric(cars_df['price'])

#___________________________ ADDRESS/REGION COLUMN ____________________________

# Import Excel-file with columns
postal_codes_df = pd.read_excel("development\postal_codes_regions.xlsx") 

# Extract postal codes 
cars_df['address'] = cars_df['address'].str.extract(r'(\b\d{4}\b)').astype(int)

# Create a dictionary mapping postal codes to regions
postal_code_region_map = dict(zip(postal_codes_df['POSTNR'], postal_codes_df['ADRESSERINGSNAVN']))
# Map the postal codes to regions and replace values in the address column
cars_df['address'] = cars_df['address'].map(postal_code_region_map)

# Rename column
cars_df.rename(columns = {'address':'region'}, inplace = True) 

# Count number of each region
cars_df['region'].value_counts()

# One-hot-encoding the address column
cars_df = pd.get_dummies(cars_df, columns = ['region'])

# Remove column to avoid multicollinearity
cars_df = cars_df.drop(['region_Region Syddanmark'], axis=1)
#______________________________ YEAR/AGE COLUMN _______________________________

# Rename column
cars_df.rename(columns = {'Modelår':'Age'}, inplace = True) 

# Get current year
today = datetime.date.today()
current_year = today.year

# Calculate age of the car based on the current year
cars_df['Age'] = current_year - cars_df['Age'] 

#______________________________ KILOMETER COLUMN ______________________________

# Rename column
cars_df.rename(columns = {'Kilometertal':'kilometers'}, inplace = True) 

# Replace all "ny-bil" (new car) values with 0 kilometers
cars_df['kilometers'].replace('(ny bil)', int(0), inplace=True)

# Remove non-numeric characters from the kilometers column
cars_df['kilometers'] = cars_df['kilometers'].str.replace(r'\D', '', regex=True)

# Convert the kilometers column to integer
cars_df['kilometers'] = pd.to_numeric(cars_df['kilometers'])

#______________________________ PROPELLANT COLUMN _____________________________

# Rename column
cars_df.rename(columns = {'Drivmiddel':'propellant'}, inplace = True) 

# Count number of each brand
cars_df['propellant'].value_counts()

# Combine different hybrid values into a single 'Hybrid' value
cars_df['propellant'].replace(['Plug-in hybrid Benzin', 'Hybrid (Benzin + El)', 'Plug-in hybrid Diesel'], 'Hybrid', inplace=True)

# One-hot-encoding the propellant column
cars_df = pd.get_dummies(cars_df, columns = ['propellant'])

# Remove column to avoid multicollinearity
cars_df = cars_df.drop(['propellant_Hybrid'], axis=1)

#___________________________ FUEL CONSUMPTION COLUMN __________________________

# Rename column
cars_df.rename(columns = {'Brændstofforbrug':'fuel_consumption'}, inplace = True) 

# Remove non-numeric characters from the fuel_consumption column
cars_df['fuel_consumption'] = cars_df['fuel_consumption'].str.replace(r'\D', '', regex=True)

# Convert the kilometers column to integer
cars_df['fuel_consumption'] = pd.to_numeric(cars_df['fuel_consumption'])
cars_df['fuel_consumption'] = cars_df['fuel_consumption'] / 10

#______________________________ TOP SPEED COLUMN ______________________________

# Rename column
cars_df.rename(columns = {'Tophastighed':'top_speed'}, inplace = True) 

# Remove non-numeric characters from the top_speed column
cars_df['top_speed'] = cars_df['top_speed'].str.replace(r'\D', '', regex=True)

# Convert the top_speed column to integer
cars_df['top_speed'] = pd.to_numeric(cars_df['top_speed'])

#______________________________ GEARTYPE COLUMN _______________________________

# Rename column
cars_df.rename(columns = {'Geartype':'gear_type'}, inplace = True) 

# Count number of each gear type
cars_df['gear_type'].value_counts()

# One-hot-encoding the gear_type column
cars_df = pd.get_dummies(cars_df, columns = ['gear_type'])

# Remove column to avoid multicollinearity
cars_df = cars_df.drop(['gear_type_Manuel'], axis=1)

#_______________________________ GEARS COLUMN _________________________________

# Rename column
cars_df.rename(columns = {'Antal gear':'gears'}, inplace = True) 

# Count number of each gear 
cars_df['gears'].value_counts()

# Convert the gears column to integer
cars_df['gears'] = pd.to_numeric(cars_df['gears'])

#___________________________ FINAL DATA CLEANUPS ______________________________

# Convert inf and NA values to 0
cars_df.fillna(0, inplace=True)  
cars_df.replace([np.inf, -np.inf], 0, inplace=True)  

# Convert all true/false values to numeric binary values (1/0)
cars_df = cars_df.astype(int)

###############################################################################
############################## MULTICOLLINEARITY ##############################
###############################################################################

# Drop the target variable 
X = cars_df.drop(columns=['price']) 

# Calculate VIF for each predictor variable 
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Sort the VIF data from highest to lowest
vif_data_sorted = vif_data.sort_values(by='VIF', ascending=False)

# Print the sorted VIF data
print("\nVIF (Sorted):")
print(vif_data_sorted)

###############################################################################
########################## MODEL CREATION / VALIDATION ########################
###############################################################################

# Define price as the target variable (y)
y = cars_df['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Initialize the models that will be trained and tested
models = {
    'XGBRegressor': XGBRegressor(random_state=42),
    'RandomForestRegressor': RandomForestRegressor(random_state=42),
    'GradientBoostingRegressor': GradientBoostingRegressor(random_state=42),
    'LightGBMRegressor': LGBMRegressor(random_state=42),
    'ExtraTreesRegressor': ExtraTreesRegressor(random_state=42)
}

#______________________________ BENCHMARKING __________________________________

# Function for training, testing and evaluating models
def train_test_models(X_train, X_test, y_train, y_test, model):
    
    # Extract the name of the model
    model_name = type(model).__name__
    # Train the model
    model.fit(X_train, y_train)
    # Make predictions
    predictions = model.predict(X_test)
    
    # Evaluate model
    r2 = r2_score(y_test, predictions) # Compute R^2 scores
    rmse = np.sqrt(mean_squared_error(y_test, predictions)) # Compute RMSE
    mae = mean_absolute_error(y_test, predictions) # Compute MAE
    mse = mean_squared_error(y_test, predictions) # Compute MSE
    
    # Input the evaluations in a dict
    bench_eval_dict = {'Model': model_name,
                 'r2': r2,
                 'rmse': rmse,
                 'mae': mae,
                 'mse': mse}
    
    return bench_eval_dict
    
# Initialize empty lists to store evaluation dictionaries
bench_eval_dicts = []

# Iterate over models
for model_name, model in models.items():
    # Call train_test_models function for each model
    bench_eval_dict = train_test_models(X_train, X_test, y_train, y_test, model)
    bench_eval_dicts.append(bench_eval_dict)


# Create dataframe from evaluation dictionaries
bench_eval_df = pd.DataFrame(bench_eval_dicts)
# Make MSE more readable
bench_eval_df['mse'] = bench_eval_df['mse'].apply(lambda x: f"{x:.2f}")


#______________________________ CROSS VALIDATION ______________________________

# Function for performing K-fold cross-validation
def evaluate_model_cross_val(model, X, y, folds=5):
    
    # Extract the name of the model
    model_name = type(model).__name__
    
    # Configure the cross validation 
    cv = KFold(n_splits=folds, shuffle=True, random_state=42)
    
    # Evaluate model
    mae_scores = -cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    mse_scores = -cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
    rmse_scores = np.sqrt(mse_scores)
    r2_scores = cross_val_score(model, X, y, scoring='r2', cv=cv, n_jobs=-1)
    
    # Input the evaluations in a dict
    cross_eval_dict = {'Model': model_name,
                 'r2': np.mean(r2_scores),
                 'rmse': np.mean(rmse_scores),
                 'mae': np.mean(mae_scores),
                 'mse': np.mean(mse_scores)}
    return cross_eval_dict


# Initialize empty lists to store evaluation dictionaries
cross_eval_dicts = []

# Iterate over models
for name, model in models.items():
    # Call evaluate_model_cross_val function for each model
    cross_eval_dict = evaluate_model_cross_val(model, X, y)
    cross_eval_dicts.append(cross_eval_dict)

# Create dataframe from evaluation dictionaries
cross_eval_df = pd.DataFrame(cross_eval_dicts)
# Make MSE more readable
cross_eval_df['mse'] = cross_eval_df['mse'].apply(lambda x: f"{x:.2f}")

#___________________________ HYPERPARAMETER TUNING ____________________________

# Initialize the model
best_eval_model = XGBRegressor(random_state=42)

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200, 300, 400], 
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.25], 
    'max_depth': [3, 5, 7, 9, 11],  
    'subsample': [0.7, 0.8, 0.9, 1.0]
}


# Setup Grid Search
grid_search = GridSearchCV(estimator=best_eval_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')

# Fit Grid Search
grid_search.fit(X_train, y_train)

# Best parameters found
print("Best parameters found: ", grid_search.best_params_)

# Use the best estimator to make predictions
best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)

# Evaluate the best model
r2 = r2_score(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)

print(f"R2: {r2}, RMSE: {rmse}, MAE: {mae}, MSE: {mse}")

# Save the model to a joblib file
dump(best_model, 'deployment/best_model.joblib')
