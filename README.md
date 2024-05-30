# Danish Car Price Predictor
Try it out live at: https://danish-car-price-predictor-dot-sylvan-mode-413619.nw.r.appspot.com/  

![image](https://github.com/allanreda/Danish-Car-Price-Predictor/assets/89948110/da13c16e-6393-448d-81ae-e987c3bcc500)  


## Overview
Your traditional car-price-predictor-project but with a Danish twist.  
This repository covers the creation of a web-based ML-application, from data gathering all the way to deployment.
The "Danish twist" comes from the training data which is initially scraped from Bilbasen.dk - Denmarks largest website for sale of used cars. Feature engineering is then conducted on the raw data to create variables for the machine learning models. The models are evaluated and the best one is chosen for deployment. The interface for the model is built simply with Gradio and deployed on App Engine in Google Cloud. 

## Explanation
### development/Webscraping.py
This script uses Selenium to scrape Bilbasen.dk. It scrapes all car data from each page and goes through the website page by page.
The primary function that is activated to do the actual scraping is shown below. It utilizes multiple pre-built functions (detailed in the actual document), each with its own functionality.
```python
def go_through_pages(pages=int):
    
    # Start timer
    start_time = time.time()
    
    # Set up the Chrome WebDriver with the webdriver_manager
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)
    
    driver.get("https://www.bilbasen.dk/brugt/bil?includeengroscvr=true&includeleasing=false&page=1")
    accept_cookies(driver)
    
    # Loop thorugh all listing pages
    for page in range(1, pages):
        # Visit the website
        driver.get(f"https://www.bilbasen.dk/brugt/bil?includeengroscvr=true&includeleasing=false&page={page}")
        
        time.sleep(2)
        
        # Scrape single page
        all_cars_on_page = scrape_all_cars_on_page(driver)
        
        # Time spent scraping the current page
        current_time = time.time()
        elapsed_time = current_time - start_time  # Total elapsed time from start to current point
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Scraped page {page}/{pages} successfully. Time spent so far: {int(hours)} hours, {int(minutes)} minutes, and {int(seconds)} seconds.")
        
        yield all_cars_on_page

    # Total time spent
    end_time = time.time()
    total_elapsed_time = end_time - start_time
    total_hours, remainder = divmod(total_elapsed_time, 3600)
    total_minutes, total_seconds = divmod(remainder, 60)
    print(f"Finished scraping. Total time spent: {int(total_hours)} hours, {int(total_minutes)} minutes, and {int(total_seconds)} seconds.")
```
You might notice that this script scrapes alot more car details, than what is actually used as variables in the later model creation. This is because, at this stage of the project, it wasn't yet decided which variables would be used in the model creation. I chose to play it safe and scrape as many car details as possible, considering the script needs to run for a long time to scrape all pages (1480 pages in this case). It was later choosen to only include the car details that was thought to be "common knowledge" about one's own car and use them as variables.

### development/feature_engineering_and_model_creation.py
As the name of the file states, it consists of two parts, which is feature engineering of the scraped car information, to create variables to be used in the models, and the actual model creation.  
For the feature engineering part, it mostly consists of simple data cleaning and one-hot-encoding, like the examples below.  
```python
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
```
Before the model creation, a quick check for multicollinearity was done using a VIF matrix.  
As can be seen in the screenshot below, a high correlation was discovered on the variables "top_speed" and "gears".  
![image](https://github.com/allanreda/Danish-Car-Price-Predictor/assets/89948110/82340e0c-02cf-48b9-b8dd-2731b3f6e0ca)  
A good data science practice would be to remove one or both of those variables. However, the main focus of this project was the most accurate predictions, which is why both scenarios was tested. Including both variables proved to provide the highest prediction accuracy, which is why they were included anyway. Also, I had decided to only use tree-based models beforehand, because of their robustness to outliers and high predictive powers. In general, tree-based models do not rely on the assumption of no multicollinearity among variables. 

I proceeded to split the data into training and test data using a 80/20 split, and benchmarked five different tree-based models, using the following function.
```python
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
```
The resulting evaluations looked like this:  
![image](https://github.com/allanreda/Danish-Car-Price-Predictor/assets/89948110/cb76b4e5-b516-4a54-92aa-021c6ad8138d)  
As can be seen in the screenshot, the XGBRegressor-model performed best on all parameters when it came to benchmarking.
However, to have a more robust foundation for choosing the XGBRegressor-model instead of the other four models, I needed to ensure that it still performed better across different subsets of the data, and not just the 80/20 split that was done. This is especially important, considering the relatively small size of the dataset, which could easily lead to overfitting.  
I therefore proceeded to do a K-fold cross-validation, using the function below.
```python
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
```
The resulting evaluations from the cross validation looked like this:  
![image](https://github.com/allanreda/Danish-Car-Price-Predictor/assets/89948110/39b9b625-a648-4bc9-9b5a-bfdae66747f7)  
The XGBRegressor-model still seemed to be the be the best choice, even though it was closely followed by the GradientBoostingRegressor.  

I chose the XGBRegressor and decided to do a grid search on it to tune its hyperparameters and therefore possibly improve the model performance, by finding the best combination of hyperparameters.  
The parameter grid was set up and ran on the grid search. The grid search itself was set up using five cross validation folds and MSE (Mean Squared Error) as the scoring metric. 
```python
# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200, 300, 400], 
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.25], 
    'max_depth': [3, 5, 7, 9, 11],  
    'subsample': [0.7, 0.8, 0.9, 1.0]
}

# Setup Grid Search
grid_search = GridSearchCV(estimator=best_eval_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')

# Fit Grid Search
grid_search.fit(X_train, y_train)
```
As shown in the screenshot below, it was possible to slightly improve the model's performance, by tuning its hyperparameters.  
![image](https://github.com/allanreda/Danish-Car-Price-Predictor/assets/89948110/0ad9a6e2-5953-4127-a165-ab1fe968990a)  
  
Lastly, this model was saved as a .joblib file, intended to be used for the application.
```python
# Save the model to a file
dump(best_model, 'deployment/best_model.joblib')
```
### deployment/app.py  
The application is built using the Gradio framework for building a simple interface which the user can interact with and use to predict their cars price. 
The file contains two main parts which together makes the application.  

First part of the application is the backend function that utilizes the XGBRegressor-model built earlier, to make predictions based on its inputs.  
The full function can be seen in the file.
```python
# Function to predict car prices
def predict(age, kilometers, fuel_consumption, top_speed, gears, brand, region, 
            propellant, gear_type):
```
Here, the categorical variables are defined differently from the numerical. Each category needs to be defined, and the category that was initially removed in the feature engineering process to avoid multicollinearity, is handled so that if it's chosen, all other categories is set to zero.  
Below is an example of this.  
```python
# Add region one-hot encoded data
    region_categories =['Region Hovedstaden', 'Region Midtjylland', 'Region Nordjylland', 
                        'Region Sjælland', 'Region Syddanmark']
    
    if region == 'Region Syddanmark':
        # If 'Region Syddanmark', all specific dummy variables are 0
        input_data += [0, 0, 0, 0]
    else:
        input_data += [int(region == r) for r in region_categories if r != 'Region Syddanmark']
```
The second part of the application is where Gradio is used for building the UI. Here, the Gradio classes Blocks(), Row() and Column() is utilized to build the overall structure.  
The input fields are defined either as numerical input fields, where the user can type a number:
```python
kilometers = gr.Number(label="Kilometers")
```
or as categorical input fields where the user can choose an option from a dropdown field:
```python
kilometers = gr.Number(label="Kilometers")
```
## Technologies
The project is built using:  
-Scikit-Learn, xgboost and lightgbm for machine learning and validation  
-Selenium for web scraping  
-Gradio for interface building  
-App Engine (Google Cloud) and YAML for deployment  
-Pandas and NumPy for data manipulation  
