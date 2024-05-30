import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service


#______________________________________________________________________________
        
# Function for handling table data
def handle_table_data(data, car_data_dict):
    # Split the data into lines
    lines = data.split('\n')
    # Skip the first line if it is a headline
    start_index = 1 if lines[0] == 'Detaljer' else 0
    # Iterate over the lines skipping "Detaljer" and checking if the next index is within the list range
    for i in range(start_index, len(lines), 2):
        label = lines[i]  # The label
        # Check if the next index exists in the list before assigning the value
        if i + 1 < len(lines):  # Checking if the next index is within bounds
            value = lines[i + 1]  # The value
            car_data_dict[label] = value  # Add to dictionary
        else:
            print(f"Missing value for label: {label}")

# Function for scraping info about a single car
def scrape_single_car(driver):
    
    # Save the scraped values in a dict
    car_data = {'brand': '',
                'model': '',
                'sub_model': '',
                'price': '',
                'address': ''}
    
    # Brand
    try:
        brand = '.bas-MuiBreadcrumbs-li:nth-child(1) .bas-MuiTypography-colorPrimary'
        car_data['brand'] = driver.find_element(By.CSS_SELECTOR, brand).text
    except Exception as e:
        print(f'An error occurred while finding brand: {e}')
        
    # Model
    try:
        model = '.bas-MuiBreadcrumbs-li:nth-child(3) .bas-MuiTypography-colorPrimary'
        car_data['model'] = driver.find_element(By.CSS_SELECTOR, model).text
    except Exception as e:
        print(f'An error occurred while finding model: {e}')

    # Sub model
    try:
        sub_model = '.bas-MuiBreadcrumbs-li:nth-child(5) .bas-MuiTypography-colorPrimary'
        car_data['sub_model'] = driver.find_element(By.CSS_SELECTOR, sub_model).text
    except Exception as e:
        print(f'An error occurred while finding sub model: {e}')

    # Price
    try:
        price = '.bas-MuiCarPriceComponent-value'
        car_data['price'] = driver.find_element(By.CSS_SELECTOR, price).text
    except Exception as e:
        print(f'An error occurred while finding price: {e}')
        
    # Sellers address
    try:
        address = '.bas-MuiSellerInfoComponent-addressWrapper'
        car_data['address'] = driver.find_element(By.CSS_SELECTOR, address).text
    except Exception as e:
        print(f'An error occurred while finding address: {e}')
        
    # Scrape details table 
    try:
        details_table = '//*[@id="root"]/div[3]/div[2]/div[2]/article/main/div[4]'
        details_table_data = driver.find_element(By.XPATH, details_table).text
    except Exception as e:
        details_table_data = ''
        print(f'An error occurred while finding car details: {e}')
    
    # Scrape table about general model info
    try:
        model_info_table = '//*[@id="root"]/div[3]/div[2]/div[2]/article/main/div[6]/div/table'
        model_info_table_data = driver.find_element(By.XPATH, model_info_table).text
    except Exception as e:
        model_info_table_data = ''
        print(f'An error occurred while finding model info: {e}')

    if details_table_data:
        try:
            # Handle table data for car details and include it in the dict
            handle_table_data(details_table_data, car_data)
        except Exception as e:
            print(f'An error occurred in handling details_table_data: {e}')
    
    if model_info_table_data:
        try:
            # Handle table data for model info and include it in the dict
            handle_table_data(model_info_table_data, car_data)
        except Exception as e:
            print(f'An error occurred in handling model_info_table_data: {e}')
    
    return car_data

# Function to go through each car link on a page and utilize the scrape_single_car function
def scrape_all_cars_on_page(driver, links = 30):
    
    # Initialize empty list
    all_cars_on_page = []
    
    # Loop through all car links on the listing page
    for link in range(1, links):
        # Click on car link
        try:
            time.sleep(2)
            car_link = f'//*[@id="__next"]/div[3]/div[2]/div/main/section[1]/article[{link}]/div/div[2]/div/div[2]/div[1]/a'
            driver.find_element(By.XPATH, car_link).click()
            
        except Exception as e: 
            print(f"Scraping car link failed: {e}")
            continue
        
        # Scrape all data for that car
        car_data = scrape_single_car(driver)
        
        # Append car data to the list
        all_cars_on_page.append(car_data)
        
        # Go back to the listing page
        driver.back()
    
    return all_cars_on_page
    
# Function to accept cookies to enter the website
def accept_cookies(driver):
    try:
        # Wait for the iframe that contains the cookie consent to load and switch to it
        WebDriverWait(driver, 10).until(
            EC.frame_to_be_available_and_switch_to_it((By.CSS_SELECTOR, "iframe[id^='sp_message_iframe_']"))
        )

        # Now that we're in the iframe, locate the accept button (update the selector as needed)
        accept_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "body > div.message-safe-area-holder.message-safe-area-holder--safe-bottom > div > div > div > div.message-component.message-row.buttons-container.modal-inset > button.message-component.message-button.no-children.focusable.secondary-button.only-necessary-button.sp_choice_type_REJECT_ALL")) # Update #accept to the correct button ID or selector
        )
        accept_button.click()

        # Switch back to the default content
        driver.switch_to.default_content()
        print('Cookies accepted succesfully')
    except Exception as e:
        print(f"An exception occurred while accepting cookies: {e}")

# Main function that goes through all pages
def go_through_pages(pages=int):
    
    # Start timer
    start_time = time.time()
    
    # Set up the Chrome WebDriver with the webdriver_manager
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)
    
    # Open the first page to prompt the cookie box
    driver.get("https://www.bilbasen.dk/brugt/bil?includeengroscvr=true&includeleasing=false&page=1")
    # Accept cookies
    accept_cookies(driver)
    
    # Loop thorugh all listing pages
    for page in range(1, pages):
        # Open the page (which is now possible because the cookies are accepted)
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
    
    

def convert_dicts_to_df(all_cars_data):
    # Initialize empty dataframe
    all_cars_df = pd.DataFrame()

    # Loop through all lists in all_cars_data
    for i in range(1, len(all_cars_data)):
        # Convert list into dataframe
        cars_on_page = pd.DataFrame(all_cars_data[i])
        # Concatenate dataframe to the main dataframe
        all_cars_df = pd.concat([all_cars_df, cars_on_page], axis = 0)
        
    return all_cars_df


# Initialize an empty list to hold all car data lists
all_cars_data = []

# Loop through all existing pages
for page in go_through_pages(1480):
    all_cars_data.append(page)

# Convert the dict to a dataframe
all_cars_df = convert_dicts_to_df(all_cars_data)

# Only keep the variables that will be used for the model
all_cars_df = all_cars_df[['brand', 'price', 'address', 'Modelår', 'Kilometertal', 
                           'Drivmiddel', 'Brændstofforbrug', 'Tophastighed', 
                           'Geartype', 'Antal gear']]

# Save dataframe as Excel
all_cars_df.to_excel('all_cars_data.xlsx', engine='openpyxl', index=False)
