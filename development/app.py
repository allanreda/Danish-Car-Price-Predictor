import gradio as gr
from joblib import load

# Load the model
model = load('best_xgb_model.joblib')

# Function to predict car prices
def predict(age, kilometers, fuel_consumption, top_speed, gears, brand, region, 
            propellant, gear_type):
    # Initialize input data with numeric fields
    input_data = [age, kilometers, fuel_consumption, top_speed, gears]

    # Add brand one-hot encoded data
    input_data += [int(brand == b) for b in ['Abarth', 'Alfa Romeo', 'Aston Martin', 'Audi', 'BMW', 'Chevrolet',
                                             'Citroën', 'Cupra', 'DS', 'Dacia', 'Dodge', 'Ferrari', 'Fiat', 'Ford',
                                             'Honda', 'Hyundai', 'Jaguar', 'Jeep', 'Kia', 'Land Rover', 'MG', 'MINI',
                                             'Maserati', 'Mazda', 'Mercedes', 'Mitsubishi', 'Nissan', 'Opel', 'Peugeot',
                                             'Porsche', 'Renault', 'Seat', 'Skoda', 'Subaru', 'Suzuki', 'Toyota', 'VW',
                                             'Volvo']]

    # Add region one-hot encoded data
    region_categories =['Region Hovedstaden', 'Region Midtjylland', 'Region Nordjylland', 
                        'Region Sjælland', 'Region Syddanmark']
    
    if region == 'Region Syddanmark':
        # If 'Region Syddanmark', all specific dummy variables are 0
        input_data += [0, 0, 0, 0]
    else:
        input_data += [int(region == r) for r in region_categories if r != 'Region Syddanmark']

    # Add propellant one-hot encoded data
    propellant_categories = ['Benzin', 'Diesel', 'Hybrid'] 
    if propellant == 'Hybrid':
        # If 'Hybrid', all specific propellant dummy variables are 0
        input_data += [0, 0]
    else:
        input_data += [int(propellant == p) for p in propellant_categories if p != 'Hybrid']

    # Add gear_type one-hot encoded data
    gear_type_categories = ['Automatisk', 'Manuel']
    if gear_type == 'Manuel':
        # If 'Manuel', the 'Automatisk' dummy variables are 0
        input_data += [0]
    else:
        input_data += [int(gear_type == g) for g in gear_type_categories if g != 'Manuel']
    

    # Convert to 2D array as expected by the model
    input_data = [input_data]
    prediction = model.predict(input_data)
    return prediction[0]

#______________________________________________________________________________

# CSS to center title
css = """
h1 {
    text-align: center;
    display: block;
}

"""
# Define Gradio blocks 
with gr.Blocks(theme='snehilsanyal/scikit-learn', title = 'Danish Car Price Predictor', css = css) as app:
    # Set title
    gr.Markdown("# Danish Car Price Predictor")
    
    # Define row for input fields
    with gr.Row():
        # First column of input fields
        with gr.Column():
            age = gr.Number(label="Age")
            kilometers = gr.Number(label="Kilometers")
            fuel_consumption = gr.Number(label="Fuel Consumption")
        # Second column of input fields
        with gr.Column():
            top_speed = gr.Number(label="Top Speed")
            gears = gr.Number(label="Gears")
            gear_type = gr.Dropdown(choices=['Automatisk', 'Manuel'], label="Gear Type")
        # Third column of input fields
        with gr.Column():
            brand = gr.Dropdown(choices=['Abarth', 'Alfa Romeo', 'Aston Martin', 'Audi', 'BMW', 'Chevrolet',
                                         'Citroën', 'Cupra', 'DS', 'Dacia', 'Dodge', 'Ferrari', 'Fiat', 'Ford',
                                         'Honda', 'Hyundai', 'Jaguar', 'Jeep', 'Kia', 'Land Rover', 'MG', 'MINI',
                                         'Maserati', 'Mazda', 'Mercedes', 'Mitsubishi', 'Nissan', 'Opel', 'Peugeot',
                                         'Porsche', 'Renault', 'Seat', 'Skoda', 'Subaru', 'Suzuki', 'Toyota', 'VW',
                                         'Volvo'], label="Brand")
            region = gr.Dropdown(choices=['Region Hovedstaden', 'Region Midtjylland', 'Region Nordjylland', 
                                          'Region Sjælland', 'Region Syddanmark'], label="Region")
            propellant = gr.Dropdown(choices=['Benzin', 'Diesel', 'Hybrid'], label="Propellant")
    # Define row for submit and output field
    with gr.Row():
        # Single column for submit and output field
        with gr.Column():
         submit_button = gr.Button("Submit")
         prediction = gr.Text(label="Prediction")
    # Define arguments for the submit button
    submit_button.click(
        fn=predict, # The 'predict' function
        inputs=[age, kilometers, fuel_consumption, top_speed, gears, brand, 
                region, propellant, gear_type], # Inputs
        outputs=prediction # Output
    )

app.launch()


