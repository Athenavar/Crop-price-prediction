from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from statistics import mode

app = Flask(__name__)  # Corrected app initialization

# Step 1: Load the dataset
def load_data(file_path):
    df = pd.read_csv(file_path)  # File path as a string
    
    df['Date'] = pd.to_datetime(df['Date'])  # Ensure 'Date' is in datetime format
    return df

# Step 2: Preprocess the dataset
def preprocess_data(df):
    df = df.dropna()  # Remove missing data
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    return df

# Step 3: Train the model and predict prices
def train_and_predict(df):
    df['DayOfYear'] = df['Date'].dt.dayofyear  # Extract day of the year as a feature
    X = df[['DayOfYear']]
    y = df['Price']
    
    # Train the RandomForest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Predict prices and store in a new column
    df['Predicted_Price'] = model.predict(X)
    return df

# Step 4: Calculate modal prices
def calculate_monthly_modal_price(df, selected_center, selected_commodity):
    # Filter the data
    filtered_df = df[(df['Centre Name'] == selected_center) & (df['Commodity'] == selected_commodity)]
    
    # Group by month and calculate modal price
    monthly_modal_prices = (
        filtered_df.groupby(filtered_df['Date'].dt.to_period('M'))['Predicted_Price']
        .apply(lambda x: mode(x.round()))
        .reset_index()
    )
    monthly_modal_prices.columns = ['Month', 'Modal_Price']
    monthly_modal_prices['Month'] = monthly_modal_prices['Month'].astype(str)  # Convert Period to string
    return monthly_modal_prices

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from form
    selected_center = request.form.get('center')
    selected_commodity = request.form.get('commodity')
    # Load, preprocess, and predict
    df = load_data('Agmarknet_Price_Report.csv')  # Update the file path
    preprocessed_data = preprocess_data(df)
    predicted_data = train_and_predict(preprocessed_data)

    # Calculate modal prices
    monthly_modal_prices = calculate_monthly_modal_price(predicted_data, selected_center, selected_commodity)

    # Render results
    return render_template('index.html', predictions=monthly_modal_prices.values)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
