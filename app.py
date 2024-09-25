from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load your dataset
df = pd.read_csv("amazon_stock_data_with_details.csv")

# Prepare data
X = df[['Open', 'High', 'Low', 'Volume']]  # Features
y = df['Close']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    open_price = float(request.form['open'])
    high_price = float(request.form['high'])
    low_price = float(request.form['low'])
    volume = float(request.form['volume'])

    # Prepare data for prediction
    features = pd.DataFrame([[open_price, high_price, low_price, volume]], columns=['Open', 'High', 'Low', 'Volume'])

    # Make prediction
    prediction = int(model.predict(features)[0])

    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
