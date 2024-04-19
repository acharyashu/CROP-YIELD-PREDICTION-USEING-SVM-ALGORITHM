from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

app = Flask(__name__)
bootstrap = Bootstrap(app)

# Read the dataset from a file
file_path = 'Dataset.csv'  # Replace 'your_dataset.csv' with the actual path to your dataset file
data = pd.read_csv(file_path)

# Select relevant features and target variable
X = data[['Temperature', 'Rainfall', 'PH', 'Nitrogen']]
y = data['Yield']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Support Vector Regression model
svr = SVR(kernel='rbf')  # Radial basis function kernel is commonly used for SVR
svr.fit(X_train, y_train)

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_yield = None
    if request.method == 'POST':
        # Get input values from the form
        temperature = float(request.form['temperature'])
        rainfall = float(request.form['rainfall'])
        ph = float(request.form['ph'])
        nitrogen = float(request.form['nitrogen'])
        
        # Predict yield for the input values
        new_input = [[temperature, rainfall, ph, nitrogen]]
        predicted_yield = svr.predict(new_input)[0]

    return render_template('index.html', predicted_yield=predicted_yield)

if __name__ == '__main__':
    app.run(debug=True)
