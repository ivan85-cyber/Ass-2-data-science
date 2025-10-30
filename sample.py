# Import libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Create a sample dataset
data = {
    'Speed': [80, 40, 100, 60, 90, 50],
    'Weather': [0, 1, 0, 1, 0, 1],   # 0=Clear, 1=Rainy
    'Road_Type': [1, 0, 1, 0, 1, 0], # 1=Highway, 0=Rural
    'Driver_Age': [25, 40, 30, 22, 28, 35],
    'Accident_Severity': [3, 1, 4, 2, 3, 2]
}

df = pd.DataFrame(data)

# Define independent (X) and dependent (y) variables
X = df[['Speed', 'Weather', 'Road_Type', 'Driver_Age']]
y = df['Accident_Severity']

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Save model for future use
joblib.dump(model, 'road_accident_model.pkl')

# Predict for a hypothetical scenario
hypothetical_case = pd.DataFrame({
    'Speed': [85],
    'Weather': [1],   # Rainy
    'Road_Type': [0], # Rural
    'Driver_Age': [30]
})

prediction = model.predict(hypothetical_case)
print(f"Predicted Accident Severity: {prediction[0]:.2f}")
