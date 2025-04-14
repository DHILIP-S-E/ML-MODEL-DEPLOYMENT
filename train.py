import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import pickle

# Expanded car price dataset
data = {
    'Year': [2015, 2012, 2019, 2017, 2013, 2018, 2020, 2014, 2016, 2011, 2021, 2017, 2019, 2014, 2015],
    'Km_Driven': [50000, 70000, 20000, 40000, 60000, 15000, 10000, 80000, 40000, 90000, 5000, 30000, 25000, 70000, 45000],
    'Fuel_Type': ['Petrol', 'Diesel', 'Petrol', 'Diesel', 'Petrol', 'Diesel', 'Petrol', 'Diesel', 'Petrol', 'Diesel', 'Petrol', 'Diesel', 'Petrol', 'Diesel', 'Petrol'],
    'Price': [350000, 280000, 500000, 420000, 300000, 550000, 600000, 250000, 350000, 220000, 700000, 400000, 520000, 280000, 370000]
}

# Create DataFrame
df = pd.DataFrame(data)

# Convert categorical column 'Fuel_Type' into numerical (Petrol = 0, Diesel = 1)
label_encoder = LabelEncoder()
df['Fuel_Type'] = label_encoder.fit_transform(df['Fuel_Type'])

# Features (X) and target (y)
X = df[['Year', 'Km_Driven', 'Fuel_Type']]
y = df['Price']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
with open("car_price_model.pkl", "wb") as file:
    pickle.dump(model, file)

# Evaluate the model
score = model.score(X_test, y_test)
print(f"Model Accuracy: {score*100:.2f}%")
