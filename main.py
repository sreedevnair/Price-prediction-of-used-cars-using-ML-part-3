import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

df = pd.read_csv('car_data.csv')

# Encoding the categorical data
df.replace({'Fuel_Type':{'Petrol':0, 'Diesel':1, 'CNG':2}, 
            'Seller_Type':{'Dealer':0, 'Individual':1}, 
            'Transmission':{'Manual':0, 'Automatic':1}}, inplace=True)

new_df = df.groupby('Car_Name').filter(lambda x: len(x) >= 3)


dummies = pd.get_dummies(new_df['Car_Name'], dtype=int)
df1 = pd.concat([new_df, dummies], axis=1)

X = df1.drop(['Selling_Price', 'Car_Name'], axis=1)  
y = df1['Selling_Price'] 

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Defining the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(X_train.shape[1],), activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)  # Output layer (for regression tasks)
])


model.compile(optimizer='adam',  
              loss='mean_squared_error', 
              metrics=[tf.keras.metrics.RootMeanSquaredError()])  

# Train the model
history = model.fit(X_train, y_train, 
                    validation_data=(X_test, y_test), 
                    epochs=10)

# Evaluate the model
loss, rmse = model.evaluate(X_test, y_test)

print("Root Mean Squared Error (RMSE) of the model is:", rmse)
print("Loss (MSE) of the model is:", loss)
