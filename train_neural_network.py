import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import joblib

# Load the prepared data from 'Team Important Stats 2010-2025.csv'
data = pd.read_csv('Team Important Stats 2010-2025.csv')

# Extract features (X) and target (y)
X = data[['OBP', 'SLG', 'ERA']]
y = data['Win Rate']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler for future use
joblib.dump(scaler, 'scaler.pkl')

# Revert to the original model architecture
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
model.fit(
    X_train, y_train,
    epochs=50,  # Original number of epochs
    batch_size=16,  # Original batch size
    validation_split=0.2
)

# Save the trained model
model.save('win_rate_predictor_model.h5')

print("Model training complete and saved as 'win_rate_predictor_model.h5'.")
# Evaluate the model on the test set    
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Test MAE: {mae}")
print(f"Test MSE: {mse}")
print(f"Test RMSE: {rmse}")