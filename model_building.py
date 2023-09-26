import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data from an Excel file
df = pd.read_excel('Bending-modulus-updated.xlsx')
X_text = df['Lipid composition (molar)']
y = df['kappa, kT (q^-4)']

# Convert text-based lipid compositions to numerical arrays (one-hot encoding)
mlb = MultiLabelBinarizer()
X_numeric = mlb.fit_transform(X_text.apply(lambda x: set(x.split('; '))))

# Define a deep learning model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_numeric.shape[1],)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)  # Output layer for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Train the model
model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=2)

# Evaluate Model Performance
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"R-squared: {r2}")


# Create a scatter plot with regression line
plt.figure(figsize=(8, 8))
sns.regplot(x=y_test, y=y_pred, scatter_kws={'alpha': 0.5})
plt.xlabel("Actual Kappa Values")
plt.ylabel("Predicted Kappa Values")
plt.title(f"Actual vs. Predicted Kappa Values\nR-squared = {r2:.2f}")
plt.grid(True)
# plt.show()

# Predict Kappa for New Inputs (Inference)
# Assuming you have your new lipid composition as a list of molar percentages

new_input_compositions = ["100% POPC","100% DAPC"]
predicted_kappas = []

# Use the same scaler as used for the training data
# Do not re-fit the scaler, only transform the new input compositions
for new_input_composition in new_input_compositions:

    # Convert the new lipid composition to a numerical format using MultiLabelBinarizer
    new_input_numeric = mlb.transform([new_input_composition])

    # Standardize the new input using the same StandardScaler (transform, not fit_transform)
    new_input_scaled = scaler.transform(new_input_numeric)

    # Use the trained model to make predictions for the standardized new input
    predicted_kappa = model.predict(new_input_scaled)

    predicted_kappas.append(predicted_kappa[0][0])

# Print the predicted kappa values for the new inputs
for i, composition in enumerate(new_input_compositions):
    print(f"Predicted Kappa for '{composition}': {predicted_kappas[i]}")
