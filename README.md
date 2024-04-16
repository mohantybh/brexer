import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Sample historical data: [Temperature, Humidity]
# You can replace this with your actual data.
data = np.array([[25, 60],
                 [28, 55],
                 [30, 58],
                 [32, 62],
                 [29, 59],
                 [27, 57]])

# Split the data into features (X) and target (y)
X = data[:, 1].reshape(-1, 1)  # Using Humidity as the feature
y = data[:, 0]                  # Temperature as the target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Predict the temperature for test data
y_pred = model.predict(X_test)

# Print the predicted temperatures
print("Predicted temperatures:", y_pred)

# Evaluate the model (This is a simple evaluation for illustration purposes)
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
