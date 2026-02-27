# Fuel-Prices-Linear-Regression-AI
The code has been modified to project UK petrol prices and predict the price for 2025 using Artificial Intelligence, this was made using Google Colab.

My orginal task was to import the 'matplotlib.pyplot' library to the code and 'sklearn.linear_model' for the AI library.

This was the orginal code (Orginal Training Data) before it was modifed, it was for a prediction for house prices based off sq foot:

# X must be in a specific format for the library [[value1], [value2]]
X = np.array([[600], [800], [1000], [1200], [1500], [1800], [2000]])
y = np.array([150000, 180000, 210000, 250000, 300000, 340000, 400000])

# Let's look at our data points
plt.scatter(X, y, color='blue')
plt.title('House Size vs Price')
plt.xlabel('Square Feet')
plt.ylabel('Price ($)')
plt.show()

model = LinearRegression()
model.fit(X, y)

# Draw the line the AI created
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red', linewidth=2)
plt.title('AI Line of Best Fit')
plt.show()

# Try to predict a 1700 sq ft house
unknown_house = [[1700]]
predicted_price = model.predict(unknown_house)

print(f"The AI predicts a 1700 sq ft house will cost: ${predicted_price[0]:,.2f}")

Here's the orginal output for house prices:
<img width="1089" height="1386" alt="image" src="https://github.com/user-attachments/assets/b4246e3e-cd89-42a7-9df6-352670692bc0" />

Here's the new output now that we have changed the code entirely:
<img width="987" height="1432" alt="image" src="https://github.com/user-attachments/assets/8a5899c9-e752-45ab-9108-b45a7b0e9e74" />

