# %% [markdown]
# Import the neccessary libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import recall_score,f1_score,r2_score,accuracy_score,mean_squared_error,precision_score

# %% [markdown]
# Data Loading

# %%
print("Loading data from audusd.csv......")

try:
    df = pd.read_csv("audusd.csv")
    print("Data loaded successfully!")
    print("First five rows of the dataset")
    print(df.head().to_string())
except FileNotFoundError:
    print("Error: The file 'forex-1year.csv' was not found. Please ensure it is in the same directory.")
    exit()

# %% [markdown]
# Data Preprocessing and Feature Engineering

# %%
# Check for missing values
df_missing = df.isnull().sum()
print("Missing values")
print(df_missing)

# Check for duplicate rows
df_duplicated = df.duplicated().sum()
print("Duplcated rows")
print(df_duplicated)

# Define the features (X) and target (y)
X = df[["open_audusd","high_audusd","low_audusd"]]
y = df["close_audusd"]

print("Shape of features (X):",X.shape)
print("Shape of target (y):",y.shape)

# %% [markdown]
# Data Splitting

# %%
# We split the data into a training set and a testing set. The model learns
# from the training data and is then evaluated on the testing data. We'll use the 70/30 split.

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

print("Number of samples in training set:",len(X_train))
print("Number of samples in testing set:",len(X_test))

# %% [markdown]
# Model Training

# %%
# We create an instance of the RandomForestRegressor model and train it using the
# training data. We use n_estimators=100, which means the model will build 100
# decision trees

print("Training the Random Forest Regressor model......")
model = RandomForestRegressor(n_estimators=100,random_state=42)
model.fit(X_train,y_train)
print("Model training complete!")

# %% [markdown]
# Model Evaluation

# %%
# Now we make predictions on the test set and evaluate the models performance
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

print("Mean Squared Error:",mse)
print("R-Squared:",r2)

# %% [markdown]
# Making a New Prediction

# %%
print("Enter the following values to predict the Close price")
while True:
    try:
        open_price_input = input("Enter the Open price(or type 'exit' to quit):")
        if open_price_input.lower() == "exit":
            break
        open_price = float(open_price_input)
        high_price = float(input("Enter the High price:"))
        low_price = float(input("Enter the Low price:"))

        # We must reshape the input to a 2D array, even for a single sample
        # The order of the features must match the order used during training
        new_prices = np.array([[open_price,high_price,low_price]])

        predicted_price = model.predict(new_prices)

        print(f"For the given prices, the predicted Close price is: {predicted_price[0]:.4f}")

    except ValueError:
        print("Invalid Input. Please enter valid numbers for all three fields")


