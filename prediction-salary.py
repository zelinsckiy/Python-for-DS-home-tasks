import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. Data, create DataFrame
data = {
    "experience": [1, 2, 3, 5, 7, 10, 12, 15],
    "salary": [30000, 35000, 40000, 55000, 70000, 90000, 105000, 130000]
}

df = pd.DataFrame(data)
df.info()
missing_values = df.isnull().sum()
df.shape
df.describe().round()
# 2. Statistics
print(df.describe())

# 3. Graphic
plt.scatter(df["experience"], df["salary"],
            color = "blue", label = "salaries = experience")

# 4. Finding correlations
print(df.corr())

# 5. Learning model Linear Regression
X = df[["experience"]]
y = df["salary"]
model = LinearRegression()
model.fit(X, y)

# 6. Prediction salary for person with 8 years experience
experience = 8
pred = model.predict([[experience]])
print(f"\nPerson experience {experience}years -> salary $ {pred[0]:,.0f}")

# 7. R**2
print(f"\nТочность модели R²: {model.score(X, y):.2f}")

# 8. Final Graphic with points and line of the model
plt.scatter(df["experience"], df["salary"],
            color="blue", label="data")
x_line = np.linspace(3, 25, 10)
y_line = model.predict(x_line.reshape(-1, 1))
plt.plot(x_line, y_line, color="red", label="model")
plt.title("Prediction salary")
plt.xlabel("experience years")
plt.ylabel("salary ($)")
plt.legend()
plt.grid(True)
plt.show()