import sys

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

if len(sys.argv) != 2:
    print('Usage: python3 {} <FILE>'.format(sys.argv[0]))
    print('Wrong number of command-line arguments')
    sys.exit(1)

path = sys.argv[1]
df = pd.read_parquet(path)

y = df.target
X = df.drop(columns=['target'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print('MSE: {:.2f}'.format(mean_squared_error(y_test, y_pred)))
print('Average value of target: {:.2f}'.format(y.mean()))

n = 5000
plt.scatter(y_test[:n], y_pred[:n], s=1)
plt.title('Зависимость между реальным значением target и предсказанием линейной модели')
plt.xlabel('y_true')
plt.ylabel('y_pred')
plt.savefig('baseline.png')
