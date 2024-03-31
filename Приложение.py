#!/usr/bin/env python
# coding: utf-8

# In[21]:


import argparse
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd


# In[22]:


def train_model(X_train, y_train):
    model = KNeighborsRegressor()
    model.fit(X_train, y_train)
    return model


# In[23]:


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return mse, r2, mae


# In[24]:


def get_user_input(X_columns):
    user_input = {}
    for column in X_columns:
        value = input(f"Введите значение для признака '{column}': ")
        user_input[column] = float(value)  # Предполагаем, что все значения числовые
    return user_input


# In[25]:


def main():
    # Обработка аргументов командной строки
    parser = argparse.ArgumentParser(description='KNN Regression Prediction')
    parser.add_argument('data_file', type=str, help='Path to data file')
    parser.add_argument('--n_neighbors', type=int, default=5, help='Number of neighbors for KNN')
    args = parser.parse_args()

    # Загрузка данных из файла
    data = pd.read_csv(args.data_file)
    X = data.drop(columns=['Tensile Elastic Modulus', 'Tensile Strength'])
    y = data['Tensile Elastic Modulus']

    # Разделение данных на обучающий и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Обучение модели
    model = train_model(X_train, y_train, args.n_neighbors)

    # Оценка модели
    mse, r2, mae = evaluate_model(model, X_test, y_test)

    # Вывод результатов
    print("Mean Squared Error (MSE):", mse)
    print("R-squared (R2):", r2)
    print("Mean Absolute Error (MAE):", mae)

    # Получение пользовательского ввода и предсказание
    user_input = get_user_input(X.columns)
    user_input_df = pd.DataFrame([user_input])
    prediction = model.predict(user_input_df)
    print("Предсказанное значение:", prediction[0])

if __name__ == "main":
    main()


# In[27]:


get_ipython().run_cell_magic('writefile', 'my_script.py', 'import argparse\nfrom sklearn.neighbors import KNeighborsRegressor\nfrom sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error\nfrom sklearn.model_selection import train_test_split\nimport pandas as pd\n\ndef train_model(X_train, y_train):\n    model = KNeighborsRegressor()\n    model.fit(X_train, y_train)\n    return model\n\ndef evaluate_model(model, X_test, y_test):\n    y_pred = model.predict(X_test)\n    mse = mean_squared_error(y_test, y_pred)\n    r2 = r2_score(y_test, y_pred)\n    mae = mean_absolute_error(y_test, y_pred)\n    return mse, r2, mae\n\ndef get_user_input(X_columns):\n    user_input = {}\n    for column in X_columns:\n        value = input(f"Введите значение для признака \'{column}\': ")\n        user_input[column] = float(value)  # Предполагаем, что все значения числовые\n    return user_input\n\ndef main():\n    # Обработка аргументов командной строки\n    parser = argparse.ArgumentParser(description=\'KNN Regression Prediction\')\n    parser.add_argument(\'data_file\', type=str, help=\'Path to data file\')\n    parser.add_argument(\'--n_neighbors\', type=int, default=5, help=\'Number of neighbors for KNN\')\n    args = parser.parse_args()\n\n    # Загрузка данных из файла\n    data = pd.read_csv(args.data_file)\n    X = data.drop(columns=[\'Tensile Elastic Modulus\', \'Tensile Strength\'])\n    y = data[\'Tensile Elastic Modulus\']\n\n    # Разделение данных на обучающий и тестовый наборы\n    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n\n    # Обучение модели\n    model = train_model(X_train, y_train, args.n_neighbors)\n\n    # Оценка модели\n    mse, r2, mae = evaluate_model(model, X_test, y_test)\n\n    # Вывод результатов\n    print("Mean Squared Error (MSE):", mse)\n    print("R-squared (R2):", r2)\n    print("Mean Absolute Error (MAE):", mae)\n\n    # Получение пользовательского ввода и предсказание\n    user_input = get_user_input(X.columns)\n    user_input_df = pd.DataFrame([user_input])\n    prediction = model.predict(user_input_df)\n    print("Предсказанное значение:", prediction[0])\n\nif __name__ == "main":\n    main()\n')

