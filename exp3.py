import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

customers_df = pd.read_csv("customers.csv")
orders_df = pd.read_csv("orders.csv")
products_df = pd.read_csv("products.csv")

customers_df.fillna({'age': customers_df['age'].mean(), 'email': 'N/A'}, inplace=True)

merged_df = customers_df.merge(orders_df, on='customer_id').merge(products_df, on='product_id')

merged_df['total_price'] = merged_df['quantity'] * merged_df['price']
merged_df['Feed_back'] = np.where(merged_df['quantity'] > 1, "Good", "Bad")

print("Cleaned, Integrated, and Transformed Data:")
print(merged_df)

features_encoded = OrdinalEncoder().fit_transform(merged_df)
target_encoded = LabelEncoder().fit_transform(merged_df['Feed_back'])

print("Features:\n", features_encoded)
print("Target:\n", target_encoded)
