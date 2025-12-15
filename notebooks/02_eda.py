import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

# ---------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------
# Loading the local CSV file into a pandas DataFrame
try:
    df = pd.read_csv('../data/health_vitals_1000.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: File not found. Please check the path.")

# ---------------------------------------------------------
# 2. Basic Exploration
# ---------------------------------------------------------
# Display the first few rows
print("\n--- First 5 Rows ---")
print(df.head())

# Display data summary (columns, non-null counts, data types)
print("\n--- DataFrame Info ---")
print(df.info())

# Statistical summary of numerical columns
print("\n--- Statistical Summary ---")
print(df.describe())

# ---------------------------------------------------------
# 3. Visualization
# ---------------------------------------------------------
# Set the visual style
sns.set_theme(style="whitegrid")

# Plot 1: Distribution of Heart Rate
plt.figure(figsize=(10, 6))
sns.histplot(df['heart_rate'], kde=True, color='blue')
plt.title('Distribution of Heart Rate')
plt.xlabel('Heart Rate (bpm)')
plt.ylabel('Frequency')
plt.show()

# Plot 2: Correlation Matrix
# We filter only numeric columns to avoid errors
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

plt.figure(figsize=(10, 8))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Health Vitals')
plt.show()

# ---------------------------------------------------------
# 4. Register Data Asset to Azure ML (Critical for DP-100)
# ---------------------------------------------------------
print("\n--- Connecting to Azure Machine Learning Workspace ---")

# Connect to the workspace using the config.json file in the root directory
# Ensure config.json is downloaded from Azure Portal and placed in the project root
try:
    credential = DefaultAzureCredential()
    ml_client = MLClient.from_config(credential=credential)
    print(f"Connected to Workspace: {ml_client.workspace_name}")
except Exception as e:
    print(f"Failed to connect to Azure ML: {e}")

# Define the Data Asset
# This wraps the local file into a managed cloud asset
my_data = Data(
    path="../data/health_vitals_1000.csv",
    type=AssetTypes.URI_FILE,
    description="Synthetic health vitals dataset for anomaly detection project.",
    name="health-vitals-dataset",
    version="1"
)

# Create or Update the data asset in the workspace
print("\n--- Registering Data Asset ---")
try:
    created_data = ml_client.data.create_or_update(my_data)
    print(f"Data Asset created successfully.")
    print(f"Name: {created_data.name}")
    print(f"Version: {created_data.version}")
    print(f"ID: {created_data.id}")
except Exception as e:
    print(f"Error registering data asset: {e}")