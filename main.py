import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('DermaMNIST_C.csv') # Load the CSV file
npz = np.load('DermaMNIST_Corrected_224.npz') # Load the NPZ file

# Display the basic information about the dataset
print("CSV rows:", len(df))
print("NPZ train images:", npz['train_images'].shape[0])
print("NPZ val images:", npz['val_images'].shape[0])
print("NPZ test images:", npz['test_images'].shape[0])

# Display the first few rows of the Dataset
print(df.columns)
print(df.head())

# Display the classes in the dataset
clas_count = df['dx'].value_counts()
print("\nImages per class:")
print(clas_count)

# Display the gender in each class
print("\nImages per class and sex:")
gender_each_class = df.groupby(['dx', 'sex']).size().unstack(fill_value=0)
print(gender_each_class)

# Plot displaying the number of images in each class
plt.figure(figsize=(10, 6))
clas_count.plot(kind='bar', color='skyblue')
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.title('Number of Images in each Class')
plt.tight_layout()
plt.show()

# Gender distribution in each class
gender_each_class.plot(kind='bar', stacked=True, color=['lightpink', 'lightblue'])
plt.ylabel('Number of Images')
plt.title('Gender Distribution in Each Class')
plt.tight_layout()
plt.show()

# Display the Localization distribution
localization_counts = df['localization'].value_counts()
plt.figure(figsize=(8,5))
localization_counts.plot(kind='bar', color='lightgreen')
plt.xlabel('Localization')
plt.ylabel('Number of Images')
plt.title(' Localization Distribution')
plt.tight_layout()
plt.show()