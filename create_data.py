import pandas as pd
import numpy as np
import random

# Settings
num_records = 500

# Lists for random generation
names = [f"Person_{i}" for i in range(1, num_records + 1)]
workclasses = ['Private', 'Self-emp', 'Govt', 'Private', 'Private', 'Without-pay']
sexes = ['Male', 'Female']
marital_statuses = ['Married', 'Single', 'Divorced', 'Widowed', 'Separated']
countries = ['USA', 'Mexico', 'India', 'Philippines', 'Germany', 'Canada']
incomes = ['<=50K', '>50K']

# Generate Data
data = {
    'Name': names,
    'Age': np.random.randint(18, 75, size=num_records),
    'Workclass': [random.choice(workclasses) for _ in range(num_records)],
    'Hours_per_week': np.random.randint(20, 70, size=num_records),
    'Sex': [random.choice(sexes) for _ in range(num_records)],
    'Marital-status': [random.choice(marital_statuses) for _ in range(num_records)],
    'Native-country': [random.choice(countries) for _ in range(num_records)],
    # Logic: If older and works more, higher chance of >50K (to make the model learnable)
    'Income': [] 
}

# Create semi-realistic income logic so the accuracy isn't random
for i in range(num_records):
    age = data['Age'][i]
    hours = data['Hours_per_week'][i]
    
    if age > 30 and hours > 40:
        data['Income'].append(random.choice(['>50K', '>50K', '<=50K'])) # Higher chance of >50k
    else:
        data['Income'].append(random.choice(['<=50K', '<=50K', '>50K'])) # Higher chance of <=50k

# Save to CSV
df = pd.DataFrame(data)
df.to_csv('adult_small.csv', index=False)
print("SUCCESS: 'adult_small.csv' with 500 records has been created!")