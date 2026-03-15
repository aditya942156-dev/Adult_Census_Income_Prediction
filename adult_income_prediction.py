# ===============================================
# Adult Census Income Prediction using KNN
# Data Mining Project - 500 Records
# ===============================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Rich library for professional output formatting
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

# -------------------------------
# 1. Load Dataset
# -------------------------------
try:
    data = pd.read_csv("adult_small.csv")
    console.print(Panel("Adult Census Income Prediction System Started", style="bold white on blue"))
    console.print("\nDATASET PREVIEW", style="bold cyan")
    console.print(data.head())
except FileNotFoundError:
    console.print("[bold red]Error:[/bold red] 'adult_small.csv' not found. Please run the data generator script first.")
    exit()

# -------------------------------
# 2. Dataset Information
# -------------------------------
console.print("\nDATASET INFORMATION", style="bold cyan")
console.print(f"Total Records : {data.shape[0]}")
console.print(f"Total Columns : {data.shape[1]}")

# -------------------------------
# 3. Preprocessing: Remove Name
# -------------------------------
# Name is not useful for mathematical prediction
if "Name" in data.columns:
    data = data.drop("Name", axis=1)

# -------------------------------
# 4. Encoding: Text to Numbers
# -------------------------------
encoder = LabelEncoder()
# Note: Age and Hours_per_week are already numbers, so we don't encode them
categorical_columns = ['Workclass','Sex','Marital-status','Native-country','Income']

for col in categorical_columns:
    if col in data.columns:
        data[col] = encoder.fit_transform(data[col])

console.print("\nDATA AFTER ENCODING (Text -> Numbers)", style="bold cyan")
console.print(data.head())

# -------------------------------
# 5. Features (X) & Target (y)
# -------------------------------
X = data.drop("Income", axis=1) # Attributes
y = data["Income"]              # Target Label

# -------------------------------
# 6. Train-Test Split (80% Train, 20% Test)
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
console.print(f"\n[green]Training Records : {len(X_train)}[/green]")
console.print(f"[yellow]Testing Records  : {len(X_test)}[/yellow]")

# -------------------------------
# 7. Normalization (Scaling)
# -------------------------------
# Essential for KNN so that Age (e.g., 50) and Hours (e.g., 40) represent similar distances
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# 8. Apply KNN Algorithm
# -------------------------------
k_value = 5
knn = KNeighborsClassifier(n_neighbors=k_value)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# -------------------------------
# 9. Model Evaluation
# -------------------------------
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

console.print("\nMODEL RESULT", style="bold green")
console.print(f"Algorithm Used : K-Nearest Neighbors (KNN)")
console.print(f"K Value        : {k_value}")
console.print(f"Model Accuracy : {accuracy*100:.2f}%")

# Create Confusion Matrix Table
cm_table = Table(title="Confusion Matrix", style="bold magenta")
cm_table.add_column("Actual \\ Predicted", justify="center")
cm_table.add_column("<=50K (Predicted)", justify="center")
cm_table.add_column(">50K (Predicted)", justify="center")
cm_table.add_row("<=50K (Actual)", str(cm[0][0]), str(cm[0][1]))
cm_table.add_row(">50K (Actual)", str(cm[1][0]), str(cm[1][1]))
console.print(cm_table)

# Create Classification Report Table
report_table = Table(title="Detailed Classification Report", style="bold magenta")
report_table.add_column("Class", justify="center")
report_table.add_column("Precision", justify="center")
report_table.add_column("Recall", justify="center")
report_table.add_column("F1-Score", justify="center")

for cls in report:
    if cls in ['accuracy', 'macro avg', 'weighted avg']:
        continue
    # Convert 0/1 back to labels for display if possible, or just use 0/1
    label = "<=50K" if cls == '0' else ">50K"
    report_table.add_row(
        label,
        f"{report[cls]['precision']:.2f}",
        f"{report[cls]['recall']:.2f}",
        f"{report[cls]['f1-score']:.2f}"
    )
console.print(report_table)

# -------------------------------
# 10. Test Different K Values (Optimization)
# -------------------------------
console.print("\nTESTING DIFFERENT K VALUES", style="bold green")
accuracy_list = []
k_range = range(1, 15)

for k in k_range:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    accuracy_list.append(acc)
    # Print bar for visual effect
    bar = "█" * int(acc * 20)
    console.print(f"K = {k:02d} | Accuracy = {acc*100:.2f}% | {bar}")

# -------------------------------
# 11. Graph Representation
# -------------------------------
console.print("\n[bold white on blue] Generating Graph... [/bold white on blue]")

plt.figure(figsize=(10,6))
plt.plot(k_range, accuracy_list, marker='o', color='purple', linestyle='dashed', linewidth=2, markersize=8)
plt.title("KNN Performance: K Value vs Accuracy")
plt.xlabel("K Value (Number of Neighbors)")
plt.ylabel("Model Accuracy")
plt.grid(True)
plt.show()

console.print("\n[bold green]Project Execution Completed Successfully![/bold green]")