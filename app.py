import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# ==========================================
# 1. Page Configuration
# ==========================================
st.set_page_config(
    page_title="Census Income Predictor",
    page_icon="💰",
    layout="wide"
)

# Title and Header
st.title("💰 Adult Census Income Prediction System")
st.markdown("""
This application uses the **K-Nearest Neighbors (KNN)** algorithm to predict if an individual's annual income 
exceeds **$50,000**. Adjust the parameters in the sidebar to test the model.
""")

# ==========================================
# 2. Load and Prepare Data
# ==========================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("adult_small.csv")
        return df
    except FileNotFoundError:
        st.error("File 'adult_small.csv' not found. Please run the data generation script first.")
        return None

df = load_data()

if df is not None:
    # Remove Name column
    if 'Name' in df.columns:
        df = df.drop('Name', axis=1)

    # Dictionary to store encoders
    encoders = {} 
    categorical_cols = ['Workclass', 'Sex', 'Marital-status', 'Native-country', 'Income']

    # Encode the Dataset
    df_encoded = df.copy()
    for col in categorical_cols:
        le = LabelEncoder()
        # Handle cases where new data might have unseen labels
        df_encoded[col] = le.fit_transform(df[col])
        encoders[col] = le

    # Separate Features and Target
    X = df_encoded.drop('Income', axis=1)
    y = df_encoded['Income']

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train the Model
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    
    # Calculate Accuracy
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # ==========================================
    # 3. Sidebar - User Inputs
    # ==========================================
    st.sidebar.header("User Input Parameters")

    def user_input_features():
        # Sliders
        age = st.sidebar.slider("Age", 18, 90, 30)
        hours = st.sidebar.slider("Hours per Week", 10, 80, 40)
        
        # Categorical Selectboxes
        workclass = st.sidebar.selectbox("Workclass", df['Workclass'].unique())
        
        # --- CHANGED HERE: Label is 'Gender', but logic maps to 'Sex' ---
        gender = st.sidebar.selectbox("Gender", df['Sex'].unique()) 
        
        marital = st.sidebar.selectbox("Marital Status", df['Marital-status'].unique())
        country = st.sidebar.selectbox("Native Country", df['Native-country'].unique())

        data = {
            'Age': age,
            'Workclass': workclass,
            'Hours_per_week': hours,
            'Sex': gender, # Map the 'Gender' input to the 'Sex' column expected by model
            'Marital-status': marital,
            'Native-country': country
        }
        return pd.DataFrame(data, index=[0])

    input_df = user_input_features()

    # ==========================================
    # 4. Main Panel - Prediction Logic
    # ==========================================
    
    tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Model Performance", "📂 Dataset"])

    with tab1:
        st.subheader("Predict User Income")
        
        # Display User Input (Renaming Sex -> Gender for display purposes)
        st.write(" **Parameters you selected:**")
        display_df = input_df.rename(columns={'Sex': 'Gender'})
        st.dataframe(display_df, hide_index=True)

        # Process User Input for the Model
        input_encoded = input_df.copy()
        
        # Apply the saved Encoders
        for col in ['Workclass', 'Sex', 'Marital-status', 'Native-country']:
            input_encoded[col] = encoders[col].transform(input_encoded[col])
        
        # Apply the Scaler
        input_scaled = scaler.transform(input_encoded)

        # Make Prediction
        if st.button("Predict Income Class", type="primary"):
            prediction = knn.predict(input_scaled)
            probability = knn.predict_proba(input_scaled)
            
            result_text = encoders['Income'].inverse_transform(prediction)[0]

            if result_text == ">50K":
                st.success(f"Result: Income is **{result_text}**")
                st.balloons()
            else:
                st.info(f"Result: Income is **{result_text}**")
            
            st.write(f"Confidence: {np.max(probability)*100:.2f}%")

    with tab2:
        st.subheader("Model Evaluation Metrics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Model Accuracy", value=f"{accuracy*100:.2f}%")
            st.write(f"**Algorithm:** K-Nearest Neighbors (KNN)")
            st.write(f"**K-Value:** 5")
            st.write(f"**Test Data Size:** {len(X_test)} records")
        
        with col2:
            st.write("**Confusion Matrix**")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            st.pyplot(fig)

    with tab3:
        st.subheader("Raw Dataset (First 50 rows)")
        # Rename column in the raw view as well for consistency
        st.dataframe(df.rename(columns={'Sex': 'Gender'}).head(50))