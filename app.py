import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
st.set_page_config(page_title="Music Popularity Classifier", page_icon="🎵", layout="wide")
st.title("🎵 Music Records Popularity Classifier")
st.markdown("An interactive app to analyze the 'Popularity of music records' dataset, perform classification to predict Top10 hits, and visualize results. Operations are triggered via buttons for lazy execution.")

# Sidebar for parameters
st.sidebar.header("Model Parameters")
penalty = st.sidebar.selectbox("Penalty", ["l2", "l1", "elasticnet", "none"], index=0)
C = st.sidebar.slider("C (Inverse of regularization strength)", 0.01, 10.0, 1.0, 0.01)
solver = st.sidebar.selectbox("Solver", ["lbfgs", "liblinear", "newton-cg", "sag", "saga"], index=0)

if penalty == "none":
    penalty = None

# Step 1: Load Data
st.header("Step 1: Load Dataset")
if st.button("Load Dataset"):
    @st.cache_data
    def load_data():
        dataset = pd.read_csv("data.csv", encoding='latin-1', header=0)
        dataset.drop_duplicates(inplace=True)
        dataset.drop_duplicates(subset=["songID"], inplace=True)
        dataset = dataset.reset_index(drop=True)
        return dataset
    
    st.session_state.dataset = load_data()
    st.success("Dataset loaded!")

if st.session_state.dataset is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Dataset Overview")
        st.write(f"Shape: {st.session_state.dataset.shape}")
        st.write(st.session_state.dataset.dtypes.value_counts())
    with col2:
        st.subheader("Missing Values")
        st.write(st.session_state.dataset.isna().sum().sum())
    
    # Data Exploration
    st.subheader("Data Exploration")
    if st.button("Show Sample Data"):
        st.dataframe(st.session_state.dataset.head(10))
    
    if st.button("Show Top10 Distribution"):
        fig, ax = plt.subplots()
        st.session_state.dataset["Top10"].value_counts().plot(kind='bar', ax=ax)
        ax.set_title("Distribution of Top10")
        ax.set_ylabel("Count")
        st.pyplot(fig)

# Step 2: Qualitative to Quantitative (Artist ID)
st.header("Step 2: Convert Artist ID to Numeric")
if st.button("Convert Artist ID"):
    if st.session_state.dataset is not None:
        dataset_copy = st.session_state.dataset.copy()
        dataset_copy["artistID"] = dataset_copy["artistID"].astype("category").cat.codes
        st.session_state.preprocessed_dataset = dataset_copy
        st.success("Artist ID converted!")
    else:
        st.error("Load dataset first!")

# Step 3: Feature Engineering
st.header("Step 3: Feature Engineering")
if st.button("Perform Feature Engineering"):
    if st.session_state.preprocessed_dataset is not None:
        dataset_copy = st.session_state.preprocessed_dataset.copy()
        
        # title_length
        dataset_copy["title_length"] = dataset_copy["songtitle"].apply(len)
        
        # artist_length
        dataset_copy["artist_length"] = dataset_copy["artistname"].apply(len)
        
        # is_love
        arr = []
        for i in range(len(dataset_copy)):
            if any(word.lower() in dataset_copy["songtitle"].iloc[i].lower() for word in ["love", "heart", "baby", "darling"]):
                arr.append(1)
            else:
                arr.append(0)
        dataset_copy["is_love"] = pd.Series(arr)
        
        # is_she
        arr2 = []
        for i in range(len(dataset_copy)):
            if any(word.lower() in dataset_copy["songtitle"].iloc[i].lower() for word in ["she", "girl", "woman", "lady"]):
                arr2.append(1)
            else:
                arr2.append(0)
        dataset_copy["is_she"] = pd.Series(arr2)
        
        # is_dance
        arr3 = []
        for i in range(len(dataset_copy)):
            if any(word.lower() in dataset_copy["songtitle"].iloc[i].lower() for word in ["dance", "party", "shake", "move"]):
                arr3.append(1)
            else:
                arr3.append(0)
        dataset_copy["is_dance"] = pd.Series(arr3)
        
        dataset_copy["Age"] = 2022 - dataset_copy["year"]
        st.session_state.preprocessed_dataset = dataset_copy
        st.success("Feature engineering completed!")
    else:
        st.error("Convert Artist ID first!")

# Step 4: Preprocessing and Split
st.header("Step 4: Preprocess and Split Data")
if st.button("Preprocess and Split"):
    if st.session_state.preprocessed_dataset is not None:
        dataset_copy = st.session_state.preprocessed_dataset.drop(["songID", "artistname", "songtitle", "year"], axis=1)
        dataset_copy.dropna(inplace=True)  # Handle missing values
        train, test = train_test_split(dataset_copy, shuffle=True, test_size=0.2, random_state=42)
        st.session_state.train_y = train["Top10"]
        st.session_state.train_x = train.drop("Top10", axis=1)
        st.session_state.test_y = test["Top10"]
        st.session_state.test_x = test.drop("Top10", axis=1)
        
        # Scale features
        scaler = StandardScaler()
        st.session_state.train_x = pd.DataFrame(scaler.fit_transform(st.session_state.train_x), columns=st.session_state.train_x.columns)
        st.session_state.test_x = pd.DataFrame(scaler.transform(st.session_state.test_x), columns=st.session_state.test_x.columns)
        
        st.success("Data split and scaled!")
    else:
        st.error("Perform feature engineering first!")

# Step 5: Train Logistic Regression Model
st.header("Step 5: Train Logistic Regression Model")
if st.button("Train Logistic Regression"):
    if st.session_state.train_x is not None:
        model = LogisticRegression(penalty=penalty, C=C, solver=solver, random_state=42, max_iter=1000)
        model.fit(st.session_state.train_x, st.session_state.train_y)
        pred = model.predict(st.session_state.test_x)
        acc = accuracy_score(st.session_state.test_y, pred)
        st.session_state.model = model
        st.session_state.accuracy = acc
        st.session_state.predictions = pred
        st.success("Logistic Regression trained!")
    else:
        st.error("Split data first!")

# Display Results if available
if st.session_state.accuracy is not None:
    st.header("Results")
    st.write(f"Logistic Regression Accuracy: {st.session_state.accuracy:.4f}")

# Graphs Section
if st.session_state.dataset is not None:
    st.header("Visualizations")
    tab1, tab2, tab3 = st.tabs(["Target Distribution", "Correlation Heatmap", "Confusion Matrix"])
    
    with tab1:
        if st.button("Generate Target Distribution"):
            fig, ax = plt.subplots(figsize=(8, 6))
            st.session_state.dataset["Top10"].value_counts().plot(kind='bar', ax=ax)
            ax.set_title("Distribution of Top10")
            ax.set_ylabel("Count")
            st.pyplot(fig)
    
    with tab2:
        if st.button("Generate Correlation Heatmap"):
            if st.session_state.train_x is not None:
                corr_data = st.session_state.train_x.corrwith(st.session_state.train_y).sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(10, 6))
                corr_data.plot(kind='bar', ax=ax)
                ax.set_title("Feature Correlation with Top10")
                st.pyplot(fig)
    
    with tab3:
        if st.button("Generate Confusion Matrix"):
            if st.session_state.predictions is not None:
                cm = confusion_matrix(st.session_state.test_y, st.session_state.predictions)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_title("Confusion Matrix - Logistic Regression")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)

# Example Prediction
if st.session_state.model is not None:
    st.header("Try a Prediction")
    if st.button("Predict on Test Set (Sample)"):
        sample_idx = np.random.randint(0, len(st.session_state.test_x))
        sample_features = st.session_state.test_x.iloc[sample_idx:sample_idx+1]
        sample_true = st.session_state.test_y.iloc[sample_idx]
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Sample Features:**")
            st.dataframe(sample_features)
        with col2:
            pred = st.session_state.model.predict(sample_features)[0]
            st.write(f"**Logistic Regression Prediction:** {pred}")
            st.write(f"**True Label:** {sample_true}")