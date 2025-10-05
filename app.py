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

# Initialize session state
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'preprocessed_dataset' not in st.session_state:
    st.session_state.preprocessed_dataset = None
if 'train_x' not in st.session_state:
    st.session_state.train_x = None
if 'train_y' not in st.session_state:
    st.session_state.train_y = None
if 'test_x' not in st.session_state:
    st.session_state.test_x = None
if 'test_y' not in st.session_state:
    st.session_state.test_y = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'accuracy' not in st.session_state:
    st.session_state.accuracy = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

# Sidebar for parameters
st.sidebar.header("Model Parameters")
penalty = st.sidebar.selectbox("Penalty", ["l2", "l1", "elasticnet", "none"], index=0)
C = st.sidebar.slider("C (Inverse of regularization strength)", 0.01, 10.0, 1.0, 0.01)

solver_options = {
    "Limited-memory BFGS (lbfgs)": "lbfgs",
    "Linear by Coordinate Descent (liblinear)": "liblinear",
    "Newton Conjugate Gradient (newton-cg)": "newton-cg",
    "Stochastic Average Gradient (sag)": "sag",
    "Stochastic Average Gradient with Averaging (saga)": "saga"
}
solver_display = st.sidebar.selectbox("Solver", list(solver_options.keys()), index=0)
solver = solver_options[solver_display]

# Solver compatibility checker
solver_compatibility = {
    "lbfgs": {"penalties": ["l2", "none"], "best_for": "Small/medium datasets", "multiclass": "Yes"},
    "liblinear": {"penalties": ["l1", "l2"], "best_for": "Small datasets", "multiclass": "One-vs-rest"},
    "newton-cg": {"penalties": ["l2", "none"], "best_for": "Medium datasets", "multiclass": "Yes"},
    "sag": {"penalties": ["l2", "none"], "best_for": "Large datasets", "multiclass": "Yes"},
    "saga": {"penalties": ["l1", "l2", "elasticnet", "none"], "best_for": "Large datasets, sparse & L1/L2", "multiclass": "Yes"}
}

# Display compatibility warning
current_penalty = penalty if penalty != "none" else "none"
if current_penalty not in solver_compatibility[solver]["penalties"]:
    st.sidebar.error(f"⚠️ Incompatible! {solver_display.split('(')[0].strip()} doesn't support {current_penalty} penalty.")
    st.sidebar.info(f"**{solver_display.split('(')[0].strip()}** supports: {', '.join(solver_compatibility[solver]['penalties'])}")
else:
    st.sidebar.success(f"✅ Compatible: {solver} with {current_penalty}")

# Display solver information
with st.sidebar.expander("ℹ️ Solver Information"):
    st.write(f"**Best for:** {solver_compatibility[solver]['best_for']}")
    st.write(f"**Regularization:** {', '.join(solver_compatibility[solver]['penalties'])}")
    st.write(f"**Multiclass:** {solver_compatibility[solver]['multiclass']}")

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
        fig, ax = plt.subplots(figsize=(8, 6))
        st.session_state.dataset["Top10"].value_counts().plot(kind='bar', ax=ax)
        ax.set_title("Distribution of Top10")
        ax.set_ylabel("Count")
        st.pyplot(fig, use_container_width=True)

# Step 2: Qualitative to Quantitative (Artist ID)
st.header("Step 2: Convert Artist ID to Numeric")
if st.button("Convert Artist ID"):
    if st.session_state.dataset is not None:
        try:
            dataset_copy = st.session_state.dataset.copy()
            dataset_copy["artistID"] = dataset_copy["artistID"].astype("category").cat.codes
            st.session_state.preprocessed_dataset = dataset_copy
            st.success("Artist ID converted!")
        except Exception as e:
            st.error(f"Error converting Artist ID: {str(e)}")
    else:
        st.error("Load dataset first!")

# Step 3: Feature Engineering
st.header("Step 3: Feature Engineering")
if st.button("Perform Feature Engineering"):
    if st.session_state.preprocessed_dataset is not None:
        try:
            dataset_copy = st.session_state.preprocessed_dataset.copy()
            
            # title_length
            dataset_copy["title_length"] = dataset_copy["songtitle"].apply(lambda x: len(x) if isinstance(x, str) else 0)
            
            # artist_length
            dataset_copy["artist_length"] = dataset_copy["artistname"].apply(lambda x: len(x) if isinstance(x, str) else 0)
            
            # is_love
            dataset_copy["is_love"] = dataset_copy["songtitle"].apply(
                lambda x: 1 if isinstance(x, str) and any(word in x.lower() for word in ["love", "heart", "baby", "darling"]) else 0
            )
            
            # is_she
            dataset_copy["is_she"] = dataset_copy["songtitle"].apply(
                lambda x: 1 if isinstance(x, str) and any(word in x.lower() for word in ["she", "girl", "woman", "lady"]) else 0
            )
            
            # is_dance
            dataset_copy["is_dance"] = dataset_copy["songtitle"].apply(
                lambda x: 1 if isinstance(x, str) and any(word in x.lower() for word in ["dance", "party", "shake", "move"]) else 0
            )
            
            dataset_copy["Age"] = 2022 - dataset_copy["year"]
            st.session_state.preprocessed_dataset = dataset_copy
            st.success("Feature engineering completed!")
        except Exception as e:
            st.error(f"Error in feature engineering: {str(e)}")
    else:
        st.error("Convert Artist ID first!")

# Step 4: Preprocessing and Split
st.header("Step 4: Preprocess and Split Data")
if st.button("Preprocess and Split"):
    if st.session_state.preprocessed_dataset is not None:
        try:
            dataset_copy = st.session_state.preprocessed_dataset.drop(["songID", "artistname", "songtitle", "year"], axis=1)
            dataset_copy.dropna(inplace=True)  # Handle missing values
            
            if len(dataset_copy) == 0:
                st.error("No data remaining after removing missing values!")
            else:
                train, test = train_test_split(dataset_copy, shuffle=True, test_size=0.2, random_state=42)
                st.session_state.train_y = train["Top10"]
                st.session_state.train_x = train.drop("Top10", axis=1)
                st.session_state.test_y = test["Top10"]
                st.session_state.test_x = test.drop("Top10", axis=1)
                
                # Scale features
                scaler = StandardScaler()
                st.session_state.train_x = pd.DataFrame(
                    scaler.fit_transform(st.session_state.train_x), 
                    columns=st.session_state.train_x.columns,
                    index=st.session_state.train_x.index
                )
                st.session_state.test_x = pd.DataFrame(
                    scaler.transform(st.session_state.test_x), 
                    columns=st.session_state.test_x.columns,
                    index=st.session_state.test_x.index
                )
                
                st.success(f"Data split and scaled! Training samples: {len(train)}, Test samples: {len(test)}")
        except Exception as e:
            st.error(f"Error in preprocessing: {str(e)}")
    else:
        st.error("Perform feature engineering first!")

# Step 5: Train Logistic Regression Model
st.header("Step 5: Train Logistic Regression Model")
if st.button("Train Logistic Regression"):
    if st.session_state.train_x is not None and st.session_state.train_y is not None:
        try:
            model = LogisticRegression(penalty=penalty, C=C, solver=solver, random_state=42, max_iter=1000)
            model.fit(st.session_state.train_x, st.session_state.train_y)
            pred = model.predict(st.session_state.test_x)
            acc = accuracy_score(st.session_state.test_y, pred)
            st.session_state.model = model
            st.session_state.accuracy = acc
            st.session_state.predictions = pred
            st.success(f"Logistic Regression trained successfully! Accuracy: {acc:.4f}")
        except Exception as e:
            st.error(f"Error training model: {str(e)}")
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
            st.pyplot(fig, use_container_width=True)
    
    with tab2:
        if st.button("Generate Correlation Heatmap"):
            if st.session_state.train_x is not None:
                corr_data = st.session_state.train_x.corrwith(st.session_state.train_y).sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(10, 6))
                corr_data.plot(kind='bar', ax=ax)
                ax.set_title("Feature Correlation with Top10")
                st.pyplot(fig, use_container_width=True)
    
    with tab3:
        if st.button("Generate Confusion Matrix"):
            if st.session_state.predictions is not None:
                cm = confusion_matrix(st.session_state.test_y, st.session_state.predictions)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_title("Confusion Matrix - Logistic Regression")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig, use_container_width=True)

# Example Prediction
if st.session_state.model is not None:
    st.header("Try a Prediction")
    if st.button("Predict on Test Set (Sample)"):
        try:
            sample_idx = np.random.randint(0, len(st.session_state.test_x))
            sample_features = st.session_state.test_x.iloc[sample_idx:sample_idx+1]
            sample_true = st.session_state.test_y.iloc[sample_idx]
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Sample Features:**")
                st.dataframe(sample_features)
            with col2:
                pred = st.session_state.model.predict(sample_features)[0]
                prob = st.session_state.model.predict_proba(sample_features)[0]
                st.write(f"**Logistic Regression Prediction:** {pred}")
                st.write(f"**Prediction Probability:** Not Top10: {prob[0]:.2%}, Top10: {prob[1]:.2%}")
                st.write(f"**True Label:** {sample_true}")
                
                if pred == sample_true:
                    st.success("✅ Correct Prediction!")
                else:
                    st.error("❌ Incorrect Prediction")
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
