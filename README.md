# 🎵 Music Records Popularity Classifier

An interactive Streamlit web application that analyzes music records dataset and uses Logistic Regression to predict whether a song will become a Top10 hit. The app provides data exploration, feature engineering, model training, and visualization capabilities.

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Workflow](#workflow)
- [Model Parameters](#model-parameters)
- [Features Explained](#features-explained)
- [Technologies Used](#technologies-used)

## ✨ Features

- **Interactive Data Loading**: Load and explore the music records dataset
- **Data Preprocessing**: Convert categorical variables and handle missing values
- **Feature Engineering**: Create new features from song titles and metadata
- **Configurable Model Parameters**: Customize penalty, regularization strength (C), and solver
- **Model Training**: Train Logistic Regression classifier with customizable hyperparameters
- **Visualizations**: 
  - Target distribution plots
  - Feature correlation heatmaps
  - Confusion matrix visualization
- **Real-time Predictions**: Test model on random samples from test set
- **Lazy Execution**: All operations are triggered via buttons for better control

## 🔧 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Steps

1. Clone the repository:
```bash
git clone https://github.com/AnishKMBtech/Music-logistic-regression.git
cd Music-logistic-regression
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

### Dependencies

- streamlit
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn

## 🚀 Usage

1. Ensure you have the `data.csv` file in the project directory

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Open your web browser and navigate to the URL displayed in the terminal (typically `http://localhost:8501`)

4. Follow the step-by-step workflow in the app:
   - **Step 1**: Load Dataset
   - **Step 2**: Convert Artist ID to Numeric
   - **Step 3**: Perform Feature Engineering
   - **Step 4**: Preprocess and Split Data
   - **Step 5**: Train Logistic Regression Model

5. Explore visualizations and test predictions

## 📊 Dataset

The application expects a CSV file named `data.csv` with the following columns:

- `songID`: Unique identifier for each song
- `artistID`: Artist identifier
- `artistname`: Name of the artist
- `songtitle`: Title of the song
- `year`: Release year of the song
- `Top10`: Target variable (1 = Top10 hit, 0 = Not a Top10 hit)
- Additional features related to song characteristics

The dataset should be encoded in 'latin-1' format.

## 📝 Workflow

### Step 1: Load Dataset
- Loads the CSV file
- Removes duplicate entries
- Displays dataset overview, shape, and missing values
- Provides options to view sample data and Top10 distribution

### Step 2: Convert Artist ID to Numeric
- Converts categorical `artistID` to numeric codes for model compatibility

### Step 3: Feature Engineering
Creates the following engineered features:

- **title_length**: Length of the song title
- **artist_length**: Length of the artist name
- **is_love**: Binary flag for love-related keywords (love, heart, baby, darling)
- **is_she**: Binary flag for female-related keywords (she, girl, woman, lady)
- **is_dance**: Binary flag for dance-related keywords (dance, party, shake, move)
- **Age**: Years since release (2022 - release year)

### Step 4: Preprocess and Split Data
- Drops non-predictive columns (songID, artistname, songtitle, year)
- Handles missing values
- Splits data into 80% training and 20% testing sets
- Applies StandardScaler to normalize features

### Step 5: Train Logistic Regression Model
- Trains a Logistic Regression classifier with selected parameters
- Evaluates accuracy on test set
- Stores predictions for further analysis

## ⚙️ Model Parameters

Configure these parameters in the sidebar:

### Penalty
Regularization type to prevent overfitting:
- **l2**: Ridge regularization (default)
- **l1**: Lasso regularization (feature selection)
- **elasticnet**: Combination of L1 and L2
- **none**: No regularization

### C (Inverse of Regularization Strength)
- Range: 0.01 to 10.0 (default: 1.0)
- Smaller values = stronger regularization
- Larger values = weaker regularization

### Solver
Algorithm used to optimize the model:

| Solver | Compatible Penalties | Best For | Multiclass Support |
|--------|---------------------|----------|-------------------|
| **lbfgs** | l2, none | Small/medium datasets | Yes |
| **liblinear** | l1, l2 | Small datasets | One-vs-rest |
| **newton-cg** | l2, none | Medium datasets | Yes |
| **sag** | l2, none | Large datasets | Yes |
| **saga** | l1, l2, elasticnet, none | Large datasets, sparse & L1/L2 | Yes |

The app automatically checks solver-penalty compatibility and displays warnings for incompatible combinations.

## 🎯 Features Explained

### Engineered Features

1. **title_length**: Longer titles might correlate with different genres or eras
2. **artist_length**: Artist name length as a potential feature
3. **is_love**: Songs about love are popular across genres
4. **is_she**: Gender-specific pronouns may indicate certain song types
5. **is_dance**: Dance-related songs have specific characteristics
6. **Age**: Older songs might have different popularity patterns

### Visualizations

1. **Target Distribution**: Bar chart showing balance of Top10 vs non-Top10 songs
2. **Correlation Heatmap**: Shows which features correlate most with Top10 status
3. **Confusion Matrix**: Visualizes model performance (True Positives, False Positives, etc.)

### Prediction Testing

The app provides a "Try a Prediction" feature that:
- Randomly selects a sample from the test set
- Displays the feature values
- Shows model prediction and confidence probabilities
- Compares prediction with actual label
- Indicates whether the prediction was correct

## 🛠️ Technologies Used

- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning library
  - LogisticRegression: Classification model
  - train_test_split: Data splitting
  - StandardScaler: Feature scaling
  - accuracy_score, confusion_matrix: Evaluation metrics
- **Matplotlib**: Static visualizations
- **Seaborn**: Statistical data visualization

## 📈 Model Performance

The app displays:
- Overall accuracy score
- Confusion matrix for detailed performance analysis
- Per-sample prediction probabilities

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

AnishKMBtech

---

**Note**: This is an educational project demonstrating machine learning classification with an interactive Streamlit interface.
