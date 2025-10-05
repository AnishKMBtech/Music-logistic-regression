# 🎵 Music Records Popularity Classifier

An interactive machine learning web application built with Streamlit to predict whether a music record will be a Top 10 hit based on various audio features and metadata.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Details](#model-details)
- [Solver-Penalty Compatibility](#solver-penalty-compatibility)
- [Project Structure](#project-structure)
- [Screenshots](#screenshots)
- [Requirements](#requirements)
- [License](#license)

## 🎯 Overview

This project analyzes music records data and uses **Logistic Regression** to classify whether a song will make it to the Top 10 charts. The application provides an interactive interface where users can:
- Load and explore music datasets
- Engineer custom features from song metadata
- Train models with different hyperparameters
- Visualize results with interactive charts
- Make predictions on test samples

## ✨ Features

### 🔧 Machine Learning Pipeline
- **Data Loading**: Supports CSV files with automatic duplicate removal
- **Feature Engineering**: Creates advanced features like:
  - Title and artist name length
  - Keyword detection (love, dance, etc.)
  - Song age calculation
- **Preprocessing**: StandardScaler normalization for optimal model performance
- **Model Training**: Logistic Regression with customizable hyperparameters

### 🎨 Interactive UI
- **Sidebar Controls**: Adjust penalty type, regularization strength (C), and solver algorithm
- **Compatibility Checker**: Real-time validation of solver-penalty combinations
- **Step-by-Step Workflow**: Guided process from data loading to prediction
- **Lazy Execution**: Operations triggered via buttons for better control

### 📊 Visualizations
- Target distribution plots
- Feature correlation heatmaps
- Confusion matrix with color-coded results
- Prediction probabilities with confidence scores

### 🛡️ Error Handling
- Comprehensive try-except blocks for robust error management
- Session state management to prevent runtime errors
- Data validation at each pipeline step
- Solver compatibility warnings

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or download this repository**
   ```bash
   cd "ml project"
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv lib
   ```

3. **Activate the virtual environment**
   - Windows (PowerShell):
     ```powershell
     .\lib\Scripts\Activate.ps1
     ```
   - Windows (Command Prompt):
     ```cmd
     lib\Scripts\activate.bat
     ```
   - Linux/Mac:
     ```bash
     source lib/bin/activate
     ```

4. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

1. **Ensure your dataset is in the project directory**
   - File should be named `data.csv`
   - Must contain columns: songID, songtitle, artistname, artistID, year, Top10, and audio features

2. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

3. **Follow the step-by-step workflow:**
   - **Step 1**: Load Dataset → Click "Load Dataset"
   - **Step 2**: Convert Artist ID → Click "Convert Artist ID"
   - **Step 3**: Feature Engineering → Click "Perform Feature Engineering"
   - **Step 4**: Preprocess and Split → Click "Preprocess and Split"
   - **Step 5**: Train Model → Click "Train Logistic Regression"

4. **Adjust model parameters in the sidebar:**
   - **Penalty**: Choose regularization type (L1, L2, ElasticNet, None)
   - **C**: Inverse of regularization strength (0.01 - 10.0)
   - **Solver**: Select optimization algorithm

5. **Explore visualizations and make predictions!**

## 📊 Dataset

### Required Columns
- `songID`: Unique identifier for songs
- `songtitle`: Title of the song
- `artistname`: Name of the artist
- `artistID`: Artist identifier (categorical)
- `year`: Release year
- `Top10`: Target variable (0 or 1)
- Audio features: `loudness`, `tempo`, `timesignature`, `key`, `duration`, etc.

### Sample Data
The project includes `new_data.csv` with sample records for testing.

## 🤖 Model Details

### Algorithm: Logistic Regression

**Why Logistic Regression?**
- Interpretable results with clear feature importance
- Fast training and prediction times
- Excellent for binary classification tasks
- Supports various regularization techniques

### Hyperparameters

| Parameter | Options | Description |
|-----------|---------|-------------|
| **Penalty** | L2, L1, ElasticNet, None | Regularization type to prevent overfitting |
| **C** | 0.01 - 10.0 | Inverse regularization strength (smaller = stronger) |
| **Solver** | lbfgs, liblinear, newton-cg, sag, saga | Optimization algorithm |

### Feature Engineering

The model uses both original and engineered features:

**Original Features:**
- Audio metrics: loudness, tempo, timesignature, key, duration
- Metadata: artistID, year

**Engineered Features:**
- `title_length`: Character count in song title
- `artist_length`: Character count in artist name
- `is_love`: Binary flag for romantic keywords
- `is_she`: Binary flag for feminine keywords
- `is_dance`: Binary flag for dance-related keywords
- `Age`: Years since release (2022 - year)

## 🔍 Solver-Penalty Compatibility

Not all solver-penalty combinations are valid. The app includes a built-in compatibility checker:

| Solver | Best For | Regularization | Multiclass |
|--------|----------|----------------|------------|
| **lbfgs** | Small/medium datasets | L2, None | Yes |
| **liblinear** | Small datasets | L1, L2 | One-vs-rest |
| **newton-cg** | Medium datasets | L2, None | Yes |
| **sag** | Large datasets | L2, None | Yes |
| **saga** | Large datasets, sparse | L1, L2, ElasticNet, None | Yes |

### Compatibility Rules:
- ✅ **lbfgs**: Works with L2 and None
- ✅ **liblinear**: Works with L1 and L2
- ✅ **newton-cg**: Works with L2 and None
- ✅ **sag**: Works with L2 and None
- ✅ **saga**: Works with all penalty types

## 📁 Project Structure

```
ml project/
│
├── app.py                  # Main Streamlit application
├── data.csv                # Original dataset (large)
├── new_data.csv            # Sample dataset for testing
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation (this file)
│
└── lib/                   # Virtual environment (auto-generated)
    ├── Scripts/           # Activation scripts
    ├── Lib/              # Installed packages
    └── Include/          # Header files
```

## 📦 Requirements

```
streamlit>=1.50.0
numpy>=2.3.0
pandas>=2.3.0
scikit-learn>=1.7.0
matplotlib>=3.10.0
seaborn>=0.13.0
```

Install all dependencies with:
```bash
pip install -r requirements.txt
```

## 🎨 Screenshots

### Main Interface
- Interactive sidebar with model parameters
- Step-by-step workflow buttons
- Real-time compatibility checking

### Visualizations
- **Target Distribution**: Bar chart showing Top10 vs non-Top10 songs
- **Correlation Heatmap**: Feature importance visualization
- **Confusion Matrix**: Model performance metrics

### Predictions
- Sample feature display
- Prediction probabilities
- Validation indicators (✅/❌)

## 🐛 Troubleshooting

### Common Issues

**Issue**: `FileNotFoundError: data.csv not found`
- **Solution**: Ensure `data.csv` is in the same directory as `app.py`

**Issue**: `AttributeError: session_state has no attribute 'dataset'`
- **Solution**: Restart the Streamlit app (Ctrl+C and rerun)

**Issue**: Incompatible solver-penalty combination error
- **Solution**: Check the sidebar for compatibility warnings and adjust parameters

**Issue**: All predictions are the same class
- **Solution**: Try different C values or add more features

## 🔮 Future Enhancements

- [ ] Support for multiple ML algorithms (Random Forest, XGBoost)
- [ ] Hyperparameter grid search with cross-validation
- [ ] Feature importance visualization
- [ ] Model export/import functionality
- [ ] Batch prediction on uploaded CSV files
- [ ] ROC curve and AUC score visualization
- [ ] Time-series analysis of music trends

## 👨‍💻 Author

**Anish K M**

## 📄 License

This project is open-source and available for educational purposes.

## 🙏 Acknowledgments

- Dataset: Music Records Popularity Dataset
- Framework: Streamlit for interactive web apps
- ML Library: scikit-learn for machine learning algorithms

---

**Happy Predicting! 🎵🎉**

For questions or suggestions, feel free to open an issue or contribute to the project.
