# 🍷 Wine Quality Prediction with Linear Regression

This Go project implements a **Multiple Linear Regression** model to predict wine quality using chemical properties from a dataset. It includes:

* Data preprocessing with normalization
* Model training using gradient descent
* 5-fold cross-validation
* Learning curve and feature importance visualization using [go-echarts](https://github.com/go-echarts/go-echarts)

## 📦 Features

* 📊 **Train/Test Split** with configurable ratio
* 🔁 **K-Fold Cross-Validation** (default: 5 folds)
* 🔍 **R², RMSE, MSE** evaluation metrics
* 📈 **Learning Curve Visualization**
* 📉 **Feature Importance Visualization**
* ⚙️ **Z-score Normalization**

## 📁 Dataset

The dataset must be a CSV file named `wine.csv` located in the same directory as the program. The CSV should follow this structure:

```csv
Fixed Acidity,Volatile Acidity,Citric Acid,Residual Sugar,Chlorides,...
7.4,0.7,0.0,1.9,0.076,...
```

The last column should be the wine quality score (numeric). The first row is assumed to be the header and will be skipped during parsing.

## 🚀 Getting Started

### Prerequisites

* Go 1.18+
* `go-echarts` for plotting

Install dependencies:

```bash
go get github.com/go-echarts/go-echarts/v2
```

### Run the Program

```bash
go run main.go
```

Make sure `wine.csv` is in the same directory.

## 📊 Output

After execution, two HTML visualizations will be created:

* `learning_curve.html`: Plots MSE over training epochs
* `feature_importance.html`: Bar chart of absolute feature weights

You'll also see output in the console including:

* R² scores for cross-validation
* Final model performance on training and test sets
* Learned weights and bias

## 📂 Project Structure

```
.
├── main.go                # Main source code
├── wine.csv              # Wine quality dataset (not included)
├── learning_curve.html   # MSE visualization over epochs
├── feature_importance.html # Feature weights visualization
└── go.mod                # Module dependencies
```

## 🧠 Model Details

* **Model**: Multiple Linear Regression
* **Optimizer**: Gradient Descent
* **Normalization**: Z-score (mean = 0, std = 1)
* **Loss**: Mean Squared Error (MSE)

## 🧪 Example Metrics

```
Cross-validation R² scores:
Fold 1: 0.3432
Fold 2: 0.3621
...
Mean R²: 0.3569

Training RMSE: 0.7123
Testing RMSE: 0.7432
Training R²: 0.4682
Testing R²: 0.3950
```

## 📌 TODO

* Support for different datasets via CLI
* Add regularization (L2/Ridge Regression)
* Export model weights to a file

## 📜 License

This project is open source and available under the [MIT License](LICENSE).
