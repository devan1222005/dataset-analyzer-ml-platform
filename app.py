from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from werkzeug.utils import secure_filename
import joblib

app = Flask(__name__)


# Import models after initializing db
from models import *
# Define the upload folder
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

UPLOAD_FOLDER = 'uploads'
MODELS_FOLDER = 'models'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

current_dataset = None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    global current_dataset
    file = request.files["file"]
    if file and file.filename.endswith(".csv"):
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(file.filename))
        file.save(filepath)
        current_dataset = pd.read_csv(filepath)
        flash("File uploaded successfully!", "success")
        return redirect(url_for("analyze"))
    else:
        flash("Invalid file format. Please upload a CSV file.", "danger")
        return redirect(url_for("index"))


@app.route('/analyze')
def analyze():
    global current_dataset
    if current_dataset is None:
        flash('No dataset found. Please upload a dataset first.')
        return redirect(url_for('index'))
    
    analysis = {
        'columns': list(current_dataset.columns),
        'data_types': current_dataset.dtypes.to_dict(),
        'missing_values': current_dataset.isnull().sum().to_dict(),
        'shape': current_dataset.shape,
        'describe': current_dataset.describe().to_html(classes='table table-striped'),
        'head': current_dataset.head().to_html(classes='table table-striped'),
        'tail': current_dataset.tail().to_html(classes='table table-striped'),
    }
    return render_template('analyze.html', analysis=analysis)

@app.route("/visualize")
def visualize():
    global current_dataset

    if current_dataset is None:
        flash("No dataset found. Please upload a dataset first.")
        return redirect(url_for("index"))

    # Create a 'static/plots' folder if it doesn't exist
    plots_dir = "static/plots"
    os.makedirs(plots_dir, exist_ok=True)

    # Generate Pairplot
    pairplot_path = os.path.join(plots_dir, "pairplot.png")
    try:
        sns.pairplot(current_dataset.select_dtypes(include=[np.number]))
        plt.savefig(pairplot_path)
    except Exception as e:
        flash(f"Error generating pairplot: {str(e)}")

    # Generate Correlation Matrix
    corr_matrix_path = os.path.join(plots_dir, "corr_matrix.png")
    plt.figure(figsize=(10, 8))
    sns.heatmap(current_dataset.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.savefig(corr_matrix_path)

    # Generate Box Plot
    boxplot_path = os.path.join(plots_dir, "boxplot.png")
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=current_dataset.select_dtypes(include=[np.number]))
    plt.xticks(rotation=90)
    plt.title("Box Plot of Numerical Features")
    plt.savefig(boxplot_path)

    return render_template("visualize.html",
                           pairplot=pairplot_path,
                           corr_matrix=corr_matrix_path,
                           boxplot=boxplot_path)
@app.route("/train", methods=["GET", "POST"])
def train():
    global model, target_column, current_dataset

    if current_dataset is None:
        flash('No dataset found. Please upload a dataset first.')
        return redirect(url_for('index'))

    analysis = {
        'columns': list(current_dataset.columns),
        'data_types': current_dataset.dtypes.to_dict(),
        'missing_values': current_dataset.isnull().sum().to_dict(),
        'shape': current_dataset.shape,
        'describe': current_dataset.describe().to_html(classes='table table-striped'),
        'head': current_dataset.head().to_html(classes='table table-striped'),
        'tail': current_dataset.tail().to_html(classes='table table-striped'),
    }

    if request.method == "POST":
        # Ensure target column is set
        target_column = request.form.get("target_column")
        if not target_column or target_column not in current_dataset.columns:
            flash("Error: Please select a valid target column.")
            return render_template("train.html", analysis=analysis, columns=analysis['columns'])

        X = current_dataset.drop(columns=[target_column])
        y = current_dataset[target_column]

        # Splitting dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Feature scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Model selection
        model_type = request.form.get("model_type")

        if model_type == "classification":
            model = RandomForestClassifier()
        else:
            model = LinearRegression()

        # Training the model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Handle classification/regression evaluation
        if model_type == "classification":
            if len(set(y_test)) > 2:  # Multi-class
                y_pred = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else np.round(y_pred).astype(int)
            else:  # Binary classification
                y_pred = (y_pred > 0.5).astype(int)

            accuracy = accuracy_score(y_test, y_pred) * 100
            result_msg = f"Model trained successfully with Accuracy: {accuracy:.2f}%"
        else:  # Regression
            mse = mean_squared_error(y_test, y_pred)
            result_msg = f"Model trained successfully with MSE: {mse:.2f}"

        # Save the trained model
        joblib.dump(model, "model.pkl")

        return render_template("train.html", message=result_msg, analysis=analysis, columns=analysis['columns'])

    return render_template("train.html", analysis=analysis, columns=analysis['columns'])


@app.route('/preprocess', methods=['GET', 'POST'])  # Allow both GET and POST
def preprocess():
    global current_dataset
    if current_dataset is None:
        flash('No dataset found. Please upload a dataset first.')
        return redirect(url_for('index'))

    analysis = {
        'columns': list(current_dataset.columns),
        'data_types': current_dataset.dtypes.to_dict(),
        'missing_values': current_dataset.isnull().sum().to_dict(),
        'shape': current_dataset.shape,
        'describe': current_dataset.describe().to_html(classes='table table-striped'),
        'head': current_dataset.head().to_html(classes='table table-striped'),
        'tail': current_dataset.tail().to_html(classes='table table-striped'),
    }

    if request.method == "POST":
        action = request.form.get("action")

        if action == "drop_missing":
            current_dataset.dropna(inplace=True)
        elif action == "fill_missing":
            current_dataset.fillna(current_dataset.mean(), inplace=True)
        elif action == "normalize":
            for col in current_dataset.select_dtypes(include=['number']).columns:
                current_dataset[col] = (current_dataset[col] - current_dataset[col].min()) / (current_dataset[col].max() - current_dataset[col].min())
        elif action == "Label_encoding":
            le = LabelEncoder()
            for col in current_dataset.select_dtypes(include=['object']).columns:
                current_dataset[col] = le.fit_transform(current_dataset[col])

        elif action == "delete_column":
            column_to_delete = request.form.get("column_to_delete")
            if column_to_delete in current_dataset.columns:
                current_dataset.drop(columns=[column_to_delete], inplace=True)
                flash(f"Column '{column_to_delete}' deleted successfully!", "success")
            else:
                flash("Invalid column selected.", "danger")

        flash("Preprocessing applied successfully!", "success")
        return redirect(url_for("preprocess"))

    return render_template("preprocess.html", analysis=analysis, columns=analysis['columns'])

if __name__ == "__main__":
    app.run(debug=True)
