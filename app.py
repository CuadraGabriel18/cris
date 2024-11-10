from flask import Flask, render_template, jsonify, request
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import base64
import io
import graphviz

app = Flask(__name__)

# Variables globales simuladas
df = None
clf_tree_reduced = None
X_train_reduced = None
y_train = None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/load_data', methods=['POST'])
def load_data():
    global df
    df = pd.DataFrame({
        'min_flowpktl': np.random.rand(100),
        'flow_fin': np.random.rand(100),
        'label': np.random.randint(0, 3, 100)
    })
    return jsonify({"data": df.head().to_html()})


@app.route('/train_tree', methods=['POST'])
def train_tree():
    global clf_tree_reduced, X_train_reduced, y_train
    if df is None:
        return jsonify({"error": "No data loaded."}), 400
    
    # Preparar datos
    X = df[['min_flowpktl', 'flow_fin']]
    y = df['label']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_reduced = scaler.fit_transform(X_train)
    X_train_reduced = pd.DataFrame(X_train_reduced, columns=X.columns)
    
    # Entrenar árbol
    clf_tree_reduced = DecisionTreeClassifier(max_depth=3)
    clf_tree_reduced.fit(X_train_reduced, y_train)

    # Exportar gráfico del árbol
    dot_data = export_graphviz(clf_tree_reduced, out_file=None,
                               feature_names=X_train_reduced.columns,
                               class_names=["normal", "adware", "malware"],
                               filled=True, rounded=True,
                               special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.format = 'png'
    graph.render("decision_tree", cleanup=True)

    with open("decision_tree.png", "rb") as image_file:
        tree_image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

    return jsonify({"message": "Tree trained successfully!", "tree_image": tree_image_base64})


@app.route('/decision_boundary', methods=['POST'])
def decision_boundary():
    global clf_tree_reduced, X_train_reduced, y_train
    if clf_tree_reduced is None or X_train_reduced is None:
        return jsonify({"error": "Tree not trained or data not prepared."}), 400

    mins = X_train_reduced.min(axis=0) - 1
    maxs = X_train_reduced.max(axis=0) + 1
    x1, x2 = np.meshgrid(np.linspace(mins[0], maxs[0], 1000),
                         np.linspace(mins[1], maxs[1], 1000))
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf_tree_reduced.predict(X_new).reshape(x1.shape)

    custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])

    plt.figure(figsize=(12, 6))
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    plt.plot(X_train_reduced.values[:, 0][y_train == 0], 
             X_train_reduced.values[:, 1][y_train == 0], "yo", label="normal")
    plt.plot(X_train_reduced.values[:, 0][y_train == 1], 
             X_train_reduced.values[:, 1][y_train == 1], "bs", label="adware")
    plt.plot(X_train_reduced.values[:, 0][y_train == 2], 
             X_train_reduced.values[:, 1][y_train == 2], "g^", label="malware")
    plt.xlabel('min_flowpktl', fontsize=14)
    plt.ylabel('flow_fin', fontsize=14, rotation=90)
    plt.legend(loc="upper left", fontsize=10)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()

    return jsonify({"decision_boundary_image": image_base64})


if __name__ == '__main__':
    app.run(debug=True)