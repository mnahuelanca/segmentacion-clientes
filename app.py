import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, f1_score

st.set_page_config(page_title="Segmentación de Clientes", layout="wide")
st.title("Segmentación de Clientes")

sns.set_style("whitegrid")

@st.cache_data
def load_data():
    return pd.read_csv("dfSegmentado.csv")

@st.cache_data
def prepare_data(df):
    X = df.drop(["Identificador", "Segmento"], axis=1)
    y = df["Segmento"]

    X = X.copy()
    X["Graduado"] = X["Graduado"].map({"No": 0, "Yes": 1})
    X["Estado_Civil"] = X["Estado_Civil"].map({"No": 0, "Yes": 1})
    X["Puntuacion_Gasto"] = X["Puntuacion_Gasto"].map({"Low": 0, "Average": 1, "High": 2})

    X_encoded = pd.get_dummies(X, columns=["Genero", "Profesion"], drop_first=True)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X_encoded, y_encoded, label_encoder

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    return rf, X_train, X_test, y_train, y_test

def compute_learning_curve(estimator, X, y):
    return learning_curve(
        estimator,
        X,
        y,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        train_sizes=np.linspace(0.2, 1.0, 6),
    )

def compute_cv_scores(estimator, X, y):
    return cross_val_score(estimator, X, y, cv=5, scoring="accuracy", n_jobs=-1)


def plot_feature_importance(model, columns):
    importances = model.feature_importances_
    df_importance = pd.DataFrame({"feature": columns, "importance": importances})
    df_importance = df_importance.sort_values(by="importance", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="importance", y="feature", data=df_importance, palette="viridis", ax=ax)
    ax.set_title("Importancia de las variables")
    ax.set_xlabel("Importancia relativa")
    ax.set_ylabel("Variable")
    plt.tight_layout()
    return fig


def plot_confusion_matrix(confusion, labels):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title("Matriz de confusión")
    ax.set_xlabel("Predicción del modelo")
    ax.set_ylabel("Segmento real")
    plt.tight_layout()
    return fig


def plot_f1_by_segment(report, labels):
    f1_scores = [report[label]["f1-score"] for label in labels]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#e74c3c", "#c0392b", "#95a5a6", "#2ecc71"][: len(labels)]
    sns.barplot(x=labels, y=f1_scores, palette=colors, ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("F1-Score por segmento")
    ax.set_ylabel("F1-Score")
    ax.set_xlabel("Segmento")
    for i, score in enumerate(f1_scores):
        ax.text(i, score + 0.02, f"{score:.3f}", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    return fig


def plot_roc_curves(y_test, y_score, labels):
    y_test_bin = label_binarize(y_test, classes=np.arange(len(labels)))
    n_classes = y_test_bin.shape[1]

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = sns.color_palette("tab10", n_classes)

    for i, color in enumerate(colors[:n_classes]):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{labels[i]} (AUC={auc(fpr, tpr):.2f})")

    fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    ax.plot(fpr_micro, tpr_micro, color="deeppink", linestyle="--", linewidth=2, label=f"Micro-average (AUC={auc(fpr_micro, tpr_micro):.2f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title("Curvas ROC multiclase")
    ax.legend(loc="lower right", fontsize="small")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_learning_curve(train_sizes, train_scores, test_scores):
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_sizes, train_mean, marker="o", color="#e74c3c", label="Accuracy entrenamiento")
    ax.plot(train_sizes, test_mean, marker="o", color="#2ecc71", label="Accuracy validación")
    ax.set_title("Curva de aprendizaje")
    ax.set_xlabel("Tamaño de entrenamiento")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.3, 1.0)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    return fig


def main():
    df = load_data()
    X, y, label_encoder = prepare_data(df)
    rf, X_train, X_test, y_train, y_test = train_model(X, y)

    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)

    accuracy_test = accuracy_score(y_test, y_pred)
    accuracy_train = accuracy_score(y_train, rf.predict(X_train))
    f1_macro = f1_score(y_test, y_pred, average="macro")
    cv_scores = compute_cv_scores(rf, X, y)
    learning_results = compute_learning_curve(rf, X, y)

    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
    conf_mat = confusion_matrix(y_test, y_pred)
    labels = list(label_encoder.classes_)

    st.markdown("## Resumen ejecutivo")
    with open("resumen.md", "r", encoding="utf-8") as f:
        st.markdown(f.read())

    st.markdown("---")
    st.markdown("## Métricas clave")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy prueba", f"{accuracy_test:.1%}")
    col2.metric("Accuracy entrenamiento", f"{accuracy_train:.1%}")
    col3.metric("F1 Macro", f"{f1_macro:.1%}")
    col4.metric("CV promedio", f"{cv_scores.mean():.1%}", delta=f"±{cv_scores.std():.2%}")

    st.markdown("---")
    st.markdown("## Gráficos de apoyo")

    st.subheader("Importancia de las variables")
    st.pyplot(plot_feature_importance(rf, X.columns))

    st.subheader("Matriz de confusión")
    st.pyplot(plot_confusion_matrix(conf_mat, labels))

    st.subheader("F1-Score por segmento")
    st.pyplot(plot_f1_by_segment(report, labels))

    st.subheader("Curvas ROC multiclase")
    st.pyplot(plot_roc_curves(y_test, y_proba, labels))

    st.subheader("Validación cruzada y estabilidad")
    st.write("Accuracy por fold (5-fold CV):")
    st.write([float(f"{s:.4f}") for s in cv_scores])
    st.line_chart(pd.DataFrame({"Accuracy": cv_scores}))

    st.subheader("Curva de aprendizaje")
    train_sizes, train_scores, test_scores = learning_results
    st.pyplot(plot_learning_curve(train_sizes, train_scores, test_scores))

    st.markdown("---")
    st.subheader("Diagnóstico rápido")
    st.write(
        "- El modelo base con Random Forest alcanza ~" + f"{accuracy_test:.1%}" + 
        ", con un gap de entrenamiento/prueba de " + f"{accuracy_train - accuracy_test:.1%}."
    )
    st.write(
        "- El F1 Macro y la estabilidad de validación cruzada muestran si el comportamiento del modelo coincide con el resumen del análisis."    )
    st.write(
        "- En esta aplicación, el resumen escrito en `resumen.md` se acompaña con las mismas métricas centrales y los gráficos de `modelosGEM.ipynb`."
    )

if __name__ == "__main__":
    main()