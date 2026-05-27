import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


st.set_page_config(page_title="Segmentación de Clientes", layout="wide")
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


@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=10,
        min_samples_split=2,
        random_state=42,
    )
    rf.fit(X_train, y_train)
    return rf, X_train, X_test, y_train, y_test


def encode_customer(customer, model_columns):
    row = pd.DataFrame(0, index=[0], columns=model_columns)
    row.loc[0, "Estado_Civil"] = 1 if customer["Estado_Civil"] == "Yes" else 0
    row.loc[0, "Edad"] = customer["Edad"]
    row.loc[0, "Graduado"] = 1 if customer["Graduado"] == "Yes" else 0
    row.loc[0, "Experiencia_Laboral"] = customer["Experiencia_Laboral"]
    row.loc[0, "Puntuacion_Gasto"] = {"Low": 0, "Average": 1, "High": 2}[customer["Puntuacion_Gasto"]]
    row.loc[0, "Tamano_Familiar"] = customer["Tamano_Familiar"]

    gender_column = f"Genero_{customer['Genero']}"
    profession_column = f"Profesion_{customer['Profesion']}"
    if gender_column in row.columns:
        row.loc[0, gender_column] = 1
    if profession_column in row.columns:
        row.loc[0, profession_column] = 1

    return row


def plot_feature_importance(model, columns):
    importance = (
        pd.DataFrame({"Variable": columns, "Importancia": model.feature_importances_})
        .sort_values("Importancia", ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    colors = sns.color_palette("crest", len(importance))
    ax.barh(importance["Variable"], importance["Importancia"], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Importancia relativa")
    ax.set_ylabel("")
    ax.set_title("Variables que más influyen en el modelo final")
    plt.tight_layout()
    return fig


def plot_confusion_matrix(confusion, labels):
    fig, ax = plt.subplots(figsize=(3.8, 2.9))
    sns.heatmap(
        confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Segmento real")
    ax.set_title("Matriz de confusión")
    ax.tick_params(axis="both", labelsize=9)
    plt.tight_layout()
    return fig


def plot_f1_by_segment(report, labels):
    f1_scores = pd.DataFrame(
        {"Segmento": labels, "F1": [report[label]["f1-score"] for label in labels]}
    )

    fig, ax = plt.subplots(figsize=(4.8, 3.1))
    ax.bar(f1_scores["Segmento"], f1_scores["F1"], color=["#4c78a8", "#f58518", "#54a24b", "#e45756"])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Segmento")
    ax.set_ylabel("F1-score")
    ax.set_title("Precisión por segmento")
    for index, row in f1_scores.iterrows():
        ax.text(index, row["F1"] + 0.02, f"{row['F1']:.2f}", ha="center", fontweight="bold")
    plt.tight_layout()
    return fig


def plot_segment_distribution(df):
    segment_counts = df["Segmento"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    ax.bar(segment_counts.index, segment_counts.values, color=["#4c78a8", "#f58518", "#54a24b", "#e45756"])
    ax.set_xlabel("Segmento")
    ax.set_ylabel("Clientes")
    ax.set_title("Cantidad de clientes por segmento")
    for index, value in enumerate(segment_counts.values):
        ax.text(index, value + 25, str(value), ha="center", fontweight="bold", fontsize=9)
    plt.tight_layout()
    return fig


def plot_age_by_segment(df):
    age_summary = df.groupby("Segmento", as_index=False)["Edad"].mean()

    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    ax.bar(age_summary["Segmento"], age_summary["Edad"], color=["#4c78a8", "#f58518", "#54a24b", "#e45756"])
    ax.set_xlabel("Segmento")
    ax.set_ylabel("Edad promedio")
    ax.set_title("Edad promedio por segmento")
    for index, row in age_summary.iterrows():
        ax.text(index, row["Edad"] + 0.8, f"{row['Edad']:.1f}", ha="center", fontweight="bold", fontsize=9)
    plt.tight_layout()
    return fig


def plot_spending_by_segment(df):
    spending_order = ["Low", "Average", "High"]
    spending_labels = {"Low": "Baja", "Average": "Media", "High": "Alta"}
    spending_share = pd.crosstab(df["Segmento"], df["Puntuacion_Gasto"], normalize="index")
    spending_share = spending_share.reindex(columns=spending_order, fill_value=0)

    fig, ax = plt.subplots(figsize=(5.1, 3.2))
    bottom = np.zeros(len(spending_share))
    colors = ["#6baed6", "#fdae6b", "#74c476"]
    for spending, color in zip(spending_order, colors):
        values = spending_share[spending].values
        ax.bar(spending_share.index, values, bottom=bottom, color=color, label=spending_labels[spending])
        bottom += values

    ax.set_xlabel("Segmento")
    ax.set_ylabel("Proporción")
    ax.set_title("Puntuación de gasto por segmento")
    ax.legend(title="Gasto", fontsize=8, title_fontsize=8, loc="upper right")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    return fig


def show_compact_chart(fig):
    left, center, right = st.columns([0.18, 0.64, 0.18])
    with center:
        st.pyplot(fig, use_container_width=False)


def main():
    df = load_data()
    X, y, label_encoder = prepare_data(df)
    rf, X_train, X_test, y_train, y_test = train_model(X, y)

    labels = list(label_encoder.classes_)
    y_pred = rf.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_pred)
    accuracy_train = accuracy_score(y_train, rf.predict(X_train))
    f1_macro = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
    conf_mat = confusion_matrix(y_test, y_pred)

    st.title("Segmentación de Clientes")
    st.caption("Modelo final: Random Forest optimizado para predecir el segmento A, B, C o D de un cliente.")

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Accuracy prueba", f"{accuracy_test:.1%}")
    metric_2.metric("F1 macro", f"{f1_macro:.1%}")
    metric_3.metric("Accuracy entrenamiento", f"{accuracy_train:.1%}")
    metric_4.metric("Clientes analizados", f"{len(df):,}".replace(",", "."))

    st.markdown("### Usar el modelo final")
    form_col, result_col = st.columns([1.1, 0.9], gap="large")

    with form_col:
        st.markdown("Ajustá el perfil del cliente y observá cómo cambia la predicción.")
        slider_col_1, slider_col_2, slider_col_3 = st.columns(3)
        edad = slider_col_1.slider("Edad", int(df["Edad"].min()), int(df["Edad"].max()), int(df["Edad"].median()))
        experiencia = slider_col_2.slider(
            "Experiencia laboral",
            int(df["Experiencia_Laboral"].min()),
            int(df["Experiencia_Laboral"].max()),
            int(df["Experiencia_Laboral"].median()),
        )
        familia = slider_col_3.slider(
            "Tamaño familiar",
            int(df["Tamano_Familiar"].min()),
            int(df["Tamano_Familiar"].max()),
            int(df["Tamano_Familiar"].median()),
        )

        input_col_1, input_col_2, input_col_3 = st.columns(3)
        genero = input_col_1.selectbox("Género", sorted(df["Genero"].unique()))
        estado_civil = input_col_2.selectbox("Estado civil", ["No", "Yes"], format_func=lambda x: "Casado/a" if x == "Yes" else "No casado/a")
        graduado = input_col_3.selectbox("Graduado", ["No", "Yes"], format_func=lambda x: "Sí" if x == "Yes" else "No")

        input_col_4, input_col_5 = st.columns([1.2, 0.8])
        profesion = input_col_4.selectbox("Profesión", sorted(df["Profesion"].unique()))
        gasto = input_col_5.select_slider(
            "Puntuación de gasto",
            options=["Low", "Average", "High"],
            value="Low",
            format_func={"Low": "Baja", "Average": "Media", "High": "Alta"}.get,
        )

    customer = {
        "Genero": genero,
        "Estado_Civil": estado_civil,
        "Edad": edad,
        "Graduado": graduado,
        "Profesion": profesion,
        "Experiencia_Laboral": float(experiencia),
        "Puntuacion_Gasto": gasto,
        "Tamano_Familiar": float(familia),
    }
    encoded_customer = encode_customer(customer, X.columns)
    predicted_code = rf.predict(encoded_customer)[0]
    predicted_segment = label_encoder.inverse_transform([predicted_code])[0]
    probabilities = pd.DataFrame(
        {
            "Segmento": labels,
            "Probabilidad": rf.predict_proba(encoded_customer)[0],
        }
    ).sort_values("Probabilidad", ascending=False)

    with result_col:
        st.markdown("Predicción")
        st.metric("Segmento estimado", predicted_segment)
        st.progress(float(probabilities.iloc[0]["Probabilidad"]))
        st.caption(f"Confianza del segmento principal: {probabilities.iloc[0]['Probabilidad']:.1%}")
        st.bar_chart(probabilities.set_index("Segmento"))

    st.markdown("### Variables utilizadas")
    st.info(
        "El modelo trabaja con un perfil básico del cliente: datos de su edad, "
        "actividad laboral, nivel de gasto y composición familiar."
    )

    st.markdown("### Gráficos del modelo final")
    selected_chart = st.selectbox(
        "Seleccioná un gráfico para visualizar",
        [
            "Variables que más influyen en el modelo",
            "Precisión por segmento",
            "Matriz de confusión",
            "Cantidad de clientes por segmento",
            "Edad promedio por segmento",
            "Puntuación de gasto por segmento",
        ],
    )

    if selected_chart == "Variables que más influyen en el modelo":
        show_compact_chart(plot_feature_importance(rf, X.columns))
    elif selected_chart == "Precisión por segmento":
        show_compact_chart(plot_f1_by_segment(report, labels))
    elif selected_chart == "Matriz de confusión":
        show_compact_chart(plot_confusion_matrix(conf_mat, labels))
    elif selected_chart == "Cantidad de clientes por segmento":
        show_compact_chart(plot_segment_distribution(df))
    elif selected_chart == "Edad promedio por segmento":
        show_compact_chart(plot_age_by_segment(df))
    else:
        show_compact_chart(plot_spending_by_segment(df))


if __name__ == "__main__":
    main()
