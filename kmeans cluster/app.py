import pandas as pd
import plotly.express as px
from dash import Input, Output, dcc, html, Dash
from jupyter_dash import JupyterDash
from scipy.stats.mstats import trimmed_var
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

JupyterDash.infer_jupyter_proxy_config()

def wrangle(filepath):
    df = pd.read_csv(filepath)
    mask = (df["TURNFEAR"] == 1) & (df["NETWORTH"] < 2e6)
    df = df[mask]
    return df

df = wrangle("SCFP2019.csv.gz")
app = Dash(__name__)

def get_high_var_features(trimmed=True, return_feat_names=True):
    if trimmed:
        top_five_features = df.apply(trimmed_var).sort_values().tail(5)
    else:
        top_five_features = df.var().sort_values().tail(5)
    if return_feat_names:
        top_five_features = top_five_features.index.tolist()
    return top_five_features

def get_model_and_labels(method="kmeans", trimmed=True, k=2):
    features = get_high_var_features(trimmed=trimmed, return_feat_names=True)
    X = df[features]
    X_scaled = StandardScaler().fit_transform(X)
    if method == "kmeans":
        model = KMeans(n_clusters=k, random_state=42)
    elif method == "agglomerative":
        model = AgglomerativeClustering(n_clusters=k)
    else:
        raise ValueError("Unsupported clustering method")
    labels = model.fit_predict(X_scaled)
    return X_scaled, labels

def get_cluster_centers(X_scaled, model, pca):
    centers = model.cluster_centers_
    return pca.transform(centers)

def get_ellipse(x, y, n_std=2.0, num_points=100):
    if len(x) < 2:
        return x, y
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    t = np.linspace(0, 2 * np.pi, num_points)
    ellipse = np.array([width / 2 * np.cos(t), height / 2 * np.sin(t)])
    R = np.array([[np.cos(np.radians(theta)), -np.sin(np.radians(theta))],
                  [np.sin(np.radians(theta)), np.cos(np.radians(theta))]])
    ellipse_rot = R @ ellipse
    return ellipse_rot[0] + np.mean(x), ellipse_rot[1] + np.mean(y)

@app.callback(
    Output("bar-chart", "figure"),
    Input("trim-button", "value")
)
def serve_bar_chart(trimmed=True):
    top_five_features = get_high_var_features(trimmed=trimmed, return_feat_names=False)
    fig = px.bar(x=top_five_features, y=top_five_features.index, orientation="h")
    fig.update_layout(xaxis_title="Variance", yaxis_title="Features")
    return fig

@app.callback(
    Output("pca-scatter", "figure"),
    Input("trim-button", "value"),
    Input("k-slider", "value"),
    Input("centroid-toggle", "value"),
    Input("cluster-method", "value")
)
def serve_scatter_plot(trimmed, k, centroid_toggle, method):
    features = get_high_var_features(trimmed=trimmed, return_feat_names=True)
    X = df[features]
    X_scaled = StandardScaler().fit_transform(X)

    if method == "kmeans":
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(X_scaled)
    elif method == "agglomerative":
        model = AgglomerativeClustering(n_clusters=k)
        labels = model.fit_predict(X_scaled)
    else:
        raise ValueError("Unknown method")

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    df_pca["labels"] = labels.astype(str)

    fig = px.scatter(df_pca, x="PC1", y="PC2", color="labels", title=f"PCA of {method.title()} Clusters")

    if "show" in centroid_toggle and method == "kmeans":
        for label in df_pca["labels"].unique():
            cluster = df_pca[df_pca["labels"] == label]
            x_ellipse, y_ellipse = get_ellipse(cluster["PC1"], cluster["PC2"])
            fig.add_trace(go.Scatter(
                x=x_ellipse,
                y=y_ellipse,
                mode="lines",
                line=dict(dash="dot", color="black"),
                showlegend=False
            ))

        centers_pca = get_cluster_centers(X_scaled, model, pca)
        fig.add_trace(go.Scatter(
            x=centers_pca[:, 0],
            y=centers_pca[:, 1],
            mode="markers",
            marker=dict(symbol="x", size=12, color="black"),
            name="Centroids"
        ))

    fig.update_layout(xaxis_title="PC1", yaxis_title="PC2")
    return fig

@app.callback(
    Output("comparison-plot", "figure"),
    Input("trim-button", "value"),
    Input("k-slider", "value")
)
def compare_clusters(trimmed, k):
    X_kmeans, labels_kmeans = get_model_and_labels("kmeans", trimmed, k)
    X_agg, labels_agg = get_model_and_labels("agglomerative", trimmed, k)

    pca_kmeans = PCA(n_components=2, random_state=42)
    X_pca_kmeans = pca_kmeans.fit_transform(X_kmeans)

    pca_agg = PCA(n_components=2, random_state=42)
    X_pca_agg = pca_agg.fit_transform(X_agg)

    fig = make_subplots(rows=1, cols=2, subplot_titles=["KMeans", "Agglomerative"])

    fig.add_trace(go.Scatter(x=X_pca_kmeans[:, 0], y=X_pca_kmeans[:, 1], mode="markers",
                             marker=dict(color=labels_kmeans, colorscale="Viridis", showscale=False),
                             name="KMeans"),
                  row=1, col=1)

    fig.add_trace(go.Scatter(x=X_pca_agg[:, 0], y=X_pca_agg[:, 1], mode="markers",
                             marker=dict(color=labels_agg, colorscale="Plasma", showscale=False),
                             name="Agglomerative"),
                  row=1, col=2)

    fig.update_layout(title="KMeans vs Agglomerative Clustering Comparison", showlegend=False)
    return fig

@app.callback(
    Output("metrics", "children"),
    Input("trim-button", "value"),
    Input("k-slider", "value"),
    Input("cluster-method", "value")
)
def show_metrics(trimmed, k, method):
    features = get_high_var_features(trimmed=trimmed, return_feat_names=True)
    X = df[features]
    X_scaled = StandardScaler().fit_transform(X)

    if method == "kmeans":
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(X_scaled)
        inertia = model.inertia_
        sil_score = silhouette_score(X_scaled, labels)
        return html.Div([
            html.H4(f"Inertia: {inertia:.0f}"),
            html.H4(f"Silhouette Score (KMeans): {sil_score:.3f}")
        ])
    else:
        model = AgglomerativeClustering(n_clusters=k)
        labels = model.fit_predict(X_scaled)
        sil_score = silhouette_score(X_scaled, labels)
        return html.Div([
            html.H4("Inertia: Not applicable"),
            html.H4(f"Silhouette Score (Agglomerative): {sil_score:.3f}")
        ])

@app.callback(
    Output("elbow-plot", "figure"),
    Input("trim-button", "value")
)
def show_elbow_plot(trimmed):
    features = get_high_var_features(trimmed=trimmed, return_feat_names=True)
    X = df[features]
    X_scaled = StandardScaler().fit_transform(X)

    k_values = list(range(2, 13, 2))
    inertias = []
    silhouettes_kmeans = []
    silhouettes_agg = []

    for k in k_values:
        model_kmeans = KMeans(n_clusters=k, random_state=42)
        labels_kmeans = model_kmeans.fit_predict(X_scaled)
        inertias.append(model_kmeans.inertia_)
        silhouettes_kmeans.append(silhouette_score(X_scaled, labels_kmeans))

        model_agg = AgglomerativeClustering(n_clusters=k)
        labels_agg = model_agg.fit_predict(X_scaled)
        silhouettes_agg.append(silhouette_score(X_scaled, labels_agg))

    best_k_kmeans = k_values[np.argmax(silhouettes_kmeans)]
    best_k_agg = k_values[np.argmax(silhouettes_agg)]

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Inertia (KMeans)", "Silhouette Score Comparison"])

    fig.add_trace(go.Scatter(x=k_values, y=inertias, mode="lines+markers", name="Inertia"), row=1, col=1)
    fig.add_trace(go.Scatter(x=k_values, y=silhouettes_kmeans, mode="lines+markers", name="KMeans"), row=1, col=2)
    fig.add_trace(go.Scatter(x=k_values, y=silhouettes_agg, mode="lines+markers", name="Agglomerative"), row=1, col=2)

    fig.add_trace(go.Scatter(x=[best_k_kmeans], y=[max(silhouettes_kmeans)], mode="markers+text", text=[f"Best k = {best_k_kmeans}"], textposition="top center", marker=dict(color="green", size=12), name="Best KMeans"), row=1, col=2)
    fig.add_trace(go.Scatter(x=[best_k_agg], y=[max(silhouettes_agg)], mode="markers+text", text=[f"Best k = {best_k_agg}"], textposition="bottom center", marker=dict(color="red", size=12), name="Best Agglomerative"), row=1, col=2)

    fig.update_layout(title="Elbow Curve: Inertia and Silhouette Score Comparison")
    fig.update_xaxes(title_text="k", row=1, col=1)
    fig.update_xaxes(title_text="k", row=1, col=2)
    fig.update_yaxes(title_text="Inertia", row=1, col=1)
    fig.update_yaxes(title_text="Silhouette Score", row=1, col=2)

    return fig

app.layout = html.Div([
    html.H1("Survey of Consumer Finances"),
    html.H2("High Variance Features"),
    dcc.Graph(figure=serve_bar_chart(), id="bar-chart"),

    dcc.RadioItems(
        options=[{"label": "trimmed", "value": True}, {"label": "not-trimmed", "value": False}],
        value=True,
        id="trim-button"
    ),

    html.H2("K-means Clustering"),
    html.H3("Number of Clusters (k)"),
    dcc.Slider(min=2, max=12, step=2, value=2, id="k-slider"),

    html.H3("Clustering Method"),
    dcc.Dropdown(
        options=[
            {"label": "KMeans", "value": "kmeans"},
            {"label": "Agglomerative", "value": "agglomerative"}
        ],
        value="kmeans",
        id="cluster-method",
        clearable=False
    ),

    dcc.Graph(id="pca-scatter"),

    html.H3("Show Cluster Centroids"),
    dcc.Checklist(
        options=[{"label": "Show Centroids & Ellipses", "value": "show"}],
        value=[],
        id="centroid-toggle",
        inline=True
    ),

    html.H2("Clustering Metrics"),
    html.Div(id="metrics"),

    html.H2("Elbow Curve (KMeans + Agglomerative Silhouette)"),
    dcc.Graph(id="elbow-plot"),

    html.H2("Clustering Comparison"),
    dcc.Graph(id="comparison-plot")
])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)