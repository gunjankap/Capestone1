##############################################
# STREAMLIT APP – BIKE + AQI AI ANALYSIS + CMBS VISUALS
##############################################

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor


# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AI Model Analysis + CMBS", layout="wide")


# -----------------------------
# HEADER
# -----------------------------
col1, col2 = st.columns([1, 6])

with col1:
    try:
        st.image("college_logo.jpg", width=90)
    except Exception:
        st.warning("Logo not found: college_logo.jpg")

with col2:
    st.markdown(
        """
        <div style="line-height:1.6; text-align:right;">
            <div style="font-size:16px; font-weight:700; color:#0b2e73;">
                Gunjan Kapoor
            </div>
            <div style="font-size:13px; color:#333;">
                Roll No: <b>EMBADTA24003</b>
            </div>
            <div style="font-size:13px; color:#333;">
                Mentor: <b>Dr. Manish Sarkhel</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<hr style='margin-top:8px; margin-bottom:8px;'>", unsafe_allow_html=True)

# ---------- BANNER ----------
try:
    st.image("banner1.png", use_container_width=True)
except Exception:
    st.info("Banner not found: banner1.png")


##############################################
# LOAD DATA
##############################################
@st.cache_data
def load_bike():
    day = pd.read_csv("day.csv")
    hour = pd.read_csv("hour.csv")

    drop_cols = ["instant", "dteday", "casual", "registered"]
    day = day.drop(columns=[c for c in drop_cols if c in day.columns])
    hour = hour.drop(columns=[c for c in drop_cols if c in hour.columns])
    return day, hour

@st.cache_data
def load_aqi():
    df = pd.read_csv("aqi.csv")
    df.columns = df.columns.str.strip()
    keep = df.select_dtypes(include=np.number)
    return df, keep

day, hour = load_bike()
aqi_raw, aqi = load_aqi()


##############################################
# SIDEBAR
##############################################
st.sidebar.title("📊 AI Model Analysis")

dataset_choice = st.sidebar.selectbox(
    "Choose Dataset",
    ["Bike Dataset - Day", "Bike Dataset - Hour", "AQI"]
)

model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["Linear Regression", "Decision Tree", "Random Forest (Ensemble)", "Neural Network (MLP)"]
)


##############################################
# DATASET HANDLING
##############################################
if dataset_choice == "Bike Dataset - Day":
    df = day.copy()
    target = "cnt"

elif dataset_choice == "Bike Dataset - Hour":
    df = hour.copy()
    target = "cnt"

else:
    st.sidebar.markdown("### AQI Target Selection")
    target = st.sidebar.selectbox(
        "Select Prediction Target",
        [c for c in aqi.columns if c not in ["Date", "Time"]]
    )
    df = aqi.copy()


##############################################
# SPLIT
##############################################
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Scaler for models that benefit from scaling (LR, NN)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)


##############################################
# TRAIN ONE SELECTED MODEL
##############################################
def build_model(name):
    if name == "Linear Regression":
        model = LinearRegression()
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)

    elif name == "Decision Tree":
        model = DecisionTreeRegressor(max_depth=8, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

    elif name == "Random Forest (Ensemble)":
        model = RandomForestRegressor(n_estimators=220, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

    else:  # Neural Network
        model = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=800,
            random_state=42
        )
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)

    return model, preds

model, preds = build_model(model_choice)


##############################################
# UI SUMMARY
##############################################
st.markdown(
    f"""
    <div style="font-size:18px; font-weight:600; padding:8px; 
                border-radius:8px; background:#f7f9fc;">
        Dataset: <span style="color:#2e7fe8">{dataset_choice}</span> |
        Model: <span style="color:#2e7fe8">{model_choice}</span> |
        Target: <span style="color:#2e7fe8">{target}</span>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(f"""
<div style="font-size:12px; padding:8px; background:#ffffff;
            border:1px dashed #d6e1ff;">
<b>Data Snapshot</b><br>
• Records: {df.shape[0]}<br>
• Features: {X.shape[1]}<br>
• Missing Values: {df.isna().sum().sum()}<br>
• Target Range: {round(y.min(),1)} – {round(y.max(),1)}
</div>
""", unsafe_allow_html=True)


##############################################
# METRICS
##############################################
mae = round(mean_absolute_error(y_test, preds), 2)
rmse = round(np.sqrt(mean_squared_error(y_test, preds)), 2)
r2 = round(r2_score(y_test, preds), 3)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("<h5 style='text-align:center; color:#5a6bbf;'>MAE</h5>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center; font-size:20px; color:#0b2e73; font-weight:700;'>{mae}</p>", unsafe_allow_html=True)

with c2:
    st.markdown("<h5 style='text-align:center; color:#5a6bbf;'>RMSE</h5>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center; font-size:20px; color:#0b2e73; font-weight:700;'>{rmse}</p>", unsafe_allow_html=True)

with c3:
    st.markdown("<h5 style='text-align:center; color:#5a6bbf;'>R2 Score</h5>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center; font-size:20px; color:#0b2e73; font-weight:700;'>{r2}</p>", unsafe_allow_html=True)


##############################################
# PERFORMANCE DIAGNOSTICS
##############################################
st.markdown(
    """
    <h4 style='text-align:center; color:#0b2e73;'>
        📊 Model Performance Diagnostics
    </h4>
    """,
    unsafe_allow_html=True
)

c1, c2, c3 = st.columns(3)

with c1:
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    sns.scatterplot(x=y_test, y=preds, s=12, ax=ax)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
            '--', color='gray', linewidth=1)
    ax.set_title("Actual vs Predicted", fontsize=10)
    ax.set_xlabel("Actual", fontsize=8)
    ax.set_ylabel("Predicted", fontsize=8)
    st.pyplot(fig)

with c2:
    residuals = y_test - preds
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    sns.scatterplot(x=preds, y=residuals, s=12, ax=ax)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_title("Residual Plot", fontsize=10)
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("Residuals", fontsize=8)
    st.pyplot(fig)

with c3:
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    sns.histplot(residuals, kde=True, ax=ax)
    ax.set_title("Prediction Error Distribution", fontsize=10)
    ax.set_xlabel("Error", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    st.pyplot(fig)

pred_std = float(np.std(preds))
st.markdown(f"""
<div style="font-size:12px; padding:8px; background:#fff7f0;
            border-left:4px solid #ff9f40;">
<b>Prediction Stability Indicator</b><br>
• Std Dev of Predictions: {round(pred_std,2)}<br>
• Interpretation: Higher values indicate more dispersed predictions across samples.
</div>
""", unsafe_allow_html=True)


##############################################
# FEATURE IMPORTANCE (RF ONLY)
##############################################
if model_choice == "Random Forest (Ensemble)":
    feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    top_feat = feat_imp.head(5)

    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    sns.barplot(x=top_feat.values, y=top_feat.index, ax=ax)
    ax.set_title("Top 5 Feature Importance", fontsize=9, color="#0b2e73")
    ax.set_xlabel("Importance", fontsize=8)
    ax.set_ylabel("")
    ax.tick_params(axis='x', labelsize=7)
    ax.tick_params(axis='y', labelsize=7)
    plt.tight_layout(pad=0.6)
    st.pyplot(fig)


##############################################
# GLOBAL FEATURE INFLUENCE ACROSS MODELS
##############################################
st.markdown(
    """
    <h4 style='text-align:center; color:#0b2e73;'>
        🌍 Global Feature Influence Across Models
    </h4>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="font-size:13px; padding:10px;
                background:#f7f9fc;
                border-left:5px solid #2e7fe8;
                border-radius:8px;">
    <b>Global Feature Influence Across Models (Normalized)</b><br>
    Compares dominant drivers across different model families to explain why
    outputs may converge even with different architectures.
    </div>
    """,
    unsafe_allow_html=True
)

# Train 3 models (LR uses scaled) for comparison
lr_model = LinearRegression().fit(X_train_s, y_train)
tree_model = DecisionTreeRegressor(max_depth=8, random_state=42).fit(X_train, y_train)
rf_model = RandomForestRegressor(n_estimators=200, random_state=42).fit(X_train, y_train)

features = X.columns
lr_imp = np.abs(lr_model.coef_)
tree_imp = tree_model.feature_importances_
rf_imp = rf_model.feature_importances_

imp_df = pd.DataFrame({
    "Feature": features,
    "Linear Regression": lr_imp,
    "Decision Tree": tree_imp,
    "Random Forest": rf_imp
})

# Normalize for fair comparison
imp_df.iloc[:, 1:] = imp_df.iloc[:, 1:].apply(lambda x: x / (x.max() if x.max() != 0 else 1))

imp_df["Overall"] = imp_df.iloc[:, 1:].mean(axis=1)
top_imp = imp_df.sort_values("Overall", ascending=False).head(8)

fig, ax = plt.subplots(figsize=(7, 4))
top_imp.set_index("Feature")[["Linear Regression", "Decision Tree", "Random Forest"]].plot(kind="barh", ax=ax)
ax.set_title("Top Global Drivers Across Model Families", fontsize=11)
ax.set_xlabel("Normalized Influence Score", fontsize=9)
ax.set_ylabel("")
plt.tight_layout()
st.pyplot(fig)


##############################################
# LOCAL EXPLAINABILITY OF A COLLECTIVE FAILURE (APPROX)
##############################################
st.markdown(
    """
    <h4 style='text-align:center; color:#0b2e73;'>
        🎯 Local Explainability of a Collective Failure Instance
    </h4>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="font-size:13px; padding:10px;
                background:#fff7f0;
                border-left:5px solid #ff9f40;
                border-radius:8px;">
    <b>Approximate Local Attribution</b><br>
    Picks the worst collective failure point (high error across models) and compares
    per-feature influence proxies to show shared reasoning patterns.
    </div>
    """,
    unsafe_allow_html=True
)

# Re-train models
lr_model = LinearRegression().fit(X_train_s, y_train)
tree_model = DecisionTreeRegressor(max_depth=8, random_state=42).fit(X_train, y_train)
rf_model = RandomForestRegressor(n_estimators=200, random_state=42).fit(X_train, y_train)

lr_preds = lr_model.predict(X_test_s)
tree_preds = tree_model.predict(X_test)
rf_preds = rf_model.predict(X_test)

error_df = pd.DataFrame({
    "Actual": y_test.values,
    "LR_Error": np.abs(y_test.values - lr_preds),
    "Tree_Error": np.abs(y_test.values - tree_preds),
    "RF_Error": np.abs(y_test.values - rf_preds),
})
error_df["Total_Error"] = error_df[["LR_Error", "Tree_Error", "RF_Error"]].mean(axis=1)

worst_index = int(error_df["Total_Error"].idxmax())

st.markdown(
    f"""
    ✅ Selected instance: <b>Worst collective failure point</b><br>
    • Actual = {round(float(error_df.loc[worst_index,"Actual"]),2)}<br>
    • RF Prediction = {round(float(rf_preds[worst_index]),2)}<br>
    """,
    unsafe_allow_html=True
)

# Instance row
x_instance_raw = X_test.iloc[worst_index:worst_index+1]
x_instance_s = X_test_s[worst_index:worst_index+1]

# LR local contribution (scaled-space is okay for relative comparison)
lr_contrib = lr_model.coef_ * x_instance_s[0]

local_df = pd.DataFrame({
    "Feature": X.columns,
    "LR_Local_Impact": np.abs(lr_contrib),
    "Tree_Proxy": tree_model.feature_importances_,
    "RF_Proxy": rf_model.feature_importances_
})

local_df.iloc[:, 1:] = local_df.iloc[:, 1:].apply(lambda x: x / (x.max() if x.max() != 0 else 1))
top_local = local_df.sort_values("LR_Local_Impact", ascending=False).head(6)

fig, ax = plt.subplots(figsize=(7, 4))
top_local.set_index("Feature")[["LR_Local_Impact", "Tree_Proxy", "RF_Proxy"]].plot(kind="barh", ax=ax)
ax.set_title("Shared Feature Attribution in Collective Failure Case", fontsize=11)
ax.set_xlabel("Normalized Local Contribution (Proxy)", fontsize=9)
ax.set_ylabel("")
plt.tight_layout()
st.pyplot(fig)


##############################################
# BASELINE MODEL COMPARISON
##############################################
st.markdown(
    """
    <h4 style='text-align:center; color:#0b2e73;'>
        Aggregate Baseline Performance Comparison Across Models
    </h4>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="font-size:13px; padding:10px;
                background:#f7f9fc;
                border-left:5px solid #2e7fe8;
                border-radius:8px;">
    Baseline comparison summarizes performance across model families before stress + CMBS analysis.
    </div>
    """,
    unsafe_allow_html=True
)

baseline_models = {
    "Linear Regression": ("scaled", LinearRegression()),
    "Decision Tree": ("raw", DecisionTreeRegressor(max_depth=8, random_state=42)),
    "Random Forest": ("raw", RandomForestRegressor(n_estimators=200, random_state=42)),
    "Neural Network": ("scaled", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=800, random_state=42)),
}

results = []
for name, (mode, mdl) in baseline_models.items():
    if mode == "scaled":
        mdl.fit(X_train_s, y_train)
        pred = mdl.predict(X_test_s)
    else:
        mdl.fit(X_train, y_train)
        pred = mdl.predict(X_test)

    rmse_val = float(np.sqrt(mean_squared_error(y_test, pred)))
    r2_val = float(r2_score(y_test, pred))
    results.append({"Model": name, "RMSE": rmse_val, "R2 Score": r2_val})

perf_df = pd.DataFrame(results)

fig, ax = plt.subplots(figsize=(7, 4))
sns.barplot(data=perf_df, x="Model", y="RMSE", ax=ax)
ax.set_title("Baseline RMSE Comparison Across Models", fontsize=11)
ax.set_ylabel("RMSE (Lower is Better)")
ax.set_xlabel("")
ax.tick_params(axis='x', rotation=20)
st.pyplot(fig)

st.markdown("### Baseline R² Scores (Higher is Better)")
st.dataframe(
    perf_df[["Model", "R2 Score"]].sort_values("R2 Score", ascending=False),
    use_container_width=True,
    height=180
)


##############################################
# STRESS TEST (NOISE INJECTION)
##############################################
st.markdown(
    """
    <h4 style='text-align:center; color:#0b2e73;'>
        Model Performance Under Stress Conditions
    </h4>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="font-size:13px; padding:10px;
                background:#f4f6fc;
                border-left:5px solid #d9534f;
                border-radius:8px;">
    Adds controlled noise to inputs to measure robustness. RMSE degradation highlights fragility.
    </div>
    """,
    unsafe_allow_html=True
)

stress_levels = [0.0, 0.05, 0.10, 0.20]
stress_results = []

for noise in stress_levels:
    X_stress = X_test.copy()
    X_stress = X_stress + noise * np.random.normal(0, 1, X_stress.shape)

    # scaled version for LR & NN
    X_stress_s = scaler.transform(X_stress)

    for name, (mode, mdl) in baseline_models.items():
        if mode == "scaled":
            mdl.fit(X_train_s, y_train)
            pred_stress = mdl.predict(X_stress_s)
        else:
            mdl.fit(X_train, y_train)
            pred_stress = mdl.predict(X_stress)

        rmse_val = float(np.sqrt(mean_squared_error(y_test, pred_stress)))
        stress_results.append({
            "Model": name,
            "Stress Level": f"{int(noise*100)}% Noise",
            "RMSE": rmse_val
        })

stress_df = pd.DataFrame(stress_results)

fig, ax = plt.subplots(figsize=(8, 4))
sns.lineplot(data=stress_df, x="Stress Level", y="RMSE", hue="Model", marker="o", ax=ax)
ax.set_title("Performance Degradation Under Increasing Stress", fontsize=11)
ax.set_xlabel("Stress Condition (Noise Injected into Inputs)")
ax.set_ylabel("RMSE (Higher = Worse Performance)")
ax.grid(True, linestyle="--", alpha=0.4)
st.pyplot(fig)

st.caption(
    "As noise increases, error rises unevenly across architectures, revealing robustness gaps."
)


##############################################
# BLIND SPOT / SUBGROUP ERROR ANALYSIS (selected model)
##############################################
st.markdown(
    """
    <h5 style='color:#0b2e73;'>
        ⚠️ Blind Spot / Subgroup Error Analysis (Selected Model)
    </h5>
    """,
    unsafe_allow_html=True
)

blind_df = X_test.copy()
blind_df["actual"] = y_test.values
blind_df["pred"] = preds

if dataset_choice in ["Bike Dataset - Day", "Bike Dataset - Hour"]:
    season_rmse = blind_df.groupby("season").apply(
        lambda x: np.sqrt(mean_squared_error(x["actual"], x["pred"]))
    ).reset_index(name="RMSE")

    weather_rmse = blind_df.groupby("weathersit").apply(
        lambda x: np.sqrt(mean_squared_error(x["actual"], x["pred"]))
    ).reset_index(name="RMSE")

    working_rmse = blind_df.groupby("workingday").apply(
        lambda x: np.sqrt(mean_squared_error(x["actual"], x["pred"]))
    ).reset_index(name="RMSE")

else:
    # AQI: Create bins using first two numeric columns (you can change these if needed)
    blind_df["TEMP_BIN"] = pd.qcut(blind_df.iloc[:, 0], 4, duplicates="drop")
    blind_df["HUM_BIN"] = pd.qcut(blind_df.iloc[:, 1], 4, duplicates="drop")

    temp_rmse = blind_df.groupby("TEMP_BIN").apply(
        lambda x: np.sqrt(mean_squared_error(x["actual"], x["pred"]))
    ).reset_index(name="RMSE")

    hum_rmse = blind_df.groupby("HUM_BIN").apply(
        lambda x: np.sqrt(mean_squared_error(x["actual"], x["pred"]))
    ).reset_index(name="RMSE")

st.markdown("""
<div style="font-size:13px; padding:12px;
            background:#f4f6fc;
            border-radius:8px;">
<b>Key Observations</b><br>
• High overall accuracy can hide subgroup-specific failures.<br>
• Certain conditions show higher RMSE (potential blind spots).<br>
• Subgroup diagnostics support fairness + robustness validation.
</div>
""", unsafe_allow_html=True)

st.markdown("### 📌 Subgroup RMSE Results")
if dataset_choice in ["Bike Dataset - Day", "Bike Dataset - Hour"]:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Season RMSE**")
        st.dataframe(season_rmse, use_container_width=True)
    with c2:
        st.markdown("**Weather Situation RMSE**")
        st.dataframe(weather_rmse, use_container_width=True)
    with c3:
        st.markdown("**Working Day RMSE**")
        st.dataframe(working_rmse, use_container_width=True)
else:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Temperature Bin RMSE**")
        st.dataframe(temp_rmse, use_container_width=True)
    with c2:
        st.markdown("**Humidity Bin RMSE**")
        st.dataframe(hum_rmse, use_container_width=True)


##############################################
# PREDICTION AGREEMENT ACROSS MODELS (VISUAL)
##############################################
st.markdown(
    """
    <h4 style='text-align:center; color:#0b2e73;'>
        Prediction Agreement Across Models Under Baseline Conditions
    </h4>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="font-size:13px; padding:10px;
                background:#fff7f0;
                border-left:5px solid #ff9f40;
                border-radius:8px;">
    Correlation-based agreement highlights how similarly different model families behave.
    High agreement can still hide a shared blind spot (CMBS).
    </div>
    """,
    unsafe_allow_html=True
)

models_for_agreement = {
    "Linear Regression": ("scaled", LinearRegression()),
    "Decision Tree": ("raw", DecisionTreeRegressor(max_depth=8, random_state=42)),
    "Random Forest": ("raw", RandomForestRegressor(n_estimators=200, random_state=42)),
    "Neural Network": ("scaled", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=800, random_state=42)),
}

predictions = {}
for name, (mode, mdl) in models_for_agreement.items():
    if mode == "scaled":
        mdl.fit(X_train_s, y_train)
        predictions[name] = mdl.predict(X_test_s)
    else:
        mdl.fit(X_train, y_train)
        predictions[name] = mdl.predict(X_test)

pred_df = pd.DataFrame(predictions)

st.markdown("### 🔍 Prediction Similarity (Correlation-Based Agreement)")
corr = pred_df.corr()

fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap="Blues", fmt=".2f", linewidths=0.5, ax=ax)
ax.set_title("Prediction Agreement Across Models", fontsize=11)
st.pyplot(fig)

st.caption(
    "High correlation indicates convergence in outputs, motivating deeper CMBS investigation."
)


##############################################
# CMBS CHECK — Collective Model Blind Spot (TABLES)
##############################################
st.markdown(
    """
    <h5 style='color:#0b2e73;'>
        🧠 CMBS — Collective Model Blind Spot Check (Subgroup-Level)
    </h5>
    """,
    unsafe_allow_html=True
)

st.markdown("""
<div style="font-size:13px; padding:12px;
            background:#fff;
            border:1px solid #e0e6ff;
            border-left:4px solid #d9534f;">
<b>CMBS Interpretation</b><br>
A subgroup is flagged as a Collective Model Blind Spot when multiple
independent models simultaneously show elevated error (vs baseline),
indicating a structural data/representation issue rather than a model-specific weakness.
</div>
""", unsafe_allow_html=True)

# attach multi-model preds to blind_df for subgroup RMSE by model
blind_df2 = X_test.copy()
blind_df2["actual"] = y_test.values
blind_df2["lr"] = pred_df["Linear Regression"].values
blind_df2["tree"] = pred_df["Decision Tree"].values
blind_df2["rf"] = pred_df["Random Forest"].values
blind_df2["nn"] = pred_df["Neural Network"].values

def cmbs_check(df_in, group_col, preds_cols=("lr", "tree", "rf", "nn"), threshold=0.25, min_n=8):
    results = {}
    base = float(np.sqrt(mean_squared_error(df_in["actual"], df_in["rf"])))  # baseline reference

    for g in df_in[group_col].dropna().unique():
        sub = df_in[df_in[group_col] == g]
        if len(sub) < min_n:
            continue

        row = {}
        for p in preds_cols:
            row[p] = round(float(np.sqrt(mean_squared_error(sub["actual"], sub[p]))), 2)

        row["Collective_BlindSpot"] = all(
            float(np.sqrt(mean_squared_error(sub["actual"], sub[p]))) > base * (1 + threshold)
            for p in preds_cols
        )
        row["n"] = int(len(sub))
        results[g] = row

    out = pd.DataFrame(results).T
    out.index.name = group_col
    return out.reset_index()

# Ensure bins exist for AQI
if dataset_choice in ["Bike Dataset - Day", "Bike Dataset - Hour"]:
    season_cmbs = cmbs_check(blind_df2, "season")
    weather_cmbs = cmbs_check(blind_df2, "weathersit")
    working_cmbs = cmbs_check(blind_df2, "workingday")
else:
    blind_df2["TEMP_BIN"] = pd.qcut(blind_df2.iloc[:, 0], 4, duplicates="drop")
    blind_df2["HUM_BIN"] = pd.qcut(blind_df2.iloc[:, 1], 4, duplicates="drop")
    temp_cmbs = cmbs_check(blind_df2, "TEMP_BIN")
    hum_cmbs = cmbs_check(blind_df2, "HUM_BIN")

st.markdown("### ✅ CMBS Subgroup Results")
if dataset_choice in ["Bike Dataset - Day", "Bike Dataset - Hour"]:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Season CMBS**")
        st.dataframe(season_cmbs, use_container_width=True)
    with c2:
        st.markdown("**Weather CMBS**")
        st.dataframe(weather_cmbs, use_container_width=True)
    with c3:
        st.markdown("**Working Day CMBS**")
        st.dataframe(working_cmbs, use_container_width=True)
else:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Temperature Bin CMBS**")
        st.dataframe(temp_cmbs, use_container_width=True)
    with c2:
        st.markdown("**Humidity Bin CMBS**")
        st.dataframe(hum_cmbs, use_container_width=True)


##############################################
# NEW CMBS VISUAL — INSTANCE-LEVEL RISK MAP (KEY VISUAL)
##############################################
st.markdown(
    """
    <h4 style='text-align:center; color:#0b2e73;'>
        🧠 CMBS Risk Map (Instance-Level): Agreement vs Error Overlap
    </h4>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="font-size:13px; padding:10px;
                background:#f7f9fc;
                border-left:5px solid #6f42c1;
                border-radius:8px;">
    <b>How to read this:</b><br>
    X-axis = <u>Model agreement</u> (lower std across predictions = higher agreement).<br>
    Y-axis = <u>Error overlap</u> (how many models are in the high-error zone for that instance).<br>
    <b>Top-left zone</b> (high agreement + high error overlap) is the strongest CMBS signal.
    </div>
    """,
    unsafe_allow_html=True
)

# Per-instance agreement (std across predictions); smaller = more agreement
pred_matrix = pred_df.values
agreement_std = pred_df.std(axis=1).values  # lower is higher agreement

# Per-instance errors per model
err_matrix = np.abs(pred_matrix - y_test.values.reshape(-1, 1))
err_cols = ["LR", "Tree", "RF", "NN"]

err_inst = pd.DataFrame(err_matrix, columns=err_cols)

# Define high-error threshold per model (top 15%)
q = 0.85
th = err_inst.quantile(q)

high_err_flags = (err_inst > th).astype(int)
error_overlap = high_err_flags.sum(axis=1).values  # 0..4

# Risk classification (simple)
# - CMBS Candidate: strong agreement (low std) AND error_overlap >= 3
agree_cut = np.quantile(agreement_std, 0.25)  # lowest 25% std = high agreement
cmbs_candidate = (agreement_std <= agree_cut) & (error_overlap >= 3)

# Plot: x = agreement_std, y = overlap
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.scatter(agreement_std, error_overlap, alpha=0.45, s=18, label="All instances")
ax.scatter(agreement_std[cmbs_candidate], error_overlap[cmbs_candidate],
           s=60, marker="X", label="CMBS Candidates")

# Reference lines
ax.axvline(agree_cut, linestyle="--", linewidth=1)
ax.axhline(3, linestyle="--", linewidth=1)

ax.set_title("CMBS Risk Map: Agreement (Std) vs Error Overlap", fontsize=11)
ax.set_xlabel("Agreement Proxy: Std Dev Across Model Predictions (Lower = More Agreement)")
ax.set_ylabel("Error Overlap: #Models in High-Error Zone (0–4)")
ax.set_yticks([0, 1, 2, 3, 4])
ax.grid(True, linestyle="--", alpha=0.3)
ax.legend()
st.pyplot(fig)

st.markdown(
    f"""
    ✅ CMBS candidates detected (instance-level): <b>{int(cmbs_candidate.sum())}</b><br>
    • Agreement cutoff (25th percentile std): <b>{round(float(agree_cut), 4)}</b><br>
    • Overlap rule: <b>≥ 3</b> models in high-error zone
    """,
    unsafe_allow_html=True
)

st.caption(
    "This is the most direct visual for your CMBS concept: models agree (low std) but still fail together (high overlap)."
)


##############################################
# OPTIONAL CMBS VISUAL — SUBGROUP HEATMAP (RMSE BY MODEL)
##############################################
st.markdown(
    """
    <h4 style='text-align:center; color:#0b2e73;'>
        🔥 CMBS Subgroup Heatmap (RMSE by Model)
    </h4>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="font-size:13px; padding:10px;
                background:#fff7f0;
                border-left:5px solid #ff9f40;
                border-radius:8px;">
    Heatmap shows where multiple models simultaneously have high subgroup RMSE.
    Darker cells across many columns = stronger CMBS evidence.
    </div>
    """,
    unsafe_allow_html=True
)

def subgroup_rmse_table(df_in, group_col, preds_cols=("lr", "tree", "rf", "nn"), min_n=8):
    rows = []
    for g, sub in df_in.groupby(group_col):
        if sub.shape[0] < min_n:
            continue
        row = {group_col: g, "n": int(sub.shape[0])}
        for p in preds_cols:
            row[p] = float(np.sqrt(mean_squared_error(sub["actual"], sub[p])))
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # order by RF rmse desc
    out = out.sort_values("rf", ascending=False)
    return out

if dataset_choice in ["Bike Dataset - Day", "Bike Dataset - Hour"]:
    group_col_for_heat = st.selectbox("Choose grouping for heatmap", ["season", "weathersit", "workingday"])
else:
    group_col_for_heat = st.selectbox("Choose grouping for heatmap", ["TEMP_BIN", "HUM_BIN"])

rmse_sub = subgroup_rmse_table(blind_df2, group_col_for_heat)

if rmse_sub.empty:
    st.warning("Not enough subgroup data to build heatmap (try a different grouping).")
else:
    heat = rmse_sub.set_index(group_col_for_heat)[["lr", "tree", "rf", "nn"]]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    sns.heatmap(heat, annot=True, fmt=".2f", linewidths=0.3, cmap="YlOrRd", ax=ax)
    ax.set_title(f"Subgroup RMSE Heatmap by Model — {group_col_for_heat}", fontsize=11)
    ax.set_xlabel("Model")
    ax.set_ylabel("Subgroup")
    st.pyplot(fig)

    st.dataframe(rmse_sub, use_container_width=True)


##############################################
# OVERLAPPING FAILURE REGIONS ACROSS MODELS (YOUR FIG 4.5)
##############################################
st.markdown(
    """
    <h4 style='text-align:center; color:#0b2e73;'>
        Illustration of Overlapping Failure Regions Across Models
    </h4>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="font-size:13px; padding:10px;
                background:#fff;
                border-left:5px solid #6f42c1;
                border-radius:8px;">
    Highlights instances where all models fall into their high-error zone together.
    </div>
    """,
    unsafe_allow_html=True
)

# reuse err_inst + high_err_flags
collective_failure = (high_err_flags.sum(axis=1) == 4)
n_overlap = int(collective_failure.sum())

st.markdown(
    f"""
    ✅ Overlapping failure points detected: <b>{n_overlap}</b> instances<br>
    These represent the strongest collective blind spot candidates.
    """,
    unsafe_allow_html=True
)

fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(range(len(y_test)), y_test.values, alpha=0.35, s=16, label="All test instances")
ax.scatter(np.where(collective_failure)[0], y_test.values[collective_failure],
           marker="X", s=80, label="All-model high-error overlap")

ax.set_title("Instances Where All Models Fail Together", fontsize=11)
ax.set_xlabel("Test Instance Index")
ax.set_ylabel("Actual Target Value")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.3)
st.pyplot(fig)

st.caption(
    "Overlapping failure regions: highlighted instances indicate structurally difficult zones where all model families exhibit elevated error."
)


##############################################
# CMBS FRAMEWORK OVERVIEW (FIG 6.1)
##############################################
st.markdown(
    """
    <h4 style='text-align:center; color:#0b2e73;'>
        CMBS Framework Overview
    </h4>
    """,
    unsafe_allow_html=True
)

fig, ax = plt.subplots(figsize=(8, 5))
ax.axis("off")

ax.text(0.5, 0.92, "Collective Model Blind Spot (CMBS) Framework",
        ha="center", fontsize=14, fontweight="bold")

mnames = ["Linear Model", "Tree Model", "Ensemble Model", "Neural Network"]
x_pos = [0.15, 0.38, 0.62, 0.85]

for i, m in enumerate(mnames):
    ax.text(x_pos[i], 0.78, m, ha="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightgray"))

ax.text(0.5, 0.62, "Test Data Subgroup", ha="center",
        bbox=dict(boxstyle="round,pad=0.4", fc="#d6eaff"))

for x in x_pos:
    ax.annotate("", xy=(x, 0.67), xytext=(x, 0.75),
                arrowprops=dict(arrowstyle="->", lw=1.5))

ax.text(0.2, 0.42, "Predictions\nRMSE ↑", ha="center",
        bbox=dict(boxstyle="round", fc="#ffe6e6"))
ax.text(0.5, 0.42, "Predictions\nRMSE ↑", ha="center",
        bbox=dict(boxstyle="round", fc="#ffe6e6"))
ax.text(0.8, 0.42, "Predictions\nRMSE ↑", ha="center",
        bbox=dict(boxstyle="round", fc="#ffe6e6"))

ax.text(0.5, 0.25, "Compare Subgroup RMSE Across Models", ha="center",
        bbox=dict(boxstyle="round,pad=0.4", fc="#fff2cc"))

ax.annotate("", xy=(0.5, 0.32), xytext=(0.5, 0.38),
            arrowprops=dict(arrowstyle="->", lw=2))

ax.text(0.5, 0.1, "⚠ Identify Collective Blind Spot", ha="center",
        fontsize=12, bbox=dict(boxstyle="round,pad=0.4", fc="#ffcccc"))

st.pyplot(fig)

st.caption("CMBS overview: compare multiple models across subgroups to detect shared failure regions.")


##############################################
# CONCEPTUAL COLLECTIVE BLIND SPOT (FIG 6.2)
##############################################
st.markdown(
    """
    <h4 style='text-align:center; color:#0b2e73;'>
        Conceptual Collective Blind Spot
    </h4>
    """,
    unsafe_allow_html=True
)

fig, ax = plt.subplots(figsize=(6, 6))
ax.axis("off")

safe = plt.Circle((0.5, 0.5), 0.42, color="#b6f2b6", ec="black", lw=1.5)
agree = plt.Circle((0.5, 0.5), 0.28, color="#ffe39f", ec="black", lw=1.5)
blind = plt.Circle((0.5, 0.5), 0.14, color="#ff7f7f", ec="black", lw=1.5)

ax.add_patch(safe)
ax.add_patch(agree)
ax.add_patch(blind)

ax.text(0.5, 0.75, "Safe Zone\nAccurate Predictions",
        ha="center", fontsize=11, fontweight="bold")

ax.text(0.5, 0.58, "Agreement Region\nLow Error Across Models",
        ha="center", fontsize=10)

ax.text(0.5, 0.48, "Blind Spot Zone\nHigh Error for ALL Models",
        ha="center", fontsize=10, color="white", fontweight="bold")

ax.annotate("Model A", xy=(0.55, 0.52), xytext=(0.85, 0.60),
            arrowprops=dict(arrowstyle="->", lw=2))
ax.annotate("Model B", xy=(0.55, 0.50), xytext=(0.85, 0.50),
            arrowprops=dict(arrowstyle="->", lw=2))
ax.annotate("Model C", xy=(0.55, 0.48), xytext=(0.85, 0.40),
            arrowprops=dict(arrowstyle="->", lw=2))

ax.text(0.5, 0.36, "⚠", ha="center", fontsize=22)

st.pyplot(fig)

st.caption(
    "Conceptual CMBS: models may agree, yet a central region can exist where all fail together due to shared limitations."
)


##############################################
# FOOTER
##############################################
st.markdown("""
<div style="
    background:linear-gradient(135deg, #e8f0ff, #ffffff);
    padding:12px 18px;
    border-radius:10px;
    border:1px solid #d6e1ff;
    text-align:center;
    font-size:15px;
    color:#0b2e73;
    font-weight:700;">
✨ Analysis Completed Successfully — Results Ready!
</div>
""", unsafe_allow_html=True)
