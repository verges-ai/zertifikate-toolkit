import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# 📌 Titel
st.title("📉 Bonus-Zertifikat Analyse & Simulation")

# --- Sidebar: Parameter ---
st.sidebar.header("📊 Parameter")

S0 = st.sidebar.number_input("Aktueller Kurs (S₀)", min_value=10.0, value=100.0)
bonus_level = st.sidebar.number_input("Bonus-Level (€)", min_value=10.0, value=120.0)
barriere = st.sidebar.number_input("Barriere (€)", min_value=10.0, value=80.0)
laufzeit = st.sidebar.slider("Laufzeit (Jahre)", 0.1, 5.0, 1.0, 0.1)

st.sidebar.markdown("---")
r = st.sidebar.slider("📉 Risikofreier Zinssatz (r)", 0.0, 0.05, 0.01, 0.001)
abschlag = st.sidebar.slider("🏦 Emittentenrisiko (%)", 0.0, 5.0, 1.0, 0.1) / 100

# --- Auszahlung berechnen ---
kurse = np.linspace(S0 * 0.5, S0 * 1.5, 200)
payoffs = []

for ST in kurse:
    if ST >= barriere:
        payoff = max(ST, bonus_level)
    else:
        payoff = ST
    payoffs.append(payoff)

payoffs_discounted = [p * np.exp(-r * laufzeit) for p in payoffs]
payoffs_risk_adjusted = [p * (1 - abschlag) for p in payoffs_discounted]

# --- Auszahlung visualisieren ---
st.header("📈 Auszahlung bei Fälligkeit")

fig, ax = plt.subplots()
ax.plot(kurse, payoffs_risk_adjusted, label="Zertifikat (diskontiert & Risiko)", color="blue", linewidth=2)
ax.plot(kurse, kurse, label="Aktie", linestyle="--", color="gray")
ax.axvline(barriere, color="red", linestyle=":", label="Barriere")
ax.axhline(bonus_level, color="green", linestyle=":", label="Bonus-Level")
ax.set_xlabel("Kurs bei Fälligkeit (ST)")
ax.set_ylabel("Barwert der Auszahlung (€)")
ax.set_title("Barwert-Auszahlung Bonus-Zertifikat")
ax.legend()
ax.grid(True)
st.pyplot(fig)

st.markdown("### ℹ️ Hinweise")
st.markdown("""
- **Barwert** = heutiger Wert der Auszahlung unter Annahme eines Zinssatzes `r`
- **Emittentenrisiko**: Reduziert Auszahlung proportional zur Wahrscheinlichkeit, dass der Emittent zahlungsunfähig wird
- Die Auszahlung gleicht **einer Long-Aktienposition + Put-Option mit Barriere** (strukturierte Komponente)
""")

# --- KI-Modul: Barriere-Verletzungs-Klassifikation ---

st.header("🧠 KI-Modul: Barriere-Verletzung vorhersagen")

def simulate_price_paths(S0, mu=0.05, sigma=0.2, T=1, steps=252, n_paths=500):
    dt = T / steps
    paths = np.zeros((n_paths, steps + 1))
    paths[:, 0] = S0
    for t in range(1, steps + 1):
        z = np.random.normal(size=n_paths)
        paths[:, t] = paths[:, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    return paths

def extract_features_from_paths(paths, barrier):
    min_prices = paths.min(axis=1)
    final_prices = paths[:, -1]
    breach = (min_prices < barrier).astype(int)
    features = np.column_stack([min_prices, final_prices])
    return features, breach

if 'model' not in st.session_state:
    st.session_state['model'] = None

if st.button("Trainiere und teste Modell"):
    paths = simulate_price_paths(S0, T=laufzeit)
    X, y = extract_features_from_paths(paths, barriere)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)

    st.write(f"Modell-Genauigkeit (Testset): {score:.2%}")

    # Modell speichern
    st.session_state['model'] = clf

    # Beispielvorhersage
    current_features = np.array([[S0 * 0.9, S0]])
    prediction = clf.predict(current_features)[0]
    proba = clf.predict_proba(current_features)[0, 1]

    st.write(f"Vorhersage, ob Barriere verletzt wird: {'Ja' if prediction == 1 else 'Nein'}")
    st.write(f"Wahrscheinlichkeit der Barriereverletzung: {proba:.2%}")

# Upload für eigene Kursdaten
uploaded_file = st.file_uploader("📂 Lade eigene Kursdaten (CSV mit Spalte 'Close') hoch")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Geladene Daten:", df.head())

    if 'Close' not in df.columns:
        st.error("Die CSV-Datei muss eine Spalte namens 'Close' enthalten.")
    else:
        prices = df['Close'].values
        features = np.array([[prices.min(), prices[-1]]])

        if st.session_state['model'] is None:
            st.warning("Bitte trainiere zuerst ein Modell über den Button 'Trainiere und teste Modell'.")
        else:
            model = st.session_state['model']
            pred = model.predict(features)[0]
            proba = model.predict_proba(features)[0, 1]

            st.write(f"Barriere-Verletzung vorhergesagt? {'Ja' if pred == 1 else 'Nein'}")
            st.write(f"Wahrscheinlichkeit: {proba:.2%}")

# --- Autor ---
st.caption("🧠 Projekt 2 von 3 • erstellt von Stephane Atontsa • powered by Streamlit")
