# 📉 Bonus-Zertifikate Analyse & KI-Prognose Toolkit

Ein interaktives Toolkit zur Bewertung von Bonus-Zertifikaten mit integrierter KI zur Vorhersage von Barriereverletzungen. Entwickelt mit Python & Streamlit.

---

## 🚀 Features

- **📊 Bonus-Zertifikat Simulation**
  - Berechnung des Barwerts bei Fälligkeit
  - Berücksichtigung von Zinssatz und Emittentenrisiko
  - Visualisierung der Auszahlungskurve

- **⚙️ Parameter-Eingabe**
  - Kurs (S₀), Bonus-Level, Barriere
  - Laufzeit, Zinssatz, Risikoabschlag

- **🧠 KI-Modul: Barriereverletzungsprognose**
  - Geometrische Brownsche Bewegung zur Pfadsimulation
  - Klassifikation mit Random Forest
  - Modellgüteanzeige (Testgenauigkeit)
  - Wahrscheinlichkeitsprognose für Barriereverletzung

---
## 🌍 Live Demo

👉 [Click here to open the app](https://zertifikate-toolkit-o68vkuemgqrcbquqq86fve.streamlit.app/)

---
## 🗂️ Projektstruktur
```bash

projekt2-zertifikate-toolkit/
├── app.py # Haupt-App (Streamlit)
├── products/
│ └── bonus_certificate.py # (Optional) Modul zur Produktspezifikation
├── utils/
│ └── plotting.py # (Optional) Hilfsfunktionen für Visualisierung
├── data/
│ └── beispiel_preise.csv # Beispielhafte Kursdaten
├── requirements.txt # Abhängigkeiten


