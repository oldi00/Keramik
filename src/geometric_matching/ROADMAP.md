# Geometric Matching: Roadmap

Brainstorming-Sammlung für die nächsten Schritte: Potenzielle Verbesserungen und Tasks, sortiert nach Themenbereichen.

## Allgemeines
- Synonym/Konkordanz Liste für gleiche Typologien mit unterschiedlichen Namen

## Laufzeit
- CPU-Parallelisierung (aktuell implementiert)
- Nutzung der GPU
- Vektorisierung des Codes (Nutzung von numpy, Vermeidung von Loops)
- Idee: Die Anzahl der möglichen Typologien reduzieren durch einen Filter
    - Vielleicht class prediction benutzen als typology filter um Laufzeit zu verbessern

## Algorithmus
- Einschränkungen für Rotation, Skalierung und Transformation verfeinern
    - Rotation vermutlich noch mehr einschränken (Anmerkung von Frederic)
    - Skalierung kritisch überdenken! Weil die constraints hängen von Resolution ab
- y-Penalty implementieren (Bestrafung basierend auf Höhe, da wir nur Gefäßlippen haben)
- Handzeichnungen des anderen Teams testen (irrelevante Linien sind dort entfernt)

## Preprocessing (Scherbe)
- Cropping verbessern (aktuell sind Werte hardcoded)
- Ignorieren der Bruchstelle // Unteren Teil der Scherbe nicht nutzen

## Preprocessing (Typology)
- Bilder mit besserer Auflösung speichern damit Feinheiten der Linie besser erkannt werden (Code von Fadi)

## User Interface
- Streamlit App erstellen
- Config Datei 

## Ergebnisse & Plots
- Confusion Matrix

## Stuff To Check Out
- ORB Algorithmus