# Keramik Challenge
Dieses Repository enthält den Code und die Pipeline zur Lösung der Keramik-Challenge. Es umfasst die komplette Struktur – von der Aufbereitung der rohen Bilddaten über das Setup der Pipeline bis hin zur Evaluierung und Visualisierung der Ergebnisse.

## 📂 Projektstruktur
Hier ist eine Übersicht über die wichtigsten Ordner und Dateien in diesem Repository:

```text
KERAMIK/
├── .venv/                  # Virtuelle Python-Umgebung (lokal)
├── archive/                
├── data/                   # Datenverzeichnis (wird von Git ignoriert)
│   ├── cache/
│   ├── processed/
│   ├── raw/                # Hierhin kommen die rohen Moodle-Daten
│   └── results/
├── sandbox/                # Skripte zum Testen (wird von Git ignoriert)
├── src/                    # Hauptquellcode des Projekts
├── .gitignore
├── config.yaml             # Lokale Konfiguration (muss manuell erstellt werden)
├── config_sample.yaml      # Vorlage für die Konfiguration
├── poetry.lock             
├── pyproject.toml          
└── README.md               
```

## 🛠️ Setup & Installation
Damit das Projekt bei dir lokal läuft, folge diesen Schritten:

1. Klone das Repository:
```Bash
git clone https://github.com/oldi00/Keramik.git
```
2. Navigiere ins Projektverzeichnis: Öffne dein Terminal und wechsle auf die Root-Ebene `\Keramik`
3. Erstelle eine virtuelle Umgebung:
    ```bash
    python -m venv .venv
    ```
4. Aktiviere die virtuelle Umgebung:
    ```bash
    Windows: .venv\Scripts\activate
    macOS/Linux: source .venv/bin/activate
    ```
5. Installiere Poetry (falls noch nicht vorhanden):
    ```bash
    pip install poetry
    ```
6. Installiere die Abhängigkeiten: Poetry liest die pyproject.toml und installiert alles automatisch:
    ```bash
    poetry install
    ```
7. Interpreter in VS Code auswählen: Damit du die Python-Dateien ausführen kannst, wähle den richtigen Interpreter (VS Code: Ctrl+Shift+P -> Python: Select Interpreter):
    - Windows: `.venv\Scripts\python.exe`
    - macOS/Linux: `.venv/bin/python`

### Dependencies verwalten
Um neue Packages hinzuzufügen, nutze folgenden Befehl (dies aktualisiert die pyproject.toml und poetry.lock automatisch für alle im Team):

```bash
poetry add <dependency-name>
poetry install
```

### Konfiguration einrichten
Damit das Projekt deine lokalen Pfade kennt:

1. Dupliziere die Datei `config_sample.yaml`.
2. Benenne die Kopie in `config.yaml` um.
3. Öffne die `config.yaml` und passe die Pfade und Variablen an dein lokales System an.

## 📦 Daten-Setup
**WICHTIG: KEINE DATEN IN DAS REPOSITORY ÜBERTRAGEN.**

Aus Datenschutzgründen ist der Datensatz nicht in diesem Repo enthalten. Das gesamte `data/`-Verzeichnis ist in der `.gitignore` so konfiguriert, dass absolut nichts davon von Git erfasst wird. Alle Daten bleiben rein lokal auf deinem Rechner. (Hinweis: Auch die Typology-Daten befinden sich in der Cloud und werden nicht über Git verwaltet).

Folge dieser Anleitung, um die Daten lokal einzurichten:

**Teil 1: Moodle-Daten (Scherben)**

1. Download: Lade den Datensatz `Römische Keramik - Daten` von Moodle herunter.
2. Ordner erstellen: Erstelle den Ordner `data/` im Hauptverzeichnis und den Unterordner `data/raw/`.
3. Entpacken: Entpacke bzw. kopiere alle Dateien aus dem Download-Ordner `Datachallenge_Roemische-Keramik` direkt nach `data/raw/`.
4. Aufräumen & Strukturieren:
    - Erstelle den Unterordner `data/raw/png/`.
    - Verschiebe alle Bilddateien, die nun lose in `data/raw/` liegen, in diesen neuen `png/` Ordner.

**Teil 2: Typology-Daten (Hessenbox)**

5. Download: Lade die Typology-Datenbank über den Link zur Hessenbox herunter.
6. Einordnen: Entpacke die Datei und verschiebe den extrahierten Ordner `typology/` (mit seinen Unterordnern) direkt in `data/raw/`.

Damit die Skripte einwandfrei funktionieren, muss deine lokale Datenstruktur danach exakt so aussehen:

```Plaintext
data/
└── raw/
    ├── png/                    
    ├── typology/               <-- (Die Daten aus der Hessenbox)
    │   ├── handzeichnungen/
    │   ├── auto_extrahiert/
    │   └── ...
    └── ...                     <-- (Weitere Dateien aus Moodle)
```

## ⚙️ Preprocessing

Bevor die Anwendung genutzt werden kann, müssen die Rohdaten verarbeitet und die Typology vorbereitet werden. Dieser Schritt extrahiert relevante Merkmale aus den Bildern und strukturiert die Daten für die Pipeline.

Führe dazu das Preprocessing-Skript im Terminal aus:
```Bash
python src/preprocess.py
```

## 🚀 App starten
Sobald das Setup abgeschlossen und die Daten eingerichtet sind, kannst du die Anwendung starten. Stelle sicher, dass deine virtuelle Umgebung aktiviert ist, und führe folgenden Befehl aus:
```Bash
streamlit run src/app.py
```

## 🌟 Danksagung & Credits
Ein besonderes Dankeschön geht an eine andere Projektgruppe, die uns den Ordner `handzeichnungen` (zu finden in der Hessenbox) zur Verfügung gestellt hat. Die Nutzung dieser Daten hat maßgeblich dazu beigetragen, die Performance und Genauigkeit unseres Algorithmus spürbar zu verbessern! :)

## 👥 Autoren & Contributions
Dieses Projekt wurde im Rahmen der Challenge von folgendem Team bearbeitet:
- Miles Lenz (s5368500@stud.uni-frankfurt.de): Fokus auf allgemeine Code-Infrastruktur, die Preprocessing-Pipeline, Implementierung des RANSAC-Algorithmus sowie die finale Überarbeitung und das Refactoring der Streamlit-App.
- Markus Oldenburger (s4742346@stud.uni-frankfurt.de): Vollständiger Fokus und Implementierung des ICP-Algorithmus, initiales Design der Streamlit-App sowie Forschung und Prototyping eines Siamesischen Netzwerks (Ansatz siehe `archive/`).
- Fadi Mekdad: Automatisierte Extraktion von Typologien aus der Literatur (Code aktuell extern verwaltet und nicht in diesem Repository abgebildet).

Bei Fragen zum Projekt, zum Code oder zur Installation erreicht ihr uns gerne per E-Mail.

## 📄 Lizenz
Dieses Projekt steht unter der MIT License.
