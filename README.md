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

1. Navigiere ins Projektverzeichnis: Öffne dein Terminal und wechsle auf die Root-Ebene `\Keramik`
2. Erstelle eine virtuelle Umgebung:
    ```bash
    python -m venv .venv
    ```
3. Aktiviere die virtuelle Umgebung:
    ```bash
    Windows: .venv\Scripts\activate
    macOS/Linux: source .venv/bin/activate
    ```
4. Installiere Poetry (falls noch nicht vorhanden):
    ```bash
    pip install poetry
    ```
5. Installiere die Abhängigkeiten: Poetry liest die pyproject.toml und installiert alles automatisch:
    ```bash
    poetry install
    ```
6. Interpreter in VS Code auswählen: Damit du die Python-Dateien ausführen kannst, wähle den richtigen Interpreter (VS Code: Ctrl+Shift+P -> Python: Select Interpreter):
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

Aus Datenschutzgründen ist der Datensatz nicht in diesem Repo enthalten. Das gesamte `data/`-Verzeichnis ist in der `.gitignore` so konfiguriert, dass absolut nichts davon von Git erfasst wird. Alle Daten bleiben rein lokal auf deinem Rechner. (Hinweis: Die Typology befindet sich mittlerweile in der Cloud und wird ebenfalls nicht mehr über Git verwaltet).

Folge dieser Anleitung, um die Daten lokal einzurichten:

1. Download: Lade den Datensatz von Moodle herunter.
2. Ordner erstellen: Stelle sicher, dass der Ordner `raw/` innerhalb des `data/` Verzeichnisses existiert.
3. Entpacken: Entpacke bzw. kopiere alle Dateien aus dem Download direkt nach `data/raw/`.
4. Aufräumen & Strukturieren:
    - Erstelle den Unterordner `data/raw/png/`.
    - Verschiebe alle Bilddateien, die nun lose in `data/raw/` liegen, in diesen neuen `png/` Ordner.

Damit die Skripte einwandfrei funktionieren, muss deine lokale Datenstruktur danach exakt so aussehen:

```text
data/
└── raw/
    ├── png/                <-- (Alle Bilder)
    └── ...                 <-- (Weitere Dateien aus dem Zip-Ordner)
```