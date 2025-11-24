# Keramik
Repo zum Lösen der Keramik-Challenge.

## Setup
1. Im Terminal navigiere auf die Root-Ebene zu '\Keramik'
2. Erstelle eine neue Python-Umgebung
    ```
    python -m venv .venv
    ```
3. Aktiviere die virtuelle Umgebung im Terminal
    ```
    .venv\Scripts\activate
    ```
4. Installiere poetry als dependency manager
    ```
    pip install poetry
    ```
5. Poetry liest die 'pyproject.toml' und installiert alles:
    ```
    poetry install
    ```
6. Neue Dependencies hinzufügen:
Der Befehl added neue Dependencies zur "pyproject.toml" und installiert diese
    ```
    poetry add <dependency-name>
    poetry install
    ```
Dadurch aktualisiert Poetry automatisch die pyproject.toml und poetry.lock und wir alle benutzen die gleichen Dependencies.
7. Um python files laufen zulassen, musst du einen python kernel adden. 
Folgenden Interpreter auswählen (VS Code: Ctrl+Shift+P -> Select Interpreter):
 - Windows: .venv\Scripts\python.exe
 - macOS: .venv/bin/python

## Data Setup

Aus Datenschutzgründen ist der Datensatz nicht direkt in diesem Repo enthalten. Folge den Anweisungen in [data/README.md](data/README.md), um die benötigten Daten korrekt und lokal einzurichten.