# Daten-Verzeichnis

**⚠️ WICHTIG: KEINE DATEN IN DAS REPOSITORY ÜBERTRAGEN.**

Dieses Verzeichnis ist so konfiguriert, dass private Daten (alles außer dieser README und `data/typology/`) von Git ignoriert werden. Sie bleiben rein lokal auf deinem Rechner.

## Anleitung zur Einrichtung

1.  **Download:** Lade den Datensatz von Moodle herunter.
2.  **Ordner erstellen:** Erstelle einen Ordner `raw/` innerhalb dieses `data/` Verzeichnisses.
3.  **Entpacken:** Entpacke/Kopiere alle Dateien aus dem Download direkt in `data/raw/`.
4.  **Aufräumen:**
    * Erstelle einen Unterordner: `data/raw/png/`
    * Verschiebe alle Bilddateien, die nun lose in `data/raw/` liegen, in diesen neuen `png/` Ordner.

## Erwartete Ordnerstruktur
Damit die Skripte funktionieren, muss es bei dir lokal so aussehen:
```
data/
├── README.md
├── ...
└── raw/
    ├── png/
    └── ...            <-- (Weitere Dateien aus dem Zip-Ordner)
```
