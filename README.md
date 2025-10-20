# AI-Dashboard

# How to run
## 0) (Optional) Create & activate a venv
```python -m venv .venv```\
Windows: ```.venv\Scripts\activate```\
macOS/Linux: ```source .venv/bin/activate```

## 1) Install deps
```pip install -r requirements.txt```

## 2) Set your API key
macOS/Linux:
```export OPENAI_API_KEY="sk-..."```\
Windows PowerShell:
```$env:OPENAI_API_KEY="sk-..."```

## 3) Run
```python utility_viz.py --csv ./data/optima_water_3in.csv --place "Scottsdale, AZ, US" --predict```

