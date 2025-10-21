# AI-Dashboard

# How to run
Open your terminal, locate to the folder you downloaded.

## 0) (Optional) Create & activate a venv
First run 
```python -m venv .venv```\
Then run\
Windows: ```.venv\Scripts\activate```\
macOS/Linux: ```source .venv/bin/activate```

## 1) Install deps
```pip install -r requirements.txt```

## 2) Set your API key
(replace the key code to your actual openai api key
macOS/Linux:
```export OPENAI_API_KEY="sk-..."```\
Windows PowerShell:
```$env:OPENAI_API_KEY="sk-..."```

## 3) Run in terminal
(replace 'data/optima_water_3in.csv' to the actual path of your csv file.)\
(replace '~/SBI/Optima_dashboard' to the folder where you want to store the output images)

```python test.py \--csv data/optima_water_3in.csv \--place "Scottsdale, AZ, US"  \--outdir ~/SBI/Optima_dashboard \--predict```


