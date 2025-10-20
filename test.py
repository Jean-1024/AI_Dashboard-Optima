#!/usr/bin/env python3
"""
Utility Bill Visualizer (VS Code version)
- Creates/updates an OpenAI Assistant with a local function tool for Meteostat temps
- Uploads your CSV, asks for a dual-axis usage/temperature chart
- (Optionally) asks for a SARIMA prediction on the same chart
- Saves any returned PNGs to ./outputs

Usage:
  python utility_viz.py --csv ./data/optima_water_3in.csv --place "Scottsdale, AZ, US" --predict

Env:
  OPENAI_API_KEY=sk-...
"""

from __future__ import annotations
import os, sys, time, json, subprocess, argparse
from datetime import datetime
from typing import List, Dict
from pathlib import Path

from openai import OpenAI

# -----------------------------
# Config / CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Utility Bill Visualization with OpenAI Assistants API")
    p.add_argument("--csv", required=True, help="Path to the CSV file")
    p.add_argument("--place", default=os.getenv("PLACE", "Scottsdale, AZ, US"),
                   help="City, State, Country for temperature lookup (default from $PLACE or Scottsdale)")
    p.add_argument("--assistant-name", default="Utility Bill Analyst",
                   help="Assistant name to create/use (default: Utility Bill Analyst)")
    p.add_argument("--model", default="gpt-4o", help="Model ID (default: gpt-4o)")
    p.add_argument("--no-predict", dest="predict", action="store_false",
                   help="Skip the second run that adds SARIMA prediction")
    p.add_argument("--predict", dest="predict", action="store_true",
                   help="Include the SARIMA prediction run (default)")
    p.set_defaults(predict=True)
    return p.parse_args()

# -----------------------------
# Utilities
# -----------------------------
def ensure_packages(pkgs: List[str]):
    for p in pkgs:
        try:
            __import__(p if p != "meteostat" else "meteostat")
        except ImportError:
            print(f"[setup] Installing {p} ...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", p])

def mkdir_p(path: Path):
    path.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Local function tool (runs here)
# -----------------------------
def get_meteostat_temperatures_for_months(place: str, months: List[str], units: str = "F") -> Dict:
    """
    Given a place string and a list of months like ["2023-08", "2023-09", ...],
    return monthly average temperatures for exactly those months.
    """
    ensure_packages(["meteostat", "geopy", "pandas", "python-dateutil"])

    from geopy.geocoders import Nominatim
    from meteostat import Monthly, Point
    import pandas as pd
    from dateutil.relativedelta import relativedelta

    # Parse & normalize month list
    months_dt = [datetime.strptime(m, "%Y-%m") for m in months]
    start = min(months_dt).replace(day=1)
    end = max(months_dt).replace(day=1)

    # Geocode the place → lat/lon
    geolocator = Nominatim(user_agent="utility-bill-analyst")
    loc = geolocator.geocode(place)
    if not loc:
        raise ValueError(f"Could not geocode place: {place}")
    lat, lon = loc.latitude, loc.longitude

    # Fetch monthly climate
    data = Monthly(Point(lat, lon), start, end)
    df = data.fetch()  # index is datetime (month)
    if df is None or df.empty:
        raise ValueError(f"No Meteostat data for {place} between {start:%Y-%m} and {end:%Y-%m}")

    # Choose temperature column
    if "tavg" in df.columns and not df["tavg"].isna().all():
        temps_c = df["tavg"]
    elif "tmin" in df.columns and "tmax" in df.columns:
        temps_c = (df["tmin"] + df["tmax"]) / 2.0
    else:
        raise ValueError("Meteostat data lacks tavg and (tmin,tmax) for selected period")

    # Convert
    if units.upper() == "F":
        temps = temps_c * 9.0 / 5.0 + 32.0
        unit_label = "F"
    else:
        temps = temps_c
        unit_label = "C"

    # Build exact-month mapping
    temps = temps.dropna()
    out_map = {
        dt.strftime("%Y-%m"): round(float(val), 1)
        for dt, val in temps.items()
        if dt.strftime("%Y-%m") in months
    }

    series = [{"month": m, "avg_temp": out_map.get(m, None)} for m in months]

    return {"place": place, "units": unit_label, "series": series, "lat": lat, "lon": lon}

def function_tool_spec():
    return {
        "type": "function",
        "function": {
            "name": "get_meteostat_temperatures_for_months",
            "description": (
                "Fetch monthly average air temperature for a given place and a list "
                "of months. Returns a JSON object mapping months to average temperature. "
                "Preferred units are Fahrenheit ('F') unless otherwise specified."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "place": {
                        "type": "string",
                        "description": "Location like 'City, State, Country' (e.g., 'Scottsdale, AZ, US')."
                    },
                    "months": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "pattern": "^\\d{4}-\\d{2}$"
                        },
                        "description": "List of months (YYYY-MM) to fetch."
                    },
                    "units": {"type": "string", "enum": ["F", "C"], "default": "F"}
                },
                "required": ["place", "months"]
            }
        }
    }

# -----------------------------
# OpenAI helpers
# -----------------------------
def upload_file(client: OpenAI, csv_path: Path):
    print("[openai] Uploading file ...", csv_path)
    with open(csv_path, "rb") as f:
        file = client.files.create(file=f, purpose="assistants")
    print("[openai] File uploaded:", file.id)
    return file

def get_or_create_assistant(client: OpenAI, *, name: str, model: str, code_file_id: str):
    tools = [
        {"type": "code_interpreter"},
        {"type": "file_search"},
        function_tool_spec()
    ]

    # Try to find assistant by name first
    exists = client.beta.assistants.list().data
    for a in exists:
        if (a.name or "").strip() == name.strip():
            print(f"[openai] Using existing assistant: {a.name} ({a.id})")
            # Make sure tools & file attachment are up to date
            return client.beta.assistants.update(
                assistant_id=a.id,
                tools=tools,
                tool_resources={"code_interpreter": {"file_ids": [code_file_id]}},
                model=model
            )

    # Otherwise create
    print("[openai] Creating a new assistant ...")
    assistant = client.beta.assistants.create(
        name=name,
        instructions=(
            "You are an expert data analyst. Analyze utility bill data from CSV files, answer questions, "
            "and create visualizations. When asked to create a plot, generate a single PNG and return it."
        ),
        tools=tools,
        tool_resources={"code_interpreter": {"file_ids": [code_file_id]}},
        model=model
    )
    print(f"[openai] New assistant created: {assistant.name} ({assistant.id})")
    return assistant

def handle_tool_calls(client: OpenAI, thread_id: str, run):
    """Handle requires_action -> submit_tool_outputs."""
    if run.status != "requires_action":
        return run

    tool_calls = run.required_action.submit_tool_outputs.tool_calls
    outputs = []

    for tc in tool_calls:
        name = tc.function.name
        args = json.loads(tc.function.arguments or "{}")

        if name == "get_meteostat_temperatures_for_months":
            place = args.get("place")
            months = args.get("months", [])
            units = args.get("units", "F")
            try:
                result = get_meteostat_temperatures_for_months(place, months, units)
                outputs.append({"tool_call_id": tc.id, "output": json.dumps(result)})
            except Exception as e:
                outputs.append({"tool_call_id": tc.id, "output": json.dumps({"error": str(e)})})
        else:
            outputs.append({"tool_call_id": tc.id, "output": json.dumps({"error": f"Unknown tool {name}"})})

    run = client.beta.threads.runs.submit_tool_outputs(
        thread_id=thread_id,
        run_id=run.id,
        tool_outputs=outputs
    )
    return run

def poll_run(client: OpenAI, thread_id: str, run):
    terminal = {"completed", "failed", "cancelled", "expired"}
    while run.status not in terminal:
        time.sleep(2)
        if run.status == "requires_action":
            run = handle_tool_calls(client, thread_id, run)
            continue
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
        print(f"[openai] Run status: {run.status}")
    return run

def save_images_from_thread(client: OpenAI, thread_id: str, outdir: Path):
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    saved = 0
    for msg in reversed(messages.data):  # newest -> oldest
        if msg.role != "assistant":
            continue
        for part in msg.content:
            if part.type == "text":
                print("\nAssistant says:\n", part.text.value)
            elif part.type == "image_file":
                image_id = getattr(getattr(part, "image_file", None), "file_id", None)
                if not image_id:
                    print("[warn] image_file part had no file_id; skipping.")
                    continue
                try:
                    file_stream = client.files.content(image_id)
                    bin_data = file_stream.read()
                except Exception as e:
                    print(f"[warn] Failed to fetch file {image_id}: {e}")
                    continue
                out_path = outdir / f"viz_{image_id}.png"
                with open(out_path, "wb") as f:
                    f.write(bin_data)
                print(f"[file] Saved chart to {out_path}")
                saved += 1
    if saved == 0:
        print("[info] No image files found in assistant responses.")

# -----------------------------
# Main flow
# -----------------------------
def main():
    args = parse_args()
    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        print(f"[error] CSV not found: {csv_path}")
        sys.exit(1)

    outdir = Path("./outputs")
    mkdir_p(outdir)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[error] Please set OPENAI_API_KEY in your environment.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # Upload the CSV
    file = upload_file(client, csv_path)

    # Create or update assistant
    assistant = get_or_create_assistant(
        client,
        name=args.assistant_name,
        model=args.model,
        code_file_id=file.id
    )

    # Create a thread
    thread = client.beta.threads.create()

    # First message: usage + temperature (single PNG)
    prompt1 = (
        "From the provided water bill CSV, do the following:\n"
        "1) Parse the month column (YYYY-MM or similar) and water usage; keep the original row order (no reindexing/skim).\n"
        f"2) Call the tool get_meteostat_temperatures_for_months for the SAME month list and place '{args.place}', in Fahrenheit.\n"
        "3) Create a dual-axis plot as ONE final chart:\n"
        "   - X-axis: all months (rotate labels 45° for readability)\n"
        "   - Left Y-axis: water usage (blue line with markers)\n"
        "   - Right Y-axis: average temperature in °F (green dashed line with markers)\n"
        "4) Save and return exactly ONE PNG image of this combined chart.\n"
        "Do everything in Python inside Code Interpreter. Do not print raw DataFrame tables unless they add value."
    )

    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=prompt1
    )

    run1 = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=assistant.id)
    print(f"[openai] Starting run (usage + temperature): {run1.id}")
    run1 = poll_run(client, thread.id, run1)
    print("[openai] Run1 finished:", run1.status)

    if args.predict:
        prompt2 = (
            "Using the same data, add a SARIMA-based prediction for the NEXT month’s water usage onto the SAME chart you created.\n"
            "Show the predicted point clearly (e.g., a distinct marker) and include uncertainty (e.g., confidence interval shading or a boxplot element for that point).\n"
            "Return the UPDATED chart as a single PNG, along with a short explanation of the model and the forecast value with its confidence interval.\n"
            "Return only one PNG image."
        )
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=prompt2
        )
        run2 = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=assistant.id)
        print(f"[openai] Starting run (prediction): {run2.id}")
        run2 = poll_run(client, thread.id, run2)
        print("[openai] Run2 finished:", run2.status)

    # Save any images from the thread
    save_images_from_thread(client, thread.id, outdir)

if __name__ == "__main__":
    main()
