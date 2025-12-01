import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------
# Load Models and Data
# ---------------------------------------------------------
rf = joblib.load("rf_model.pkl")
feature_cols = joblib.load("feature_cols.pkl")
df = pd.read_csv("processed_dataset.csv")

# PCA pipeline for future prediction
pca_model = joblib.load("pca_model.pkl")
pca_scaler = joblib.load("pca_scaler.pkl")
pca_feature_cols = joblib.load("pca_feature_cols.pkl")

app = FastAPI(
    title="Climate Displacement Prediction API",
    description="Predict future climate-driven displacement scenarios.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Future Climate Scenario Prediction
# ---------------------------------------------------------

class FutureInput(BaseModel):
    country: str
    year: int

    # Climate inputs
    tas_mean: float
    tasmin_mean: float
    tasmax_mean: float
    pr_sum: float
    pr_mean: float
    tas_std: float
    pr_std: float
    tas_anomaly: float
    pr_anomaly: float

    # Disaster event counts
    Drought: float = 0
    Earthquake: float = 0
    Erosion: float = 0
    Extreme_Temperature: float = 0
    Flood: float = 0
    Mass_Movement: float = 0
    Mixed_disasters: float = 0
    Sea_level_Rise: float = 0
    Storm: float = 0
    Volcanic_activity: float = 0
    Wave_action: float = 0
    Wildfire: float = 0
    
    Population: float = None


@app.post("/predict_future")
def predict_future(payload: FutureInput):
    """
    Predict future displacement using climate scenario and disaster inputs.
    """
    
    # Get country's historical data
    country_data = df[df["ISO3"] == payload.country].sort_values("Year", ascending=False)
    
    if country_data.empty:
        raise HTTPException(
            status_code=404, 
            detail=f"No historical data found for country: {payload.country}"
        )
    
    # Use median of last 5 years as baseline (robust against outliers)
    recent_years = country_data.head(5)
    baseline_displacement = float(recent_years['Total_Displacement'].median())
    baseline_year_range = f"{int(recent_years['Year'].min())}-{int(recent_years['Year'].max())}"
    
    # Use most recent year for feature extraction
    baseline = country_data.iloc[0]
    baseline_year = int(baseline['Year'])
    
    # PCA transform climate inputs
    raw_climate = [[payload.__dict__[f] for f in pca_feature_cols]]
    scaled = pca_scaler.transform(raw_climate)
    pcs = pca_model.transform(scaled)[0]
    
    # Build feature row
    new_row = {}
    
    # Add climate PCs
    for i in range(len(pcs)):
        new_row[f"Climate_PC{i+1}"] = pcs[i]
    
    # Add raw climate values
    climate_features = ['tas_mean', 'tasmin_mean', 'tasmax_mean', 'pr_sum', 'pr_mean', 
                       'tas_std', 'pr_std', 'tas_anomaly', 'pr_anomaly']
    for f in climate_features:
        new_row[f] = float(payload.__dict__[f])
    
    # Project population to target year
    if payload.Population is not None:
        new_row['Population'] = float(payload.Population)
    else:
        baseline_population = baseline['Population']
        years_diff = payload.year - baseline_year
        new_row['Population'] = baseline_population * ((1.01) ** years_diff)
    
    # Add disaster event counts
    disaster_features = ['Drought', 'Earthquake', 'Erosion', 'Extreme_Temperature', 
                        'Flood', 'Mass_Movement', 'Mixed_disasters', 'Sea_level_Rise', 
                        'Storm', 'Volcanic_activity', 'Wave_action', 'Wildfire']
    
    for f in disaster_features:
        feature_name = f.replace('_', ' ')
        new_row[feature_name] = float(payload.__dict__.get(f, 0))
    
    # Calculate aggregate disaster metrics
    total_events = sum(payload.__dict__.get(f, 0) for f in disaster_features)
    new_row['Num_Disaster_Events'] = float(total_events)
    
    # Estimate disaster displacement using weighted severity
    baseline_events = baseline['Num_Disaster_Events']
    baseline_total_disp = baseline['Total_Disaster_Displacements']
    
    if baseline_events > 0:
        avg_displacement_per_event = baseline_total_disp / baseline_events
    else:
        global_avg = df[df['Num_Disaster_Events'] > 0]
        avg_displacement_per_event = (global_avg['Total_Disaster_Displacements'].mean() / 
                                     global_avg['Num_Disaster_Events'].mean())
    
    # Severity weights by disaster type
    severity_weights = {
        'Flood': 1.2, 'Storm': 1.3, 'Drought': 0.8, 'Earthquake': 1.5,
        'Extreme_Temperature': 0.6, 'Wildfire': 0.7, 'Sea_level_Rise': 1.0,
        'Mass_Movement': 1.1, 'Volcanic_activity': 1.4, 'Erosion': 0.5,
        'Wave_action': 0.9, 'Mixed_disasters': 1.0
    }
    
    weighted_events = 0
    max_single_event = 0
    
    for disaster_type in disaster_features:
        count = payload.__dict__.get(disaster_type, 0)
        weight = severity_weights.get(disaster_type, 1.0)
        weighted_events += count * weight
        
        if count > 0:
            estimated_impact = count * weight * avg_displacement_per_event
            max_single_event = max(max_single_event, estimated_impact / count)
    
    if weighted_events > 0:
        new_row['Total_Disaster_Displacements'] = weighted_events * avg_displacement_per_event
        new_row['Max_Event_Displacement'] = max_single_event
    else:
        new_row['Total_Disaster_Displacements'] = 0
        new_row['Max_Event_Displacement'] = 0
    
    # Fill missing features
    for f in feature_cols:
        if f not in new_row:
            new_row[f] = 0
    
    # Predict
    X = pd.DataFrame([new_row])[feature_cols]
    pred_log = rf.predict(X)[0]
    pred = float(np.expm1(pred_log))
    
    # Calculate context
    change_from_baseline = pred - baseline_displacement
    change_pct = (change_from_baseline / baseline_displacement * 100) if baseline_displacement > 0 else 0
    
    population_used = new_row.get('Population', baseline['Population'])
    displacement_pct_of_pop = (pred / population_used * 100) if population_used > 0 else 0
    
    explanation = (
        f"For {payload.country}, this scenario predicts {int(pred):,} people displaced in {payload.year} "
        f"({displacement_pct_of_pop:.2f}% of population). "
        f"Based on {int(total_events)} disaster events and climate scenario inputs."
    )
    
    return {
        "country": payload.country,
        "year": payload.year,
        "prediction": pred,
        "baseline_year_range": baseline_year_range,
        "baseline_displacement": baseline_displacement,
        "change_from_baseline": change_from_baseline,
        "change_percent": change_pct,
        "displacement_percent_of_population": displacement_pct_of_pop,
        "population": float(population_used),
        "num_events": int(total_events),
        "explanation": explanation
    }

@app.get("/defaults/{iso3}")
def get_country_defaults(iso3: str):
    """Return recent-average climate and disaster defaults for a given country."""
    country_data = df[df["ISO3"] == iso3].sort_values("Year")

    if country_data.empty:
        raise HTTPException(status_code=404, detail=f"No data for country: {iso3}")

    # Use last 5 years as the baseline
    recent = country_data.tail(5)

    # Climate features
    climate_cols = [
        "tas_mean", "tasmin_mean", "tasmax_mean",
        "pr_sum", "pr_mean",
        "tas_std", "pr_std",
        "tas_anomaly", "pr_anomaly"
    ]

    climate_defaults = {
        col: float(recent[col].mean()) for col in climate_cols
        if col in recent.columns
    }

    # Calculate ranges for climate variables (from full dataset for reasonable bounds)
    climate_ranges = {}
    for col in climate_cols:
        if col in df.columns:
            col_min = float(df[col].min())
            col_max = float(df[col].max())
            # Add padding (10% on each side) for better UX
            padding = (col_max - col_min) * 0.1 if col_max > col_min else abs(col_min) * 0.1 if col_min != 0 else 1
            climate_ranges[col] = {
                "min": float(col_min - padding),
                "max": float(col_max + padding)
            }

    # Disaster features
    disaster_cols = [
        "Drought", "Earthquake", "Erosion",
        "Extreme Temperature", "Flood",
        "Mass Movement", "Mixed disasters",
        "Sea level Rise", "Storm",
        "Volcanic activity", "Wave action", "Wildfire"
    ]

    disaster_defaults = {
        col: float(recent[col].mean()) for col in disaster_cols
        if col in recent.columns
    }

    # Calculate ranges for disaster variables
    disaster_ranges = {}
    for col in disaster_cols:
        if col in df.columns:
            col_min = float(df[col].min())
            col_max = float(df[col].max())
            # Add padding for disasters too
            padding = (col_max - col_min) * 0.1 if col_max > col_min else 1
            disaster_ranges[col] = {
                "min": float(max(0, col_min - padding)),  # Disasters can't be negative
                "max": float(col_max + padding)
            }

    # Add population baseline
    population_default = float(recent["Population"].mean())

    return {
        "climate": climate_defaults,
        "disaster": disaster_defaults,
        "population": population_default,
        "ranges": {
            "climate": climate_ranges,
            "disaster": disaster_ranges
        }
    }