# Install the openeo module for accessing the Sentinel-5P data
# pip install openeo

import openeo
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# Connect to your openEO backend (replace with your actual endpoint)
connection = openeo.connect("openeo.dataspace.copernicus.eu").authenticate_oidc()

# -------------------------------
# Section A. Data Acquisition via openEO
# -------------------------------

# Define AOIs for the six cities
aois = {
    "Jakarta": {
        "type": "Point",
        "coordinates": [106.85, -6.21]
    },
    "Bangkok": {
        "type": "Point",
        "coordinates": [100.50, 13.75]
    },
    "Manila": {
        "type": "Point",
        "coordinates": [120.98, 14.60]
    },
    "New Delhi": {
        "type": "Point",
        "coordinates": [77.21, 28.61]
    },
    "Sydney": {
        "type": "Point",
        "coordinates": [151.21, -33.87]
    },
    "Beijing": {
        "type": "Point",
        "coordinates": [116.40, 39.90]
    },
}

# Define the temporal extent for 2024 (full year)
temporal_extent = ["2024-01-01", "2024-12-31"]

# Dictionary to store the openEO datasets for each city
city_datasets = {}

# Loop through each city and load Sentinel-5P NO₂ data
for city, aoi in aois.items():
    print(f"Loading data for {city}...")
    lon, lat = aoi["coordinates"]

    # Load data for the single point (using the city's coordinates)
    dataset = connection.load_collection(
        "SENTINEL_5P_L2",
        temporal_extent=temporal_extent,
        spatial_extent={"west": lon, "south": lat, "east": lon, "north": lat},
        bands=["NO2"],
    )

    # Aggregate the data by day (mean over each day)
    dataset = dataset.aggregate_temporal_period(reducer="mean", period="day")

    # Spatial aggregation using the point AOI
    dataset = dataset.aggregate_spatial(reducer="mean", geometries=aoi)

    city_datasets[city] = dataset

print("Data loading complete via openEO.")

# Execute batch processing to export each city’s data as a NetCDF file
city_jobs = {}
for city, dataset in city_datasets.items():
    output_filename = f"NO2_2024_{city}.nc"
    print(f"Executing batch job for {city} -> {output_filename} ...")
    job = dataset.execute_batch(title=f"NO2 2024 - {city}", outputfile=output_filename)
    city_jobs[city] = job

print("Batch processing initiated for all cities. (Wait for the jobs to finish and the files to be available locally.)")

# --------------------------------------------------------------------------------------
# Section B. Data Processing & Visualization (once the NetCDF files are available locally)
# --------------------------------------------------------------------------------------

# Define AOIs for the six cities
aois = {
    "Jakarta": {
        "type": "Point",
        "coordinates": [106.85, -6.21]
    },
    "Bangkok": {
        "type": "Point",
        "coordinates": [100.50, 13.75]
    },
    "Manila": {
        "type": "Point",
        "coordinates": [120.98, 14.60]
    },
    "New Delhi": {
        "type": "Point",
        "coordinates": [77.21, 28.61]
    },
    "Sydney": {
        "type": "Point",
        "coordinates": [151.21, -33.87]
    },
    "Beijing": {
        "type": "Point",
        "coordinates": [116.40, 39.90]
    },
}

# Create a dictionary to store processed DataFrames (one per city)
city_dfs = {}

for city in aois.keys():
    file_name = f"NO2_2024_{city}.nc"
    print(f"Loading NetCDF data for {city} from {file_name} ...")

    ds = xr.load_dataset(file_name)

    # Convert the time coordinate to datetime (assuming the coordinate is named 't')
    ds["t"] = pd.to_datetime(ds["t"].values)

    # Convert the xarray dataset to a pandas DataFrame
    # (Assuming the NO2 variable is stored as ds.NO2)
    df = ds[["NO2"]].to_dataframe().reset_index()

    # Create columns for day-of-week and day type (Weekday vs Weekend)
    # In Python's datetime, Monday=0 ... Sunday=6. We define weekend as Saturday (5) and Sunday (6)
    df["day_of_week"] = df["t"].dt.weekday
    df["day_type"] = np.where(df["day_of_week"] < 5, "Weekday", "Weekend")

    city_dfs[city] = df

print("Data processing complete!")

# ----------------------------------------------------------
# Section C. Timeseries Plot: 6 Subplots (One for Each City)
# ----------------------------------------------------------
fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(12, 18), sharex=True, sharey=True)
fig.suptitle("Daily NO₂ Levels in 2024", fontsize=16)

for ax, (city, df) in zip(axes, city_dfs.items()):
    # Multiply the NO2 levels by 10^4
    df["NO2_scaled"] = df["NO2"] * 1e4

    # Compute a 7-day rolling mean for the scaled data
    df["NO2_7day_scaled"] = df["NO2_scaled"].rolling(window=7, center=True).mean()

    # Plot raw daily NO₂ (scaled) as circles (no connecting lines)
    ax.plot(df["t"], df["NO2_scaled"], marker='o', linestyle='', color="lightgray", label="Daily NO₂")
    # Plot the 7-day rolling mean as a continuous blue line
    ax.plot(df["t"], df["NO2_7day_scaled"], linestyle='--', color="blue", label="7-day Rolling Mean")

    # Retrieve coordinates for the city from the aois dictionary
    coords = aois[city]["coordinates"]
    # Update the title to include the city's name and its coordinates (lon, lat)
    ax.set_title(f"{city} ({coords[0]}, {coords[1]})")

    # Compute statistics from the scaled daily NO₂ values
    mean_val = df["NO2_scaled"].mean()
    median_val = df["NO2_scaled"].median()
    max_val = df["NO2_scaled"].max()

    # Get the number of data points for the current city
    data_point_count = len(df)

    # Create a text string with the computed statistics and data point count (formatted to two decimal places)
    stats_text = f"Mean: {mean_val:.2f}\nMedian: {median_val:.2f}\nMax: {max_val:.2f}\nN Data: {data_point_count}"
    # Place the statistics text in the top left of the subplot (using axis coordinates)
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.5))

    ax.set_ylabel("NO₂ (x $10^4$ mol/m²)")
    ax.grid(True)
    ax.legend()

axes[-1].set_xlabel("Date")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('no2_timeseries.png', dpi=300)
plt.show()

# ----------------------------------------------------------
# Section D. Boxplot: Weekdays vs Weekends for All Cities
# ----------------------------------------------------------

# Combine data from all cities into a single DataFrame
combined_df = pd.concat([df.assign(city=city) for city, df in city_dfs.items()], ignore_index=True)

plt.figure(figsize=(12, 6))
sns.boxplot(x="city", y="NO2_scaled", hue="day_type", data=combined_df, palette="Set2")
plt.title("Comparison of NO₂ Levels: Weekdays vs Weekends (2024)")
plt.xlabel("City")
plt.ylabel("NO₂ (x $10^4$ mol/m²)")
plt.legend(title="Day Type")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig('no2_daytype.png', dpi=300)
plt.show()

# ----------------------------------------------------------
# Section E. Boxplot: Seasonal Analysis for All Cities
# ----------------------------------------------------------

# Example "aois" dictionary (update if necessary)
aois = {
    "Jakarta": {"type": "Point", "coordinates": [106.85, -6.21]},
    "Bangkok": {"type": "Point", "coordinates": [100.50, 13.75]},
    "Manila":  {"type": "Point", "coordinates": [120.98, 14.60]},
    "New Delhi": {"type": "Point", "coordinates": [77.21, 28.61]},
    "Sydney": {"type": "Point", "coordinates": [151.21, -33.87]},
    "Beijing": {"type": "Point", "coordinates": [116.40, 39.90]},
}

# Combine data from all cities into a single DataFrame.
combined_df = pd.concat([df.assign(city=city) for city, df in city_dfs.items()], ignore_index=True)

# Ensure the datetime column is parsed properly.
combined_df["t"] = pd.to_datetime(combined_df["t"])
combined_df["month"] = combined_df["t"].dt.month

# Define a function to assign a season based on month and the city's latitude.
def assign_season(row):
    month = row['month']
    # Get the city's latitude from the "aois" dictionary.
    lat = aois[row['city']]['coordinates'][1] if row['city'] in aois else 0
    if lat >= 0:  # Northern Hemisphere
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    else:  # Southern Hemisphere
        if month in [12, 1, 2]:
            return 'Summer'
        elif month in [3, 4, 5]:
            return 'Fall'
        elif month in [6, 7, 8]:
            return 'Winter'
        else:
            return 'Spring'

# Apply the season assignment to the combined DataFrame.
combined_df["season"] = combined_df.apply(assign_season, axis=1)

# Create a boxplot for seasonal analysis.
plt.figure(figsize=(12, 6))
sns.boxplot(x="city", y="NO2_scaled", hue="season", data=combined_df, palette="Set2")
plt.title("Comparison of NO₂ Levels by Season (2024)")
plt.xlabel("City")
plt.ylabel("NO₂ (x $10^4$ mol/m²)")
plt.legend(title="Season")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig('no2_seasonal.png', dpi=300)
plt.show()

