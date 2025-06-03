# Install the openeo module for accessing the Sentinel-5P data
# pip install openeo
# You might also need: pip install xarray netcdf4 matplotlib pandas seaborn numpy

import openeo
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import time # For script execution timing

# --- Record start time ---
script_start_time = time.perf_counter()

# Connect to your openEO backend
print("Attempting to connect and authenticate with openEO backend: openeo.dataspace.copernicus.eu ...")
try:
    # Always attempt OIDC authentication for Copernicus Dataspace
    connection = openeo.connect("openeo.dataspace.copernicus.eu").authenticate_oidc()
    
    # Perform a simple operation that requires authentication to verify
    print("Verifying authentication by listing jobs (first page)...")
    connection.list_jobs(limit=1) # This will raise an error if not authenticated
    print("Successfully connected and authenticated with openEO backend.")
    
except Exception as e:
    print(f"CRITICAL ERROR: Failed to connect or authenticate with openEO backend: {e}")
    print("Please ensure you can authenticate with Copernicus Dataspace.")
    print("This might involve a browser-based login if it's your first time or tokens have expired.")
    print("Exiting script as openEO operations cannot proceed without valid authentication.")
    exit() # Stop the script if authentication fails

# -------------------------------
# Section A. Data Acquisition via openEO
# -------------------------------
section_a_start_time = time.perf_counter()

# Define AOIs for the six cities
aois = {
    "Muscat": {"type": "Point", "coordinates": [58.54, 23.61]},
    "Salalah": {"type": "Point", "coordinates": [54.0924, 17.0151]},
    "Sohar": {"type": "Point", "coordinates": [56.7071, 24.3459]},
    "Nizwa": {"type": "Point", "coordinates": [57.5301, 22.9333]},
    "Sur": {"type": "Point", "coordinates": [59.5289, 22.5667]},
    "AlBuraimi": {"type": "Point", "coordinates": [55.7890, 24.2441]},
}

# Define the temporal extent
TARGET_YEAR = "2024" # Define target year
temporal_extent = [f"{TARGET_YEAR}-01-01", f"{TARGET_YEAR}-12-31"]

city_datasets = {}
RUN_OPENEO_DATA_ACQUISITION = True # Set to False to skip downloading if files exist

if RUN_OPENEO_DATA_ACQUISITION:
    print(f"--- Running Section A: Data Acquisition via openEO for {TARGET_YEAR} ---")
    for city, aoi_dict in aois.items():
        print(f"Preparing data definition for {city}...")
        lon, lat = aoi_dict["coordinates"]
        buffer = 0.1
        dataset = connection.load_collection(
            "SENTINEL_5P_L2",
            temporal_extent=temporal_extent,
            spatial_extent={"west": lon - buffer, "south": lat - buffer, "east": lon + buffer, "north": lat + buffer, "crs": "EPSG:4326"},
            bands=["NO2"],
        )
        dataset = dataset.aggregate_temporal_period(reducer="mean", period="day")
        feature_collection_geometry = {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]}, "properties": {"id": city}}]}
        dataset = dataset.aggregate_spatial(reducer="mean", geometries=feature_collection_geometry)
        city_datasets[city] = dataset
    print("Data definition complete via openEO.")

    city_jobs = {}
    for city, dataset_item in city_datasets.items():
        output_filename = f"NO2_{TARGET_YEAR}_{city}.nc"
        print(f"Submitting batch job for {city} -> {output_filename} ...")
        try:
            job = dataset_item.execute_batch(title=f"NO2 {TARGET_YEAR} - {city}", outputfile=output_filename)
            city_jobs[city] = job
            print(f"Job {job.job_id} for {city} submitted.")
        except openeo.rest.OpenEoApiError as e:
            print(f"ERROR submitting job for {city}: {e}")
            print("This might be due to an issue with job parameters or backend resources.")
            print("Continuing to submit other jobs if possible...")
            continue # Try to submit jobs for other cities
        except Exception as e:
            print(f"An unexpected error occurred submitting job for {city}: {e}")
            continue


    print("\nBatch processing initiated (if jobs were successfully submitted).")
    print("IMPORTANT: You will need to wait for these jobs to finish processing on the backend")
    print("and for the output NetCDF files to be downloaded to your local directory")
    print("BEFORE the rest of this script (Sections B onwards) can use them.")
    print("You can monitor job progress on the openEO platform dashboard.")

else:
    print(f"--- Skipping Section A: Data Acquisition (RUN_OPENEO_DATA_ACQUISITION is False) ---")
    print(f"Assuming NetCDF files for {TARGET_YEAR} are already downloaded.")

section_a_end_time = time.perf_counter()
print(f"Section A (openEO definition/submission) took: {section_a_end_time - section_a_start_time:.2f} seconds.\n")

# --------------------------------------------------------------------------------------
# Section B. Local Data Processing
# --------------------------------------------------------------------------------------
section_b_start_time = time.perf_counter()
print("--- Running Section B: Local Data Processing ---")
city_dfs = {}

for city_name in aois.keys():
    file_name = f"NO2_{TARGET_YEAR}_{city_name}.nc"
    print(f"Attempting to load NetCDF data for {city_name} from {file_name} ...")
    try:
        ds = xr.load_dataset(file_name)
        ds["t"] = pd.to_datetime(ds["t"].values)
        df = ds[["NO2"]].to_dataframe().reset_index() # Original NO2 data, may contain NaNs

        df["day_of_week"] = df["t"].dt.weekday
        df["day_type"] = np.where(df["day_of_week"] < 5, "Weekday", "Weekend")
        city_dfs[city_name] = df
        print(f"Successfully processed data for {city_name}.")
    except FileNotFoundError:
        print(f"ERROR: File {file_name} not found. Skipping this city.")
        continue
    except Exception as e:
        print(f"ERROR: Could not load or process {file_name} for {city_name}: {e}. Skipping.")
        continue

if not city_dfs:
    print("\nNo data was loaded. Sections C, D, E will be skipped.")
else:
    print("\nLocal data processing complete for available cities!")

section_b_end_time = time.perf_counter()
print(f"Section B (Local Data Processing) took: {section_b_end_time - section_b_start_time:.2f} seconds.\n")

if city_dfs:
    # ----------------------------------------------------------
    # Section C. Timeseries Plot
    # ----------------------------------------------------------
    section_c_start_time = time.perf_counter()
    print("--- Running Section C: Timeseries Plotting ---")
    num_cities_loaded = len(city_dfs)
    fig, axes = plt.subplots(nrows=num_cities_loaded, ncols=1, figsize=(14, 3.5 * num_cities_loaded), sharex=True, sharey=True, squeeze=False)
    axes = axes.flatten()
    fig.suptitle(f"Daily NO₂ Levels in {TARGET_YEAR}", fontsize=16, y=0.98)

    for i, (city_plot, df_plot) in enumerate(city_dfs.items()):
        ax = axes[i]
        df_plot["NO2_scaled"] = df_plot["NO2"] * 1e4
        df_plot["NO2_7day_scaled"] = df_plot["NO2_scaled"].rolling(window=7, center=True, min_periods=1).mean()

        ax.plot(df_plot["t"], df_plot["NO2_scaled"], marker='o', linestyle='', ms=4, color="dimgray", alpha=0.7, label="Daily NO₂")
        ax.plot(df_plot["t"], df_plot["NO2_7day_scaled"], linestyle='-', color="dodgerblue", lw=1.5, label="7-day Rolling Mean")

        coords = aois[city_plot]["coordinates"]
        ax.set_title(f"{city_plot} ({coords[0]:.2f}, {coords[1]:.2f})")

        mean_val = df_plot["NO2_scaled"].mean()
        median_val = df_plot["NO2_scaled"].median()
        max_val = df_plot["NO2_scaled"].max()
        data_point_count = df_plot["NO2_scaled"].count() 

        stats_text = (f"Mean: {mean_val:.2f}\n"
                      f"Median: {median_val:.2f}\n"
                      f"Max: {max_val:.2f}\n"
                      f"N Data: {data_point_count}")
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
                fontsize=8, bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.75))

        ax.set_ylabel("NO₂ (x $10^4$ mol/m²)")
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper right', fontsize=8)

    if num_cities_loaded > 0:
        axes[-1].set_xlabel("Date")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'no2_timeseries_{TARGET_YEAR}.png', dpi=300) 
    plt.show()
    print(f"Timeseries plot saved as no2_timeseries_{TARGET_YEAR}.png")
    
    section_c_end_time = time.perf_counter()
    print(f"Section C (Timeseries Plotting) took: {section_c_end_time - section_c_start_time:.2f} seconds.\n")

    # ----------------------------------------------------------
    # Section D. Boxplot: Weekdays vs Weekends
    # ----------------------------------------------------------
    section_d_start_time = time.perf_counter()
    print("--- Running Section D: Weekday/Weekend Boxplot ---")
    combined_df_list = []
    for city_data_key, df_data_val in city_dfs.items():
        if "NO2_scaled" not in df_data_val.columns: 
             df_data_val["NO2_scaled"] = df_data_val["NO2"] * 1e4
        combined_df_list.append(df_data_val.assign(city=city_data_key))
    
    if combined_df_list:
        combined_df = pd.concat(combined_df_list, ignore_index=True)
        plt.figure(figsize=(12, 7))
        sns.boxplot(x="city", y="NO2_scaled", hue="day_type", data=combined_df, palette="Set2", showfliers=False)
        plt.title(f"Comparison of NO₂ Levels: Weekdays vs Weekends ({TARGET_YEAR})", fontsize=14) 
        plt.xlabel("City", fontsize=12)
        plt.ylabel("NO₂ (x $10^4$ mol/m²)", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.legend(title="Day Type", loc='upper right')
        plt.grid(axis="y", linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'no2_daytype_{TARGET_YEAR}.png', dpi=300) 
        plt.show()
        print(f"Weekday/weekend boxplot saved as no2_daytype_{TARGET_YEAR}.png")
    else:
        print("No data available for weekday/weekend boxplot.")
        
    section_d_end_time = time.perf_counter()
    print(f"Section D (Weekday/Weekend Boxplot) took: {section_d_end_time - section_d_start_time:.2f} seconds.\n")

    # ----------------------------------------------------------
    # Section E. Boxplot: Seasonal Analysis
    # ----------------------------------------------------------
    section_e_start_time = time.perf_counter()
    print("--- Running Section E: Seasonal Boxplot ---")
    if combined_df_list:
        combined_df["t"] = pd.to_datetime(combined_df["t"])
        combined_df["month"] = combined_df["t"].dt.month

        def assign_season(row):
            month_val = row['month']
            city_latitude = aois[row['city']]['coordinates'][1]
            if city_latitude >= 0: 
                if month_val in [12, 1, 2]: return 'Winter (DJF)'
                elif month_val in [3, 4, 5]: return 'Spring (MAM)'
                elif month_val in [6, 7, 8]: return 'Summer (JJA)'
                else: return 'Fall (SON)'
            else: 
                if month_val in [12, 1, 2]: return 'Summer (DJF)'
                elif month_val in [3, 4, 5]: return 'Fall (MAM)'
                elif month_val in [6, 7, 8]: return 'Winter (JJA)'
                else: return 'Spring (SON)'
        
        combined_df["season"] = combined_df.apply(assign_season, axis=1)
        season_order = ['Winter (DJF)', 'Spring (MAM)', 'Summer (JJA)', 'Fall (SON)']

        plt.figure(figsize=(14, 7))
        sns.boxplot(x="city", y="NO2_scaled", hue="season", data=combined_df, hue_order=season_order, palette="viridis", showfliers=False)
        plt.title(f"Comparison of NO₂ Levels by Season ({TARGET_YEAR})", fontsize=14) 
        plt.xlabel("City", fontsize=12)
        plt.ylabel("NO₂ (x $10^4$ mol/m²)", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.legend(title="Season", loc='upper right')
        plt.grid(axis="y", linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'no2_seasonal_{TARGET_YEAR}.png', dpi=300) 
        plt.show()
        print(f"Seasonal boxplot saved as no2_seasonal_{TARGET_YEAR}.png")
    else:
        print("No data available for seasonal boxplot.")
    
    section_e_end_time = time.perf_counter()
    print(f"Section E (Seasonal Boxplot) took: {section_e_end_time - section_e_start_time:.2f} seconds.\n")

script_end_time = time.perf_counter()
total_script_duration = script_end_time - script_start_time
print(f"-------------------------------------------------------------")
print(f"Total script execution time: {total_script_duration:.2f} seconds ({total_script_duration / 60:.2f} minutes).")
print(f"-------------------------------------------------------------")
