# Install the openeo module for accessing the Sentinel-5P data
# pip install openeo

import openeo
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# Connect to your openEO backend
# Ensure you are authenticated if running non-interactively
try:
    print("Attempting to connect to openEO platform...")
    connection = openeo.connect("openeo.dataspace.copernicus.eu").authenticate_oidc()
    print("Successfully connected and authenticated with openEO.")
except Exception as e:
    print(f"Error connecting to openEO: {e}")
    print("Please ensure you can authenticate with openeo.dataspace.copernicus.eu")
    print("If running for the first time, follow the browser authentication steps.")
    exit()

# -------------------------------
# Section A. Configuration
# -------------------------------
print("\n--- Configuration ---")
# Define AOIs for cities in Oman
# Format: CityName: [longitude, latitude]
cities_coordinates = {
    "Muscat": [58.54, 23.61],
    "Salalah": [54.0924, 17.0151],
    "Sohar": [56.7071, 24.3459],
    "Nizwa": [57.5301, 22.9333],
    "Sur": [59.5289, 22.5667],
    "AlBuraimi": [55.7890, 24.2441]
}
print(f"Target cities: {list(cities_coordinates.keys())}")

# Define the temporal extent
temporal_extent = ["2024-01-01", "2025-01-01"]
year_str = temporal_extent[0][:4]
print(f"Temporal extent: {temporal_extent[0]} to {temporal_extent[1]}")

# Define Aerosol Index band and buffer for spatial extent
product_band = "AER_AI_354_388" # UV Aerosol Index from Sentinel-5P
product_name_for_file = "AerosolIndex354_388" # Used in filenames
buffer_degrees = 0.5 # Buffer in degrees around the point for the spatial extent
print(f"Using product band: {product_band}")

# --- Multiple Dust Event Threshold Configuration ---
# Define a list of thresholds for statistical reporting on plots
dust_event_thresholds_list = [0.75, 1.0, 1.5, 2.0, 2.5]
print(f"Statistical analysis with AI thresholds: {dust_event_thresholds_list}")

# Define thresholds for colored scatter points on timeseries plot
threshold_yellow_min = 1.0
threshold_orange_min = 1.5
threshold_red_min = 2.0
print(f"Timeseries plot color highlights: Yellow (AI > {threshold_yellow_min} to <= {threshold_orange_min}), Orange (AI > {threshold_orange_min} to <= {threshold_red_min}), Red (AI > {threshold_red_min})")


# Loop through each city for data acquisition, processing, and plotting
for city_name, coords in cities_coordinates.items():
    print(f"\n--- Processing for city: {city_name} ({coords[1]}, {coords[0]}) ---")
    lon, lat = coords

    # -------------------------------
    # Section B. Data Acquisition via openEO
    # -------------------------------
    print(f"Defining data loading for {city_name} using band {product_band}...")
    try:
        dataset = connection.load_collection(
            "SENTINEL_5P_L2", # Sentinel-5P Level 2 data
            temporal_extent=temporal_extent,
            spatial_extent={
                "west": lon - buffer_degrees, "south": lat - buffer_degrees,
                "east": lon + buffer_degrees, "north": lat + buffer_degrees,
                "crs": "EPSG:4326" # Coordinate Reference System
            },
            bands=[product_band], # Specify the Aerosol Index band
        )
        # Aggregate to daily mean values for the defined spatial extent
        dataset_daily_spatial_mean = dataset.aggregate_temporal_period(reducer="mean", period="day")

        # Further aggregate spatially to get a single time series for the point of interest (city location)
        # This effectively takes the mean of the already daily-averaged data within the feature geometry.
        feature_collection_geometry = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {"id": city_name}
            }]
        }
        processed_dataset = dataset_daily_spatial_mean.aggregate_spatial(
            reducer="mean", geometries=feature_collection_geometry
        )
        print("Data loading definition complete via openEO.")

        # Define output filename and job title for the batch job
        output_filename = f"{product_name_for_file}_{year_str}_{city_name}.nc" # e.g., AerosolIndex354_388_2023_Muscat.nc
        job_title = f"{product_name_for_file} {year_str} - {city_name}"

        print(f"Executing batch job for {city_name} -> {output_filename} ... (This may take several minutes)")
        # execute_batch submits the processing job to the openEO backend and downloads the result.
        job = processed_dataset.execute_batch(title=job_title, outputfile=output_filename,
                                             job_options={'driver-memory': '1g'}) # Example job option
        print(f"Batch job for {city_name} completed. Output NetCDF: {output_filename}")
    except Exception as e:
        print(f"ERROR during openEO data acquisition or job execution for {city_name}: {e}")
        continue # Skip to the next city if an error occurs

    # --------------------------------------------------------------------------------------
    # Section C. Data Processing (once the NetCDF file is available locally)
    # --------------------------------------------------------------------------------------
    df_plot = None # Initialize DataFrame
    print(f"Attempting to load NetCDF data for {city_name} from {output_filename} ...")
    try:
        # Load the downloaded NetCDF file using xarray
        ds = xr.load_dataset(output_filename)

        # Attempt to identify the time coordinate automatically
        time_coord_name = 't' # Default assumption
        if time_coord_name not in ds.coords:
            potential_time_coords = [c for c in ds.coords if 'time' in c.lower() or ds[c].dtype == 'datetime64[ns]']
            if potential_time_coords: time_coord_name = potential_time_coords[0]
            elif list(ds.dims): time_coord_name = list(ds.dims)[0] # Fallback to first dimension
            else: raise ValueError("No suitable time coordinate or dimension found in NetCDF.")
        print(f"Using time coordinate: {time_coord_name}")
        ds[time_coord_name] = pd.to_datetime(ds[time_coord_name].values) # Ensure datetime format

        # Attempt to identify the Aerosol Index data variable
        data_variable_name_in_file = product_band # Ideal case
        if data_variable_name_in_file not in ds.data_vars:
            print(f"Warning for {city_name}: Requested band '{data_variable_name_in_file}' not found as a direct data variable.")
            print(f"Available data variables in NetCDF: {list(ds.data_vars.keys())}")
            found_alt = False
            # Try common variations if the exact name isn't present
            for var_name in ds.data_vars:
                if "aerosol_index" in var_name.lower() or "aer_ai" in var_name.lower() or product_band.lower() in var_name.lower():
                    data_variable_name_in_file = var_name
                    print(f"Using alternative data variable from file: '{data_variable_name_in_file}'")
                    found_alt = True; break
            if not found_alt: # If still not found
                if list(ds.data_vars): # Use the first available data variable as a last resort
                    data_variable_name_in_file = list(ds.data_vars)[0]
                    print(f"Could not find a suitable aerosol index variable. Using first available data variable: '{data_variable_name_in_file}'")
                else:
                    raise ValueError("No data variables found in the NetCDF file.")
        print(f"Using data variable: {data_variable_name_in_file}")

        # Convert to pandas DataFrame for easier plotting with seaborn/matplotlib
        df = ds[[data_variable_name_in_file]].to_dataframe().reset_index()
        # Rename columns to standard names used in plotting sections
        df.rename(columns={data_variable_name_in_file: product_band, time_coord_name: 't'}, inplace=True)

        df_plot = df
        print(f"Data processing complete for {city_name}!")
    except FileNotFoundError:
        print(f"ERROR: File {output_filename} not found for {city_name}. Please ensure the openEO job completed and downloaded the file.")
        continue
    except Exception as e:
        print(f"ERROR loading or processing {output_filename} for {city_name}: {e}")
        continue

    # Proceed to plotting only if data was successfully loaded and processed
    if df_plot is None or df_plot.empty:
        print(f"No data available for visualization for {city_name}. Skipping plots.")
        continue
    else:
        # ----------------------------------------------------------
        # Section D. Timeseries Plot (Enhanced with Multi-Color Dust Event Highlighting)
        # ----------------------------------------------------------
        print(f"Generating timeseries plot for {city_name}...")
        df_plot[f"{product_band}_7day_mean"] = df_plot[product_band].rolling(window=7, center=True, min_periods=1).mean()

        plt.figure(figsize=(17, 8))
        # Plot all daily AI values as grey dots (background)
        plt.plot(df_plot["t"], df_plot[product_band], marker='o', markersize=4, linestyle='', color="lightgray", label=f"Daily {product_band}", zorder=1)

        # Define conditions for color highlighting
        cond_yellow = (df_plot[product_band] > threshold_yellow_min) & (df_plot[product_band] <= threshold_orange_min)
        cond_orange = (df_plot[product_band] > threshold_orange_min) & (df_plot[product_band] <= threshold_red_min)
        cond_red = df_plot[product_band] > threshold_red_min

        # Plot highlighted events
        if cond_yellow.any():
            plt.scatter(df_plot.loc[cond_yellow, "t"], df_plot.loc[cond_yellow, product_band],
                        color='yellow', edgecolor='darkgoldenrod', marker='o', s=50,
                        label=f"AI ({threshold_yellow_min:.1f}-{threshold_orange_min:.1f})", zorder=3, alpha=0.8)

        if cond_orange.any():
            plt.scatter(df_plot.loc[cond_orange, "t"], df_plot.loc[cond_orange, product_band],
                        color='orange', edgecolor='saddlebrown', marker='o', s=60,
                        label=f"AI ({threshold_orange_min:.1f}-{threshold_red_min:.1f})", zorder=4, alpha=0.8)

        if cond_red.any():
            plt.scatter(df_plot.loc[cond_red, "t"], df_plot.loc[cond_red, product_band],
                        color='red', edgecolor='black', marker='o', s=70,
                        label=f"AI > {threshold_red_min:.1f}", zorder=5, alpha=0.8)

        plt.plot(df_plot["t"], df_plot[f"{product_band}_7day_mean"], linestyle='--', color="blue", label="7-day Rolling Mean", zorder=2)

        plt.title(f"Daily {product_band} for {city_name} - {year_str} (Dust Intensity Highlighted)", fontsize=16)
        plt.ylabel(f"{product_band} (Unitless)")
        plt.xlabel("Date")

        mean_val = df_plot[product_band].mean()
        median_val = df_plot[product_band].median()
        max_val = df_plot[product_band].max()
        data_point_count = len(df_plot)

        stats_text_lines = [
            f"Mean AI: {mean_val:.2f}", f"Median AI: {median_val:.2f}",
            f"Max AI: {max_val:.2f}", f"Data Points: {data_point_count}",
            "--- Event Days (Stat Thresholds) ---"
        ]
        for thold in dust_event_thresholds_list: # Uses the separate list for stats
            num_event_days_thold = (df_plot[product_band] > thold).sum()
            stats_text_lines.append(f"Days AI > {thold:.2f}: {num_event_days_thold}")

        stats_text = "\n".join(stats_text_lines)
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.85))

        plt.grid(True)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plot_filename_timeseries = f'{product_name_for_file}_timeseries_intensity_{city_name}_{year_str}.png'
        plt.savefig(plot_filename_timeseries, dpi=300)
        print(f"Saved timeseries plot: {plot_filename_timeseries}")
        # plt.show() # In Colab, plots usually show automatically or use plt.close()
        plt.close() # Close the figure to free memory

        # ----------------------------------------------------------
        # Section E. Histogram of Aerosol Index Values
        # ----------------------------------------------------------
        print(f"Generating histogram for {city_name}...")
        plt.figure(figsize=(10, 6))
        sns.histplot(df_plot[product_band].dropna(), kde=True, bins=30) # dropna for robustness
        plt.title(f"Distribution of Daily {product_band} for {city_name} ({year_str})", fontsize=16)
        plt.xlabel(f"{product_band} (Unitless)")
        plt.ylabel("Frequency")
        plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_val:.2f}')
        plt.axvline(median_val, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median_val:.2f}')

        hist_threshold_colors = plt.cm.viridis(np.linspace(0, 0.8, len(dust_event_thresholds_list)))
        for i, thold in enumerate(dust_event_thresholds_list):
            plt.axvline(thold, color=hist_threshold_colors[i], linestyle='dotted', linewidth=1.5, label=f'Stat Threshold: {thold:.2f}')

        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plot_filename_hist = f'{product_name_for_file}_histogram_{city_name}_{year_str}.png'
        plt.savefig(plot_filename_hist, dpi=300)
        print(f"Saved histogram: {plot_filename_hist}")
        plt.close()

        # ----------------------------------------------------------
        # Section F. Boxplot: Seasonal Analysis
        # ----------------------------------------------------------
        print(f"Generating seasonal boxplot for {city_name}...")
        df_plot["month"] = df_plot["t"].dt.month
        def assign_season(row_month): # Simple season assignment for Northern Hemisphere
            if row_month in [12, 1, 2]: return 'Winter (DJF)'
            elif row_month in [3, 4, 5]: return 'Spring (MAM)'
            elif row_month in [6, 7, 8]: return 'Summer (JJA)'
            else: return 'Fall (SON)'
        df_plot["season"] = df_plot["month"].apply(assign_season)
        season_order = ["Winter (DJF)", "Spring (MAM)", "Summer (JJA)", "Fall (SON)"]

        plt.figure(figsize=(10, 6))
        sns.boxplot(x="season", y=product_band, data=df_plot, palette="Set3", order=season_order)
        plt.title(f"Seasonal Distribution of {product_band} for {city_name} ({year_str})", fontsize=16)
        for thold_idx, thold in enumerate(dust_event_thresholds_list):
            plt.axhline(y=thold, color=hist_threshold_colors[thold_idx], linestyle=':', linewidth=1.0, alpha=0.9)
        plt.xlabel("Season")
        plt.ylabel(f"{product_band} (Unitless)")
        plt.grid(axis="y", linestyle='--', alpha=0.7)
        plt.tight_layout()
        plot_filename_seasonal = f'{product_name_for_file}_seasonal_boxplot_{city_name}_{year_str}.png'
        plt.savefig(plot_filename_seasonal, dpi=300)
        print(f"Saved seasonal boxplot: {plot_filename_seasonal}")
        plt.close()

        # ----------------------------------------------------------
        # Section G. Boxplot: Monthly Analysis
        # ----------------------------------------------------------
        print(f"Generating monthly boxplot for {city_name}...")
        month_names_map = {
            1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
            7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
        }
        df_plot['month_name'] = pd.Categorical(df_plot['month'].map(month_names_map), categories=month_names_map.values(), ordered=True)

        plt.figure(figsize=(12, 7))
        sns.boxplot(x="month_name", y=product_band, data=df_plot, palette="coolwarm")
        plt.title(f"Monthly Distribution of {product_band} for {city_name} ({year_str})", fontsize=16)
        for thold_idx, thold in enumerate(dust_event_thresholds_list):
            plt.axhline(y=thold, color=hist_threshold_colors[thold_idx], linestyle=':', linewidth=1.0, alpha=0.9)
        plt.xlabel("Month")
        plt.ylabel(f"{product_band} (Unitless)")
        plt.grid(axis="y", linestyle='--', alpha=0.7)
        plt.tight_layout()
        plot_filename_monthly = f'{product_name_for_file}_monthly_boxplot_{city_name}_{year_str}.png'
        plt.savefig(plot_filename_monthly, dpi=300)
        print(f"Saved monthly boxplot: {plot_filename_monthly}")
        plt.close()

print("\n--- Script finished for all configured Omani cities. ---")
print("Output NetCDF files and PNG plots are saved in your Colab environment's current working directory.")
print("You can download them from the file browser panel on the left.")

