# Citizen Reporting Portal — Implementation Plan

## Goal
Enhance the existing [dashboard.py](file:///c:/Users/Acer/Desktop/Project/Hackathon/dashboard.py) by adding a **Citizen Reporting Portal**. This allows normal users to submit a new complaint (via a map click) and instantly receive a preliminary risk score for their area by running a live inference through the existing Random Forest model.

## Proposed Changes

### [MODIFY] [dashboard.py](file:///c:/Users/Acer/Desktop/Project/Hackathon/dashboard.py)
We will restructure the layout to use Streamlit Tabs to separate the Planner/Dispatch view from the Citizen view.

1. **Implement `st.tabs`:**
   Configure `tab1, tab2 = st.tabs(["🛡️ Overview & Dispatch", "📸 Citizen Reporting"])`.
   Move all existing map, heatmap, KPI metrics, and charts into `tab1`.

2. **Build Citizen Reporting Tab (`tab2`):**
   - Provide a smaller, separate Folium map using `st_folium` allowing users to click a location.
   - Use `st_folium`'s returned dictionary to capture the `last_clicked` coordinates (Latitude, Longitude).
   - Display a Form (`st.form`) below the map to collect:
     - Issue Type (Dropdown: Overgrown yard, Standing water, Illegal dumping, Pothole, etc.)
     - Description (Text input)
     - Submit button.

3. **Inference Pipeline for Citizen Report:**
   When a user clicks "Submit Report":
   - Use `auto_pipeline.assign_grid_cell(lat, lon)` to determine the grid cell.
   - Load [Dataset/feature_matrix.csv](file:///c:/Users/Acer/Desktop/Project/Hackathon/Dataset/feature_matrix.csv) to find the baseline historical features for that specific grid cell.
   - If the grid cell is completely new (not in matrix), construct an empty feature row (all zeros) but compute the `dist_to_nearest_siren_km` dynamically using the [Dataset/sirens_cleaned.csv](file:///c:/Users/Acer/Desktop/Project/Hackathon/Dataset/sirens_cleaned.csv).
   - **Simulate the impact:** Increment `complaint_count_30d` by 1. If the mapped issue type is an environmental nuisance, increment `nuisance_count_30d` by 1.
   - Filter the row to only the expected features using `model.feature_names_in_`.
   - Run `model.predict_proba(row)[0][1]` to get the raw model risk score.
   - Multiply by the current `weather_multiplier` (fetched from the active Bright Data weather pipeline in the sidebar).
   - Display a nice user-facing message:
     *"Thank you! Your report in Zone [X] has been logged. Current area risk score is **[Score]**. This has been escalated to dispatch."*

4. **Persist the Report (Optional but nice):**
   - Append the new report to `Dataset/citizen_reports.csv`.
   - Plot these citizen reports as distinct markers on the main Dispatch map in `tab1` so planners can see them.

## Verification Plan

### Manual Verification
1. Run `uv run streamlit run dashboard.py`.
2. Ensure the app loads and displays two tabs.
3. Verify the main dashboard (Tab 1) functions exactly as it did before.
4. Switch to Tab 2.
5. Click a location on the map in Montgomery (e.g., near downtown). Verify the coordinates populate.
6. Select "Standing Water / Empty Pool" and type "Mosquitoes everywhere".
7. Click **Submit Report**.
8. Verify a loading spinner appears, followed by a success message displaying a computed Risk Score (e.g., 0.65) and simulating the dispatch action.
9. (Optional) Check that the live weather multiplier correctly amplifies the citizen's baseline score if a severe weather event is manually simulated in the sidebar.
