ENRICHED SAMPLE DATASET for 'Predicting Urban Respiratory Health Risks' project

Files included under data/samples/:
- air_sample.csv          : hourly air quality measurements (pm25, pm10, no2, o3, co, aqi_est) for 3 synthetic cities over 90 days.
- weather_sample.csv      : hourly weather (temperature, humidity, wind speed, pressure)
- traffic_sample.csv      : hourly traffic index and vehicle counts
- health_sample.csv       : daily hospital admissions and respiratory_admissions per city
- data_metadata.csv       : metadata about these generated files
- districts_geo.json      : simple GeoJSON with 3 district polygons (District_A, District_B, District_C)
- population_by_district.csv : population and area estimates per district
- merged_daily.csv        : daily aggregated + merged dataset ready for modeling (features + targets)
- README_SAMPLE_DATA.txt  : this file

Notes:
- merged_daily.csv includes a next_day_respiratory_admissions target (shifted by city) and a categorical risk_category.
- The datasets are synthetic but structured to mimic real multi-source challenges: missing values, multi-dtypes, geospatial join keys, and time alignment.

Generated: 2025-11-16T20:13:19.800043
