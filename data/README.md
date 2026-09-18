# Example data

The `nyc/` directory contains the bundled NYC inputs.

The bundled demand data is a random sample of 2018 trip records from the New
York City Taxi and Limousine Commission (TLC). Charging locations are public
EV charger locations in Manhattan as of 2018.

- `nyc_sample_05.csv`: pickup and drop-off zones and trip timestamps.
- `nyc_parking.json`: candidate charging and repositioning locations.
- `nyc_taxizone_geodata_utkm.geojson`: taxi-zone geometry.

Infrastructure configurations select locations and set charger capacities.
File checksums are listed in `manifest.json`.

The Chicago adapter requires additional data that is not included.
