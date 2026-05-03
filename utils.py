import ee
import geopandas as gpd
import uuid

def load_shapefile_to_gee(shp_path):
    gdf = gpd.read_file(shp_path)
    if len(gdf) == 1:
        coords = gdf.geometry.iloc[0].__geo_interface__['coordinates'][0]
    else:
        merged_geom = gdf.unary_union
        coords = list(merged_geom.__geo_interface__['coordinates'][0])
    return ee.Geometry.Polygon(coords)

def get_utm_for_roi(roi):
    centroid = roi.centroid()
    lon = centroid.coordinates().getInfo()[0]
    zone = int((lon + 180) / 6) + 1
    hemisphere = 'N' if centroid.coordinates().getInfo()[1] >= 0 else 'S'
    zone_code = f"{zone:02d}"
    epsg_code = f'EPSG:326{zone_code}' if hemisphere == 'N' else f'EPSG:327{zone_code}'
    return epsg_code

def create_unique_asset_id(parent_path, prefix):
    unique_suffix = uuid.uuid4().hex[:8]
    return f"{parent_path}{prefix}_{unique_suffix}"

def covers_full_roi(image, roi, scale=30, band_name=None):
    roi_area = ee.Image.pixelArea().reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=roi,
        scale=scale,
        maxPixels=1e9
    ).values().get(0)

    if band_name:
        valid_mask = image.select(band_name).mask()
    else:
        band_names = image.bandNames()
        first_band = ee.List(band_names).get(0)
        valid_mask = image.select(first_band).mask()

    coverage_area = image.pixelArea().updateMask(valid_mask).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=roi,
        scale=scale,
        maxPixels=1e9
    ).values().get(0)

    return ee.Number(coverage_area).divide(roi_area).multiply(100)

def covers_full_roi_FAST(image, roi, scale=30, band_name='B4'):
    samples = image.select(band_name).sample(
        region=roi, 
        scale=scale, 
        numPixels=100, 
        dropNulls=True
    )
    
    n_samples = samples.size()
    return ee.Number(n_samples).divide(100).multiply(100)