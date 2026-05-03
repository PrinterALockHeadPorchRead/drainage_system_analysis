import ee
import os
from datetime import datetime, timedelta
import rasterio
import numpy as np
import shutil
import requests

from config import *
from utils import *
from main_project.drainage_core import drainage_test_pipeline_adapted
from segmentation_inference import run_inference_on_arrays

ee.Initialize(project=GEE_PROJECT_ID)

def get_best_s2_single(roi, date_start, date_end, min_coverage=95, cloud_max=10, scale=30):
    s2_coll = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
               .filterDate(date_start, date_end)
               .filterBounds(roi)
               .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_max)))

    def check_coverage_and_clouds(img):
        coverage_pct = covers_full_roi(img, roi, scale, 'B8')
        scl = img.select('SCL')
        cloud_mask = scl.eq(8).Or(scl.eq(9)).Or(scl.eq(10)).Or(scl.eq(3))
        roi_clouds = cloud_mask.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=roi,
            scale=20,
            maxPixels=1e9
        ).values().get(0)
        roi_clouds_pct = ee.Number(roi_clouds).multiply(100)
        return img.set('coverage_pct', coverage_pct) \
                  .set('roi_clouds_pct', roi_clouds_pct) \
                  .set('score', coverage_pct.subtract(roi_clouds_pct))

    scored_coll = s2_coll.limit(10).map(check_coverage_and_clouds)
    full_coverage = scored_coll.filter(ee.Filter.gte('coverage_pct', min_coverage))

    full_count = full_coverage.size().getInfo()

    if full_count > 0:
        best_raw = full_coverage.sort('roi_clouds_pct').first()

        coverage = best_raw.get('coverage_pct').getInfo()
        roi_clouds = best_raw.get('roi_clouds_pct').getInfo()
        date_str = ee.Date(best_raw.get('system:time_start')).format('YYYY-MM-dd').getInfo()

        print(f"S2 (тайл): {coverage:.1f}% покрытие | {roi_clouds:.1f}% облаков | {date_str}")

        s2_processed = (best_raw.select(['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12'])
                    .clip(roi)
                    .set('system:time_start', best_raw.get('system:time_start')))

        return s2_processed, date_str

    print("S2 (тайл): нет полного покрытия")
    return None, None

def get_best_s1_near_s2(roi, s2_date_ee, min_coverage=90, scale=30):
    for days in [0, 1, 3, 7, 14]:
        if days == 0:
            window_start = s2_date_ee
            window_end = s2_date_ee.advance(1, 'day')
            window_desc = "0д (тот же день)"
        else:
            window_start = s2_date_ee.advance(-days, 'day')
            window_end = s2_date_ee.advance(days, 'day')
            window_desc = f"±{days}д"

        s1_coll = (ee.ImageCollection('COPERNICUS/S1_GRD')
                  .filterDate(window_start, window_end)
                  .filterBounds(roi)
                  .filter(ee.Filter.eq('instrumentMode', 'IW')))

        s1_count = s1_coll.size().getInfo()
        print(f"S1 (тайл) {window_desc}: {s1_count} снимков")

        if s1_count > 0:
            def check_s1_coverage(img):
                coverage = covers_full_roi(img, roi, scale, 'VV')
                return img.set('coverage_pct', coverage)

            scored_s1 = s1_coll.limit(10).map(check_s1_coverage)
            best_s1 = scored_s1.sort('coverage_pct', False).first()

            coverage = best_s1.get('coverage_pct').getInfo()
            print(f"S1 (тайл) покрытие: {coverage:.1f}% ({window_desc})")

            if coverage >= min_coverage:
                polars = best_s1.get('transmitterReceiverPolarisation').getInfo()
                bands = ['VV', 'VH'] if 'VH' in polars else ['VV']
                print(f"S1 (тайл) НАЙДЕН: {bands} | {coverage:.1f}% ({window_desc})")
                return best_s1.select(bands).clip(roi)

    print("S1 (тайл): нет хорошего покрытия")
    return None
    
def collect_masks_into_collection(mask_asset_ids):
    image_list = [ee.Image(asset_id) for asset_id in mask_asset_ids if asset_id is not None]
    return ee.ImageCollection.fromImages(image_list)

def mosaic_masks(mask_collection):
    return mask_collection.mosaic()

def load_image_as_array(image, region, scale, crs, filename, folder="./temp_data/"):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    url = image.getDownloadUrl({
        'region': region,
        'crs': crs,
        'format': 'GEO_TIFF',
        'dimensions': f'{TILE_TARGET_SIZE_PX}x{TILE_TARGET_SIZE_PX}'
    })

    response = requests.get(url)
    response.raise_for_status()

    with open(filepath, 'wb') as f:
        f.write(response.content)

    with rasterio.open(filepath) as src:
        arr = src.read()
        arr = arr.transpose((1, 2, 0)) if arr.ndim > 2 else arr
    print(f"Файл {filename} загружен")
    return arr, filepath

def get_s2_s1_dem_for_target_rect(target_rect, target_date_str, resolution=TILE_TARGET_SCALE, 
                                  min_coverage=99, cloud_max=3):
    date_obj = datetime.strptime(target_date_str, '%Y-%m-%d')
    date_start = (date_obj - timedelta(days=14)).strftime('%Y-%m-%d')
    date_end = (date_obj + timedelta(days=14)).strftime('%Y-%m-%d')

    s2_img, s2_date_str = get_best_s2_single(target_rect, date_start, date_end, min_coverage, cloud_max, scale=resolution)
    if s2_img is None:
        print(f"Не найден подходящий S2 снимок для целевого прямоугольника.")
        return None, None, None, None
    
    s2_ee_date = ee.Date(s2_img.get('system:time_start'))

    s1_img = get_best_s1_near_s2(target_rect, s2_ee_date, min_coverage-5, scale=resolution)

    if s1_img is None:
        print(f"Не найден подходящий S1 снимок, синхронизированный с S2 {s2_date_str}. Используем fallback...")
        s1_coll = (ee.ImageCollection('COPERNICUS/S1_GRD')
                   .filterDate(date_start, date_end)
                   .filterBounds(target_rect)
                   .filter(ee.Filter.eq('instrumentMode', 'IW'))
                   .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')))

        s1_count = s1_coll.size().getInfo()
        if s1_count > 0:
            polars = s1_coll.first().get('transmitterReceiverPolarisation').getInfo()
            if 'VH' in polars:
                s1_img = s1_coll.median().select(['VV', 'VH']).clip(target_rect)
            else:
                s1_img = s1_coll.median().select(['VV']).clip(target_rect)
            print(f"S1 fallback: медиана {s1_count} снимков из окна")
        else:
            print(f"S1: нет данных в поисковом окне")
            s1_img = ee.Image.constant(-25).rename(['VV']).clip(target_rect)

    dem = (ee.ImageCollection('COPERNICUS/DEM/GLO30')
           .mosaic()
           .select('DEM')
           .clip(target_rect)
           .set('system:time_start', s2_img.get('system:time_start')))
    
    return s2_img, s1_img, dem, s2_date_str

def get_worldcover_filter(target_rect, utm_crs, target_date_str, temp_folder):
    worldcover_ee = (ee.ImageCollection("ESA/WorldCover/v200")
                 .first()
                 .select('Map')
                 .clip(target_rect)
                 .reproject(crs=utm_crs, scale=30))

    worldcover_arr, wc_fp = load_image_as_array(
        worldcover_ee, target_rect, 30, utm_crs,
        f"worldcover_{target_date_str}.tif", temp_folder
    )

    if worldcover_arr.ndim > 2:
        worldcover_arr = worldcover_arr.squeeze()

    ALLOWABLE_CLASSES = [30, 40] 
    land_filter = np.isin(worldcover_arr, ALLOWABLE_CLASSES).astype(np.uint8)
    return land_filter, wc_fp

def run_inference_and_create_binary_mask_for_roi(roi_geometry, target_date_str, temp_folder):
    utm_crs = get_utm_for_roi(roi_geometry)

    target_scale = TILE_TARGET_SCALE
    target_size_px = TILE_TARGET_SIZE_PX
    target_size_m = target_size_px * target_scale

    bounds = roi_geometry.bounds(proj=ee.Projection(utm_crs), maxError=1)
    coords = bounds.coordinates().getInfo()[0]
    min_x, min_y = min([c[0] for c in coords]), min([c[1] for c in coords])
    max_x, max_y = max([c[0] for c in coords]), max([c[1] for c in coords])

    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    target_rect = ee.Geometry.Rectangle(
        [center_x - target_size_m/2, center_y - target_size_m/2, center_x + target_size_m/2, center_y + target_size_m/2],
        proj=utm_crs, geodesic=False
    )

    s2_img, s1_img, dem_img, s2_date_found = get_s2_s1_dem_for_target_rect(
        target_rect, target_date_str, resolution=TILE_TARGET_SCALE, min_coverage=99, cloud_max=3
    )

    if s2_img is None or s1_img is None or dem_img is None:
        print(f"Не удалось получить все данные для ROI.")
        return None, []

    s2_final = s2_img.clip(target_rect).reproject(crs=utm_crs, scale=TILE_TARGET_SCALE)
    s1_final = s1_img.clip(target_rect).reproject(crs=utm_crs, scale=TILE_TARGET_SCALE)
    dem_final = dem_img.clip(target_rect).reproject(crs=utm_crs, scale=TILE_TARGET_SCALE)

    s2_arr, s2_fp = load_image_as_array(s2_final, target_rect, TILE_TARGET_SCALE, utm_crs, f"s2_{target_date_str}_s2date_{s2_date_found.replace('-', '')}.tif", temp_folder)
    s1_arr, s1_fp = load_image_as_array(s1_final, target_rect, TILE_TARGET_SCALE, utm_crs, f"s1_{target_date_str}_s2date_{s2_date_found.replace('-', '')}.tif", temp_folder)
    dem_arr, dem_fp = load_image_as_array(dem_final, target_rect, TILE_TARGET_SCALE, utm_crs, f"dem_{target_date_str}_s2date_{s2_date_found.replace('-', '')}.tif", temp_folder)

    h, w = s2_arr.shape[0], s2_arr.shape[1]
    if h != TILE_TARGET_SIZE_PX or w != TILE_TARGET_SIZE_PX:
         print(f"Размер ROI не {TILE_TARGET_SIZE_PX}x{TILE_TARGET_SIZE_PX}: {h}x{w}. Пропуск.")
         return None, [s2_fp, s1_fp, dem_fp]

    print(f"Запуск инференса модели сегментации")
    try:
        prediction_mask = run_inference_on_arrays(s2_arr, s1_arr, dem_arr)
        print(f"Инференс завершён")
    except Exception as e:
        print(f"Ошибка инференса: {e}")
        return None, [s2_fp, s1_fp, dem_fp] 

    mask_template = ee.Image.constant(0).select([0], ['mask']).int8()
    roi_mask_ee = mask_template.paint(roi_geometry, 1).reproject(crs=utm_crs, scale=TILE_TARGET_SCALE)
    roi_mask_arr, roi_mask_fp = load_image_as_array(roi_mask_ee, target_rect, TILE_TARGET_SCALE, utm_crs, f"roi_mask_{target_date_str}.tif", temp_folder)
    if roi_mask_arr.ndim > 2:
        roi_mask_arr = roi_mask_arr.squeeze(axis=-1)

    land_filter, wc_fp = get_worldcover_filter(target_rect, utm_crs, target_date_str, temp_folder)

    binary_drainage_mask = np.where((prediction_mask == 2) & (roi_mask_arr == 1) & (land_filter == 1), 1, 0).astype(np.uint8)

    print(f"Бинарная маска дренажа создана: {binary_drainage_mask.shape}, Пикселей дренажа: {np.sum(binary_drainage_mask)}")

    binary_mask_filename = f"binary_drainage_mask_{target_date_str}_s2date_{s2_date_found.replace('-', '')}.tif"
    binary_mask_filepath = os.path.join(temp_folder, binary_mask_filename)

    coords = target_rect.bounds(proj=ee.Projection(utm_crs), maxError=1).coordinates().getInfo()[0]
    min_x, min_y = min([c[0] for c in coords]), min([c[1] for c in coords])
    height, width = binary_drainage_mask.shape
    max_y = min_y + height * TILE_TARGET_SCALE
    transform = (TILE_TARGET_SCALE, 0.0, min_x, 0.0, -TILE_TARGET_SCALE, max_y)

    with rasterio.open(
        binary_mask_filepath,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=binary_drainage_mask.dtype,
        crs=utm_crs,
        transform=rasterio.transform.Affine(*transform[:6]),
        nodata=255
    ) as dst:
        dst.write(binary_drainage_mask, 1)

    temp_files_to_delete = [s2_fp, s1_fp, dem_fp, roi_mask_fp, wc_fp]

    return binary_mask_filepath, temp_files_to_delete
    
def main():
    shapefile_path = "shapes/shape_file.shp"
    start_date_main = "2018-04-05"
    end_date_main = "2026-04-24"
    target_date_inference = "2023-06-15" 

    roi_polygon = load_shapefile_to_gee(shapefile_path)

    temp_folder = "./temp_inference_process/"
    os.makedirs(temp_folder, exist_ok=True)

    binary_mask_path, temp_files_all = run_inference_and_create_binary_mask_for_roi(
        roi_polygon, target_date_inference, temp_folder
    )

    if binary_mask_path is None:
        print("Процесс инференса не удался")
        shutil.rmtree(temp_folder, ignore_errors=True)
        return

    _ = drainage_test_pipeline_adapted(
        roi_polygon, start_date_main, end_date_main, 
        valid_months=[3,4,5,6,7,8,9,10], max_year_diff=0,
        drainage_mask_path=binary_mask_path,
        output_folder="./drainage_analysis_strelna/"
    )

    for fp in temp_files_all:
        if os.path.exists(fp):
            os.unlink(fp)

    shutil.rmtree(temp_folder, ignore_errors=True)

if __name__ == "__main__":
    main()