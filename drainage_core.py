import os
import rasterio
import requests
import json
import ee
from datetime import datetime, timedelta
import pandas as pd
import math
from utils import *
import numpy as np
import shutil
from rasterio.features import shapes
from rasterio import mask
import pickle
import zipfile
import glob
import contextlib
import cdsapi
import xgboost as xgb

SAVE_RAW_DATA = True
era5_cache = {}
cds = cdsapi.Client()

def load_drainage_model():
    with open('PATH_TO_MODEL', 'rb') as f:
        return pickle.load(f)

def check_available_dates(start_date, end_date, roi, min_coverage=95):

    s2_collection = (ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
                    .filterDate(start_date, end_date)
                    .filterBounds(roi)
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                    .sort('CLOUDY_PIXEL_PERCENTAGE'))
    
    cs_collection = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')\
        .filterDate(start_date, end_date)\
        .filterBounds(roi)
    
    s2_cs = s2_collection.linkCollection(cs_collection, ['cs', 'cs_cdf'])

    def check_s2_cloudscore(img):
        date = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd')
        coverage = covers_full_roi_FAST(img, roi, 30, 'B4')

        cs_min = img.select('cs_cdf').reduceRegion(
            ee.Reducer.min(), roi, scale=100, maxPixels=1e6 
        ).get('cs_cdf')
        
        return ee.Feature(None, {
            'date': date, 
            'coverage': coverage,
            'cs_min': cs_min
        })
    
    results = s2_cs.map(check_s2_cloudscore)
    dates = results.aggregate_array('date').getInfo()
    coverages = results.aggregate_array('coverage').getInfo()
    cs_mins = results.aggregate_array('cs_min').getInfo()

    n = min(len(dates), len(coverages), len(cs_mins))

    s2_good = [
        dates[i] for i in range(n)
        if coverages[i] >= min_coverage and cs_mins[i] >= 0.5
    ]
    print(f"S2 покрытие ROI: {len(s2_good)} дат")

    s1_collection = (ee.ImageCollection('COPERNICUS/S1_GRD')
                    .filterDate(start_date, end_date)
                    .filterBounds(roi)
                    .filter(ee.Filter.eq('instrumentMode', 'IW'))
                    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')))

    valid_s2_dates = []
    for s2_date in s2_good:
        s2_dt = datetime.strptime(s2_date, "%Y-%m-%d")
        s1_ok = False
        for days in [1, 2]:
            s1_window_start = (s2_dt - timedelta(days=days)).strftime('%Y-%m-%d')
            s1_window_end = (s2_dt + timedelta(days=days)).strftime('%Y-%m-%d')
            s1_candidates = s1_collection.filterDate(s1_window_start, s1_window_end)
            
            if s1_candidates.size().getInfo() > 0:
                s1_img = s1_candidates.first()
                if covers_full_roi_FAST(s1_img, roi, 30, 'VV').getInfo() >= min_coverage:
                    s1_ok = True
                    break
        if s1_ok:
            valid_s2_dates.append(s2_date)

    valid_s2_dates = sorted(list(set(valid_s2_dates)), reverse=True)
    print(f"S1+S2 покрытие ROI: {len(valid_s2_dates)} дат")
    return sorted(valid_s2_dates, reverse=True)

def get_soil_drainage_class(roi):
    clay_img = ee.Image("projects/soilgrids-isric/clay_mean").select('clay_0-5cm_mean')
    sand_img = ee.Image("projects/soilgrids-isric/sand_mean").select('sand_0-5cm_mean')
    
    clay_stats = clay_img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=roi, 
        scale=250,   
        maxPixels=1e9,
        bestEffort=True
    ).getInfo()
    
    sand_stats = sand_img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=roi,
        scale=250,
        maxPixels=1e9,
        bestEffort=True
    ).getInfo()
    
    clay_pct = float(clay_stats.get('clay_0-5cm_mean', 150.0)) / 10.0
    sand_pct = float(sand_stats.get('sand_0-5cm_mean', 500.0)) / 10.0
    
    print(f"SoilGrids усреднено по ROI:")
    print(f"clay: {clay_pct:.1f}% ({clay_stats.get('clay_0-5cm_mean', 'N/A')}‰)")
    print(f"sand: {sand_pct:.1f}% ({sand_stats.get('sand_0-5cm_mean', 'N/A')}‰)")
    
    if pd.isna(clay_pct) or pd.isna(sand_pct):
        clay_pct, sand_pct = 15.0, 50.0
    #     soil_type = 'unknown'
    # elif clay_pct < 20:
    #     soil_type = 'sandy'
    # elif clay_pct < 30:
    #     soil_type = 'loamy'
    # else:
    #     soil_type = 'clay'
    soil_type = 'clay'
    
    if soil_type == 'sandy':
        wet_condition = lambda df: df['precip_3d_mm'] > 10.0
        dry_condition = lambda df: (df['precip_1d_mm'] < 1.0) & \
                                 (df['precip_3d_mm'] > 5.0) & (df['precip_3d_mm'] < 10.0)
    elif soil_type == 'loamy':
        wet_condition = lambda df: (df['precip_1d_mm'] > 3.0) & (df['precip_3d_mm'] > 8.0)
        dry_condition = lambda df: (df['precip_1d_mm'] < 1.0) & \
                                 (df['precip_3d_mm'] < 2.0) & (df['precip_7d_mm'] > 10.0)
    else: 
        wet_condition = lambda df: df['precip_3d_mm'] > 9.0
        dry_condition = lambda df: (df['precip_1d_mm'] < 1.0) & (df['precip_3d_mm'] < 2.0) & (df['precip_7d_mm'] > 10.0)
    
    return {
        'soil_type': soil_type,
        'clay_pct': clay_pct,
        'sand_pct': sand_pct,
        'wet_condition': wet_condition,
        'dry_condition': dry_condition
    }

def build_wet_dry_pools(df_met, valid_dates, wet_condition, dry_condition):
    df = df_met.copy()
    df['date'] = pd.to_datetime(df['date'])
    valid_dates_dt = set(pd.to_datetime(d).date() for d in valid_dates)
    df['is_valid_date'] = df['date'].dt.date.isin(valid_dates_dt)
    
    required_cols = ['precip_1d_mm', 'precip_3d_mm', 'precip_7d_mm']
    df_valid = df[df['is_valid_date'] & df[required_cols].notna().all(axis=1)].copy()
    
    wet_pool = df_valid[wet_condition(df_valid)]['date'].dt.date.tolist()
    dry_pool = df_valid[dry_condition(df_valid)]['date'].dt.date.tolist()

    if len(wet_pool) < 1 or len(dry_pool) < 1:
        return [], [], 'no_candidates'
    
    return wet_pool, dry_pool, 'valid'

def is_valid_month(date_obj, valid_months):
    return date_obj.month in valid_months

def priority_chrono_doy_matching_filtered(wet_pool, 
                                        dry_pool, 
                                        valid_months,
                                        max_year_diff=1):
    pairs = []
    used_wet_dates = set()
    
    for wet_date in wet_pool:
        wet_dt = datetime.combine(wet_date, datetime.min.time())
        if wet_date in used_wet_dates or not is_valid_month(wet_dt, valid_months):
            continue
        
        chrono_candidates = []
        for dry_date in dry_pool:
            dry_dt = datetime.combine(dry_date, datetime.min.time())
            days_diff = abs((dry_dt - wet_dt).days)
            
            if 2 <= days_diff <= 14 and is_valid_month(dry_dt, valid_months):
                chrono_candidates.append((dry_date, days_diff))
        
        if chrono_candidates:
            best_dry = min(chrono_candidates, key=lambda x: x[1])[0]
            days_diff = min(chrono_candidates, key=lambda x: x[1])[1]
            pairs.append((wet_date, best_dry, 'chrono_short', days_diff))
            used_wet_dates.add(wet_date)
            continue
        
        wet_doy = wet_date.timetuple().tm_yday
        doy_candidates = []
        
        for dry_date in dry_pool:
            dry_dt = datetime.combine(dry_date, datetime.min.time())
            days_total_diff = abs((dry_dt - wet_dt).days)
            
            year_diff = abs(dry_date.year - wet_date.year)
            if days_total_diff <= 2 or year_diff > max_year_diff:
                continue
            
            dry_doy = dry_date.timetuple().tm_yday
            doy_diff = min(abs(wet_doy - dry_doy), 365 - abs(wet_doy - dry_doy))
            
            if doy_diff <= 14 and is_valid_month(dry_dt, valid_months):
                doy_candidates.append((dry_date, doy_diff))
        
        if doy_candidates:
            best_dry = min(doy_candidates, key=lambda x: x[1])[0]
            doy_diff = min(doy_candidates, key=lambda x: x[1])[1]
            pairs.append((wet_date, best_dry, 'doy', doy_diff))
            used_wet_dates.add(wet_date)
    
    return pairs

def find_top_drainage_pairs(df_met, 
                          valid_dates,
                          wet_condition,
                          dry_condition,
                          valid_months=[4,5,6,7,8,9,10], 
                          top_k=5,
                          max_year_diff=0):
    print(f"Поиск пар (top_k={top_k}, месяцы={valid_months})")
    print(f"Длина valid_dates: {len(valid_dates)}")
    wet_pool, dry_pool, status = build_wet_dry_pools(df_met, valid_dates, wet_condition, dry_condition)
    
    if status == 'invalid' or not wet_pool or not dry_pool:
        print(f"Пулы пусты: wet={len(wet_pool)}, dry={len(dry_pool)}")
        return None
    
    print(f"Пулы: wet={len(wet_pool)}, dry={len(dry_pool)}")
    
    raw_pairs = priority_chrono_doy_matching_filtered(
        wet_pool, dry_pool, valid_months, max_year_diff
    )
    
    if not raw_pairs:
        print("Нет подходящих пар (chrono+doy)")
        return None
    
    print(f"Найдено пар: {len(raw_pairs)}")
    
    formatted_pairs = []
    for wet_date, dry_date, method, score in raw_pairs[:top_k * 2]: 
        try:
            if wet_date is None or dry_date is None:
                continue
            days_gap = abs((datetime.combine(dry_date, datetime.min.time()) - 
                        datetime.combine(wet_date, datetime.min.time())).days)
            
            wet_row = df_met[df_met['date'].dt.strftime('%Y-%m-%d') == wet_date.strftime('%Y-%m-%d')].iloc[0]
            dry_row = df_met[df_met['date'].dt.strftime('%Y-%m-%d') == dry_date.strftime('%Y-%m-%d')].iloc[0]
            
            formatted_pairs.append({
                'wet_date': wet_date.strftime('%Y-%m-%d'),
                'dry_date': dry_date.strftime('%Y-%m-%d'),
                'method': method,
                'days_gap': days_gap,
                'score': score,
                'wet_precip_3d': wet_row['precip_3d_mm'],
                'dry_precip_3d': dry_row['precip_3d_mm'], 
                'wet_precip_1d': wet_row['precip_1d_mm'],
                'dry_precip_1d': dry_row['precip_1d_mm']
            })
        except Exception:
            continue

    if len(formatted_pairs) == 0:
        print("formatted_pairs пуст")
        return None
    
    top_pairs = sorted(formatted_pairs, 
                      key=lambda p: (2 if p['method']=='chrono_short' else 1 if p['method']=='doy' else 0, 
                                    -p['days_gap'], p['score']),
                      reverse=True)[:top_k]
    
    result = {
        'top_pairs': top_pairs,
        'pairs_found': len(raw_pairs),
        'top_k_used': len(top_pairs),
        'wet_pool_size': len(wet_pool),
        'dry_pool_size': len(dry_pool)
    }
    
    return result

def load_s1_s2_pair(date_str, roi, target_scale=30, min_coverage=90):
    target_dt = datetime.strptime(date_str, "%Y-%m-%d")
    
    s2_start = (target_dt - timedelta(days=1)).strftime('%Y-%m-%d')
    s2_end = (target_dt + timedelta(days=1)).strftime('%Y-%m-%d')

    utm_crs = get_utm_for_roi(roi)

    def resample_to_target_scale(image):
        return image.resample('bilinear').reproject(
            crs=utm_crs,
            scale=target_scale 
        )
    
    s2_collection = (ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
                    .filterDate(s2_start, s2_end)
                    .filterBounds(roi)
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                    .select(['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12'])
                    .map(resample_to_target_scale))
    
    cs_collection = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')\
        .filterDate(s2_start, s2_end)\
        .filterBounds(roi)
    
    s2_cs = s2_collection.linkCollection(cs_collection, ['cs', 'cs_cdf'])

    def check_s2_cloudscore(img):
        coverage = covers_full_roi_FAST(img, roi, 30, 'B4')
        cs_min = img.select('cs_cdf').reduceRegion(
            ee.Reducer.min(), roi, scale=100, maxPixels=1e6 
        ).get('cs_cdf')
        return ee.Feature(None, {
            'img_id': img.get('system:index'), 
            'coverage': coverage,
            'cs_min': cs_min
        })
    
    results = s2_cs.map(check_s2_cloudscore)
    img_ids = results.aggregate_array('img_id').getInfo()
    coverages = results.aggregate_array('coverage').getInfo()
    cs_mins = results.aggregate_array('cs_min').getInfo()
    best_s2_idx = None
    for i, (cov, cs_min) in enumerate(zip(coverages, cs_mins)):
        if cov >= min_coverage and cs_min >= 0.5:
            best_s2_idx = i
            break

    if best_s2_idx is None:
        print(f"Нет S2 cs_min≥0.5 для {date_str}")
        return None, None
    
    best_img_id = img_ids[best_s2_idx]
    s2_img = (ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
             .filter(ee.Filter.eq('system:index', best_img_id))
             .map(resample_to_target_scale)
             .first())
    
    def add_s2_indices(image):
        ndvi = image.normalizedDifference(['B8','B4']).rename('sentinel2_NDVI')
        ndmi = image.normalizedDifference(['B8','B11']).rename('sentinel2_NDMI')
        band_names = image.bandNames().map(lambda name: ee.String('sentinel2_').cat(name))
        image = image.rename(band_names)
        return image.addBands([ndvi, ndmi])
    
    s2_img = add_s2_indices(s2_img)
    
    s1_start = (target_dt - timedelta(days=2)).strftime('%Y-%m-%d')
    s1_end = (target_dt + timedelta(days=2)).strftime('%Y-%m-%d')
    
    s1_collection = (ee.ImageCollection('COPERNICUS/S1_GRD')
                    .filterDate(s1_start, s1_end)
                    .filterBounds(roi)
                    .filter(ee.Filter.eq('instrumentMode', 'IW'))
                    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
                    .select(['VV', 'VH', 'angle']))
    
    s1_img = get_closest_s1_with_coverage(date_str, s1_collection, roi, target_scale)
    
    return s1_img, s2_img

def get_closest_s1_with_coverage(target_date_str, collection, roi_geometry, scale=30, band_name='VV'):
    target_date = ee.Date(target_date_str)
    
    def add_temporal_diff(image):
        diff = image.date().difference(target_date, 'day').abs()
        return image.set('temporal_diff', diff)
    
    collection_with_temp_diff = collection.map(add_temporal_diff)
    
    def check_full_coverage(image):
        mask = image.select(band_name).mask()
        coverage_check = mask.reduceRegion(
            reducer=ee.Reducer.allNonZero(),
            geometry=roi_geometry,
            scale=scale,
            maxPixels=1e9,
            tileScale=4,
            bestEffort=True
        )
        is_fully_covered = ee.Number(coverage_check.get(band_name, 0))
        return image.set('is_fully_covered', is_fully_covered)
    
    collection_with_coverage_flag = collection_with_temp_diff.map(check_full_coverage)
    fully_covered_collection = collection_with_coverage_flag.filter(ee.Filter.eq('is_fully_covered', 1))
    sorted_collection = fully_covered_collection.sort('temporal_diff')
    return sorted_collection.first()

def apply_wcm_sentinel1(image, ndvi_img):
    ndvi_eff = ndvi_img.clamp(0.0, 0.8)
    vv = image.select('VV')
    vh = image.select('VH')
    angle = image.select('angle')
    
    theta_rad = angle.multiply(math.pi / 180)
    cos_theta = theta_rad.cos()
    sec_theta = cos_theta.pow(-1)
    
    A_vv = ndvi_eff.multiply(0.12)
    B_vv = ndvi_eff.multiply(0.70)
    tau2_vv = B_vv.multiply(ndvi_eff).multiply(sec_theta).multiply(-2).exp()
    sigma_veg_vv = A_vv.multiply(ndvi_eff).multiply(cos_theta).multiply(tau2_vv.subtract(1).multiply(-1))
    vv_lin = vv.divide(10).exp()
    sigma_soil_vv = vv_lin.subtract(sigma_veg_vv).divide(tau2_vv)
    vv_soil_db = sigma_soil_vv.log10().multiply(10).rename('VV_soil')
     
    A_vh = ndvi_eff.multiply(0.05)
    B_vh = ndvi_eff.multiply(1.45)
    tau2_vh = B_vh.multiply(ndvi_eff).multiply(sec_theta).multiply(-2).exp()
    sigma_veg_vh = A_vh.multiply(ndvi_eff).multiply(cos_theta).multiply(tau2_vh.subtract(1).multiply(-1))
    vh_lin = vh.divide(10).exp()
    sigma_soil_vh = vh_lin.subtract(sigma_veg_vh).divide(tau2_vh)
    vh_soil_db = sigma_soil_vh.log10().multiply(10).rename('VH_soil')
    
    return image.addBands([vv_soil_db, vh_soil_db])

def get_era5_features(df_era5, target_date_str, lag_days=[1,3,7]):
    if df_era5.empty or 'valid_time' not in df_era5.columns:
        return {k: np.nan for k in 
               ['precip_1d_mm', 'precip_3d_mm', 'precip_7d_mm', 
                'era5_swvl1', 't2m_C', 'skin_temp_C', 'soil_temp_l1_C', 
                'wind_speed_ms', 'ssrd_Jm2_day', 'd2m_C']}

    target_date = pd.to_datetime(target_date_str).date()
    df_sorted = df_era5.sort_values('valid_time')
    
    if 'date_only' not in df_sorted.columns:
        df_sorted['date_only'] = df_sorted['valid_time'].dt.date
    
    date_mask = df_sorted['date_only'] == target_date
    if not date_mask.any():
        print(f"Нет данных ERA5 за {target_date_str}")
        return {k: np.nan for k in 
               ['precip_1d_mm', 'precip_3d_mm', 'precip_7d_mm', 
                'era5_swvl1', 't2m_C', 'skin_temp_C', 'soil_temp_l1_C', 
                'wind_speed_ms', 'ssrd_Jm2_day', 'd2m_C']}
    
    day_data = df_sorted[date_mask].copy()
    
    end_idx = df_sorted[df_sorted['date_only'] <= target_date].index[-1]
    lags = {}
    for days in lag_days:
        hours_back = days * 24
        start_idx = max(0, end_idx - hours_back + 1)
        tp_period = df_sorted.iloc[start_idx:end_idx+1]['tp'].sum()
        lags[f'precip_{days}d_mm'] = float(tp_period * 1000)
    
    ssrd_day = day_data['ssrd'].sum() if 'ssrd' in day_data.columns else 0.0
    tp_day = day_data['tp'].sum() if 'tp' in day_data.columns else 0.0
    
    t2m_mean = day_data['t2m'].mean() if 't2m' in day_data.columns else np.nan
    d2m_mean = day_data['d2m'].mean() if 'd2m' in day_data.columns else np.nan
    skt_mean = day_data['skt'].mean() if 'skt' in day_data.columns else np.nan
    stl1_mean = day_data['stl1'].mean() if 'stl1' in day_data.columns else np.nan
    swvl1_mean = day_data['swvl1'].mean() if 'swvl1' in day_data.columns else np.nan
    
    if 'u10' in day_data.columns and 'v10' in day_data.columns:
        wind_speed = np.sqrt(day_data['u10']**2 + day_data['v10']**2)
        wind_mean = wind_speed.mean()
    else:
        wind_mean = np.nan
    
    features = {
        **lags,
        'precip_1d_mm': float(tp_day * 1000), 
        'era5_swvl1': float(swvl1_mean),
        't2m_C': float(t2m_mean - 273.15),
        'd2m_C': float(d2m_mean - 273.15),
        'skin_temp_C': float(skt_mean - 273.15),
        'soil_temp_l1_C': float(stl1_mean - 273.15),
        'wind_speed_ms': float(wind_mean),
        'ssrd_Jm2_day': float(ssrd_day)
    }
    
    return features

def get_dsm(roi, target_scale=30):
    utm_crs = get_utm_for_roi(roi)
    def resample_to_target_scale(image):
        return image.resample('bilinear').reproject(
            crs=utm_crs,
            scale=target_scale
        )
    
    alos_dsm_col = ee.ImageCollection("JAXA/ALOS/AW3D30/V3_2") \
        .filterBounds(roi) \
        .select('DSM') \
        .map(resample_to_target_scale)
    
    alos_dsm = alos_dsm_col.median()
    alos_dsm_available = alos_dsm_col.size().gt(0)
    
    srtm_dem = ee.Image("USGS/SRTMGL1_003").select('elevation')\
        .rename('DSM')\
        .resample('bilinear')\
        .reproject(crs=utm_crs, scale=target_scale)
    
    return ee.Image(ee.Algorithms.If(alos_dsm_available, alos_dsm, srtm_dem))

def selective_normalize(image):
    min_max_dict = {
        'VV_soil': (-54.837955, 24.112611),
        'VH_soil': (-56.586738, 24.244538),
        'DSM': (-5.0, 3690.0)
    }
    
    bands_10000 = [
        'sentinel2_B1', 'sentinel2_B2', 'sentinel2_B3', 'sentinel2_B4', 'sentinel2_B5',
        'sentinel2_B6', 'sentinel2_B7', 'sentinel2_B8', 'sentinel2_B8A', 'sentinel2_B9',
        'sentinel2_B10', 'sentinel2_B11', 'sentinel2_B12'
    ]
    
    bands_1000 = [
        'latitude', 'longitude'
    ]
    
    normalized_bands = []
    
    for band, (band_min, band_max) in min_max_dict.items():
        norm = image.select(band).subtract(band_min).divide(band_max - band_min).multiply(2).subtract(1)
        normalized_bands.append(norm.rename(band))
    
    for band in bands_10000:
        if band in image.bandNames().getInfo():
            scaled = image.select(band).divide(10000)
            normalized_bands.append(scaled.rename(band))
    
    for band in bands_1000:
        if band in image.bandNames().getInfo():
            scaled = image.select(band).divide(1000)
            normalized_bands.append(scaled.rename(band))
    
    all_bands = image.bandNames().getInfo()
    normalized_names = list(min_max_dict.keys()) + bands_10000 + bands_1000
    remaining_bands = [b for b in all_bands if b not in normalized_names]
    
    for band in remaining_bands:
        normalized_bands.append(image.select(band))
    
    return ee.Image.cat(normalized_bands)

def predict_from_image(image, model_bundle, feature_order_model, scale=30, roi=None):
    if roi is None:
        roi = image.geometry()

    # print(f"predict_from_image: scale={scale}, ROI={roi.area().divide(1e6).getInfo():.3f}км²")
    # print(f"image bands: {image.bandNames().getInfo()}")
    # print(f"feature_order_model: {len(feature_order_model)} фич")

    feature_samples = image.sample(
        region=roi, scale=scale, dropNulls=False
    )
    sample_count = feature_samples.size().getInfo()
    # print(f"sample_count: {sample_count}")

    image_bands = image.bandNames().getInfo()
    valid_features = [f for f in feature_order_model if f in image_bands]
    # print(f"valid_features: {len(valid_features)}/{len(feature_order_model)}")
    
    feature_list = feature_samples.reduceColumns(
        ee.Reducer.toList(len(feature_order_model)), feature_order_model
    ).get('list').getInfo()
    # print(f"len_feature_list: {len(feature_list)}")
    
    tabular_data = pd.DataFrame(feature_list, columns=feature_order_model)
    
    X = tabular_data[feature_order_model].fillna(0)
    X_scaled = model_bundle['scaler'].transform(X)
    dmatrix = xgb.DMatrix(X_scaled, feature_names=feature_order_model)
    predictions = model_bundle['model'].predict(dmatrix)
    tabular_data['prediction'] = predictions
    
    tabular_data['lat_10m'] = np.round(tabular_data['latitude'] * 1000, 5)
    tabular_data['lon_10m'] = np.round(tabular_data['longitude'] * 1000, 5)
    
    grouped = tabular_data.groupby(['lat_10m', 'lon_10m'], as_index=False)['prediction'].mean()
    
    prediction_features = []
    for _, row in grouped.iterrows():
        point = ee.Geometry.Point([row['lon_10m'], row['lat_10m']])
        feature = ee.Feature(point, {'prediction': row['prediction']})
        prediction_features.append(feature)

    utm_crs = get_utm_for_roi(roi)
    prediction_fc = ee.FeatureCollection(prediction_features)
    prediction_ee = prediction_fc.reduceToImage(
        properties=['prediction'], reducer=ee.Reducer.mean()
    ).reproject(crs=utm_crs, scale=scale)
    
    kernel_radius = int(2000 / scale)
    gaussian_kernel = ee.Kernel.gaussian(radius=kernel_radius, sigma=1)
    smoothed = prediction_ee.convolve(gaussian_kernel)
    
    return smoothed

def generate_delta_image(wet_date, dry_date, roi, lon_lat, df_era5, soil_props, model_bundle, feature_order_model, target_scale=30):
    print(f"delta-карта: {wet_date} (wet) → {dry_date} (dry)")
    
    wet_s1, wet_s2 = load_s1_s2_pair(wet_date, roi, target_scale)
    if wet_s1 is None or wet_s2 is None:
        print(f"Не удалось загрузить S1/S2 для {wet_date}")
        return None
    wet_s1_corrected = apply_wcm_sentinel1(wet_s1, wet_s2.select('sentinel2_NDVI'))
    wet_full = wet_s1_corrected.addBands([lon_lat, wet_s2, get_dsm(roi)]).clip(roi)
    wet_normalized = selective_normalize(wet_full)
    
    dry_s1, dry_s2 = load_s1_s2_pair(dry_date, roi, target_scale)
    if dry_s1 is None or dry_s2 is None:
        print(f"Не удалось загрузить S1/S2 для {dry_date}")
        return None
    dry_s1_corrected = apply_wcm_sentinel1(dry_s1, dry_s2.select('sentinel2_NDVI'))
    dry_full = dry_s1_corrected.addBands([lon_lat, dry_s2, get_dsm(roi)]).clip(roi)
    dry_normalized = selective_normalize(dry_full)
    
    delta_bands = [
        ('VV_soil', 'delta_VV_soil'),
        ('VH_soil', 'delta_VH_soil'),
        ('VV', 'delta_VV'),
        ('VH', 'delta_VH'),
        
        ('sentinel2_NDVI', 'delta_NDVI'),
        ('sentinel2_NDMI', 'delta_NDMI'),
        ('sentinel2_B2', 'delta_B2'), ('sentinel2_B3', 'delta_B3'),
        ('sentinel2_B4', 'delta_B4'), ('sentinel2_B5', 'delta_B5'),
        ('sentinel2_B6', 'delta_B6'), ('sentinel2_B7', 'delta_B7'),
        ('sentinel2_B8', 'delta_B8'), ('sentinel2_B8A', 'delta_B8A'),
        ('sentinel2_B11', 'delta_B11'), ('sentinel2_B12', 'delta_B12'),
    ]
    
    delta_image_list = []
    for orig_band, delta_name in delta_bands:
        if orig_band in dry_normalized.bandNames().getInfo():
            delta = dry_normalized.select(orig_band).subtract(wet_normalized.select(orig_band))
            delta_image_list.append(delta.rename(delta_name))
    
    static_bands = ['latitude', 'longitude', 'DSM']
    for band in static_bands:
        if band in dry_normalized.bandNames().getInfo():
            delta_image_list.append(dry_normalized.select(band))
    
    delta_image = ee.Image.cat(delta_image_list).clip(roi)
    
    days_gap = abs((pd.to_datetime(dry_date) - pd.to_datetime(wet_date)).days)
    wet_era5 = get_era5_features(df_era5, wet_date)
    dry_era5 = get_era5_features(df_era5, dry_date)
    
    era5_bands = [
        ('clay_pct', soil_props['clay_pct']),
        ('sand_pct', soil_props['sand_pct']),
        ('precip_wet_1d', wet_era5['precip_1d_mm']),
        ('precip_wet_3d', wet_era5['precip_3d_mm']),
        ('precip_wet_7d', wet_era5['precip_7d_mm']),
        ('era5_swvl1_wet', wet_era5['era5_swvl1']),
        ('t2m_wet', wet_era5['t2m_C']),
        ('t2m_dry', dry_era5['t2m_C']),
        ('wind_speed_dry', dry_era5['wind_speed_ms']),
        ('ssrd_dry', dry_era5['ssrd_Jm2_day']),
        ('soil_temp_l1_dry', dry_era5['soil_temp_l1_C']),
        ('days_gap', days_gap)
    ]
    
    for band_name, value in era5_bands:
        delta_image = delta_image.addBands(ee.Image.constant(value).rename(band_name))
    
    delta_drainage = predict_from_image(delta_image, model_bundle, model_bundle['feature_columns'], target_scale, roi)
    
    return {
        'delta_drainage': delta_drainage,
        'wet_date': wet_date,
        'dry_date': dry_date,
        'days_gap': days_gap,
        'wet_era5': wet_era5,
        'dry_era5': dry_era5,
        'delta_image': delta_image
    }

def compare_moisture_maps_topk(top_pairs_result, roi, lon_lat, 
                               df_era5, soil_props, model_bundle):
    if not top_pairs_result or not top_pairs_result.get('top_pairs'):
        raise ValueError(f"Нет пар для обработки: {top_pairs_result}")
    
    all_delta_maps = []
    pair_details = []
    
    for i, pair in enumerate(top_pairs_result['top_pairs'], 1):
        print(f"{i}/{len(top_pairs_result['top_pairs'])}: {pair['wet_date']} → {pair['dry_date']} "
              f"({pair['days_gap']}д, {pair['method']})")
        
        delta_result = generate_delta_image(
            pair['wet_date'], pair['dry_date'], roi, lon_lat, 
            df_era5, soil_props, model_bundle, model_bundle['feature_columns'],
            target_scale=30
        )
        if delta_result and delta_result['delta_drainage']:
            all_delta_maps.append(delta_result['delta_drainage'])
            pair_details.append({
                'wet_date': pair['wet_date'],
                'dry_date': pair['dry_date'],
                'days_gap': pair['days_gap'],
                'method': pair['method'],
                'n_pixels': delta_result['delta_drainage'].reduceRegion(
                    ee.Reducer.count(), roi, 30
                ).getInfo()
            })
    
    mean_delta_moisture = (ee.ImageCollection(all_delta_maps)
                      .select('mean')      
                      .mean()
                      .rename('prediction_mean'))
    
    stats = mean_delta_moisture.reduceRegion(
        reducer=ee.Reducer.mean(), 
        geometry=roi,
        scale=30,
        maxPixels=1e9
    ).getInfo()

    return {
        'mean_delta_moisture': mean_delta_moisture,
        'n_pairs': len(all_delta_maps),
        'pair_details': pair_details,
        'individual_delta_maps': all_delta_maps,
        'stats': stats,  
        'top_pairs_result': top_pairs_result
    }

def download_raw_delta_maps(result, roi, prefix, folder="./drainage_analysis/", scale=30, SAVE_RAW_DATA=True):
    os.makedirs(folder, exist_ok=True)
    region_coords = roi.bounds().coordinates().getInfo()[0]
    
    downloaded_maps = {}
    raw_files = []
    
    print(f"Скачиваем усредненную delta-карту ({result['n_pairs']} пар)")
    utm_crs = get_utm_for_roi(roi)
    url_mean = result['mean_delta_moisture'].getDownloadURL({
        'region': region_coords, 
        'scale': scale, 
        'crs': utm_crs, 
        'format': 'GEO_TIFF'
    })
    
    mean_filename = f"{prefix}_delta_mean_{result['n_pairs']}pairs.tif"
    mean_filepath = os.path.join(folder, mean_filename)
    
    r = requests.get(url_mean)
    r.raise_for_status()
    with open(mean_filepath, 'wb') as f:
        f.write(r.content)
    
    raw_files.append(mean_filepath)
    
    with rasterio.open(mean_filepath) as src:
        downloaded_maps['mean_delta'] = {
            'data': src.read(1),
            'transform': src.transform,
            'filename': mean_filename,
            'stats': {
                'mean': float(np.nanmean(src.read(1))),
                'std': float(np.nanstd(src.read(1))),
                'min': float(np.nanmin(src.read(1))),
                'max': float(np.nanmax(src.read(1)))
            }
        }
    
    if SAVE_RAW_DATA and result.get('individual_delta_maps'):
        print(f"Скачиваем {len(result['individual_delta_maps'])} отдельных delta-карт")
        
        for i, (delta_map, pair_info) in enumerate(zip(result['individual_delta_maps'], result['pair_details']), 1):
            pair_prefix = f"{prefix}_pair_{i:02d}_{pair_info['wet_date'][:10].replace('-','')}_{pair_info['dry_date'][:10].replace('-','')}"
            utm_crs = get_utm_for_roi(roi)
            url_pair = delta_map.getDownloadURL({
                'region': region_coords, 
                'scale': scale, 
                'crs': utm_crs, 
                'format': 'GEO_TIFF'
            })
            
            pair_filename = f"{pair_prefix}_delta.tif"
            pair_filepath = os.path.join(folder, pair_filename)
            
            r = requests.get(url_pair)
            r.raise_for_status()
            with open(pair_filepath, 'wb') as f:
                f.write(r.content)
            
            raw_files.append(pair_filepath)
            
            with rasterio.open(pair_filepath) as src:
                downloaded_maps[f'pair_{i}'] = {
                    'data': src.read(1),
                    'transform': src.transform,
                    'filename': pair_filename,
                    'wet_date': pair_info['wet_date'],
                    'dry_date': pair_info['dry_date'],
                    'days_gap': pair_info['days_gap'],
                    'stats': {
                        'mean': float(np.nanmean(src.read(1))),
                        'std': float(np.nanstd(src.read(1))),
                        'min': float(np.nanmin(src.read(1))), 
                        'max': float(np.nanmax(src.read(1)))
                    }
                }
    
    metadata = {
        'prefix': prefix,
        'n_pairs': result['n_pairs'],
        'mean_delta_stats': downloaded_maps['mean_delta']['stats'],
        'pair_details': result['pair_details'],
        'top_pairs_info': result['top_pairs_result']
    }
    
    metadata_filename = f"{prefix}_delta_metadata.json"
    metadata_filepath = os.path.join(folder, metadata_filename)
    with open(metadata_filepath, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    if not SAVE_RAW_DATA:
        for raw_file in raw_files:
            if os.path.exists(raw_file):
                os.remove(raw_file)
        raw_files = []
    
    return {
        'maps': downloaded_maps,
        'raw_files': raw_files,
        'metadata_file': metadata_filepath,
        'folder': folder
    }

def apply_drainage_mask_local(moisture_maps, drainage_mask_path, output_folder, prefix, target_crs=None):
    os.makedirs(output_folder, exist_ok=True)
    temp_files = [] 
    
    with rasterio.open(drainage_mask_path) as src:
        image = src.read(1).astype('uint8')
        shapes_gen = shapes(image, mask=(image==1), transform=src.transform)
        geoms = [{'properties': {'raster_val': v}, 'geometry': s} for i, (s, v) in enumerate(shapes_gen)]
    
    gdf = gpd.GeoDataFrame.from_features(geoms, crs=src.crs)
    if target_crs is not None:
        drainage_vector = gdf.to_crs(target_crs)
    else:
        drainage_vector = gdf

    drainage_vector['geometry'] = drainage_vector['geometry'].buffer(0)
    
    result_paths = {}
    
    for map_name, map_info in moisture_maps.items():
        temp_tif = f"temp_{map_name}.tif"
        temp_files.append(temp_tif)

        map_crs = map_info.get('crs', str(drainage_vector.crs))
        
        with rasterio.open(temp_tif, 'w', driver='GTiff', height=map_info['data'].shape[0], 
                          width=map_info['data'].shape[1], count=1, dtype=map_info['data'].dtype,
                          crs=map_crs, transform=map_info['transform']) as dst:
            dst.write(map_info['data'], 1)
        
        out_filename = f"{prefix}_{map_name}_drainage_masked.tif"
        out_masked = os.path.join(output_folder, out_filename)
        
        try:
            with rasterio.open(temp_tif) as src_map:
                drainage_for_raster = drainage_vector.to_crs(src_map.crs)
                out_image, out_transform = rasterio.mask.mask(
                    src_map, drainage_for_raster.geometry, 
                    crop=True, nodata=np.nan
                )
            
            if out_image.ndim == 3 and out_image.shape[0] == 1:
                out_image = out_image[0]
            
            out_meta = src_map.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[0],
                "width": out_image.shape[1],
                "transform": out_transform,
                "crs": str(src_map.crs)
            })
            
            with rasterio.open(out_masked, 'w', **out_meta) as dest:
                dest.write(out_image, 1)
            
            result_paths[map_name] = out_masked
            
        except Exception as e:
            print(f"Ошибка маскировки {map_name}: {e}")
            shutil.copy(temp_tif, out_masked)
            result_paths[map_name] = out_masked
        
        finally:
            if os.path.exists(temp_tif):
                os.remove(temp_tif)
                if temp_tif in temp_files:
                    temp_files.remove(temp_tif)
    
    main_result = result_paths['mean_delta']
    
    individual_maps = {k: v for k, v in result_paths.items() if k != 'mean_delta'}
    if individual_maps and not SAVE_RAW_DATA:
        for map_name, map_path in individual_maps.items():
            if os.path.exists(map_path):
                os.remove(map_path)
        result_paths = {'mean_delta': main_result}
    
    return result_paths

def merge_era5_timeseries_csv(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("temp_era5_csv")
    
    csv_files = glob.glob("temp_era5_csv/*.csv")
    target_vars = ['d2m', 't2m', 'tp', 'ssrd', 'skt', 'stl1', 'swvl1', 'u10', 'v10']
    dfs = {}
    
    for csv_file in csv_files:
        df_temp = pd.read_csv(csv_file, parse_dates=['valid_time'])
        var_cols = [col for col in df_temp.columns 
                   if col in target_vars and col not in ['valid_time', 'latitude', 'longitude']]
        
        for var_name in var_cols:
            df_temp[var_name] = df_temp[var_name].astype(float)
            dfs[var_name] = df_temp[['valid_time', var_name]].drop_duplicates('valid_time')
    
    required = ['t2m', 'tp', 'swvl1']
    missing = [v for v in required if v not in dfs]
    if missing:
        print(f"ОТСУТСТВУЮТ ERA5 переменные: {missing}")
    
    if not dfs:
        raise ValueError("Не найдено ни одной ERA5 переменной!")
    
    first_var = list(dfs.keys())[0]
    merged_df = dfs[first_var][['valid_time']].copy()
    
    for var_name, df_var in dfs.items():
        merged_df = pd.merge(merged_df, df_var, on='valid_time', how='outer')
    
    merged_df = merged_df.sort_values('valid_time').reset_index(drop=True)
    merged_df['date_only'] = merged_df['valid_time'].dt.date
    
    shutil.rmtree("temp_era5_csv")
    return merged_df

def download_and_process_era5_timeseries_cached(lat, lon, start_date, end_date):
    key = f"{lat:.4f}_{lon:.4f}_{start_date}_{end_date}"
    if key in era5_cache and len(era5_cache[key]) > 10:
        return era5_cache[key]
    
    result = download_and_process_era5_timeseries(lat, lon, start_date, end_date)
    if not result.empty:
        era5_cache[key] = result
    return result

def download_and_process_era5_timeseries(lat, lon, start_date, end_date, max_retries=3):
    for attempt in range(max_retries):
        zip_path = f"era5_tmp_{lat:.1f}_{lon:.1f}_{attempt}.zip"
        try:
            request = {
                "variable": [
                    "2m_dewpoint_temperature", "2m_temperature",
                    "total_precipitation", "surface_solar_radiation_downwards",
                    "skin_temperature", "soil_temperature_level_1",
                    "volumetric_soil_water_level_1",
                    "10m_u_component_of_wind", "10m_v_component_of_wind"
                ],
                "location": {"longitude": lon, "latitude": lat},
                "date": f"{start_date}/{end_date}",
                "data_format": "csv"
            }

            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    cds.retrieve("reanalysis-era5-land-timeseries", request).download(zip_path)

            if not os.path.exists(zip_path):
                continue

            df_era5 = merge_era5_timeseries_csv(zip_path)
            os.remove(zip_path)
            return df_era5

        except Exception as e:
            print(f"Попытка {attempt+1} провалилась: {str(e)[:100]}")
            if os.path.exists(zip_path):
                os.remove(zip_path)

    print(f"НЕ УДАЛОСЬ получить ERA5 для {lat:.1f},{lon:.1f}")
    return pd.DataFrame()

def drainage_test_pipeline_adapted(original_roi_polygon, start_date, end_date, 
                                   valid_months=[3,4,5,6,7,8,9,10], max_year_diff=0,
                                   status=None,
                                drainage_mask_path=None,
                                output_folder="./drainage_analysis/"):
    if status:
        status.write("Шаг 1: загрузка модели...")
    model_bundle = load_drainage_model()
    
    analysis_roi = original_roi_polygon
    if status:
        status.write("Шаг 2: поиск доступных дат...")
    print("Поиск доступных дат S1+S2...")
    valid_dates = check_available_dates(start_date, end_date, analysis_roi)
    print(f"Найдено {len(valid_dates)} дат")
    
    if len(valid_dates) < 5:
        return {"error": f"Недостаточно снимков для анализа: {len(valid_dates)} < 5"}
    if status:
        status.write("Шаг 3: анализ типа почвы...")
    print("Анализ типа почвы...")
    soil_info = get_soil_drainage_class(analysis_roi)
    print(f"Тип почвы: {soil_info['soil_type']}")
    print(f"Глина: {soil_info['clay_pct']:.1f}%, Песок: {soil_info['sand_pct']:.1f}%")
    
    print("Загрузка ERA5...")
    centroid = analysis_roi.centroid().coordinates().getInfo()
    lon, lat = centroid[0], centroid[1]
    print(f"Центроид: {lat:.4f}°, {lon:.4f}°")
    if status:
        status.write("Шаг 4: загрузка ERA5 данных...")
    min_date = (pd.to_datetime(start_date) - pd.Timedelta(days=7)).strftime('%Y-%m-%d')
    df_era5 = download_and_process_era5_timeseries_cached(lat, lon, min_date, end_date)
    
    if df_era5.empty:
        return {"error": "Не удалось получить ERA5 данные"}
    
    print(f"ERA5: {len(df_era5)} записей ({df_era5['valid_time'].min().date()} → {df_era5['valid_time'].max().date()})")
    
    df_met = df_era5[['valid_time', 'tp']].copy()
    df_met['date'] = df_met['valid_time'].dt.date

    daily_precip = df_met.groupby('date')['tp'].sum().reset_index()
    daily_precip['precip_1d_mm'] = daily_precip['tp'] * 1000  
    daily_precip['date'] = pd.to_datetime(daily_precip['date'])

    df_met = daily_precip[['date', 'precip_1d_mm']].copy()
    df_met['precip_3d_mm'] = df_met['precip_1d_mm'].rolling(3, min_periods=1).sum()
    df_met['precip_7d_mm'] = df_met['precip_1d_mm'].rolling(7, min_periods=1).sum()
    df_met['valid_time'] = df_met['date']

    print(f"Длина df_met: {len(df_met)}")
    if status:
        status.write("Шаг 5: поиск пар дат...")
    print("Поиск ТОП wet/dry пар...")
    top_pairs_result = find_top_drainage_pairs(
        df_met, valid_dates, 
        soil_info['wet_condition'], soil_info['dry_condition'],
        valid_months=valid_months, 
        top_k=25,
        max_year_diff=max_year_diff
    )
    
    if not top_pairs_result or not top_pairs_result.get('top_pairs'):
        return {"error": "Подходящих wet/dry пар не найдено"}
    
    print(f"Найдено {len(top_pairs_result['top_pairs'])} ТОП-пар:")
    for i, pair in enumerate(top_pairs_result['top_pairs'], 1):
        print(f"{i}. {pair['wet_date']} ({pair['wet_precip_3d']:.0f}mm) → "
              f"{pair['dry_date']} (+{pair['days_gap']}д, score={pair['score']:.3f})")
    
    print("Формирование delta-карт по ТОП-парам...")
    lon_lat = ee.Image.pixelLonLat().select(['longitude', 'latitude'])
    if status:
        status.write("Шаг 6: формирование delta-карт...")
    comparison_result = compare_moisture_maps_topk(
        top_pairs_result, analysis_roi, lon_lat, 
        df_era5, soil_info, model_bundle
    )
    
    if status:
        status.write("Шаг 7: скачивание delta-карт...")
    prefix = f"delta_{start_date.replace('-','')}_to_{end_date.replace('-','')}"
    print("Скачивание delta-карт...")
    raw_maps_data = download_raw_delta_maps(
        comparison_result, analysis_roi, prefix,
        folder=output_folder, 
        SAVE_RAW_DATA=True 
    )
    
    if status:
        status.write("Шаг 8: применение маски дренажа...")
    final_maps = {}
    utm_crs = get_utm_for_roi(analysis_roi)
    if drainage_mask_path and os.path.exists(drainage_mask_path):
        print("Применение маски дренажа...")
        masked_paths = apply_drainage_mask_local(
            raw_maps_data['maps'], drainage_mask_path,
            os.path.join(output_folder, 'masked'),
            prefix,
            target_crs=utm_crs
        )
        final_maps = masked_paths
    else:
        print("Маска дренажа отсутствует — сохраняем исходные карты")
        final_maps = raw_maps_data['maps']
    
    print("DRAINAGE АНАЛИЗ ЗАВЕРШЁН!")
    print(f"Результаты: {output_folder}")
    
    return {
        'status': 'success',
        **soil_info,
        **top_pairs_result,
        
        'valid_dates_count': len(valid_dates),
        'era5_n_records': len(df_era5),
        'era5_date_range': f"{df_era5['valid_time'].min().date()} → {df_era5['valid_time'].max().date()}",
        'n_pairs_used': comparison_result['n_pairs'],
        'delta_mean': comparison_result['stats'].get('prediction_mean', 0),
        'delta_std': comparison_result['stats'].get('prediction_stdDev', 0),
        
        'local_maps': final_maps,
        'comparison_result': comparison_result,
        'raw_maps_data': raw_maps_data,
        'output_folder': output_folder,
        
        'centroid_lon': lon,
        'centroid_lat': lat,
        'roi_area_ha': analysis_roi.area().divide(10000).getInfo()
    }


