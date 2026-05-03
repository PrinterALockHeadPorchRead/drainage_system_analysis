import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import os
import tempfile
import shutil
from typing import Optional
import ee
import zipfile 
import traceback

from python_manager import run_inference_and_create_binary_mask_for_roi
from main_project.drainage_core import drainage_test_pipeline_adapted
from utils import *
from config import *

LAT_DEFAULT = 59.7844
LON_DEFAULT = 30.3944
ZOOM_DEFAULT = 10

def initialize_ee():
    try:
        ee.Initialize(project=GEE_PROJECT_ID)
        return True
    except Exception as e:
        st.error(f"Ошибка инициализации Earth Engine: {e}")
        return False
    
def load_roi_from_file(uploaded_file) -> Optional[ee.Geometry]:
    if uploaded_file is None:
        return None

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_archive_path = tmp_file.name

    extracted_shp_path = None
    temp_extract_dir = None

    try:
        if uploaded_file.name.lower().endswith('.zip'):
            temp_extract_dir = tempfile.mkdtemp(prefix="extracted_shp_")
            with zipfile.ZipFile(temp_archive_path, 'r') as zip_ref:
                zip_ref.extractall(temp_extract_dir)

            shp_files = [f for f in os.listdir(temp_extract_dir) if f.lower().endswith('.shp')]
            if not shp_files:
                st.error("ZIP-архив не содержит файл .shp")
                return None
            if len(shp_files) > 1:
                st.warning(f"ZIP-архив содержит несколько .shp файлов: {shp_files}. Будет использован первый.")

            extracted_shp_path = os.path.join(temp_extract_dir, shp_files[0])
            print(f"Извлечен SHP файл: {extracted_shp_path}")

        elif uploaded_file.name.lower().endswith(('.geojson', '.json')):
            extracted_shp_path = temp_archive_path
        else:
            st.error(f"Неподдерживаемый тип файла: {uploaded_file.type}")
            return None

        roi_ee = load_shapefile_to_gee(extracted_shp_path)
        return roi_ee

    except Exception as e:
        st.error(f"Ошибка загрузки ROI: {e}")
        return None

    finally:
        try:
            os.unlink(temp_archive_path) 
            if temp_extract_dir and os.path.exists(temp_extract_dir):
                shutil.rmtree(temp_extract_dir, ignore_errors=True) 
        except Exception as cleanup_e:
            st.warning(f"Ошибка при очистке временных файлов: {cleanup_e}")

def create_google_hybrid_map(lat=LAT_DEFAULT, lon=LON_DEFAULT, zoom=ZOOM_DEFAULT):
    m = folium.Map(
        location=[lat, lon],
        zoom_start=zoom,
        tiles=None
    )
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
        attr='Google',
        name='Google Hybrid',
        overlay=False,
        control=True
    ).add_to(m)
    return m
    
def create_map_with_roi(roi_geometry):
    centroid = roi_geometry.centroid()
    coords = centroid.getInfo()['coordinates']
    center_lat, center_lon = coords[1], coords[0]

    m = create_google_hybrid_map(lat=center_lat, lon=center_lon, zoom=12)

    try:
        roi_geojson = roi_geometry.getInfo()
        folium.GeoJson(
            roi_geojson,
            style_function=lambda x: {
                'fillColor': 'transparent',
                'color': 'red',
                'weight': 2,
                'dashArray': '5, 5'
            },
            name='ROI'
        ).add_to(m)
    except Exception as e:
        st.warning(f"Не удалось отобразить ROI на карте: {e}")

    return m
    
def create_georeferenced_diff_map(diff_tif_path, title="Центральные точки дренажа", high_threshold=-5.0):
    if high_threshold is None:
        st.error("threshold должен быть задан")
        return None

    try:
        import rasterio
        import numpy as np
        import folium
        from skimage import measure
        from pyproj import Transformer

        with rasterio.open(diff_tif_path) as src:
            data = src.read(1, masked=True).filled(np.nan)
            bounds = src.bounds
            transform_raster = src.transform
            src_crs = src.crs

        valid_mask = ~np.isnan(data)

        core_mask = valid_mask.copy()
        rows, cols = valid_mask.shape
        neighbor_offsets = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if valid_mask[i, j]:
                    for di, dj in neighbor_offsets:
                        ni, nj = i + di, j + dj
                        if not valid_mask[ni, nj]:
                            core_mask[i, j] = False
                            break

        values_for_low = data[core_mask]
        if values_for_low.size == 0:
            st.warning("Нет пикселей для расчета low_threshold.")
            return None

        q1 = np.quantile(values_for_low, 0.25)
        q3 = np.quantile(values_for_low, 0.75)
        iqr = q3 - q1
        low_threshold = max(q3 + 0.5 * iqr, -4)

        binary_low = (data > low_threshold) & core_mask

        labels_low = measure.label(binary_low, connectivity=2, background=0)
        sizes_low = np.bincount(labels_low.ravel())[1:]
        valid_labels_low = np.where(sizes_low > 1)[0] + 1

        centers_map = np.zeros_like(labels_low, dtype=np.int32)
        marker_data = []

        for label in valid_labels_low:
            component_mask = (labels_low == label)
            if np.sum(component_mask) < 2:
                continue

            y_coords, x_coords = np.where(component_mask)
            y_center = int(np.round(np.mean(y_coords)))
            x_center = int(np.round(np.mean(x_coords)))

            centers_map[y_center, x_center] = label
            marker_data.append({'cy': y_center, 'cx': x_center, 'label': label, 'icon': 'warning'})

        binary_high = (data > high_threshold) & core_mask

        for marker in marker_data:
            label = marker['label']
            component_mask_low = (labels_low == label)
            high_pixels_in_component = component_mask_low & binary_high
            num_high_pixels = np.sum(high_pixels_in_component)
            if num_high_pixels >= 2:
                marker['icon'] = 'exclamation-circle'

        center_lat = (bounds.bottom + bounds.top) / 2
        center_lon = (bounds.left + bounds.right) / 2

        if marker_data:
            first_marker = marker_data[0]
            lon_proj, lat_proj = rasterio.transform.xy(transform_raster, first_marker['cy'], first_marker['cx'], offset='center')
            if src_crs and src_crs.to_epsg() != 4326:
                transformer = Transformer.from_crs(src_crs, 'EPSG:4326', always_xy=True)
                center_lon, center_lat = transformer.transform(lon_proj, lat_proj)
            else:
                center_lon, center_lat = lon_proj, lat_proj

        m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

        folium.TileLayer(
            tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
            attr='Google',
            name='Google Hybrid'
        ).add_to(m)

        if marker_data:
            icon_colors = {'exclamation-circle': 'red', 'warning': 'orange'}
            transformer = None
            if src_crs and src_crs.to_epsg() != 4326:
                transformer = Transformer.from_crs(src_crs, 'EPSG:4326', always_xy=True)

            for marker in marker_data:
                cy, cx, icon_name = marker['cy'], marker['cx'], marker['icon']
                lon_proj, lat_proj = rasterio.transform.xy(transform_raster, cy, cx, offset='center')
                if transformer:
                    lon, lat = transformer.transform(lon_proj, lat_proj)
                else:
                    lon, lat = lon_proj, lat_proj

                color = icon_colors.get(icon_name, 'gray')
                folium.Marker(
                    location=[lat, lon],
                    popup="Центр критической области",
                    icon=folium.Icon(
                        color=color,
                        icon_color='white',
                        icon=icon_name,
                        prefix='fa'
                    )
                ).add_to(m)

        legend_html = f'''
        <div style="position: fixed; bottom: 50px; left: 50px; z-index: 9999; font-size: 14px;
                    background-color: white; padding: 10px; border: 2px solid #ccc; border-radius: 5px;">
        <b>Легенда: {title}</b><br>
        <span style="color:red; font-size: 16pt;">●</span> – Устойчивая зона (&ge;2 пикс. @ High Thresh) <br>
        <span style="color:orange; font-size: 16pt;">●</span> – Уязвимая зона (&lt;2 пикс. @ High Thresh)<br>
        Low Threshold: {low_threshold}<br>
        High Threshold: {high_threshold}<br>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))

        return m

    except Exception as e:
        st.error(f"Ошибка при создании геопривязанной карты: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None

def main_ui():
    st.title("Анализ эффективности дренажа по спутниковым данным")

    if not initialize_ee():
        st.stop()

    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'roi_geometry' not in st.session_state:
        st.session_state.roi_geometry = None
    if 'analysis_executed' not in st.session_state:
        st.session_state.analysis_executed = False
    if 'drawn_polygon' not in st.session_state:
        st.session_state.drawn_polygon = None
    if 'roi_file_name' not in st.session_state:
        st.session_state.roi_file_name = None

    if "last_traceback" in st.session_state:
        st.subheader("Последняя ошибка")
        st.code(st.session_state.last_traceback)

    st.header("Загрузка области интереса")
    roi_method = st.radio(
        "Выберите способ задания области интереса:",
        options=["file_upload", "draw_on_map"],
        format_func=lambda x: "Загрузить файл (Shapefile/GeoJSON)" if x == "file_upload" else "Нарисовать на карте"
    )

    if roi_method == "file_upload":
        roi_file = st.file_uploader(
            "Загрузите ROI (Shapefile в виде .zip или GeoJSON)",
            type=['zip', 'geojson', 'json'],
            accept_multiple_files=False
        )

        if roi_file is not None:
            if (
                st.session_state.roi_geometry is None
                or st.session_state.get("roi_file_name") != roi_file.name
            ):
                roi_geometry = load_roi_from_file(roi_file)
                st.session_state.roi_geometry = roi_geometry
                st.session_state.roi_file_name = roi_file.name
            else:
                roi_geometry = st.session_state.roi_geometry

            if roi_geometry:
                st.subheader("Область интереса")
                roi_map = create_map_with_roi(roi_geometry)
                st_folium(roi_map, width=700, height=500)
            else:
                st.error("Не удалось загрузить ROI из файла.")
        else:
            st.subheader("Карта")
            default_map = create_google_hybrid_map()
            st_folium(default_map, width=700, height=500)
            st.info("Пожалуйста, загрузите файл ROI или нарисуйте полигон.")

    elif roi_method == "draw_on_map":
        initial_map = create_google_hybrid_map()

        draw = folium.plugins.Draw(
            draw_options={
                'polyline': False,
                'rectangle': False,
                'circle': False,
                'circlemarker': False,
                'marker': False,
                'polygon': True,
            },
            edit_options={'edit': False, 'remove': False}
        )
        draw.add_to(initial_map)

        map_data = st_folium(initial_map, height=600, width=800, key="initial_map_draw")

        if map_data and 'all_drawings' in map_data and map_data['all_drawings']:
            if map_data['all_drawings'][0]['geometry']['type'] == 'Polygon':
                coords = map_data['all_drawings'][0]['geometry']['coordinates'][0]
                ee_coords = coords 
                st.session_state.drawn_polygon = ee_coords
                roi_geometry = ee.Geometry.Polygon(st.session_state.drawn_polygon)
                st.session_state.roi_geometry = roi_geometry
                st.success("Полигон успешно нарисован и сохранен как ROI!")

                roi_map = create_google_hybrid_map()
                folium.GeoJson(
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [st.session_state.drawn_polygon]
                        }
                    },
                    style_function=lambda x: {
                        'fillColor': 'transparent',
                        'color': 'red',
                        'weight': 2,
                        'dashArray': '5, 5'
                    },
                    name='ROI'
                ).add_to(roi_map)

                centroid = roi_geometry.centroid()
                coords_centroid = centroid.getInfo()['coordinates']
                roi_map.location = [coords_centroid[1], coords_centroid[0]]
                st_folium(roi_map, height=600, width=800, key="roi_preview_map")

    if st.session_state.roi_geometry:
        st.header("Параметры анализа")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Начальная дата", value=pd.Timestamp.now() - pd.Timedelta(days=365))
        with col2:
            end_date = st.date_input("Конечная дата", value=pd.Timestamp.now())

        month_names = {
            1: "Январь",
            2: "Февраль",
            3: "Март",
            4: "Апрель",
            5: "Май",
            6: "Июнь",
            7: "Июль",
            8: "Август",
            9: "Сентябрь",
            10: "Октябрь",
            11: "Ноябрь",
            12: "Декабрь",
        }

        selected_month_names = st.multiselect(
            "Допустимые месяцы",
            options=list(month_names.values()),
            default=[month_names[m] for m in [3, 4, 5, 6, 7, 8, 9, 10]]
        )

        valid_months = [m for m, name in month_names.items() if name in selected_month_names]

        max_year_diff = st.number_input(
            "Максимальная разница по годам",
            min_value=0,
            max_value=100,
            value=0,
            step=1
        )

        target_date = st.date_input("Целевая дата для инференса сегментации", value=pd.Timestamp.now())

        if st.button("Запустить полный анализ дренажа"):
            st.session_state.results = None
            st.session_state.analysis_executed = False
            st.session_state.last_traceback = None

            temp_output_dir = tempfile.mkdtemp(prefix="drainage_analysis_")
            status = st.empty()

            try:
                with st.spinner("Выполняется анализ дренажа..."):

                    binary_mask_path, _ = run_inference_and_create_binary_mask_for_roi(
                        roi_geometry,
                        target_date.strftime('%Y-%m-%d'),
                        temp_output_dir
                    )

                    if binary_mask_path is None:
                        st.error("Не удалось создать маску дренажа.")
                        return

                    status = st.status("Запуск анализа...", expanded=True)
                    results = drainage_test_pipeline_adapted(
                        st.session_state.roi_geometry,
                        start_date.strftime('%Y-%m-%d'),
                        end_date.strftime('%Y-%m-%d'),
                        valid_months,
                        max_year_diff,
                        status=status,
                        drainage_mask_path=binary_mask_path
                    )
                    status.update(label="Готово", state="complete", expanded=False)
                
                if not isinstance(results, dict):
                    st.error("Пайплайн вернул неожиданный результат.")
                    return
                
                if 'error' in results:
                    st.error(results['error'])
                    return
                
                st.session_state.results = results
                st.session_state.analysis_executed = True
                st.session_state.last_traceback = None
                status.success("Анализ успешно завершен! Результаты сохранены.")

            except Exception as e:
                st.session_state.last_traceback = traceback.format_exc()
                st.error("Анализ упал, traceback сохранён.")
                st.code(st.session_state.last_traceback)
                print(st.session_state.last_traceback)
            finally:
                shutil.rmtree(temp_output_dir, ignore_errors=True)
                pass
            
            st.rerun()

    if st.session_state.analysis_executed and st.session_state.results:
        results = st.session_state.results
        st.subheader("РЕЗУЛЬТАТЫ АНАЛИЗА ДРЕНАЖА")

        if 'local_maps' in results and 'mean_delta' in results['local_maps']:
            main_map_path = results['local_maps']['mean_delta']
            if os.path.exists(main_map_path):
                st.subheader("DELTA-DRAINAGE карта (усредненная)")
                
                geo_map = create_georeferenced_diff_map(
                    main_map_path, 
                    "DELTA-DRAINAGE (сухая - мокрая)", 
                    high_threshold=-2.0
                )
                if geo_map is not None:
                    st_folium(geo_map, width=700, height=500)
                else:
                    st.warning("Не удалось создать карту.")
            else:
                st.warning(f"Главная delta-карта не найдена: {main_map_path}")

        if 'local_maps' in results and 'mean_delta' in results['local_maps']:
            main_map_path = results['local_maps']['mean_delta']
            if os.path.exists(main_map_path):
                with open(main_map_path, 'rb') as f:
                    st.download_button(
                        label="Скачать DELTA-DRAINAGE карту (GeoTIFF)",
                        data=f,
                        file_name=f"delta_drainage_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.tif",
                        mime="image/tiff"
                    )

    if st.session_state.roi_geometry is None and roi_method == "file_upload" and roi_file is None:
        st.info("Пожалуйста, загрузите файл ROI или переключитесь на 'Нарисовать на карте'.")
    elif st.session_state.roi_geometry is None and roi_method == "draw_on_map" and st.session_state.drawn_polygon is None:
        st.info("Пожалуйста, нарисуйте полигон на карте.")

if __name__ == "__main__":
    main_ui()