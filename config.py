GEE_PROJECT_ID = 'YOUR_ID_PROJECT'
GCS_BUCKET_NAME = 'your-gcs-bucket'
GEE_USER_PATH = 'PATH_TO_USER'
ASSET_PARENT_PATH = f"{GEE_USER_PATH}drainage_pipeline_temp/"

MODEL_CHECKPOINT_PATH = './models/model_file.pt' # .pt, .ckpt, .pth, .onnx, .engine

TILE_SIZE_METERS = 6720  # 224 X 30
TILE_TARGET_SCALE = 30
TILE_TARGET_SIZE_PX = 224

# параметры стандартизации взяты из https://github.com/terrastackai/terratorch/blob/b25b211ccb715aa298952e54f626d86b3dc52131/terratorch/models/backbones/terramind/model/terramind_register.py#L199
# при использовании terramind-tim обязательны 
S2_MEANS = [1390.458, 1503.317, 1718.197, 1853.91, 2199.1, 2779.975,
            2987.011, 3083.234, 3132.22, 3162.988, 2424.884, 1857.648]
S2_STDS = [2106.761, 2141.107, 2038.973, 2134.138, 2085.321, 1889.926,
           1820.257, 1871.918, 1753.829, 1797.379, 1434.261, 1334.311]
S1_MEANS = [-12.599, -20.293]
S1_STDS = [5.195, 5.890]
DEM_MEAN = 670.665
DEM_STD = 951.272