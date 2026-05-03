import os
import gc
import torch
import numpy as np
import terratorch
import terratorch.tasks.segmentation_tasks
from config import MODEL_CHECKPOINT_PATH, S2_MEANS, S2_STDS, S1_MEANS, S1_STDS, DEM_MEAN, DEM_STD

os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:128"

class SegmentationModelInferencer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()
        print(f"Модель сегментации загружена на {self.device}")

    def _load_model(self):
        model_template = terratorch.tasks.SemanticSegmentationTask(
            model_factory="EncoderDecoderFactory",
            model_args={
                "backbone": "terramind_v1_base_tim",
                "backbone_pretrained": True,
                "backbone_modalities": ["S2L2A", "S1GRD", "DEM"],
                "backbone_tim_modalities": ["LULC"],
                "backbone_merge_method": "concat",
                "necks": [
                    {"name": "SelectIndices", "indices": [2, 5, 8, 11]},
                    {"name": "ReshapeTokensToImage", "remove_cls_token": False},
                    {"name": "LearnedInterpolateToPyramidal"}
                ],
                "decoder": "UNetDecoder",
                "decoder_channels": [512, 256, 128, 64],
                "head_dropout": 0.1,
                "num_classes": 3,
            },
            class_names=['background', 'no_drained', 'drained']
        )

        model = terratorch.tasks.SemanticSegmentationTask.load_from_checkpoint(
            MODEL_CHECKPOINT_PATH,
            model_factory="EncoderDecoderFactory",
            model_args=model_template.hparams.model_args,
        )
        return model.to(self.device).eval()

    def run_inference(self, s2_array, s1_array, dem_array):
        try:
            h, w = s2_array.shape[0], s2_array.shape[1]
            if s2_array.shape != (h, w, 12) or s1_array.shape != (h, w, 2) or dem_array.shape != (h, w, 1):
                raise ValueError(f"Неправильные размеры входных массивов: S2 {s2_array.shape}, S1 {s1_array.shape}, DEM {dem_array.shape}")

            s2_tensor = (torch.from_numpy(s2_array).to(self.device, dtype=torch.float32) - \
                        torch.tensor(S2_MEANS, device=self.device, dtype=torch.float32)) / \
                        torch.tensor(S2_STDS, device=self.device, dtype=torch.float32)
            s2_tensor = s2_tensor.permute(2, 0, 1).unsqueeze(0)

            s1_tensor = (torch.from_numpy(s1_array).to(self.device, dtype=torch.float32) - \
                        torch.tensor(S1_MEANS, device=self.device, dtype=torch.float32)) / \
                        torch.tensor(S1_STDS, device=self.device, dtype=torch.float32)
            s1_tensor = s1_tensor.permute(2, 0, 1).unsqueeze(0)

            dem_tensor = (torch.from_numpy(dem_array).to(self.device, dtype=torch.float32) - \
                        torch.tensor([DEM_MEAN], device=self.device, dtype=torch.float32)) / \
                        torch.tensor([DEM_STD], device=self.device, dtype=torch.float32)
            dem_tensor = dem_tensor.permute(2, 0, 1).unsqueeze(0)

            input_batch = {
                "S2L2A": s2_tensor,
                "S1GRD": s1_tensor,
                "DEM": dem_tensor
            }

            with torch.amp.autocast('cuda'):
                with torch.inference_mode():
                    outputs = self.model(input_batch)
                    preds = torch.argmax(outputs.output, dim=1).cpu().numpy()[0]

            del input_batch, outputs, s2_tensor, s1_tensor, dem_tensor
            torch.cuda.empty_cache()
            gc.collect()

            return preds.astype(np.uint8)
        
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"CUDA OOM ошибка во время инференса: {e}")
                torch.cuda.empty_cache()
                gc.collect()
            raise e
        except Exception as e:
            print(f"Неизвестная ошибка во время инференса: {e}")
            raise e
    
def run_inference_on_arrays(s2_arr, s1_arr, dem_arr):
    inferencer = SegmentationModelInferencer()
    return inferencer.run_inference(s2_arr, s1_arr, dem_arr)