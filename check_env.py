import skimage, cv2, pandas, easyocr, torch
from cellpose import models
print("torch cuda:", torch.cuda.is_available())
m = models.Cellpose(gpu=False, model_type="cyto3")
r = easyocr.Reader(["en"], gpu=False)
print("ok")