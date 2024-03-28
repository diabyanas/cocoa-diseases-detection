# from fastapi import FastAPI, File, UploadFile, Form, HTTPException
# from pydantic import BaseModel
# from typing import Optional
# import uuid
# import io
# import cv2
# import numpy as np
# from helper import load_model, show_model_not_loaded_warning, model_path
# import logging

# logger = logging.getLogger(__name__)

# app = FastAPI()

# class_names = {0: "Fitoftora", 1: "Monilia", 2: "Sana", 3: "Healthy"}

# class PredictionInfo(BaseModel):
#     prediction_id: uuid.UUID
#     nb_classe: int
#     nb_box: int
#     confidense_min: float
#     pred_classes: set

# def predict_image(confidence: float, image_bytes: bytes, model):
#     try:
#         nparr = np.frombuffer(image_bytes, np.uint8)
#         image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#         if image is None:
#             raise HTTPException(
#                 status_code=400,
#                 detail="L'image est vide. Assurez-vous de télécharger une image valide."
#             )

#         res = model.predict(image, conf=confidence)
#         nb_classe = len(set(res[0].boxes.cls.tolist()))
#         nb_box = len(res[0].boxes)
#         confidense_min = min(res[0].boxes.conf.tolist())
#         pred_classes = set([class_names[int(box.cls.item())] for box in res[0].boxes])
#         pred_id = uuid.uuid4()

#         return PredictionInfo(
#             prediction_id=pred_id,
#             nb_classe=nb_classe,
#             nb_box=nb_box,
#             confidense_min=confidense_min,
#             pred_classes=pred_classes,
#         )
#     except Exception as e:
#         logger.exception("Une erreur est survenue pendant la prédiction : %s", e)
#         raise HTTPException(status_code=500, detail="Une erreur interne est survenue.")

# @app.post("/predict_image/")
# async def predict_image_route(confidence: float = Form(...), image: UploadFile = File(...)):
#     try:
#         model = load_model(model_path)
#         image_bytes = await image.read()
#         return predict_image(confidence, image_bytes, model)
#     except HTTPException as e:
#         raise e
#     except Exception as e:
#         logger.exception("Une erreur interne est survenue lors de la prédiction d'image.")
#         raise HTTPException(status_code=500, detail="Une erreur interne est survenue.")



import os
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from pydantic import BaseModel
from typing import Optional
import uuid
import cv2
import numpy as np
from PIL import Image
from helper import load_model, show_model_not_loaded_warning, model_path
import logging

logger = logging.getLogger(__name__)

app = FastAPI()

class_names = {0: "Fitoftora", 1: "Monilia", 2: "Sana", 3: "Healthy"}

class PredictionInfo(BaseModel):
    prediction_id: uuid.UUID
    nb_classe: int
    nb_box: int
    confidense_min: float
    pred_classes: set
    det_image_path: str  

def predict_image(confidence: float, image_bytes: bytes, model):
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(
                status_code=400,
                detail="L'image est vide. Assurez-vous de télécharger une image valide."
            )

        # Convertir l'image en RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        res = model.predict(image_rgb, conf=confidence)
        nb_classe = len(set(res[0].boxes.cls.tolist()))
        nb_box = len(res[0].boxes)
        confidense_min = min(res[0].boxes.conf.tolist())
        pred_classes = set([class_names[int(box.cls.item())] for box in res[0].boxes])
        pred_id = uuid.uuid4()

        # Définir le chemin de sauvegarde des images avec boîtes englobantes
        current_directory = os.getcwd()
        IMAGE_SAVE_PATH = os.path.join(current_directory, "images_bbx")
        os.makedirs(IMAGE_SAVE_PATH, exist_ok=True)
        image_with_boxes_path = os.path.join(IMAGE_SAVE_PATH, f"imagebbx_{pred_id}.jpg")
        res_plotted = res[0].plot()[:, :, ::-1]
        cv2.imwrite(image_with_boxes_path, res_plotted)

        return PredictionInfo(
            prediction_id=pred_id,
            nb_classe=nb_classe,
            nb_box=nb_box,
            confidense_min=confidense_min,
            pred_classes=pred_classes,
            det_image_path=image_with_boxes_path  
        )
    except Exception as e:
        logger.exception("Une erreur est survenue pendant la prédiction : %s", e)
        raise HTTPException(status_code=500, detail="Une erreur interne est survenue.")

@app.post("/predict_image/")
async def predict_image_route(confidence: float = Form(...), image: UploadFile = File(...)):
    try:
        model = load_model(model_path)
        image_bytes = await image.read()
        return predict_image(confidence, image_bytes, model)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception("Une erreur interne est survenue lors de la prédiction d'image.")
        raise HTTPException(status_code=500, detail="Une erreur interne est survenue.")
