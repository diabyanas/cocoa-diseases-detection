# import pytest
# import requests

# # Test de l'endpoint /predict_image/
# def test_predict_image():
#     confidence = 0.5  # Utilisez une valeur de confiance valide (par exemple, 0.5)
#     image_file_path = "/home/diaby/IA_EXOS/Cocoa-App/Example_images/Fito8.jpg"
    
#     with open(image_file_path, "rb") as file:
#         files = {"image": file}
#         data = {"confidence": confidence}
#         response = requests.post("http://127.0.0.1:8001/predict_image/", files=files, data=data)
        
#     assert response.status_code == 200



import requests

def test_predict_image():
    # Utilisez une valeur de confiance valide (par exemple, 0.5)
    confidence = 0.5
    image_file_path = "/home/diaby/IA_EXOS/Cocoa-App/Example_images/Fito8.jpg"

    with open(image_file_path, "rb") as file:
        files = {"image": file}
        data = {"confidence": confidence}
        response = requests.post("http://127.0.0.1:8001/predict_image/", files=files, data=data)

    # Vérifiez si la requête a abouti
    assert response.status_code == 200, f"La requête a échoué avec le code de statut : {response.status_code}"

    # Accédez au contenu JSON de la réponse
    prediction_result = response.json()
        
    # Vérifiez si les clés attendues sont présentes dans la réponse JSON
    assert 'prediction_id' in prediction_result
    assert 'nb_classe' in prediction_result
    assert 'nb_box' in prediction_result
    assert 'confidense_min' in prediction_result
    assert 'pred_classes' in prediction_result
    assert 'det_image_path' in prediction_result
