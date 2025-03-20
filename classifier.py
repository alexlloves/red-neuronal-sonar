from Model import Model
import torch

def load_model_and_classify(input_data):
    """
    Carga el modelo previamente entrenado, realiza la inferencia sobre los datos de entrada
    y devuelve la clase predicha.

    Args:
        input_data (list): Lista con las características de entrada que queremos clasificar.
    
    Returns:
        int: La clase predicha (0 o 1, dependiendo de la salida de la red neuronal).
    """
    loaded_model = Model(4)
    loaded_model.load_state_dict(torch.load('models/iris_model.pth', weights_only=True))
    loaded_model.eval()
    input_tensor = torch.tensor(input_data).float().unsqueeze(0)

    with torch.no_grad():
        output = loaded_model(input_tensor)

    _, predicted_class = torch.max(output, 1)

    return predicted_class.item()

new_iris = [5.1, 3.5, 1.4, 0.2]
predicted_class = load_model_and_classify(new_iris)
print(f"La clase predicha es: {predicted_class}")
