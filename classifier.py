from Model import Model
import torch

def load_model_and_classify(input_data):

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
