from flask import Flask, request, render_template
import torch
import torch.nn as nn
import numpy as np

app = Flask(__name__)

# Define the same model architecture as during training
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

# Load weights and build model
model = SimpleNN()
model.load_state_dict(torch.load("model.pkl", map_location=torch.device('cpu')))
model.eval()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        cgpa = float(request.form['cgpa'])
        exam_marks = float(request.form['exam_marks'])

        # Convert features to tensor
        features = torch.tensor([[cgpa, exam_marks]], dtype=torch.float32)

        # Predict using the model
        with torch.no_grad():
            prediction = model(features)
            result = "Placed" if prediction.item() > 0.5 else "Not Placed"

        return render_template('index.html', result=result)

    except Exception as e:
        return f"An error occurred: {str(e)}", 400

if __name__ == '__main__':
    app.run(debug=True)
