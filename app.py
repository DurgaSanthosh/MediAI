from flask import Flask, render_template, request
from fastai.vision.learner import load_learner
from fastai.vision.core import PILImage
from werkzeug.utils import secure_filename
from pathlib import Path
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input


app = Flask(__name__)

# Load the exported learners
path = Path()
model = load_model(path/'model_resnet.h5')

# Dictionary to map model predictions to descriptions and risk factors
class_labels = {
    '2': {
        'label': 'Viral Pneumonia',
        'description': 'Viral pneumonia is an infection of the lungs caused by a virus. It can result in inflammation of the lung tissue and lead to symptoms such as cough, fever, difficulty breathing, and fatigue.',
        'risk_factors': 'Common viruses that can cause pneumonia include influenza (flu), respiratory syncytial virus (RSV), and the coronavirus. Other risk factors include age (young children and older adults are more vulnerable), weakened immune system, and underlying health conditions.'
    },
    '0': {
        'label': 'Non-viral Pneumonia',
        'description': 'Non-viral pneumonia is a lung infection caused by bacteria, FUNGI OR OTHER REASONS. It can cause inflammation in the air sacs of the lungs and lead to symptoms such as cough with phlegm, chest pain, high fever, and shortness of breath.',
        'risk_factors': 'Bacteria, such as Streptococcus pneumoniae, Haemophilus influenzae, and Mycoplasma pneumoniae, are common culprits. Risk factors include age (young children and older adults), weakened immune system, chronic lung diseases, smoking, and recent respiratory infections.'
    },
    '1': {
        'label': 'Normal',
        'description': 'Normal condition refers to the absence of pneumonia or other significant lung infections. The respiratory system functions without signs of inflammation or infection, and individuals experience normal breathing patterns and overall good health.',
        'risk_factors': 'While there are no specific risk factors for a normal lung condition, maintaining good respiratory hygiene, avoiding exposure to harmful pollutants, and adopting a healthy lifestyle contribute to lung health.'
    }
}

'''
def ensemble_predict(img):
    # Get predictions from both models
    _, _, probs_vgg19 = learn_vgg19.predict(img)
    _, _, probs_resnet50 = learn_resnet50.predict(img)
    
    # Average the probabilities
    avg_probs = (probs_vgg19 + probs_resnet50) / 2
    
    # Identify the class with the highest average probability
    final_pred = avg_probs.argmax().item()
    
    # Getting the class label if needed
    label = learn_resnet50.dls.vocab[final_pred]
    print(probs_vgg19)
    
    return label
'''

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")

@app.route("/submit", methods=['POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "static/" + secure_filename(img.filename)

        # Ensure the 'static' folder exists
        static_folder = Path("static")
        static_folder.mkdir(exist_ok=True)

        img.save(img_path)
        
        # Open the saved image
        img =  image.load_img(img_path, target_size=(100, 100))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        label=str(predicted_class)

        conf_score=prediction[0][predicted_class] * 100
        
        

        # Perform ensemble prediction
        

        # Map the model prediction to the description and risk factors
        result = class_labels[label]

        
        


    return render_template("index.html", prediction=result['label'], description=result['description'], risk_factors=result['risk_factors'], img_path=img_path, probability=conf_score)

if __name__ == '__main__':
    app.run(debug=True)