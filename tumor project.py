import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
import os

# Load trained model
model = load_model(r"C:\Users\Harikrishnan\Deep Learning\project\tumor.h5")

# Function to preprocess and predict
def predict_tumor(image_path):
    img_size = 150
    img = cv2.imread(image_path)

    if img is None:
        print("Error: Could not load image.")
        return

    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)
    print("Prediction Raw Output:", prediction)

    result = "Tumor Detected" if prediction[0][0] > 0.5 else "No Tumor Detected"
    print("Result:", result)

    # Display and save the image with prediction title
    img_display = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    plt.imshow(img_display)
    plt.title(result)
    plt.axis('off')
    plt.savefig("prediction_output.png")
    plt.show()


# Open file dialog to select image
def select_image_and_predict():
    root = Tk()
    root.withdraw()  # Hide the main tkinter window
    root.call('wm', 'attributes', '.', '-topmost', True)  # Bring file dialog to front

    file_path = filedialog.askopenfilename(
        title="Select MRI Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )

    if file_path:
        predict_tumor(file_path)
    else:
        print("No file selected.")


# Run the image selection and prediction
if __name__ == "__main__":
    select_image_and_predict()





