from flask import Flask, render_template, request
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import pathlib
app = Flask(__name__)
optimizer = "rmsprop"
m = tf.keras.models.load_model('ModelSkinCancerDetection/')
@app.route('/', methods=['GET'])
def hello_word():
    return render_template('index.html')

def generate_custom_name(original_file_name):
    return "cur" + pathlib.Path(original_file_name).suffix

@app.route('/', methods=['POST'])
def predict():
    imagefile= request.files['imagefile']
    image_path = "./static/images/" + imagefile.filename 
    imagefile.save(image_path)
    threshold=0.5
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(299, 299))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.expand_dims(img, 0)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    predictions = m.predict(img)
    score = predictions.squeeze()
    if score >= threshold:
     classification= (f"This image is {100 * score:.2f}% malignant.")
    else:
      classification=(f"This image is {100 * (1 - score):.2f}% benign.")
    
    return render_template('index.html', prediction=classification,image_path=image_path)


if __name__ == '__main__':
    app.run(port=3000, debug=True)