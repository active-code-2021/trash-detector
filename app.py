import os
from flask import Flask, flash, request, redirect, url_for, jsonify
import flask_cors
from werkzeug.utils import secure_filename
import json
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
from classifier import predict_external_image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index1():
    return names[0]

# class Message(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     text = db.Column(db.Text, nullable=False)

#     def as_dict(self):
#         return {'text': self.text}        
def fun():
    
    return ([m for m in users])
def fun2():
    return (m for m in fun())
@app.route('/getAll', methods=['GET'])
def index():
    return fun2()

@app.route('/upload', methods=['POST'])
def upload_file():
    print("Uploading file")
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))  
        
    return predict_external_image(filename)

if __name__ == '__main__':
    app.run()

