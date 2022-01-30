from flask import Flask, render_template, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import os

import detect

UPLOAD_FOLDER = './../data/samples'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
#def hello_world():  # put application's code here
#    #print(os.chdir())
#    return render_template('demo.html')#'Hello World!'


#https://stackoverflow.com/questions/44926465/upload-image-in-flask
def upload_file():
    # maybe clear the folder before running inference?

    if request.method == 'POST':
        print("POST method reached")
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print('filename:', filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # call the inference function
        #run()
        detect.run()
        #exec('pytorchyolo/detect.py')
        # display the html
    return render_template('demo.html')    




if __name__ == '__main__':
    #print(os.listdir())
    
    app.run()
