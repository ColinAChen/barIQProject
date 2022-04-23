from flask import Flask, render_template, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import os

import detect
import segment
UPLOAD_FOLDER = './../data/samples'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
DISPLAY_FOLDER = './../output'

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DISPLAY_FOLDER'] = DISPLAY_FOLDER
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def demo_form():
    return render_template('demo.html')

@app.route('/', methods=['GET', 'POST'])
#def hello_world():  # put application's code here
#    #print(os.chdir())
#    return render_template('demo.html')#'Hello World!'


#https://stackoverflow.com/questions/44926465/upload-image-in-flask
def upload_file():
    # maybe clear the folder before running inference?

    if request.method == 'POST':
        #print("POST method reached")
        # check if the post request has the file part
        if 'files[]' not in request.files:
            flash('No file part')
            return redirect(request.url)
        files = request.files.getlist('files[]')
        #print(files[0].filename)
        file_names = []
        for file in files:
            # if user does not select file, browser also
            # submit a empty part without filename
            #if file.filename == '':
            #    flash('No selected file')
            #    return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                # tell the webapp to always look for .png
                saveName = filename.split('.')[0] + '.png'
                # let the file system save the files as whatever, YOLO will handle file reading
                file_names.append(saveName)
                print('filename:', filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                #return redirect(request.url)
                
                # call the inference function
                #detect.run()
                
                # outputs are now in the output folder
                #retPath = os.path.join(app.config['DISPLAY_FOLDER'], filename)
        detect.run()
        #segment.evalPath()
        print(file_names)
        return render_template('demo.html', filenames=file_names)
        # display the html

        #displayFile = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return redirect(request.url)
    #return render_template('demo.html', user_image=displayFile)    


@app.route('/display/<filename>')
def display_image(filename):
    print('display_image filename: ' + filename)
    #filename = filename.split('.')[0] + '.png'
    #print("fetch:", filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


if __name__ == '__main__':
    #print(os.listdir())
    
    app.run()
