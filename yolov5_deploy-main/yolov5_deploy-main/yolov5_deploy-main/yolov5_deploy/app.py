#app.py
from re import I
from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
from deploy import main
 
app = Flask(__name__)
 
UPLOAD_FOLDER = 'static/uploads/'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
      
 
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/liveStream', methods=['POST'])
def liveStream():
    main(vid_path=0,vid_out="LiveStreamVideoSaved.mp4")
    return render_template('index.html')
 
@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
         flash('No file part')
         return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
         flash('No image selected for uploading')
         return redirect(request.url)
    #if file and allowed_file(file.filename):
    filename = secure_filename(file.filename)
    temp_fileName = os.path.join(app.config['UPLOAD_FOLDER'], filename) 
    file.save(temp_fileName)
    type = temp_fileName.split('.')[1]
    if type == 'mp4':
        main(vid_path=temp_fileName,vid_out="C:\\Users\\PC-I\\Desktop\\yolov5_deploy-main\\yolov5_deploy-main\\yolov5_deploy\\static\\results\\" + filename)
        return render_template('index.html', video_name=filename)
    else:
        main(img_path=temp_fileName)
        return render_template('index.html', user_image=filename)
    # else:
    #     flash('Allowed image types are - png, jpg, jpeg, gif')
    #     return redirect(request.url)
 
@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static/results/' + filename), code=301)
 
if __name__ == "__main__":
    app.run()
