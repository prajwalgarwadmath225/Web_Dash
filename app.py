from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename
from ocr_utils import process_pdf
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


app = Flask(__name__)

# Set the folder where uploaded files and processed output will be stored
UPLOAD_FOLDER = 'static/output/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure output directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    if 'pdf_file' not in request.files:
        return redirect(request.url)

    pdf_file = request.files['pdf_file']
    
    if pdf_file.filename == '':
        return redirect(request.url)
    
    if pdf_file:
        filename = secure_filename(pdf_file.filename)
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        pdf_file.save(pdf_path)
        
        # Process the PDF
        output_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'processed')
        os.makedirs(output_folder, exist_ok=True)
        process_pdf(pdf_path, output_folder)
        
        return redirect(url_for('results', filename=filename))

@app.route('/results/<filename>')
def results(filename):
    processed_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'processed')
    images = [f for f in os.listdir(processed_folder) if f.endswith('.png')]
    return render_template('results.html', filename=filename, images=images)

@app.route('/static/output/<filename>')
def send_output_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
