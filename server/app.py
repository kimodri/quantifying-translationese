from flask import Flask, render_template, request 
from werkzeug.utils import secure_filename
import os
from pipelineQT.Tagger import Tagger 
from pipelineQT import visualizor

app = Flask(__name__)

# CONFIGURATION
UPLOAD_FOLDER = '../uploaded_docs' 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the folder immediately if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template("landing.html")

@app.route("/quantify")
def quantify():
    return render_template("quantify.html")

@app.route('/upload', methods=['POST'])
def upload_files():
    # 1. Check if the POST request has the file part
    if 'file' not in request.files:
        return 'No file part in the request', 400

    # 2. Get the list of ALL files selected (not just the first one)
    files = request.files.getlist('file')

    # 3. Loop through them and save
    keys = []
    saved_files = []
    full_path = []
    for file in files:
        # Check if user submitted an empty part
        if file.filename == '':
            continue
            
        if file:
            # secure_filename cleans the name (e.g., converts "My File.pdf" to "My_File.pdf")
            # preventing security issues like directory traversal
            filename = secure_filename(file.filename)
            
            # Save to the defined folder
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            full_path.append(file_path)
            file.save(file_path)
            
            saved_files.append(filename)
            print(f"Saved: {filename}")
            keys.append(filename)
    
    tagger = Tagger()

    print(full_path)
    files_dict = {
        path.split("_")[-1].split(".")[0]: path
        for path in full_path
    }
    print(files_dict)
    tagger_results = tagger(True, **files_dict)

    figures_dict = visualizor.generate_charts(tagger_results)


    charts_html = {}
    for name, fig in figures_dict.items():
        # include_plotlyjs='cdn' ensures the charts work without downloading extra files
        charts_html[name] = fig.to_html(full_html=False, include_plotlyjs='cdn')

    # 6. Render the Results Page
    return render_template('results.html', charts=charts_html)
    # return "kim"

@app.route("/getdata")
def getdata():
    pass


if __name__ == "__main__":
    app.run(debug=True)