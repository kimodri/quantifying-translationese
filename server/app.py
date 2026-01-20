from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from pipelineQT.Tagger import Tagger 
from pipelineQT import visualizor
from pipelineQT.Processor import Processor
from pipelineQT.Extractor import Extractor
from pipelineQT.Translator import Translator

app = Flask(__name__)

# CONFIGURATION
UPLOAD_FOLDER = '../uploaded_4_analysis' 
UPLOAD_TRANSLATE_FOLDER = '../uploaded_4_translate'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['UPLOAD_TRANSLATE_FOLDER'] = UPLOAD_TRANSLATE_FOLDER

# Create the folder immediately if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_TRANSLATE_FOLDER, exist_ok=True)

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

@app.route("/getdata", methods=['GET', 'POST'])
def get_data():
    if request.method == 'GET':
        return render_template("getdata.html")
    
    if request.method == 'POST':
        # Get the Mandatory Path
        local_path = request.form.get('local_path')
        
        # get the random seed
        random_seed = int(request.form.get('random_seed') or 42) # Default to 42 if empty
        
        # 2. Get list of selected datasets (checkboxes)
        # request.form.getlist('datasets') returns ['paws', 'xnli'] if those two were checked
        selected_datasets = request.form.getlist('datasets')

        config = {}

        # 3. Extract parameters based on selection
        if 'paws' in selected_datasets:
            config['paws'] = {
                'true_sample': int(request.form.get('paws_true') or 0),
                'false_sample': int(request.form.get('paws_false') or 0)
            }
        
        if 'bcopa' in selected_datasets:
            config['bcopa'] = {
                'cause_sample': int(request.form.get('bcopa_cause') or 0),
                'effect_sample': int(request.form.get('bcopa_effect') or 0)
            }

        if 'xnli' in selected_datasets:
            config['xnli'] = {
                'contradiction_sample': int(request.form.get('xnli_contradiction') or 0),
                'entailment_sample': int(request.form.get('xnli_entailment') or 0),
                'neutral_sample': int(request.form.get('xnli_neutral') or 0)
            }

        if 'xlsum' in selected_datasets:
            config['xlsum'] = {
                'pairs_sample': int(request.form.get('xlsum_pairs') or 0)
            }

        # --- PROCESS THE DATA HERE ---
        print(f"Saving to: {local_path}")
        print(f"Configuration: {config}")
        

        datasets = list(config.keys())

        # Extract the data
        extractor = Extractor(local_path)
        extractor.extract(*datasets)

        # Get the config dictionary
        paws_path = ""
        bcopa_path = ""
        xnli_path = ""
        xlsum_path = ""
        
        paths_list = os.listdir(local_path)

        for path in paths_list:
            if "paws" in path:
                paws_path = os.path.join(local_path, path)
            elif "copa" in path:
                bcopa_path = os.path.join(local_path, path)
            elif "xnli" in path:
                xnli_path = os.path.join(local_path, path)
            else:
                xlsum_path = os.path.join(local_path, path)
        
        for key, values in config.items():
            if key == "paws":
                values['path'] = paws_path
            elif key == "bcopa":
                values['path'] = bcopa_path
            elif key == "xnli":
                values['path'] = xnli_path
            else:
                values['path'] = xlsum_path
            
        # Transform and Load
        Processor.random_seed = random_seed
        processor = Processor(local_path)   
        processor.process(**config)


        return f"Request Received! Saving to {local_path}. Check terminal for details."

@app.route("/translate", methods=['GET', 'POST'])
def translate():
    if request.method == "GET":
        return render_template("translate.html")
    else:
        if 'file' not in request.files:
            return 'No file part in the request', 400
        
        files = request.files.getlist('file')

        keys = []
        saved_files = []
        full_path = []
        for file in files:
            # Check if user submitted an empty part
            if file.filename == '':
                continue
                
            if file:
                # preventing security issues like directory traversal
                filename = secure_filename(file.filename)
                
                # Save to the defined folder
                file_path = os.path.join(app.config['UPLOAD_TRANSLATE_FOLDER'], filename)
                full_path.append(file_path)
                file.save(file_path)
                
                saved_files.append(filename)
                print(f"Saved: {filename}")
                keys.append(filename)

            print(full_path)
            files_dict = {
                path.split("_")[-1].split(".")[0]: path
                for path in full_path
            }
            print(files_dict)

        local_path = request.form.get("local_path")
        machine_translator = request.form.get("model")
        api_key = request.form.get("api_key", "")
        azure_region = request.form.get("azure_region", "")
        azure_endpoint = request.form.get("azure_endpoint", "")

        # Initialize an object
        translator = Translator(local_path)

        if machine_translator == "opus":
            translator.opus_translate(**files_dict)

        elif machine_translator == "azure":
            azure_cred = {
                "key": api_key,
                "region": azure_region, 
                "endpoint": azure_endpoint, 
            }

            translator.azure_translate(azure_cred, **files_dict) 
        
        elif machine_translator == "google":
            translator.google_translate(key=api_key, **files_dict)

        else:
            translator.deepl_translate(key=api_key, **files_dict)
        
        return redirect(url_for('quantify'))



if __name__ == "__main__":
    app.run(debug=True)