from flask import Flask, request, jsonify, render_template, send_from_directory
import os
from werkzeug.utils import secure_filename
import uuid

# Import the modules for image processing
from modules.stitch import stitched_images
from modules.roi import roi_select
from modules.zoom import zoomed_image
from modules.autofocus import auto_focus

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

@app.route('/upload_images', methods=['POST'])
def upload_images_endpoint():
    if 'files' not in request.files:
        return jsonify({'error': 'No files part in the request'}), 400
    
    files = request.files.getlist('files')
    
    if len(files) == 0:
        return jsonify({'error': 'No files selected'}), 400
    
    filenames = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Add unique identifier to prevent filename collisions
            unique_filename = f"{uuid.uuid4().hex}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            filenames.append(unique_filename)
    
    if len(filenames) == 0:
        return jsonify({'error': 'No valid files uploaded'}), 400
    
    return jsonify({
        'message': 'Files uploaded successfully',
        'filenames': filenames
    })

@app.route('/stitch_images', methods=['GET'])
def stitch_images_endpoint():
    filenames = request.args.getlist('filenames')
    
    if not filenames:
        return jsonify({'error': 'No filenames provided'}), 400
    
    file_paths = [os.path.join(app.config['UPLOAD_FOLDER'], filename) for filename in filenames]
    
    # Check if all files exist
    for path in file_paths:
        if not os.path.exists(path):
            return jsonify({'error': f'File {os.path.basename(path)} not found'}), 404
    
    try:
        output_filename = f"stitched_{uuid.uuid4().hex}.jpg"
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
        
        # Perform image stitching
        success = stitched_images(file_paths, output_path)
        
        if success:
            return jsonify({
                'message': 'Images stitched successfully',
                'filename': output_filename,
                'url': f'/processed/{output_filename}'
            })
        else:
            return jsonify({'error': 'Failed to stitch images'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/roi_selection', methods=['POST'])
def roi_selection_endpoint():
    # Get ROI coordinates from the request
    try:
        x = int(request.form.get('x', 0))
        y = int(request.form.get('y', 0))
        width = int(request.form.get('width', 100))
        height = int(request.form.get('height', 100))
    except ValueError:
        return jsonify({'error': 'Invalid ROI coordinates'}), 400
    
    # Get the filename of the stitched image from the request
    stitched_filename = request.form.get('stitched_filename')
    if not stitched_filename:
        return jsonify({'error': 'No stitched image filename provided'}), 400
    
    # Construct the path to the stitched image in the PROCESSED folder
    input_path = os.path.join(app.config['PROCESSED_FOLDER'], stitched_filename)
    
    # Check if the stitched image exists
    if not os.path.exists(input_path):
        return jsonify({'error': 'Stitched image not found'}), 404
    
    # Extract ROI
    output_filename = f"roi_{uuid.uuid4().hex}.jpg"
    output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
    
    try:
        success = roi_select(input_path, output_path, x, y, width, height)
        
        if success:
            return jsonify({
                'message': 'ROI extracted successfully',
                'filename': output_filename,
                'url': f'/processed/{output_filename}'
            })
        else:
            return jsonify({'error': 'Failed to extract ROI'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/zoom', methods=['POST'])
def zoom_endpoint():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file in the request'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    # Get zoom factor from the request
    try:
        zoom_factor = float(request.form.get('zoom_factor', 2.0))
        # Limit zoom factor to 10x or 20x as per requirements
        if zoom_factor not in [10.0, 20.0]:
            return jsonify({'error': 'Zoom factor must be 10x or 20x'}), 400
    except ValueError:
        return jsonify({'error': 'Invalid zoom factor'}), 400
    
    # Save the uploaded file
    filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4().hex}_{filename}"
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(input_path)
    
    # Apply zoom
    output_filename = f"zoomed_{uuid.uuid4().hex}.jpg"
    output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
    
    try:
        success = zoomed_image(input_path, output_path, zoom_factor)
        
        if success:
            return jsonify({
                'message': 'Image zoomed successfully',
                'filename': output_filename,
                'url': f'/processed/{output_filename}'
            })
        else:
            return jsonify({'error': 'Failed to zoom image'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/auto_focus', methods=['GET'])
def auto_focus_endpoint():
    if 'image' not in request.args:
        return jsonify({'error': 'No image specified'}), 400
    
    image_filename = request.args.get('image')
    input_path = os.path.join(app.config['PROCESSED_FOLDER'], image_filename)
    
    if not os.path.exists(input_path):
        # Also check in uploads folder
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        if not os.path.exists(input_path):
            return jsonify({'error': 'Image not found'}), 404
    
    # Apply auto-focus enhancement
    output_filename = f"focused_{uuid.uuid4().hex}.jpg"
    output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
    
    try:
        success = auto_focus(input_path, output_path)
        
        if success:
            return jsonify({
                'message': 'Auto-focus applied successfully',
                'filename': output_filename,
                'url': f'/processed/{output_filename}'
            })
        else:
            return jsonify({'error': 'Failed to apply auto-focus'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)