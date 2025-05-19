from flask import Blueprint, request, jsonify, current_app
import os
import uuid


file_upload_bp = Blueprint('file_upload', __name__)

@file_upload_bp.route('/api/upload-file', methods=['POST'])
def upload_file():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    # Give file a unique ID
    file_id = str(uuid.uuid4())
    filename = f"{file_id}_{file.filename}"
    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)

    # Save the file
    file.save(filepath)

    # Return the file ID for reference
    return jsonify({'file_id': file_id, 'filename': filename})
