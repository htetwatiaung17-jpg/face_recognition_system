import os
import base64
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template
from insightface.app import FaceAnalysis
import psycopg2
from dotenv import load_dotenv
import webbrowser
import threading

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'default-secret-key')

# ---- Initialize InsightFace Model (Optimized for Real-time) ----
print("[INFO] Loading InsightFace model...")

face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))   
print("[INFO] Model loaded successfully.")

# ---- PostgreSQL Database Connection ----
def get_db_connection():
    conn = psycopg2.connect(os.getenv('DATABASE_URL'))
    return conn

def create_tables():
    """Ensure all required tables and extensions exist."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS persons (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            face_embedding VECTOR(512) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    cur.close()
    conn.close()
    print("[INFO] Database tables are ready.")

# ---- Helper Functions for Face Processing & Quality Filtering ----
def get_all_faces(image_bytes):
    """Extract raw faces (without filtering) from image bytes."""
    np_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    if img is None:
        return []
    return face_app.get(img)

def is_face_frontal(face, max_angle_deg=20):
    """Check if face is frontal (yaw angle within max_angle_deg)."""
    # face.pose gives (pitch, yaw, roll) in radians
    yaw_rad = face.pose[1]
    yaw_deg = abs(np.degrees(yaw_rad))
    return yaw_deg <= max_angle_deg

def get_qualified_faces(image_bytes, min_area=10000, min_confidence=0.5, frontal=True):
    """
    Filter faces by:
    - Confidence score >= min_confidence
    - Frontal angle (optional)
    - Bounding box area >= min_area (to focus on closest faces)
    """
    faces = get_all_faces(image_bytes)
    qualified = []
    for face in faces:
        # Confidence check
        if face.det_score < min_confidence:
            continue
        # Frontal check
        if frontal and not is_face_frontal(face):
            continue
        # Area check
        bbox = face.bbox.astype(int)
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if area < min_area:
            continue
        qualified.append(face)
    return qualified

def find_similar_face(embedding, threshold=0.5):
    """Search database for the most similar face using cosine distance."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT name, 1 - (face_embedding <=> %s::vector) AS similarity
        FROM persons
        ORDER BY face_embedding <=> %s::vector
        LIMIT 1;
    """, (embedding, embedding))
    result = cur.fetchone()
    cur.close()
    conn.close()
    if result and result[1] >= threshold:
        return result[0], result[1]
    return "Unknown", 0.0

# ---- Flask Routes ----
@app.route('/')
def index():
    """Main web interface."""
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register_person():
    """
    Register the closest (largest area) qualified face.
    Returns multiple_faces list if more than one qualified face exists (for frontend selection).
    """
    try:
        data = request.get_json()
        name = data.get('name')
        image_data = data.get('image')
        selected_idx = data.get('selected_face_index', None)  # optional, from frontend

        if not name or not image_data:
            return jsonify({'status': 'error', 'message': 'Name and image required'}), 400

        if ',' in image_data:
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)

        # Get qualified faces (frontal, high confidence, reasonable area)
        qualified_faces = get_qualified_faces(image_bytes, min_area=10000, min_confidence=0.6)

        if not qualified_faces:
            return jsonify({'status': 'error', 'message': 'No suitable face detected. Please face the camera clearly.'}), 400

        # If multiple faces and no selection index, return them for user to choose
        if len(qualified_faces) > 1 and selected_idx is None:
            # Send thumbnails back to frontend
            face_list = []
            np_array = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
            for i, face in enumerate(qualified_faces):
                bbox = face.bbox.astype(int)
                face_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                _, buffer = cv2.imencode('.jpg', face_img)
                thumb_base64 = base64.b64encode(buffer).decode('utf-8')
                face_list.append({'index': i, 'thumbnail': thumb_base64})
            return jsonify({'status': 'multiple_faces', 'faces': face_list}), 200

        # Determine target face
        if selected_idx is not None and 0 <= selected_idx < len(qualified_faces):
            target_face = qualified_faces[selected_idx]
        else:
            # Default: largest area (closest)
            target_face = max(qualified_faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))

        embedding = target_face.embedding.tolist()
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO persons (name, face_embedding)
            VALUES (%s, %s::vector)
            RETURNING id;
        """, (name, embedding))
        person_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()

        return jsonify({'status': 'success', 'message': f'{name} registered successfully!', 'id': person_id})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/recognize', methods=['POST'])
def recognize_face():
    """
    Recognize all qualified faces in the frame.
    Returns list of names, similarities, and bounding boxes.
    """
    try:
        data = request.get_json()
        image_data = data.get('image')
        if not image_data:
            return jsonify({'status': 'error', 'message': 'Image required'}), 400

        if ',' in image_data:
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)

        # Use qualified faces with slightly relaxed area (to also detect smaller faces)
        qualified_faces = get_qualified_faces(image_bytes, min_area=5000, min_confidence=0.5, frontal=False)
        results = []
        for face in qualified_faces:
            name, similarity = find_similar_face(face.embedding.tolist())
            bbox = face.bbox.astype(int).tolist()
            results.append({
                'name': name,
                'similarity': round(float(similarity), 4),
                'bbox': bbox
            })
        return jsonify({'status': 'success', 'results': results})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# ---- Management Routes ----
@app.route('/list_persons', methods=['GET'])
def list_persons():
    """Return all registered persons (id, name, created_at)."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, name, created_at FROM persons ORDER BY created_at DESC")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        persons = [{'id': r[0], 'name': r[1], 'created_at': r[2].strftime('%Y-%m-%d %H:%M:%S')} for r in rows]
        return jsonify(persons)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/delete_person/<int:person_id>', methods=['DELETE'])
def delete_person(person_id):
    """Delete a person by ID."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM persons WHERE id = %s", (person_id,))
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({'status': 'success', 'message': 'Deleted successfully'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/update_person/<int:person_id>', methods=['PUT'])
def update_person(person_id):
    """Update a person's name."""
    try:
        data = request.get_json()
        new_name = data.get('name')
        if not new_name:
            return jsonify({'status': 'error', 'message': 'Name required'}), 400
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("UPDATE persons SET name = %s WHERE id = %s RETURNING id", (new_name, person_id))
        if cur.fetchone() is None:
            return jsonify({'status': 'error', 'message': 'Person not found'}), 404
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({'status': 'success', 'message': 'Name updated'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# ---- Database Initialization on Startup ----
with app.app_context():
    create_tables()

# ---- Auto-open browser (optional) ----
def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")

if __name__ == '__main__':
    threading.Timer(1, open_browser).start()
    app.run(host='0.0.0.0', port=5000, debug=True)
