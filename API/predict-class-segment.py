from flask import Flask, request, jsonify, send_file
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
import os
import gdown
import io
from io import BytesIO
import base64

app = Flask(__name__)

#device setup
device = torch.device('cpu')

# Persiapkan direktori upload
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Fungsi untuk persiapan gambar klalifikasi
def prepare_image_classify(img, target_size=(256, 256)):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


#fungsi klasifikasi
def classify_image(img):
    CLASS_MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
    CLASS_MODEL_NAME = "predict_realFakemix(100 epoch weighted).keras"
    CLASS_MODEL_PATH = os.path.join(CLASS_MODEL_DIR, CLASS_MODEL_NAME)

    # Google Drive file ID from your link
    GDRIVE_FILE_ID = "13wFTPGWNh-YcnGU1MR5l_8Ygk_4BH2iq"
    GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

    if not os.path.exists(CLASS_MODEL_PATH):
        os.makedirs(CLASS_MODEL_DIR, exist_ok=True)
        print("Model not found. Downloading from Google Drive...")
        gdown.download(GDRIVE_URL, CLASS_MODEL_PATH, quiet=False)

    # Load model
    modelClassify = keras.models.load_model(CLASS_MODEL_PATH, compile=False)

    # Kelas output dari model
    class_names = ["Normal", "Benign", "Malignant"]

    """Melakukan klasifikasi pada gambar dan mengembalikan label, confidence, dan array prediksi."""
    img_array = prepare_image_classify(img)
    prediction = modelClassify.predict(img_array)
    confidenceClasify = float(np.max(prediction))
    label_index = int(np.argmax(prediction))
    label = class_names[label_index]
    return label, confidenceClasify, label_index


# SEGMENTASI
segment_base_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, num_classes=3)
# def get_model_instance_segmentation(num_classes):
#     model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
#     return model

def get_transform_single_image():
    """Transformasi untuk satu gambar inferensi (tanpa augmentasi)."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

#fungsi segmentasi
def segmentation(img, score_threshold=0.5):
    # Load model
    SEG_MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
    SEG_MODEL_NAME = "segmentation(512, filter kalsif -25%).pth"
    SEG_MODEL_PATH = os.path.join(SEG_MODEL_DIR, SEG_MODEL_NAME)

    SEG_MODEL_ID = "1ADYlp-ZlmYY69lNSPR5SDc7ejLxRRkIQ"
    SEG_MODEL_URL = f"https://drive.google.com/uc?id={SEG_MODEL_ID}"

    if not os.path.exists(SEG_MODEL_PATH):
        print("Downloading segmentation model...")
        gdown.download(SEG_MODEL_URL, SEG_MODEL_PATH, quiet=False)

    # num_classes = 3 # Background, Kanker, Kalsifikasi
    labels_map = {0: 'Background', 1: 'Kanker', 2: 'Kalsifikasi'} # Pastikan ini sesuai

    loaded_model = segment_base_model
    loaded_model.load_state_dict(torch.load(SEG_MODEL_PATH, map_location=device))
    loaded_model.to(device)
    loaded_model.eval()

    # Load image
    img_pil_original = img
    # memastikan mode RGB
    if img_pil_original.mode != "RGB":
        img_pil_original = img_pil_original.convert("RGB")
    
    transform = get_transform_single_image()
    img_tensor = transform(img_pil_original).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        prediction = loaded_model(img_tensor)[0]

    pred_boxes = prediction['boxes'].cpu().numpy()
    pred_labels = prediction['labels'].cpu().numpy()
    pred_scores = prediction['scores'].cpu().numpy()

    # Prepare figure for drawing
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(img_pil_original)
    ax.axis('off')

    valid_scores = []
    for j in range(len(pred_boxes)):
        score = float(pred_scores[j])
        if score >= score_threshold:
            valid_scores.append(score)
            x_min, y_min, x_max, y_max = pred_boxes[j]
            label_idx = pred_labels[j]
            label_text = labels_map.get(label_idx, f'Unknown ({label_idx})')

            color = 'red'
            if label_idx == 1:
                color = 'lime'
            elif label_idx == 2:
                color = 'orange'

            rect = patches.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=2,
                edgecolor=color,
                facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(x_min, y_min - 10, f'{label_text} ({score:.2f})', color='black',
                    fontsize=10, bbox=dict(facecolor=color, alpha=0.6, edgecolor='none', pad=1.5))

    avg_confidence = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

    # Convert Matplotlib figure to PIL Image
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)  # Close the figure to free memory
    buf.seek(0)
    segmented_img = Image.open(buf)

    return segmented_img, avg_confidence


# Fungsi watermark
def add_watermark(original_img, text):
    width, height = original_img.size
    font_size = max(12, width // 40)
    bar_height = max(40, height // 15)

    result_img = Image.new("RGB", (width, height + bar_height), "white")
    result_img.paste(original_img, (0, 0))
    draw = ImageDraw.Draw(result_img)

    font_path = os.path.join(os.path.dirname(__file__), "font", "DejaVuSans.ttf")
    font = ImageFont.truetype(font_path, size=font_size)

    # Teks utama
    text_bbox = draw.textbbox((0, 0), text, font=font)
    x = (width - text_bbox[2]) // 2
    y = height + (bar_height - text_bbox[3]) // 4
    draw.text((x, y), text, fill="black", font=font)

    # Branding
    mammo_text = "MammoDetect"
    mammo_bbox = draw.textbbox((0, 0), mammo_text, font=font)
    mammo_x = width - mammo_bbox[2] - (bar_height / 10)
    mammo_y = height + bar_height - mammo_bbox[3] - (bar_height / 9)
    draw.text((mammo_x, mammo_y), mammo_text, fill="black", font=font)

    return result_img

# Endpoint untuk prediksi
@app.route('/predict', methods=['POST'])
def predict():
    confidenceSegment = 0

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    image_stream_data = file.read()
    file.close()
    
    img = Image.open(BytesIO(image_stream_data))
    # file.stream.close() # Close the stream explicitly
    # del file # Delete the file object

    # Panggil fungsi klasifikasi terpisah
    label, confidenceClasify, _ = classify_image(img.copy())

    # if not normal do segmentation
    if label == "Benign" or label == "Malignant":
        segmented_img, avg_confidence = segmentation(img.copy(), score_threshold=0.5)

        if avg_confidence:
            confidenceSegment = avg_confidence
        if segmented_img:
            img = segmented_img
        #done segmentasi

    max_width = 640
    if img.width > max_width:
        ratio = max_width / img.width
        new_height = int(img.height * ratio)
        img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
        
    # add watermark when all done        
    result_text = f"Prediction: {label}"
    watermarked_img = add_watermark(img, result_text)

    # Simpan ke buffer
    buffer = io.BytesIO()
    watermarked_img.save(buffer, format='PNG')
    buffer.seek(0)

    # Base64 encode
    img_base64 = base64.b64encode(buffer.getvalue()).decode()

    # Metadata
    metadata = {
        "Klasifikasi": label,
        "ConfidenceClasify": round(confidenceClasify * 100, 2),
        "ConfidenceSegment": round(confidenceSegment * 100, 2),
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "watermarked_image_base64": img_base64
    }

    return jsonify(metadata)

if __name__ == '__main__':
    app.run(debug=False)
