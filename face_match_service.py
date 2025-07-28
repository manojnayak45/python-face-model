from flask import Flask, request, jsonify
from deepface import DeepFace
import os
from werkzeug.utils import secure_filename
import cv2
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Folder paths
BASE_UPLOAD_FOLDER = os.path.abspath("../backedn-new/uploads")
GUEST_TEMP_FOLDER = os.path.abspath("temp")
app.config['UPLOAD_FOLDER'] = BASE_UPLOAD_FOLDER

# Ensure guest temp folder exists
os.makedirs(GUEST_TEMP_FOLDER, exist_ok=True)

# ✅ Load FaceNet model once
print("⏳ Loading FaceNet model...")
model = DeepFace.build_model("Facenet512")
print("✅ Model loaded")

# Resize function
def resize_image(path, size=(224, 224)):
    try:
        img = cv2.imread(path)
        if img is not None:
            h, w, _ = img.shape
            if h != size[1] or w != size[0]:
                img = cv2.resize(img, size)
                cv2.imwrite(path, img)
    except Exception as e:
        print(f"❌ Resize failed for {path}: {e}")

@app.route('/match', methods=['POST'])
def match_faces():
    start_time = time.time()

    guest_photo = request.files.get('guestPhoto')
    event_folder = request.form.get('eventFolder')
    guest_email = request.form.get('email')

    if not guest_photo or not event_folder or not guest_email:
        return jsonify({"error": "Missing data"}), 400

    folder_path = os.path.join(app.config['UPLOAD_FOLDER'], event_folder)
    guest_filename = secure_filename(guest_photo.filename)
    guest_path = os.path.join(GUEST_TEMP_FOLDER, guest_filename)

    if not os.path.exists(folder_path):
        return jsonify({"error": f"Folder '{folder_path}' not found"}), 400

    guest_photo.save(guest_path)
    resize_image(guest_path)
    print("📸 Guest photo saved and resized:", guest_path)

    matches = []
    for img in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img)
        try:
            resize_image(img_path)
            result = DeepFace.verify(
                guest_path,
                img_path,
                model_name="Facenet512",
                enforce_detection=False
            )
            print(f"✅ Compared: {img} | Verified: {result['verified']} | Distance: {result['distance']:.4f}")
            if result["verified"]:
                matches.append(img)
        except Exception as e:
            print(f"❌ Compare error with {img}: {e}")

    # Cleanup guest image
    try:
        os.remove(guest_path)
    except:
        pass

    # ✅ Send email with inline attached images
    if matches:
        try:
            msg = MIMEMultipart("related")
            msg["Subject"] = "Your Matching Wedding Photos 🎉"
            msg["From"] = os.getenv("EMAIL_USER")
            msg["To"] = guest_email

            msg_alternative = MIMEMultipart("alternative")
            msg.attach(msg_alternative)

            html = "<h2>Hi! We found your matching photos 🎯</h2>"

            for idx, m in enumerate(matches):
                cid = f"img{idx}"
                html += f'<img src="cid:{cid}" style="width:300px;margin:10px;" />'

            msg_alternative.attach(MIMEText(html, "html"))

            # Attach image files using Content-ID
            for idx, m in enumerate(matches):
                img_path = os.path.join(folder_path, m)
                with open(img_path, "rb") as f:
                    mime_img = MIMEImage(f.read())
                    mime_img.add_header("Content-ID", f"<img{idx}>")
                    msg.attach(mime_img)

            # Send email via Gmail SMTP
            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()
                server.login(os.getenv("EMAIL_USER"), os.getenv("EMAIL_PASS"))
                server.sendmail(msg["From"], msg["To"], msg.as_string())
                print("📩 Email sent to:", guest_email)

        except Exception as e:
            print("❌ Email error:", e)
    else:
        print("⚠️ No matches found — no email sent.")

    end_time = time.time()
    print(f"⏱️ Match completed in {round(end_time - start_time, 2)} seconds")

    return jsonify({"matches": matches})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
