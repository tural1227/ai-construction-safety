import os
import cv2
import time
import json
import threading
import numpy as np
from datetime import datetime
from flask import Flask, render_template, Response, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
VIOLATIONS_FOLDER = os.path.join(BASE_DIR, 'violations')
PPE_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'Hansung-Cho', 'best.pt')
PERSON_MODEL_PATH = os.path.join(BASE_DIR, 'yolo11s.pt')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VIOLATIONS_FOLDER, exist_ok=True)

DEFAULT_VIDEO = os.path.join(BASE_DIR, 'viii.mp4')
if os.path.exists(DEFAULT_VIDEO):
    import shutil
    dest = os.path.join(UPLOAD_FOLDER, 'viii.mp4')
    if not os.path.exists(dest):
        shutil.copy(DEFAULT_VIDEO, dest)

def verify_construction_hardhat(crop):
    """
    Qara beretka, parça kepka, qəribə saçlar və keçəl başları
    inşaat kaskasından ayıran fiziki rəng və parlaqlıq filtri.
    İnşaat kaskaları əsasən SARI və AĞ (və ya Mavi/Narıncı) rəngdə olur.
    """
    if crop is None or crop.size < 40:
        return False, "Kiçik"

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h_channel, s_channel, v_channel = cv2.split(hsv)
    total_pixels = crop.shape[0] * crop.shape[1]

    avg_v = np.mean(v_channel)

    # 1. TÜND / QARA BAŞ ÖRTÜKLƏRİ (Qara beretka, tünd papaq, tünd saç)
    # Sarı və ya Ağ inşaat kaskası heç vaxt qara və ya tünd ola bilməz!
    if avg_v < 60:
        return False, "Qara beretka / tünd papaq / saç"

    # 2. SARI KASKA (Genişləndirilmiş real gün işığı spektri)
    yellow_mask = cv2.inRange(hsv, np.array([14, 25, 70]), np.array([44, 255, 255]))
    yellow_ratio = np.sum(yellow_mask > 0) / total_pixels

    # 3. AĞ KASKA (Yüksək lüminans, aşağı doymuşluq)
    white_mask = cv2.inRange(hsv, np.array([0, 0, 135]), np.array([180, 55, 255]))
    white_ratio = np.sum(white_mask > 0) / total_pixels

    # 4. İNSAN DƏRİSİ VƏ KEÇƏL BAŞ (Skin Tone)
    skin_mask = cv2.inRange(hsv, np.array([0, 20, 60]), np.array([25, 160, 230]))
    skin_ratio = np.sum(skin_mask > 0) / total_pixels

    # 5. MAVİ VƏ YA NARINCI KASKA
    blue_mask = cv2.inRange(hsv, np.array([90, 50, 70]), np.array([135, 255, 255]))
    blue_ratio = np.sum(blue_mask > 0) / total_pixels

    orange_mask = cv2.inRange(hsv, np.array([8, 100, 90]), np.array([18, 255, 255]))
    orange_ratio = np.sum(orange_mask > 0) / total_pixels

    if yellow_ratio >= 0.10:
        return True, "Sarı Kaska"
    
    if white_ratio >= 0.14 and skin_ratio < 0.35:
        return True, "Ağ Kaska"

    if blue_ratio >= 0.12:
        return True, "Mavi Kaska"

    if orange_ratio >= 0.12:
        return True, "Narıncı Kaska"

    if skin_ratio >= 0.38:
        return False, "Keçəl Baş (Dəri)"

    return False, "Adi papaq / saç"

class StreamManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.source_type = None
        self.source_path = None
        self.cap = None
        self.conf_thresh = 0.25
        self.last_violation_time = 0
        self.violation_cooldown = 1.5
        
        self.stats = {
            "total_persons": 0,
            "hardhat_count": 0,
            "no_hardhat_count": 0,
            "vest_count": 0,
            "status": "GÖZLƏMƏDƏDİR",
            "fps": 0.0,
            "source_name": "Seçilməyib"
        }
        
        print("[*] Sənaye Modelləri yüklənir...")
        self.person_model = YOLO(PERSON_MODEL_PATH)
        self.ppe_model = YOLO(PPE_MODEL_PATH)
        print("[*] YOLO11s High-Res + Dual PPE Fusion hazırdır!")

    def set_source(self, source, is_camera=False):
        with self.lock:
            if self.cap:
                self.cap.release()
            self.source_type = 'camera' if is_camera else 'file'
            self.source_path = source
            source_name = source if is_camera else os.path.basename(source)
            self.stats["source_name"] = source_name
            self.init_capture()

    def init_capture(self):
        if not self.source_path:
            self.cap = None
            return
        if self.source_type == 'camera':
            src = int(self.source_path) if str(self.source_path).isdigit() else self.source_path
            self.cap = cv2.VideoCapture(src)
        else:
            self.cap = cv2.VideoCapture(self.source_path)

    def get_frame(self):
        with self.lock:
            if not self.source_path or self.cap is None or not self.cap.isOpened():
                if self.source_path:
                    self.init_capture()
                
                if not self.source_path or self.cap is None or not self.cap.isOpened():
                    time.sleep(0.1)
                    # Standby placeholder canvas
                    placeholder = np.zeros((720, 1280, 3), dtype=np.uint8)
                    placeholder[:] = (248, 250, 252)
                    
                    cv2.rectangle(placeholder, (300, 180), (980, 520), (226, 232, 240), 2)
                    cv2.rectangle(placeholder, (302, 182), (978, 518), (255, 255, 255), -1)
                    
                    cv2.putText(placeholder, "GORUNTU MENBEYI GOZLENILIR", (360, 320), cv2.FONT_HERSHEY_DUPLEX, 1.1, (15, 23, 42), 2, cv2.LINE_AA)
                    cv2.putText(placeholder, "Zehmet olmasa sol panelden video yukleyin ve ya IP kamera qosun", (330, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (100, 116, 139), 1, cv2.LINE_AA)
                    cv2.putText(placeholder, "Fayl secildikden sonra AI analizi derhal baslayacaq", (400, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (148, 163, 184), 1, cv2.LINE_AA)
                    
                    self.stats["total_persons"] = 0
                    self.stats["hardhat_count"] = 0
                    self.stats["no_hardhat_count"] = 0
                    self.stats["status"] = "GÖZLƏMƏDƏDİR"
                    self.stats["fps"] = 0.0
                    
                    _, buffer = cv2.imencode('.jpg', placeholder, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    return buffer.tobytes()

            ret, frame = self.cap.read()
            if not ret:
                if self.source_type == 'file':
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.05)
                    return None

            t0 = time.time()
            img_h, img_w, _ = frame.shape

            # 1. Bütün Şəxsləri Yüksək Dəqiqliklə Aşkarlayırıq (YOLO11s at imgsz=960)
            person_results = self.person_model(
                frame,
                imgsz=960,
                conf=self.conf_thresh,
                classes=[0],
                verbose=False
            )[0]
            
            person_boxes = []
            for b in person_results.boxes:
                person_boxes.append((b.xyxy[0].cpu().numpy(), float(b.conf[0])))

            # 2. PPE Modelini Yüksək Həllolma ilə İcra Edirik (imgsz=960)
            ppe_results = self.ppe_model(frame, imgsz=960, conf=0.25, verbose=False)[0]
            candidate_hardhat_boxes = []
            direct_no_hardhat_boxes = []
            vest_boxes = []

            for b in ppe_results.boxes:
                cls_name = self.ppe_model.names[int(b.cls[0])]
                conf = float(b.conf[0])
                coords = b.xyxy[0].cpu().numpy()
                if cls_name == "Hardhat":
                    candidate_hardhat_boxes.append((coords, conf))
                elif cls_name == "NO-Hardhat":
                    direct_no_hardhat_boxes.append((coords, conf))
                elif cls_name == "Safety Vest":
                    vest_boxes.append((coords, conf))

            # 3. Yalnız Həqiqi İnşaat Kaskalarını Təsdiqlə (Rəng & Parlaqlıq Analizi)
            verified_hardhats = []
            for h_box, conf in candidate_hardhat_boxes:
                hx1, hy1, hx2, hy2 = [int(v) for v in h_box]
                crop_x1, crop_y1 = max(0, hx1), max(0, hy1)
                crop_x2, crop_y2 = min(img_w, hx2), min(img_h, hy2)
                h_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

                is_real_hardhat, reason = verify_construction_hardhat(h_crop)
                if is_real_hardhat:
                    verified_hardhats.append((h_box, conf, reason))

            COLOR_RED = (40, 40, 235)      # Kaskasız (Təhlükə)
            COLOR_GREEN = (40, 200, 40)    # Kaskalı
            COLOR_YELLOW = (0, 215, 255)   # Jilet
            COLOR_WHITE = (255, 255, 255)

            hardhat_cnt = 0
            no_hardhat_cnt = 0
            vest_cnt = len(vest_boxes)
            violations_detected = []

            # 4. Hər bir işçinin baş və kaska analizi
            for idx, (pb, p_conf) in enumerate(person_boxes, start=1):
                px1, py1, px2, py2 = [int(v) for v in pb]
                pw = px2 - px1
                ph = py2 - py1

                # Çox kiçik ləkələri keç
                if ph < 15 or pw < 8:
                    continue

                # Baş nahiyəsi: yuxarı 35%
                head_y2 = py1 + int(0.35 * ph)

                # A) Birbaşa NO-Hardhat varmı?
                direct_no_helmet = False
                for nhb, nh_conf in direct_no_hardhat_boxes:
                    nx1, ny1, nx2, ny2 = [int(v) for v in nhb]
                    if max(px1 - 15, nx1) < min(px2 + 15, nx2) and max(py1 - 20, ny1) < min(head_y2 + 20, ny2):
                        direct_no_helmet = True
                        break

                # B) Təsdiqlənmiş qanuni inşaat kaskası varmı?
                has_valid_helmet = False
                helmet_label_text = ""
                matched_conf = 0.0

                if not direct_no_helmet:
                    for hb, h_conf, reason in verified_hardhats:
                        hx1, hy1, hx2, hy2 = [int(v) for v in hb]
                        # Kəsişmə
                        if max(px1 - 15, hx1) < min(px2 + 15, hx2) and max(py1 - 30, hy1) < min(head_y2 + 25, hy2):
                            has_valid_helmet = True
                            helmet_label_text = reason
                            matched_conf = h_conf
                            break

                font = cv2.FONT_HERSHEY_SIMPLEX
                id_prefix = f"#{idx} "

                if has_valid_helmet:
                    # KASKA VAR (TƏHLÜKƏSİZ)
                    hardhat_cnt += 1
                    label = f"{id_prefix}{helmet_label_text}: {matched_conf:.2f}"
                    cv2.rectangle(frame, (px1, py1), (px2, py2), COLOR_GREEN, 2)
                    (tw, th), _ = cv2.getTextSize(label, font, 0.45, 1)
                    bg_y = max(0, py1 - th - 6)
                    cv2.rectangle(frame, (px1, bg_y), (px1 + tw + 6, py1), COLOR_GREEN, -1)
                    cv2.putText(frame, label, (px1 + 3, py1 - 3), font, 0.45, COLOR_WHITE, 1, cv2.LINE_AA)
                else:
                    # KASKA YOXDUR! (QAYDAPOZUNTUSU)
                    no_hardhat_cnt += 1
                    violations_detected.append(([px1, py1, px2, py2], p_conf))

                    cv2.rectangle(frame, (px1, py1), (px2, py2), COLOR_RED, 3)
                    cv2.rectangle(frame, (px1, py1), (px2, head_y2), (0, 0, 255), 2)

                    label = f"{id_prefix}KASKA YOXDUR!: {p_conf:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, font, 0.52, 1)
                    bg_y = max(0, py1 - th - 6)
                    cv2.rectangle(frame, (px1, bg_y), (px1 + tw + 6, py1), COLOR_RED, -1)
                    cv2.putText(frame, label, (px1 + 3, py1 - 3), font, 0.52, COLOR_WHITE, 1, cv2.LINE_AA)

                    cx = int((px1 + px2) / 2)
                    cy = max(20, py1 - 18)
                    cv2.circle(frame, (cx, cy), 13, COLOR_RED, -1)
                    cv2.putText(frame, "!", (cx - 4, cy + 5), font, 0.6, COLOR_WHITE, 2)


            now = time.time()
            if no_hardhat_cnt > 0 and (now - self.last_violation_time > self.violation_cooldown):
                self.last_violation_time = now
                timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                snap_name = f"violation_{timestamp_str}.jpg"
                snap_path = os.path.join(VIOLATIONS_FOLDER, snap_name)
                cv2.imwrite(snap_path, frame)

            dt = time.time() - t0
            fps = 1.0 / dt if dt > 0 else 0.0

            self.stats["total_persons"] = len(person_boxes)
            self.stats["hardhat_count"] = hardhat_cnt
            self.stats["no_hardhat_count"] = no_hardhat_cnt
            self.stats["vest_count"] = vest_cnt
            self.stats["status"] = "TEHLUKE!" if no_hardhat_cnt > 0 else "NORMAL"
            self.stats["fps"] = round(fps, 1)

            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            return buffer.tobytes()

manager = StreamManager()

@app.route('/')
def index():
    return render_template('index.html')

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

def gen_frames():
    while True:
        frame_bytes = manager.get_frame()
        if frame_bytes is None:
            time.sleep(0.04)
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def get_stats():
    return jsonify(manager.stats)

@app.route('/violations')
def list_violations():
    files = []
    if os.path.exists(VIOLATIONS_FOLDER):
        for f in sorted(os.listdir(VIOLATIONS_FOLDER), reverse=True):
            if f.endswith(('.jpg', '.png', '.jpeg')):
                files.append({
                    "filename": f,
                    "url": f"/violations/{f}",
                    "time": f.replace("violation_", "").replace(".jpg", "").replace("_", " ")
                })
    return jsonify(files)

@app.route('/violations/<filename>')
def get_violation_image(filename):
    return send_from_directory(VIOLATIONS_FOLDER, filename)

@app.route('/clear_violations', methods=['POST'])
def clear_violations():
    for f in os.listdir(VIOLATIONS_FOLDER):
        p = os.path.join(VIOLATIONS_FOLDER, f)
        if os.path.isfile(p):
            os.remove(p)
    return jsonify({"status": "cleared"})

@app.route('/available_videos')
def available_videos():
    vids = []
    if os.path.exists(UPLOAD_FOLDER):
        for f in os.listdir(UPLOAD_FOLDER):
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                vids.append(f)
    return jsonify(vids)

@app.route('/set_source', methods=['POST'])
def set_source():
    data = request.json or {}
    source_type = data.get('type', 'file')
    if source_type == 'camera':
        rtsp_url = data.get('url', '').strip()
        if not rtsp_url:
            return jsonify({"error": "Kamera URL qeyd olunmayıb"}), 400
        manager.set_source(rtsp_url, is_camera=True)
        return jsonify({"status": "ok", "source": rtsp_url})
    else:
        video_name = data.get('video_name')
        if not video_name:
            return jsonify({"error": "Video adı qeyd olunmayıb"}), 400
        full_path = os.path.join(UPLOAD_FOLDER, secure_filename(video_name))
        if not os.path.exists(full_path):
            return jsonify({"error": "Fayl tapılmadı"}), 404
        manager.set_source(full_path, is_camera=False)
        return jsonify({"status": "ok", "source": video_name})

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "Video faylı tapılmadı"}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "Fayl seçilməyib"}), 400
    
    filename = secure_filename(file.filename)
    dest_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(dest_path)
    manager.set_source(dest_path, is_camera=False)
    return jsonify({"status": "uploaded", "filename": filename})

@app.route('/set_conf', methods=['POST'])
def set_conf():
    data = request.json or {}
    conf = float(data.get('confidence', 0.25))
    manager.conf_thresh = max(0.1, min(0.9, conf))
    return jsonify({"status": "ok", "confidence": manager.conf_thresh})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    print(f"[*] Professional HSE Server ishe salir: http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, threaded=True, debug=False)
