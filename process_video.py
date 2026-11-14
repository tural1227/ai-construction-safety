import os
import cv2
import time
import json
from datetime import datetime
from ultralytics import YOLO

def process_construction_video(
    video_path="viii.mp4",
    output_path="output_viii_analyzed.mp4",
    model_path="models/Hansung-Cho/best.pt",
    conf_thresh=0.35,
    save_violations=True,
    violations_dir="violations"
):
    print(f"[*] AI İnşaat Təhlükəsizlik Sistemi işə düşür...")
    print(f"[*] Video faylı: {video_path}")
    print(f"[*] Model yüklənir: {model_path}...")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video faylı tapılmadı: {video_path}")

    if save_violations:
        os.makedirs(violations_dir, exist_ok=True)

    # YOLO PPE Model
    model = YOLO(model_path)
    print(f"[*] Model uğurla yükləndi. Aşkarlama sinifləri: {model.names}")

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Video yazıcı təyin edilir
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"[*] Video parametrləri: {width}x{height} @ {fps:.1f} FPS, Cəmi kadr: {total_frames}")

    frame_num = 0
    violation_count_total = 0
    violation_records = []
    last_saved_time = 0

    # Rənglər (BGR formatında)
    COLOR_RED = (40, 40, 235)      # Kaskasız (Təhlükə)
    COLOR_GREEN = (40, 200, 40)    # Kaskalı (Təhlükəsiz)
    COLOR_YELLOW = (0, 215, 255)   # Jilet
    COLOR_BLUE = (255, 140, 0)     # Şəxs
    COLOR_WHITE = (255, 255, 255)
    COLOR_BLACK = (0, 0, 0)

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # YOLO PPE aşkarlaması
        results = model(frame, conf=conf_thresh, verbose=False)[0]

        hardhat_count = 0
        no_hardhat_count = 0
        vest_count = 0
        no_vest_count = 0
        person_count = 0

        current_frame_violations = []

        # Detalları ayırd et
        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])
            xyxy = [int(v) for v in box.xyxy[0]]
            x1, y1, x2, y2 = xyxy

            # Sayğaclar
            if cls_name == "Hardhat":
                hardhat_count += 1
                box_color = COLOR_GREEN
                label = f"KASKA VAR: {conf:.2f}"
            elif cls_name == "NO-Hardhat":
                no_hardhat_count += 1
                box_color = COLOR_RED
                label = f"TEHLUKE! KASKA YOXDUR: {conf:.2f}"
                current_frame_violations.append((xyxy, conf))
            elif cls_name == "Safety Vest":
                vest_count += 1
                box_color = COLOR_YELLOW
                label = f"Jilet: {conf:.2f}"
            elif cls_name == "NO-Safety Vest":
                no_vest_count += 1
                continue  # Vizual səliqəlilik üçün yalnız kaskaya və əsas obyektlərə fokuslanırıq
            elif cls_name == "Person":
                person_count += 1
                box_color = COLOR_BLUE
                label = f"Isci: {conf:.2f}"
            else:
                continue

            # Çərçivə və vizual xəbərdarlıq
            thickness = 3 if cls_name == "NO-Hardhat" else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)

            # Etiket fonu və mətni
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55 if cls_name == "NO-Hardhat" else 0.45
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, 1)

            label_bg_y1 = max(0, y1 - text_h - 8)
            label_bg_y2 = y1
            cv2.rectangle(frame, (x1, label_bg_y1), (x1 + text_w + 10, label_bg_y2), box_color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 4), font, font_scale, COLOR_WHITE, 1, cv2.LINE_AA)

            # Əgər kaska yoxdursa, başın üstündə diqqətçəkən yanıb-sönən xəbərdarlıq nişanı
            if cls_name == "NO-Hardhat":
                cv2.circle(frame, (int((x1 + x2) / 2), max(20, y1 - 20)), 12, COLOR_RED, -1)
                cv2.putText(frame, "!", (int((x1 + x2) / 2) - 4, max(20, y1 - 15)), font, 0.6, COLOR_WHITE, 2)

        # Əgər pozuntu varsa və müəyyən aralıq keçibsə (saniyədə 1 dəfə snapshot), qeydə al
        if current_frame_violations and save_violations and (frame_num - last_saved_time > int(fps)):
            last_saved_time = frame_num
            violation_count_total += len(current_frame_violations)
            snap_filename = f"violation_frame_{frame_num}.jpg"
            snap_path = os.path.join(violations_dir, snap_filename)
            cv2.imwrite(snap_path, frame)

            record = {
                "frame": frame_num,
                "timestamp": current_time_str,
                "no_hardhat_workers": len(current_frame_violations),
                "snapshot": snap_filename
            }
            violation_records.append(record)
            print(f" [!] XEBERDARLIQ: Kadr #{frame_num} - Kaskasiz {len(current_frame_violations)} nefer askarlandi! Snapshot: {snap_filename}")

        # --- ÜST MƏLUMAT VƏ STATİSTİKA PANELİ (Overlay HUD) ---
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 85), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Panel Mətnləri
        title = "INSAAT SAHESI TEHLUKESIZLIK MONITORINQI (AI HSE)"
        cv2.putText(frame, title, (20, 28), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 220, 255), 2, cv2.LINE_AA)

        time_fps_text = f"Vaxt: {current_time_str} | Kadr: {frame_num}/{total_frames}"
        cv2.putText(frame, time_fps_text, (width - 420, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # Statistika xətti
        stats_kaska = f"Kaskali: {hardhat_count}"
        cv2.putText(frame, stats_kaska, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_GREEN, 2, cv2.LINE_AA)

        stats_kaskasiz = f"Kaskasiz (POZUNTU): {no_hardhat_count}"
        color_alert = COLOR_RED if no_hardhat_count > 0 else (180, 180, 180)
        cv2.putText(frame, stats_kaskasiz, (220, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color_alert, 2, cv2.LINE_AA)

        stats_vest = f"Jiletli: {vest_count}"
        cv2.putText(frame, stats_vest, (560, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_YELLOW, 2, cv2.LINE_AA)

        if no_hardhat_count > 0:
            status_alert = "DIQQET: QAYDAPOZUNTUSU!"
            cv2.putText(frame, status_alert, (width - 340, 65), cv2.FONT_HERSHEY_DUPLEX, 0.65, COLOR_RED, 2, cv2.LINE_AA)
        else:
            status_ok = "TEHLUKESIZLIK VEZIYYETI: NORMAL"
            cv2.putText(frame, status_ok, (width - 380, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_GREEN, 1, cv2.LINE_AA)

        # Video yazıcıya kadrı göndər
        out.write(frame)

        if frame_num % 30 == 0 or frame_num == total_frames:
            elapsed = time.time() - start_time
            fps_proc = frame_num / elapsed if elapsed > 0 else 0
            print(f"[*] Emal olunur: {frame_num}/{total_frames} kadr ({frame_num/total_frames*100:.1f}%) - {fps_proc:.1f} FPS")

    cap.release()
    out.release()

    # Pozuntuların tarixçəsini JSON-a yaz
    log_file = os.path.join(violations_dir, "violations_log.json")
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(violation_records, f, indent=4, ensure_ascii=False)

    print("\n" + "="*50)
    print(f"[+] EMAL TAMAMLANDI!")
    print(f"[+] Netice videosu: {output_path}")
    print(f"[+] Qeyde alinan kaskasiz pozuntu snapshot-lari: {violations_dir}/")
    print(f"[+] Pozuntu log fayli: {log_file}")
    print("="*50)

if __name__ == "__main__":
    process_construction_video()
