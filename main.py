import cv2
import numpy as np
from ultralytics import YOLO
import os

MODEL_PATH = "best.pt"
EARRING_PATH = "earring.png"

if not os.path.exists(MODEL_PATH):
    print(f"ОШИБКА: Модель не найдена по пути: {MODEL_PATH}")
    exit(1)

# Загрузка модели
print("Загрузка модели YOLO...")
best_model = YOLO(MODEL_PATH)
print("Модель загружена успешно!")

# Загрузка сережки
earring_img = cv2.imread(EARRING_PATH, cv2.IMREAD_UNCHANGED)
if earring_img is not None:
    h, w = earring_img.shape[:2]
    print(f"Загружена сережка: {w}x{h} пикселей")
    
    if max(h, w) > 300:
        scale_factor = 300.0 / max(h, w)
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        earring_img = cv2.resize(earring_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        print(f"Новый размер: {new_w}x{new_h} пикселей")

earring_h, earring_w = earring_img.shape[:2]

def overlay_earring_safe(background, earring, x, y, scale=1.0):
    if earring.shape[2] != 4:
        earring = cv2.cvtColor(earring, cv2.COLOR_BGR2BGRA)
    
    h, w = earring.shape[:2]
    
    if scale != 1.0:
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        earring_scaled = cv2.resize(earring, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        earring_scaled = earring
    
    h_scaled, w_scaled = earring_scaled.shape[:2]
    
    y1 = max(0, y - h_scaled//2)
    y2 = min(background.shape[0], y + h_scaled//2)
    x1 = max(0, x - w_scaled//2)
    x2 = min(background.shape[1], x + w_scaled//2)
    
    if y1 >= y2 or x1 >= x2:
        return background
    
    region_h = y2 - y1
    region_w = x2 - x1
    
    earring_y1 = max(0, h_scaled//2 - y)
    earring_y2 = earring_y1 + region_h
    earring_x1 = max(0, w_scaled//2 - x)
    earring_x2 = earring_x1 + region_w
    
    earring_y2 = min(h_scaled, earring_y2)
    earring_x2 = min(w_scaled, earring_x2)
    
    earring_part = earring_scaled[earring_y1:earring_y2, earring_x1:earring_x2]
    
    if earring_part.shape[0] != region_h or earring_part.shape[1] != region_w:
        earring_part = cv2.resize(earring_part, (region_w, region_h))
    
    if earring_part.shape[2] == 4:
        alpha = earring_part[:, :, 3] / 255.0
        for c in range(3):
            background[y1:y2, x1:x2, c] = alpha * earring_part[:, :, c] + (1 - alpha) * background[y1:y2, x1:x2, c]
    else:
        background[y1:y2, x1:x2] = earring_part
    
    return background

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ОШИБКА: Не удалось открыть веб-камеру!")
    exit(1)

# Получаем параметры камеры
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Камера: {width}x{height} пикселей, {fps} FPS")

print("\nЗапуск обработки в реальном времени...")
print("Нажмите 'q' для выхода")
print("Нажмите 's' для сохранения скриншота")
print("Нажмите 'f' для переключения полноэкранного режима")

frame_count = 0
processed_count = 0
last_positions = []  # Для сглаживания движения
show_debug = False  # Показывать ли отладочную информацию
fullscreen = False  # Полноэкранный режим

# Создаем окно
cv2.namedWindow('Virtual Earring', cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("ОШИБКА: Не удалось получить кадр с камеры")
        break
    
    frame_count += 1
    
    # Зеркальное отражение
    frame = cv2.flip(frame, 1)
    
    # Детекция уха
    results = best_model(frame, verbose=False)
    
    ear_detected = False
    scale_used = 0.3
    ear_position = None
    
    for result in results:
        if result.keypoints is not None and len(result.keypoints.data) > 0:
            kpt_data = result.keypoints.data[0].cpu().numpy()
            
            if len(kpt_data) > 0 and kpt_data[0, 2] > 0.3:
                x, y = int(kpt_data[0, 0]), int(kpt_data[0, 1])
                
                # Сглаживание движения
                last_positions.append((x, y))
                if len(last_positions) > 5:
                    last_positions.pop(0)
                
                if len(last_positions) > 0:
                    avg_x = int(np.mean([p[0] for p in last_positions]))
                    avg_y = int(np.mean([p[1] for p in last_positions]))
                    x, y = avg_x, avg_y
                
                # Расчет масштаба
                if result.boxes is not None and len(result.boxes) > 0:
                    bbox = result.boxes.data[0][:4].cpu().numpy()
                    ear_width = bbox[2] - bbox[0]
                    
                    target_size = ear_width * 0.75
                    scale_used = target_size / max(earring_w, earring_h)
                    scale_used = max(0.3, min(1.0, scale_used))
                
                # Накладываем сережку
                frame = overlay_earring_safe(frame, earring_img, x, y, scale_used)
                ear_detected = True
                processed_count += 1
                ear_position = (x, y)
                
                # Отладочная информация
                if show_debug:
                    if result.boxes is not None:
                        bbox = result.boxes.data[0][:4].cpu().numpy().astype(int)
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    
                    cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
                    cv2.putText(frame, f"Ear", (x+10, y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    status_color = (0, 255, 0) if ear_detected else (0, 0, 255)
    status_text = "EAR DETECTED" if ear_detected else "NO EAR"
    
    cv2.putText(frame, status_text, (20, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
    
    cv2.putText(frame, f"FPS: {frame_count % 100}", (20, 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.putText(frame, f"Scale: {scale_used:.2f}", (20, 110), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.putText(frame, "Q: Quit | S: Save | F: Fullscreen | D: Debug", 
               (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    cv2.imshow('Virtual Earring', frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):  # Выход
        break
    elif key == ord('s'):  # Сохранить скриншот
        timestamp = cv2.getTickCount()
        filename = f"screenshot_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Скриншот сохранен: {filename}")
    elif key == ord('f'):  # Переключить полноэкранный режим
        fullscreen = not fullscreen
        if fullscreen:
            cv2.setWindowProperty('Virtual Earring', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty('Virtual Earring', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    elif key == ord('d'):  # Переключить отладку
        show_debug = not show_debug
        print(f"Отладочный режим: {'ВКЛ' if show_debug else 'ВЫКЛ'}")
    elif key == ord('+'):  # Увеличить сережку
        earring_img = cv2.resize(earring_img, (int(earring_w*1.1), int(earring_h*1.1)))
        earring_h, earring_w = earring_img.shape[:2]
        print(f"Увеличена сережка: {earring_w}x{earring_h}")
    elif key == ord('-'):  # Уменьшить сережку
        earring_img = cv2.resize(earring_img, (int(earring_w*0.9), int(earring_h*0.9)))
        earring_h, earring_w = earring_img.shape[:2]
        print(f"Уменьшена сережка: {earring_w}x{earring_h}")

cap.release()
cv2.destroyAllWindows()

print(f"\n{'='*60}")
print("СТАТИСТИКА РАБОТЫ:")
print(f"{'='*60}")
print(f"Всего обработано кадров: {frame_count}")
print(f"Кадров с сережкой: {processed_count}")
if frame_count > 0:
    print(f"Успешность: {processed_count/frame_count*100:.1f}%")
print(f"{'='*60}")