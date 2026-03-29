import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import sqlite3
from PIL import Image

# ==========================================
# 1. БАЗА ДАННИ
# ==========================================
def init_db(db_name="signatures_system.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS identification_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            true_user TEXT NOT NULL,
            predicted_user TEXT NOT NULL,
            confidence_score REAL NOT NULL,
            result TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def log_result(true_user, predicted_user, score, result):
    conn = sqlite3.connect("signatures_system.db")
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO identification_logs (true_user, predicted_user, confidence_score, result)
        VALUES (?, ?, ?, ?)
    ''', (true_user, predicted_user, score, result))
    conn.commit()
    conn.close()

# ==========================================
# 2. ПРЕПРОЦЕСИНГ (БЕЗ ЧЕРНИ ЛЕНТИ - ДИРЕКТНО РАЗПЪВАНЕ)
# ==========================================
def preprocess_for_math(image_path, target_size=(300, 150)):
    pil_img = Image.open(image_path).convert('RGBA')
    img_np = np.array(pil_img)
    alpha = img_np[:, :, 3:4] / 255.0
    rgb = img_np[:, :, :3]
    white = np.ones_like(rgb, dtype=np.uint8) * 255
    img_rgb = (rgb * alpha + white * (1 - alpha)).astype(np.uint8)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(thresh)
    
    if coords is None:
        return np.zeros((target_size[1], target_size[0]), dtype=np.uint8), img_gray

    x, y, w, h = cv2.boundingRect(coords)
    cropped_thresh = thresh[y:y+h, x:x+w]
    cropped_original = img_gray[y:y+h, x:x+w] 

    canvas_math = cv2.resize(cropped_thresh, target_size)
    canvas_vis = cv2.resize(cropped_original, target_size)

    return canvas_math, canvas_vis

# ==========================================
# 3. ВИЗУАЛИЗАЦИЯ (ЗА ДОКУМЕНТАЦИЯТА)
# ==========================================
def generate_doc_example(test_vis, ref_vis, math_test, math_ref, test_name, ref_name, is_match, filename):
    ssim_score, diff_map = ssim(math_test, math_ref, full=True)
    mse_score = mse(math_test, math_ref)
    
    result_text = "СЪЩИЯТ ЧОВЕК (Одобрен)" if is_match else "РАЗЛИЧЕН ЧОВЕК (Отхвърлен)"
    result_color = '#2ecc71' if is_match else '#e74c3c'

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Сравнение за документация: {test_name} срещу профила на {ref_name}", fontsize=14, fontweight='bold')

    axes[0, 0].imshow(ref_vis, cmap='gray')
    axes[0, 0].set_title(f"Еталон ({ref_name})", fontsize=11)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(test_vis, cmap='gray')
    axes[0, 1].set_title(f"Тестов подпис ({test_name})", fontsize=11)
    axes[0, 1].axis('off')

    axes[1, 0].imshow(diff_map, cmap='hot')
    axes[1, 0].set_title("SSIM карта на разликите (върху мастилото)", fontsize=11)
    axes[1, 0].axis('off')

    axes[1, 1].axis('off')
    info_text = (
        f"=== АНАЛИЗ НА СИСТЕМАТА ===\n\n"
        f"Базов алгоритъм: MSE\n"
        f"  Грешка: {mse_score:.2f}\n\n"
        f"Основен алгоритъм: SSIM\n"
        f"  Структурно сходство: {ssim_score:.4f}\n\n"
        f"== Решение ==\n"
        f"  {result_text}"
    )
    axes[1, 1].text(0.1, 0.95, info_text, transform=axes[1, 1].transAxes,
                     fontsize=12, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor=result_color, alpha=0.2))

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

# ==========================================
# 4. ОСНОВНА СИСТЕМА (ИДЕНТИФИКАЦИЯ 1:N)
# ==========================================
def main():
    init_db()
    dataset_dir = "dataset"
    
    if not os.path.exists(dataset_dir):
        print(f"ГРЕШКА: Папката '{dataset_dir}' не съществува!")
        return

    users = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    
    profiles_math = {}
    profiles_vis = {}
    tests_math = {}
    tests_vis = {}

    print("============================================================")
    print(" ИНИЦИАЛИЗИРАНЕ НА СИСТЕМАТА (10 TRAIN / 10 TEST)")
    print("============================================================")
    for user in users:
        user_dir = os.path.join(dataset_dir, user)
        images = sorted(os.listdir(user_dir))
        
        if len(images) < 20:
            print(f"[ВНИМАНИЕ] Потребител {user} няма 20 снимки (Намерени: {len(images)}). Може да има грешки!")
            continue
            
        profiles_math[user], profiles_vis[user] = [], []
        tests_math[user], tests_vis[user] = [], []
        
        for img in images[:10]:
            m, v = preprocess_for_math(os.path.join(user_dir, img))
            profiles_math[user].append(m)
            profiles_vis[user].append(v)
            
        for img in images[10:20]:
            m, v = preprocess_for_math(os.path.join(user_dir, img))
            tests_math[user].append(m)
            tests_vis[user].append(v)

    y_true, y_pred = [], []
    
    # ПРОМЯНАТА: Променливи за следене на най-добрите и най-лошите сравнения
    best_match_data = {'score': -1, 'args': None}
    worst_mismatch_data = {'score': float('inf'), 'args': None}

    print("\nЗапочва процесът по идентификация...\n")
    for actual_user, test_imgs in tests_math.items():
        for i, test_img in enumerate(test_imgs):
            # print(f"--- Тестване на непознат подпис (Истина: [{actual_user}], Снимка {i+1}) ---")
            
            best_match_user = None
            highest_avg_score = -1
            
            for profile_user, train_imgs in profiles_math.items():
                scores = [ssim(test_img, t_img, full=True)[0] for t_img in train_imgs]
                avg_score = np.mean(scores)
                # print(f"  -> Сходство с профил [{profile_user}]: {avg_score:.4f}")
                
                if avg_score > highest_avg_score:
                    highest_avg_score = avg_score
                    best_match_user = profile_user
            
            is_correct = (actual_user == best_match_user)
            result_str = "УСПЕШНО" if is_correct else "ГРЕШКА"
            print(f"Истина: [{actual_user}] -> Решение: [{best_match_user}] ({result_str}) | Среден SSIM: {highest_avg_score:.4f}")
            
            y_true.append(actual_user)
            y_pred.append(best_match_user)
            log_result(actual_user, best_match_user, highest_avg_score, result_str)

            # --- ЛОГИКА ЗА НАМИРАНЕ НА РЕКОРДИТЕ ЗА ДОКУМЕНТАЦИЯТА ---
            # Взимаме първата снимка от профила, който системата е избрала
            ref_vis = profiles_vis[best_match_user][0]
            test_vis = tests_vis[actual_user][i]
            ref_math = profiles_math[best_match_user][0]
            
            # Изчисляваме 1:1 SSIM точно за тези две картинки
            current_1v1_ssim = ssim(test_img, ref_math, full=True)[0]
            plot_args = (test_vis, ref_vis, test_img, ref_math, actual_user, best_match_user)

            if is_correct:
                if current_1v1_ssim > best_match_data['score']:
                    best_match_data['score'] = current_1v1_ssim
                    best_match_data['args'] = plot_args
            else:
                if current_1v1_ssim < worst_mismatch_data['score']:
                    worst_mismatch_data['score'] = current_1v1_ssim
                    worst_mismatch_data['args'] = plot_args

    # Генерираме графиките ЧАК НАКРАЯ, след като сме намерили рекордите
    print("\n============================================================")
    print(" ГЕНЕРИРАНЕ НА ГРАФИКИ ЗА ДОКУМЕНТАЦИЯТА")
    print("============================================================")
    if best_match_data['args'] is not None:
        generate_doc_example(*best_match_data['args'], True, "doc_example_BEST_MATCH.png")
        print(f"-> Генериран 'doc_example_BEST_MATCH.png' (Най-категорично разпознат със SSIM: {best_match_data['score']:.4f})")
        
    if worst_mismatch_data['args'] is not None:
        generate_doc_example(*worst_mismatch_data['args'], False, "doc_example_WORST_MISMATCH.png")
        print(f"-> Генериран 'doc_example_WORST_MISMATCH.png' (Най-брутална грешка/отхвърляне със SSIM: {worst_mismatch_data['score']:.4f})")

    # ==========================================
    # 5. МАТРИЦА НА ГРЕШКИТЕ
    # ==========================================
    cm = confusion_matrix(y_true, y_pred, labels=users)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=users, yticklabels=users, annot_kws={"size": 16})
    plt.title("Многокласова Матрица на Грешките (10 Train / 10 Test)", fontsize=16)
    plt.ylabel('Истински Потребител', fontsize=14)
    plt.xlabel('Решение на Системата', fontsize=14)
    plt.tight_layout()
    plt.savefig("final_confusion_matrix.png", dpi=300)
    
    print("-> Генерирана 'final_confusion_matrix.png'")
    print("============================================================")
    print("Всички операции приключиха успешно!")

if __name__ == "__main__":
    main()