import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
import os
from PIL import Image
from itertools import combinations
import sqlite3
from datetime import datetime

# ==========================================
# БАЗА ДАННИ (SQLite)
# ==========================================
def init_db(db_name="signatures.db"):
    """Създава базата данни и таблицата, ако не съществуват."""
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    # Създаваме таблица за история на верификациите
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS verification_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pair_names TEXT NOT NULL,
            ssim_score REAL NOT NULL,
            mse_score REAL NOT NULL,
            result TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
    print("[DB] Базата данни е инициализирана успешно.")

def log_verification(pair_names, ssim_score, mse_score, result, db_name="signatures.db"):
    """Записва резултата от конкретна проверка в базата данни."""
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO verification_logs (pair_names, ssim_score, mse_score, result)
        VALUES (?, ?, ?, ?)
    ''', (pair_names, ssim_score, mse_score, result))
    conn.commit()
    conn.close()
    print(f"[DB] Записано в базата: {pair_names} | Резултат: {result}")

def show_all_logs(db_name="signatures.db"):
    """Извежда цялата история от базата данни (за проверка)."""
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM verification_logs")
    rows = cursor.fetchall()
    print("\n--- ИСТОРИЯ НА ВЕРИФИКАЦИИТЕ В БАЗАТА ---")
    for row in rows:
        print(f"ID: {row[0]:<3} | Двойка: {row[1]:<20} | SSIM: {row[2]:.4f} | MSE: {row[3]:>8.2f} | Резултат: {row[4]:<12} | Време: {row[5]}")
    conn.close()

# ==========================================
# 1. ПРЕДВАРИТЕЛНА ОБРАБОТКА И СЕГМЕНТИРАНЕ
# ==========================================
def load_image_pil(image_path):
    """Зарежда изображение чрез PIL и обработва прозрачен фон."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Изображението не е намерено: {image_path}")
    try:
        pil_img = Image.open(image_path).convert('RGBA')
        img_np = np.array(pil_img)
    except Exception as e:
        raise ValueError(f"ГРЕШКА: Снимката е повредена или форматът не се поддържа: {e}")

    # Заместване на прозрачния фон с бял
    alpha = img_np[:, :, 3:4] / 255.0
    rgb = img_np[:, :, :3]
    white = np.ones_like(rgb, dtype=np.uint8) * 255
    img_rgb = (rgb * alpha + white * (1 - alpha)).astype(np.uint8)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return img_gray

def segment_signature(img_gray):
    """Сегментира подписа чрез бинаризация (Otsu) и Bounding Box."""
    _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Морфологично затваряне за свързване на фрагментирани щрихи
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Морфологично отваряне за премахване на шум
    noise_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, noise_kernel, iterations=1)

    # Намиране на Bounding Box
    coords = cv2.findNonZero(cleaned)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        cropped = img_gray[y:y+h, x:x+w]
        return cropped, (x, y, w, h)
    return img_gray, (0, 0, img_gray.shape[1], img_gray.shape[0])

# ==========================================
# 2. МАТЕМАТИЧЕСКО СРАВНЕНИЕ (SSIM & MSE)
# ==========================================
# ==========================================
# 2. МАТЕМАТИЧЕСКО СРАВНЕНИЕ (SSIM & MSE)
# ==========================================
def compare_signatures(path1, path2, target_size=(300, 150)):
    """Сравнява два подписа чрез SSIM и MSE индекси с правилен препроцесинг."""
    img1_gray = load_image_pil(path1)
    img2_gray = load_image_pil(path2)

    sig1, _ = segment_signature(img1_gray)
    sig2, _ = segment_signature(img2_gray)

    # --- НОВИЯТ ДВИГАТЕЛ (Негатив + Запазване на пропорциите) ---
    def process_for_ssim(cropped_img, target_size):
        # 1. Обръщаме в негатив (светещо бяло мастило на черен фон)
        inverted = cv2.bitwise_not(cropped_img)
        
        # 2. Запазваме пропорциите (Aspect Ratio)
        h, w = inverted.shape
        scale = min(target_size[0] / w, target_size[1] / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(inverted, (new_w, new_h))
        
        # 3. Слагаме ги в центъра на чисто ЧЕРНО платно
        canvas = np.zeros((target_size[1], target_size[0]), dtype=np.uint8)
        x_offset = (target_size[0] - new_w) // 2
        y_offset = (target_size[1] - new_h) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas

    sig1_processed = process_for_ssim(sig1, target_size)
    sig2_processed = process_for_ssim(sig2, target_size)

    # Изчисляване на SSIM
    ssim_score, diff_map = ssim(sig1_processed, sig2_processed, full=True)
    
    # Изчисляване на MSE (База за сравнение)
    mse_score = mse(sig1_processed, sig2_processed)
    
    # Връщаме обработените снимки, за да се покажат в графиката (изглеждат много 'хакерски' на черен фон)
    return ssim_score, mse_score, diff_map, sig1_processed, sig2_processed

# ==========================================
# 3. ВИЗУАЛИЗАЦИЯ И РЕГИСТРИРАНЕ
# ==========================================
def verify_pair(path1, path2, name1="Подпис 1", name2="Подпис 2", ssim_threshold=0.55):
    """Сравнява два подписа, визуализира резултатите и записва в базата."""
    ssim_score, mse_score, diff_map, sig1_vis, sig2_vis = compare_signatures(path1, path2)
    
    ssim_authentic = ssim_score >= ssim_threshold
    ssim_result = "Автентичен" if ssim_authentic else "Фалшификат"

    # --- Визуализация ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Сравнение: {name1} vs {name2}", fontsize=14, fontweight='bold')

    axes[0, 0].imshow(sig1_vis, cmap='gray')
    axes[0, 0].set_title(f"{name1} (Референтен)", fontsize=11)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(sig2_vis, cmap='gray')
    axes[0, 1].set_title(f"{name2} (Тестов)", fontsize=11)
    axes[0, 1].axis('off')

    axes[1, 0].imshow(diff_map, cmap='hot')
    axes[1, 0].set_title("SSIM карта на разликите", fontsize=11)
    axes[1, 0].axis('off')

    axes[1, 1].axis('off')
    result_color = '#2ecc71' if ssim_authentic else '#e74c3c'
    
    # Текстът вече показва MSE и SSIM
    results_text = (
        f"=== РЕЗУЛТАТИ ===\n\n"
        f"Базов алгоритъм: MSE\n"
        f"  (Mean Squared Error)\n"
        f"  Стойност: {mse_score:.2f}\n"
        f"  (По-ниско е по-добре)\n\n"
        f"Основен алгоритъм: SSIM\n"
        f"  (Structural Similarity)\n"
        f"  Стойност: {ssim_score:.4f}\n"
        f"  Праг: {ssim_threshold}\n\n"
        f"== Заключение ==\n"
        f"  {ssim_result}"
    )
    
    axes[1, 1].text(0.1, 0.95, results_text, transform=axes[1, 1].transAxes,
                     fontsize=12, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor=result_color, alpha=0.15))

    plt.tight_layout()
    plt.savefig(f"result_{name1}_vs_{name2}.png", dpi=150, bbox_inches='tight')
    plt.close() # Затваряме фигурата, за да не трупаме прозорци, ако са много снимки

    # ЗАПИС В БАЗАТА ДАННИ
    log_verification(f"{name1} vs {name2}", ssim_score, mse_score, ssim_result)

    return {
        'pair': f"{name1} vs {name2}",
        'mse_score': mse_score,
        'ssim_score': ssim_score,
        'ssim_result': ssim_result
    }

def run_full_test(image_dir="test_images"):
    """Тества всички комбинации от подписи и генерира обобщена таблица."""
    extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    images = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(extensions)
    ])

    if len(images) < 2:
        print("Необходими са поне 2 изображения в папката.")
        return

    print(f"\nНамерени {len(images)} изображения: {images}")
    print(f"Общо комбинации за сравнение: {len(list(combinations(images, 2)))}")
    print("=" * 60)

    results = []

    for img1_name, img2_name in combinations(images, 2):
        path1 = os.path.join(image_dir, img1_name)
        path2 = os.path.join(image_dir, img2_name)
        name1 = os.path.splitext(img1_name)[0]
        name2 = os.path.splitext(img2_name)[0]

        print(f"Сравняване: {name1} vs {name2}...")
        result = verify_pair(path1, path2, name1, name2)
        results.append(result)

    # Обобщена таблица
    print("\n" + "=" * 60)
    print("ОБОБЩЕНА ТАБЛИЦА С РЕЗУЛТАТИ")
    print("=" * 60)
    print(f"{'Двойка':<20} {'MSE':<12} {'SSIM':<10} {'Решение':<15}")
    print("-" * 60)
    for r in results:
        print(f"{r['pair']:<20} {r['mse_score']:<12.2f} {r['ssim_score']:<10.4f} {r['ssim_result']:<15}")
    print("=" * 60)

    # Графика на SSIM стойностите
    fig, ax = plt.subplots(figsize=(10, 5))
    pairs = [r['pair'] for r in results]
    scores = [r['ssim_score'] for r in results]
    colors = ['#2ecc71' if r['ssim_result'] == 'Автентичен' else '#e74c3c' for r in results]

    bars = ax.bar(range(len(pairs)), scores, color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0.55, color='orange', linestyle='--', linewidth=2, label='Праг (0.55)')
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(pairs, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('SSIM индекс')
    ax.set_title('Сравнение на всички двойки подписи (SSIM метод)')
    ax.legend()
    ax.set_ylim(0, 1)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{score:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig("results_summary.png", dpi=150, bbox_inches='tight')
    plt.show()

    return results

# ==========================================
# 4. ИЗПЪЛНЕНИЕ НА ПРОГРАМАТА
# ==========================================
if __name__ == "__main__":
    init_db()

    print("=" * 50)
    print(" СИСТЕМА ЗА ВЕРИФИКАЦИЯ НА ПОДПИСИ")
    print(" АЛГОРИТМИ: SSIM (Основен) vs MSE (Базов)")
    print("=" * 50)

    if os.path.exists("test_images"):
        results = run_full_test("test_images")
        print("\nВсички проверки са приключени и графиките са запазени.")
        
        # Покажи какво има в базата накрая
        show_all_logs()
    else:
        print("\nГРЕШКА: Папката 'test_images' не е намерена.")
        print("Моля, създайте я и добавете снимки на подписи.")