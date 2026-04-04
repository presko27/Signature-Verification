import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import sqlite3
import time

# ==========================================
# КОНФИГУРАЦИЯ
# ==========================================
CONFIG = {
    # Размер за нормализация на подписите
    'target_size': (300, 150),
    
    # Тежести за комбиниран score (SSIM + MSE)
    'ssim_weight': 0.85,
    'mse_weight': 0.15,
    
    # MSE нормализация (максимална очаквана MSE стойност)
    'mse_max': 10000.0,
    
    # Праг за "непознат потребител" (под него = UNKNOWN)
    'unknown_threshold': 0.65,
    
    # Морфологични ядра
    'closing_kernel': (7, 7),
    'closing_iterations': 3,
    'opening_kernel': (3, 3),
    'opening_iterations': 1,
    

    
    # Разделение train/test
    'train_count': 10,
    'test_count': 10,
    
    # База данни
    'db_name': 'signatures_system.db',
    
    # Изходна папка за графики
    'output_dir': 'results'
}


# ==========================================
# 1. БАЗА ДАННИ (SQLite)
# ==========================================
def init_db():
    """Създава базата данни и таблицата за одитен журнал."""
    conn = sqlite3.connect(CONFIG['db_name'])
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS identification_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            true_user TEXT NOT NULL,
            predicted_user TEXT NOT NULL,
            ssim_score REAL NOT NULL,
            mse_score REAL NOT NULL,
            combined_score REAL NOT NULL,
            result TEXT NOT NULL,
            processing_time_ms REAL NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
    print("[DB] Базата данни е инициализирана успешно.")


def log_result(true_user, predicted_user, ssim_score, mse_score, combined_score, result, proc_time_ms):
    """Записва резултат от идентификация в базата данни."""
    conn = sqlite3.connect(CONFIG['db_name'])
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO identification_logs 
        (true_user, predicted_user, ssim_score, mse_score, combined_score, result, processing_time_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (true_user, predicted_user, ssim_score, mse_score, combined_score, result, proc_time_ms))
    conn.commit()
    conn.close()


def show_all_logs():
    """Извежда цялата история от одитния журнал."""
    conn = sqlite3.connect(CONFIG['db_name'])
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM identification_logs ORDER BY id")
    rows = cursor.fetchall()
    conn.close()
    
    print("\n" + "=" * 110)
    print(" ОДИТЕН ЖУРНАЛ (AUDIT TRAIL)")
    print("=" * 110)
    print(f"{'ID':<5} {'Истина':<12} {'Решение':<12} {'SSIM':<8} {'MSE':<10} {'Комб.':<8} {'Резултат':<10} {'Време(ms)':<10} {'Timestamp'}")
    print("-" * 110)
    for row in rows:
        print(f"{row[0]:<5} {row[1]:<12} {row[2]:<12} {row[3]:<8.4f} {row[4]:<10.2f} {row[5]:<8.4f} {row[6]:<10} {row[7]:<10.2f} {row[8]}")
    print("=" * 110)
    return rows


# ==========================================
# 2. ПРЕПРОЦЕСИНГ И СЕГМЕНТИРАНЕ
# ==========================================
def load_and_prepare(image_path):
    """Зарежда изображение, обработва алфа канал, връща сиво изображение."""
    from PIL import Image
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Файлът не е намерен: {image_path}")
    
    pil_img = Image.open(image_path).convert('RGBA')
    img_np = np.array(pil_img)
    
    # Алфа смесване - прозрачен фон -> бял фон
    alpha = img_np[:, :, 3:4] / 255.0
    rgb = img_np[:, :, :3]
    white = np.ones_like(rgb, dtype=np.uint8) * 255
    img_rgb = (rgb * alpha + white * (1 - alpha)).astype(np.uint8)
    
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return img_gray


def segment_signature(img_gray):
    """Сегментира подписа: Otsu бинаризация + морфология + Bounding Box."""
    # Бинаризация по метода на Otsu
    _, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Морфологично затваряне (свързва фрагментирани щрихи)
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, CONFIG['closing_kernel'])
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k_close, iterations=CONFIG['closing_iterations'])
    
    # Морфологично отваряне (премахва шум)
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, CONFIG['opening_kernel'])
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, k_open, iterations=CONFIG['opening_iterations'])
    
    # Bounding Box
    coords = cv2.findNonZero(cleaned)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        cropped_gray = img_gray[y:y+h, x:x+w]
        cropped_bin = cleaned[y:y+h, x:x+w]
        return cropped_gray, cropped_bin
    
    return img_gray, cleaned




def center_and_scale(img, target_size, bg_color=0):
    h, w = img.shape
    
    scale = min(target_size[0] / w, target_size[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(img, (new_w, new_h))
    
    # Създаваме платното с искания цвят (черен=0, бял=255)
    canvas = np.full((target_size[1], target_size[0]), bg_color, dtype=np.uint8)
    
    x_offset = (target_size[0] - new_w) // 2
    y_offset = (target_size[1] - new_h) // 2
    
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas


def preprocess_signature(image_path):
    """
    Пълен pipeline: зареждане -> сегментиране -> нормализация.
    Връща: (math_img, vis_img)
      - math_img: изображение за математическо сравнение (SSIM/MSE)
      - vis_img: изображение за визуализация (оригинални сиви тонове)
    """
    target = CONFIG['target_size']
    
    img_gray = load_and_prepare(image_path)
    cropped_gray, cropped_bin = segment_signature(img_gray)
    
    # Визуално изображение (за документацията)
    vis_img = center_and_scale(cropped_gray, target, bg_color=255)
    math_img = center_and_scale(cropped_bin, target)
    
    return math_img, vis_img


# ==========================================
# 3. КОМБИНИРАН SCORE (SSIM + MSE)
# ==========================================
def compute_combined_score(img1, img2):
    """
    Изчислява комбиниран score от SSIM и нормализиран MSE.
    Връща: (combined_score, ssim_score, mse_score, ssim_diff_map)
    """
    ssim_score, diff_map = ssim(img1, img2, full=True)
    mse_score = mse(img1, img2)
    
    # Нормализация на MSE в диапазон [0, 1] (обърнат - по-високо = по-добре)
    mse_normalized = max(0.0, 1.0 - (mse_score / CONFIG['mse_max']))
    
    # Комбиниран score
    combined = CONFIG['ssim_weight'] * ssim_score + CONFIG['mse_weight'] * mse_normalized
    
    return combined, ssim_score, mse_score, diff_map


def compare_against_profile(test_img, profile_imgs):
    """
    Сравнява тестов подпис срещу всички еталони на един потребител.
    Връща средния комбиниран score и средните SSIM/MSE.
    """
    combined_scores = []
    ssim_scores = []
    mse_scores = []
    
    for train_img in profile_imgs:
        comb, ss, ms, _ = compute_combined_score(test_img, train_img)
        combined_scores.append(comb)
        ssim_scores.append(ss)
        mse_scores.append(ms)
    
    return np.mean(combined_scores), np.mean(ssim_scores), np.mean(mse_scores)


# ==========================================
# 4. ВИЗУАЛИЗАЦИИ
# ==========================================
def generate_comparison_plot(test_vis, ref_vis, test_math, ref_math, 
                             test_name, ref_name, is_match, filename):
    """Генерира детайлна визуализация на сравнение (за документация)."""
    combined, ssim_score, mse_score, diff_map = compute_combined_score(test_math, ref_math)
    
    result_text = "СЪЩИЯТ ЧОВЕК (Успешно)" if is_match else "РАЗЛИЧЕН ЧОВЕК (Грешка)"
    result_color = '#2ecc71' if is_match else '#e74c3c'
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Сравнение: {test_name} срещу профила на {ref_name}", fontsize=14, fontweight='bold')
    
    axes[0, 0].imshow(ref_vis, cmap='gray')
    axes[0, 0].set_title(f"Еталон ({ref_name})", fontsize=11)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(test_vis, cmap='gray')
    axes[0, 1].set_title(f"Тестов подпис ({test_name})", fontsize=11)
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(diff_map, cmap='hot')
    axes[1, 0].set_title("SSIM карта на разликите", fontsize=11)
    axes[1, 0].axis('off')
    
    axes[1, 1].axis('off')
    info_text = (
        f"=== АНАЛИЗ НА СИСТЕМАТА ===\n\n"
        f"Базов алгоритъм: MSE\n"
        f"  Грешка: {mse_score:.2f}\n\n"
        f"Основен алгоритъм: SSIM\n"
        f"  Сходство: {ssim_score:.4f}\n\n"
        f"Комбиниран Score:\n"
        f"  {CONFIG['ssim_weight']:.0%} SSIM + {CONFIG['mse_weight']:.0%} MSE(norm)\n"
        f"  Стойност: {combined:.4f}\n\n"
        f"Праг за непознат: {CONFIG['unknown_threshold']}\n\n"
        f"== Решение ==\n"
        f"  {result_text}"
    )
    axes[1, 1].text(0.05, 0.95, info_text, transform=axes[1, 1].transAxes,
                     fontsize=11, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor=result_color, alpha=0.2))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def generate_ssim_heatmap(users, profiles_math, output_path):
    """
    Генерира SSIM heatmap: среден SSIM между всеки двама потребители.
    Показва ясно как подписите на един човек са сходни помежду си,
    а подписите на различни хора — различни.
    """
    n = len(users)
    heatmap = np.zeros((n, n))
    
    for i, user_a in enumerate(users):
        for j, user_b in enumerate(users):
            scores = []
            for img_a in profiles_math[user_a]:
                for img_b in profiles_math[user_b]:
                    s = ssim(img_a, img_b)
                    scores.append(s)
            heatmap[i, j] = np.mean(scores)
    
    plt.figure(figsize=(9, 7))
    sns.heatmap(heatmap, annot=True, fmt=".3f", cmap="RdYlGn", 
                xticklabels=users, yticklabels=users,
                vmin=0, vmax=1, annot_kws={"size": 13},
                linewidths=0.5, linecolor='gray')
    plt.title("Средно SSIM сходство между потребителите\n(Диагонал = същия човек, Извън = различни хора)", fontsize=13)
    plt.ylabel("Потребител A", fontsize=12)
    plt.xlabel("Потребител B", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> Генерирана '{output_path}'")


def generate_confusion_matrix(y_true, y_pred, labels, output_path):
    """Генерира матрица на грешките (абсолютни стойности + нормализирана)."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Абсолютни стойности
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels, annot_kws={"size": 15}, ax=axes[0],
                linewidths=0.5, linecolor='gray')
    axes[0].set_title("Матрица на грешките (Абсолютни)", fontsize=13)
    axes[0].set_ylabel("Истински потребител", fontsize=12)
    axes[0].set_xlabel("Решение на системата", fontsize=12)
    
    # Нормализирана (проценти по ред)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    sns.heatmap(cm_norm, annot=True, fmt=".0%", cmap="Greens", cbar=False,
                xticklabels=labels, yticklabels=labels, annot_kws={"size": 15}, ax=axes[1],
                vmin=0, vmax=1, linewidths=0.5, linecolor='gray')
    axes[1].set_title("Матрица на грешките (Нормализирана %)", fontsize=13)
    axes[1].set_ylabel("Истински потребител", fontsize=12)
    axes[1].set_xlabel("Решение на системата", fontsize=12)
    
    fig.suptitle(f"Идентификация 1:N  ({CONFIG['train_count']} Train / {CONFIG['test_count']} Test)", fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> Генерирана '{output_path}'")


def generate_metrics_report(y_true, y_pred, labels, output_path):
    """Генерира текстов и визуален отчет с всички метрики."""
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    
    # Изчисляване на FAR и FRR за всеки потребител
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    n_users = len(labels)
    total_tests = len(y_true)
    
    far_per_user = {}
    frr_per_user = {}
    
    for i, user in enumerate(labels):
        # FRR = грешно отхвърлени / общо легитимни опити за този потребител
        true_positives = cm[i, i]
        total_genuine = cm[i, :].sum()
        frr_per_user[user] = 1.0 - (true_positives / total_genuine) if total_genuine > 0 else 0.0
        
        # FAR = грешно приети за този потребител / общо чужди опити
        false_accepts = cm[:, i].sum() - cm[i, i]
        total_impostor = total_tests - total_genuine
        far_per_user[user] = false_accepts / total_impostor if total_impostor > 0 else 0.0
    
    avg_far = np.mean(list(far_per_user.values()))
    avg_frr = np.mean(list(frr_per_user.values()))
    
    # Текстов отчет
    lines = []
    lines.append("=" * 60)
    lines.append(" ОТЧЕТ ЗА ПРОИЗВОДИТЕЛНОСТ НА СИСТЕМАТА")
    lines.append("=" * 60)
    lines.append(f"\nОбща точност (Accuracy): {acc:.2%}")
    lines.append(f"Среден FAR (False Acceptance Rate): {avg_far:.2%}")
    lines.append(f"Среден FRR (False Rejection Rate): {avg_frr:.2%}")
    lines.append(f"\nОбщо тестови проби: {total_tests}")
    lines.append(f"Правилно идентифицирани: {int(acc * total_tests)}")
    lines.append(f"Грешно идентифицирани: {total_tests - int(acc * total_tests)}")
    
    lines.append(f"\n{'Потребител':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'FAR':<10} {'FRR':<10}")
    lines.append("-" * 68)
    for user in labels:
        r = report.get(user, {})
        lines.append(f"{user:<12} {r.get('precision', 0):<12.4f} {r.get('recall', 0):<12.4f} "
                     f"{r.get('f1-score', 0):<12.4f} {far_per_user[user]:<10.2%} {frr_per_user[user]:<10.2%}")
    
    lines.append("-" * 68)
    lines.append(f"{'СРЕДНО':<12} {report['macro avg']['precision']:<12.4f} {report['macro avg']['recall']:<12.4f} "
                 f"{report['macro avg']['f1-score']:<12.4f} {avg_far:<10.2%} {avg_frr:<10.2%}")
    
    report_text = "\n".join(lines)
    print(report_text)
    
    # Запис във файл
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\n-> Записан отчет в '{output_path}'")
    
    return acc, avg_far, avg_frr


def generate_score_distribution(all_same_scores, all_diff_scores, output_path):
    """
    Генерира хистограма на разпределението на combined score:
    - Сравнения "същия човек" vs "различен човек"
    Показва колко добре системата разделя двете групи.
    """
    plt.figure(figsize=(10, 6))
    
    if all_same_scores:
        plt.hist(all_same_scores, bins=20, alpha=0.7, color='#2ecc71', 
                label=f'Същият човек (n={len(all_same_scores)})', edgecolor='black')
    if all_diff_scores:
        plt.hist(all_diff_scores, bins=20, alpha=0.7, color='#e74c3c', 
                label=f'Различен човек (n={len(all_diff_scores)})', edgecolor='black')
    
    plt.axvline(x=CONFIG['unknown_threshold'], color='orange', linestyle='--', 
                linewidth=2, label=f"Праг за непознат ({CONFIG['unknown_threshold']})")
    
    plt.xlabel("Комбиниран Score (SSIM + MSE)", fontsize=12)
    plt.ylabel("Брой сравнения", fontsize=12)
    plt.title("Разпределение на Score: Същият човек vs Различен човек", fontsize=14)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> Генерирана '{output_path}'")


# ==========================================
# 5. ОСНОВНА СИСТЕМА (ИДЕНТИФИКАЦИЯ 1:N)
# ==========================================
def main():
    init_db()
    
    dataset_dir = "dataset"
    output_dir = CONFIG['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(dataset_dir):
        print(f"ГРЕШКА: Папката '{dataset_dir}' не съществува!")
        print("Очаквана структура:")
        print("  dataset/")
        print("    User1/")
        print("      img001.png ... img020.png")
        print("    User2/")
        print("      img001.png ... img020.png")
        return
    
    users = sorted([d for d in os.listdir(dataset_dir) 
                    if os.path.isdir(os.path.join(dataset_dir, d))])
    
    if len(users) < 2:
        print("ГРЕШКА: Необходими са поне 2 потребителя в dataset/")
        return
    
    train_n = CONFIG['train_count']
    test_n = CONFIG['test_count']
    
    profiles_math = {}
    profiles_vis = {}
    tests_math = {}
    tests_vis = {}
    
    print("=" * 65)
    print(" СИСТЕМА ЗА ИДЕНТИФИКАЦИЯ НА ПОДПИСИ (1:N)")
    print(f" Алгоритми: SSIM (основен) + MSE (базов) = Комбиниран Score")
    print(f" Тежести: SSIM={CONFIG['ssim_weight']:.0%}, MSE(norm)={CONFIG['mse_weight']:.0%}")
    print(f" Праг за непознат: {CONFIG['unknown_threshold']}")
    print(f" Разделение: {train_n} Train / {test_n} Test")
    print("=" * 65)
    
    # Зареждане на данните
    print(f"\nНамерени потребители: {users}")
    for user in users:
        user_dir = os.path.join(dataset_dir, user)
        images = sorted(os.listdir(user_dir))
        
        needed = train_n + test_n
        if len(images) < needed:
            print(f"[ВНИМАНИЕ] {user}: намерени {len(images)} снимки, нужни {needed}. Пропуснат!")
            continue
        
        profiles_math[user], profiles_vis[user] = [], []
        tests_math[user], tests_vis[user] = [], []
        
        for img in images[:train_n]:
            m, v = preprocess_signature(os.path.join(user_dir, img))
            profiles_math[user].append(m)
            profiles_vis[user].append(v)
        
        for img in images[train_n:train_n + test_n]:
            m, v = preprocess_signature(os.path.join(user_dir, img))
            tests_math[user].append(m)
            tests_vis[user].append(v)
        
        print(f"  [{user}] Заредени {train_n} еталона + {test_n} тестови")
    
    active_users = sorted(profiles_math.keys())
    if len(active_users) < 2:
        print("ГРЕШКА: Необходими са поне 2 активни потребителя!")
        return
    
    # --- ИДЕНТИФИКАЦИЯ ---
    y_true, y_pred = [], []
    all_same_scores = []  # Score при сравнение "същия човек"
    all_diff_scores = []  # Score при сравнение "различен човек"
    
    best_match_data = {'score': -1, 'args': None}
    worst_mismatch_data = {'score': float('inf'), 'args': None}
    
    print(f"\n{'=' * 65}")
    print(" ПРОЦЕС НА ИДЕНТИФИКАЦИЯ")
    print(f"{'=' * 65}\n")
    
    for actual_user in active_users:
        for i, test_img in enumerate(tests_math[actual_user]):
            start_time = time.time()
            
            best_user = None
            best_combined = -1
            best_ssim = 0
            best_mse = 0
            
            # Сравняваме с всеки профил
            for profile_user in active_users:
                avg_combined, avg_ssim, avg_mse = compare_against_profile(
                    test_img, profiles_math[profile_user]
                )
                
                if avg_combined > best_combined:
                    best_combined = avg_combined
                    best_user = profile_user
                    best_ssim = avg_ssim
                    best_mse = avg_mse
            
            proc_time = (time.time() - start_time) * 1000  # в милисекунди
            
            # Проверка за праг на непознат
            if best_combined < CONFIG['unknown_threshold']:
                predicted = "UNKNOWN"
            else:
                predicted = best_user
            
            is_correct = (actual_user == predicted)
            result_str = "ВЯРНО" if is_correct else "ГРЕШКА"
            
            print(f"  [{actual_user}] Тест #{i+1:02d} -> Решение: [{predicted}] "
                  f"| Комб: {best_combined:.4f} | SSIM: {best_ssim:.4f} | MSE: {best_mse:.1f} "
                  f"| {result_str} | {proc_time:.1f}ms")
            
            y_true.append(actual_user)
            y_pred.append(predicted)
            log_result(actual_user, predicted, best_ssim, best_mse, best_combined, result_str, proc_time)
            
            # Събиране на score-ове за хистограмата
            if actual_user == best_user:
                all_same_scores.append(best_combined)
            else:
                all_diff_scores.append(best_combined)
            
            # Намиране на най-добро и най-лошо сравнение за документацията
            if best_user is not None:
                ref_vis = profiles_vis[best_user][0]
                test_vis = tests_vis[actual_user][i]
                ref_math = profiles_math[best_user][0]
                plot_args = (test_vis, ref_vis, test_img, ref_math, actual_user, best_user)
                
                if is_correct and best_combined > best_match_data['score']:
                    best_match_data['score'] = best_combined
                    best_match_data['args'] = plot_args
                
                if not is_correct and best_combined < worst_mismatch_data['score']:
                    worst_mismatch_data['score'] = best_combined
                    worst_mismatch_data['args'] = plot_args
    
    # --- ГЕНЕРИРАНЕ НА ВСИЧКИ ВИЗУАЛИЗАЦИИ ---
    print(f"\n{'=' * 65}")
    print(" ГЕНЕРИРАНЕ НА ГРАФИКИ И ОТЧЕТИ")
    print(f"{'=' * 65}")
    
    # Определяне на labels (включително UNKNOWN ако е имало такова решение)
    all_labels = active_users[:]
    if "UNKNOWN" in y_pred:
        all_labels.append("UNKNOWN")
    
    # 1. Матрица на грешките
    generate_confusion_matrix(y_true, y_pred, all_labels, 
                              os.path.join(output_dir, "confusion_matrix.png"))
    
    # 2. SSIM Heatmap между потребителите
    generate_ssim_heatmap(active_users, profiles_math,
                          os.path.join(output_dir, "ssim_heatmap.png"))
    
    # 3. Разпределение на score-овете
    generate_score_distribution(all_same_scores, all_diff_scores,
                                os.path.join(output_dir, "score_distribution.png"))
    
    # 4. Примерни сравнения за документацията
    if best_match_data['args'] is not None:
        generate_comparison_plot(*best_match_data['args'], True,
                                 os.path.join(output_dir, "example_BEST_MATCH.png"))
        print(f"-> Генериран 'example_BEST_MATCH.png' (Score: {best_match_data['score']:.4f})")
    
    if worst_mismatch_data['args'] is not None:
        generate_comparison_plot(*worst_mismatch_data['args'], False,
                                 os.path.join(output_dir, "example_WORST_MISMATCH.png"))
        print(f"-> Генериран 'example_WORST_MISMATCH.png' (Score: {worst_mismatch_data['score']:.4f})")
    
    # 5. Метрики
    print()
    acc, avg_far, avg_frr = generate_metrics_report(y_true, y_pred, all_labels,
                                                     os.path.join(output_dir, "metrics_report.txt"))
    
    # 6. Одитен журнал
    show_all_logs()
    
    print(f"\n{'=' * 65}")
    print(" ВСИЧКИ ОПЕРАЦИИ ПРИКЛЮЧИХА УСПЕШНО!")
    print(f" Резултатите са в папка: '{output_dir}/'")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()