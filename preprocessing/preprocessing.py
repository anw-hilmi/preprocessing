import os
import shutil
import random
import cv2
import pandas as pd
import numpy as np
from skimage.transform import rotate
from skimage.exposure import adjust_gamma
from skimage import io, img_as_ubyte
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- 1. SETUP DIREKTORI ---
if not os.path.exists("animals"):
    print("Error: Folder 'animals' tidak ditemukan!")
    exit()

train_dir = "animals/train"
val_dir = "animals/val"
combined_dir = "animals/dataset"
os.makedirs(combined_dir, exist_ok=True)

# --- 2. PENGGABUNGAN DATASET ---
for folder in [train_dir, val_dir]:
    if os.path.exists(folder):
        for category in os.listdir(folder):
            category_dir = os.path.join(folder, category)
            if os.path.isdir(category_dir):
                shutil.copytree(category_dir, os.path.join(combined_dir, category), dirs_exist_ok=True)

# --- 3. FUNGSI AUGMENTASI ---
def resize_img(img):
    return cv2.resize(img, (150, 150))

def subtle_rotation(img):
    img = resize_img(img)
    sudut = random.randint(-30, 30)
    return rotate(img, sudut, preserve_range=True)

def flip_left_right(img):
    img = resize_img(img)
    return np.fliplr(img)

def blur_image(img):
    img = resize_img(img)
    return cv2.GaussianBlur(img, (5, 5), 0)

def add_brightness(img):
    img = resize_img(img)
    gamma_val = random.uniform(0.7, 1.3)
    return adjust_gamma(img, gamma=gamma_val, preserve_range=True)

def enhance_contrast(img):
    img = resize_img(img)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)

transformations = {
    'subtle rotation': subtle_rotation,
    'blurring image': blur_image,
    'add brightness': add_brightness,
    'flip horizontal': flip_left_right,
    'resize_img': resize_img,
    'enhace_contrast': enhance_contrast
}

# --- 4. PROSES AUGMENTASI ---
target_folders = sorted(os.listdir(combined_dir))[:5]
for folder in target_folders:
    images_path = os.path.join(combined_dir, folder)
    images = [os.path.join(images_path, im) for im in os.listdir(images_path) if im.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    target_total = 4000
    current_count = len(images)
    images_to_generate = target_total - current_count

    print(f"Memproses {folder}: Butuh {max(0, images_to_generate)} gambar tambahan.")

    for i in tqdm(range(1, max(0, images_to_generate) + 1)):
        image_path = random.choice(images)
        try:
            original_image = io.imread(image_path)
            if len(original_image.shape) == 2:
                original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
            elif original_image.shape[2] == 4:
                original_image = cv2.cvtColor(original_image, cv2.COLOR_RGBA2RGB)

            transformed_image = original_image
            for _ in range(random.randint(1, 3)):
                key = random.choice(list(transformations))
                transformed_image = transformations[key](transformed_image)

            new_image_path = os.path.join(images_path, f"aug_{folder}_{i}.jpg")
            transformed_image = img_as_ubyte(transformed_image / transformed_image.max() if transformed_image.max() > 1 else transformed_image)
            cv2.imwrite(new_image_path, cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR))
        except Exception as e:
            continue

# --- 5. PEMBAGIAN DATA ---
data_list = []
for folder in target_folders:
    folder_path = os.path.join(combined_dir, folder)
    for name in os.listdir(folder_path):
        if name.lower().endswith(('.jpg', '.png', '.jpeg')):
            data_list.append({'path': os.path.join(folder_path, name), 'labels': folder})

df = pd.DataFrame(data_list)
X_train, X_test, y_train, y_test = train_test_split(df['path'], df['labels'], test_size=0.2, random_state=300)

df_tr = pd.DataFrame({'path': X_train, 'labels': y_train, 'set': 'train'})
df_te = pd.DataFrame({'path': X_test, 'labels': y_test, 'set': 'test'})
df_all = pd.concat([df_tr, df_te], ignore_index=True)

# --- 6. PEMINDAHAN KE FOLDER FINAL ---
dataset_final_path = "Dataset-Final"
for _, row in tqdm(df_all.iterrows(), total=len(df_all)):
    dest_dir = os.path.join(dataset_final_path, row['set'], row['labels'])
    os.makedirs(dest_dir, exist_ok=True)
    shutil.copy2(row['path'], os.path.join(dest_dir, os.path.basename(row['path'])))

print(f"Proses Selesai! Hasil tersimpan di folder: {dataset_final_path}")