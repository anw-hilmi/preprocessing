import json
import os

# Mengambil path absolut folder tempat script ini berada
current_dir = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(current_dir, "Dataset_animals.ipynb")

try:
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Hapus metadata widgets yang rusak
    if "widgets" in data.get("metadata", {}):
        del data["metadata"]["widgets"]
        print("Metadata widgets berhasil dihapus!")
    else:
        print("Metadata widgets tidak ditemukan atau sudah bersih.")

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
        print("File berhasil diperbarui!")

except FileNotFoundError:
    print(f"Error: File tidak ditemukan di {filename}")
    print("Pastikan file .ipynb berada di folder yang sama dengan script ini.")