
import os
import gdown

MODEL_PATH = "campusguard_phishing_model.joblib"
FILE_ID = "1rWi719gwN_rGESJI75W_3K-dtr8nu75B"

if not os.path.exists(MODEL_PATH):
    print("Downloading trained CampusGuard model...")
    gdown.download(
        id=FILE_ID,
        output=MODEL_PATH,
        quiet=False
    )

if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
    raise RuntimeError("Model download failed.")

print("Model file is ready.")
