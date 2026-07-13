"""
Teri kasalliklarini aniqlash uchun backend server.
EfficientNet-B0 asosidagi model (best_model.pth) yordamida rasm klassifikatsiya qilinadi.

Ishga tushirish:
    pip install -r requirements.txt
    uvicorn app:app --host 0.0.0.0 --port 8000
"""

import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ------------------------------------------------------------------
# 1) Klasslar (train paytida ImageFolder papkalarni alifbo tartibida
#    o'qigani uchun aynan shu tartib saqlanishi SHART)
# ------------------------------------------------------------------
CLASS_NAMES = ["BCC", "MEL", "NV", "SCC", "UNK"]

CLASS_INFO = {
    "BCC": {
        "full_name": "Bazal-hujayrali karsinoma (Basal Cell Carcinoma)",
        "risk": "malignant",
    },
    "MEL": {
        "full_name": "Melanoma",
        "risk": "malignant",
    },
    "NV": {
        "full_name": "Nevus (oddiy xol)",
        "risk": "benign",
    },
    "SCC": {
        "full_name": "Yassi hujayrali karsinoma (Squamous Cell Carcinoma)",
        "risk": "malignant",
    },
    "UNK": {
        "full_name": "Noma'lum / aniqlanmagan",
        "risk": "unknown",
    },
}

IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------------
# 2) Model arxitekturasi — notebook'dagi bilan bir xil bo'lishi SHART
# ------------------------------------------------------------------
class SkinClassifier(nn.Module):
    def __init__(self, base_model, num_classes):
        super(SkinClassifier, self).__init__()
        self.base = base_model
        self.bn = nn.BatchNorm1d(1280)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.base(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def load_model():
    base_model = models.efficientnet_b0(weights=None)
    base_model.classifier = nn.Identity()
    model = SkinClassifier(base_model, num_classes=len(CLASS_NAMES))
    state_dict = torch.load("best_model.pth", map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


model = load_model()

inference_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# ------------------------------------------------------------------
# 3) FastAPI ilova
# ------------------------------------------------------------------
app = FastAPI(title="Skin Lesion Classifier API")

# Mobil ilova istalgan manzildan so'rov yubora olishi uchun (dev rejimida)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health_check():
    return {"status": "ok", "device": str(DEVICE), "classes": CLASS_NAMES}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Fayl rasm formatida bo'lishi kerak")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Rasmni o'qib bo'lmadi")

    tensor = inference_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    results = []
    for idx, cls in enumerate(CLASS_NAMES):
        results.append(
            {
                "class": cls,
                "full_name": CLASS_INFO[cls]["full_name"],
                "risk": CLASS_INFO[cls]["risk"],
                "probability": round(float(probs[idx]) * 100, 2),
            }
        )
    results.sort(key=lambda r: r["probability"], reverse=True)

    top = results[0]
    return {
        "prediction": top["class"],
        "full_name": top["full_name"],
        "risk": top["risk"],
        "confidence": top["probability"],
        "all_classes": results,
        "disclaimer": (
            "Bu natija faqat ma'lumot uchun. Yakuniy tashxis uchun dermatolog "
            "shifokorga murojaat qiling."
        ),
    }
