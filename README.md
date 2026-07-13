# Skin Lesion Classifier API

EfficientNet-B0 asosidagi teri kasalliklarini klassifikatsiya qiluvchi FastAPI backend.

## Endpointlar
- `GET /` — holat tekshiruvi
- `POST /predict` — rasm yuborib klassifikatsiya natijasini olish (multipart/form-data, `file` maydoni)

## Klasslar
BCC, MEL, NV, SCC, UNK

## Render.com'da joylashtirish
Bu repo `Dockerfile` orqali Render.com'da avtomatik quriladi va ishga tushadi.
Render `PORT` muhit o'zgaruvchisini o'zi belgilaydi — Dockerfile buni avtomatik o'qiydi.
