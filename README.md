# Plant Disease Prediction Model

This project trains a deep learning model to classify plant diseases from the PlantVillage dataset in:

- `c:\Users\asus\Downloads\archive (1)\PlantVillage`

The training script automatically detects and avoids the duplicated nested dataset folder (`PlantVillage/PlantVillage`) so images are not double-counted.

## 1) Install dependencies

```bash
pip install -r requirements.txt
```

## 2) Train the model

```bash
python train_disease_model.py --epochs 10
```

Optional arguments:

- `--data-dir "c:\\Users\\asus\\Downloads\\archive (1)\\PlantVillage"`
- `--output-dir artifacts`
- `--image-size 224`
- `--batch-size 32`
- `--test-size 0.15`
- `--val-size 0.15`

## 3) Predict disease for one image

```bash
python predict_disease.py --image "path\to\leaf_image.jpg"
```

Optional:

- `--model artifacts/plant_disease_model.keras`
- `--labels artifacts/class_names.json`
- `--image-size 224`

## 4) Run FastAPI locally

Start the API server:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Prediction request:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
	-H "accept: application/json" \
	-H "Content-Type: multipart/form-data" \
	-F "file=@path/to/leaf_image.jpg"
```

## 5) Deploy with Docker

Build image:

```bash
docker build -t plant-disease-api .
```

Run container:

```bash
docker run --rm -p 8000:8000 plant-disease-api
```

If your model artifacts are generated outside the image build context, mount them at runtime:

```bash
docker run --rm -p 8000:8000 \
	-e MODEL_PATH=artifacts/plant_disease_model.keras \
	-e LABELS_PATH=artifacts/class_names.json \
	-v "${PWD}/artifacts:/app/artifacts" \
	plant-disease-api
```

Environment variables supported by the API:

- `MODEL_PATH` default: `artifacts/plant_disease_model.keras`
- `LABELS_PATH` default: `artifacts/class_names.json`
- `IMAGE_SIZE` default: `224`
- `TOP_K` default: `3`

## Output artifacts

Training writes these files in `artifacts/`:

- `plant_disease_model.keras` (final model)
- `best_model.keras` (best validation checkpoint)
- `class_names.json` (index-to-class mapping)
- `train_history.json` (loss/accuracy history)
