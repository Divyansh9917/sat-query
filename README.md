<p align="center">
  <h1 align="center">🛰️ SatQuery AI</h1>
  <p align="center">
    <strong>Agentic Multimodal Remote-Sensing Assistant</strong>
  </p>
  <p align="center">
    Query-driven vision-language AI for analyzing single, cross-modal (Optical + SAR),<br/>
    and bi-temporal remote-sensing images through natural language.
  </p>
</p>

---

## ✨ Key Capabilities

| Capability | Description | Models/Methods |
|:---|:---|:---|
| 🔍 **Visual Question Answering** | Answer natural-language questions about RS imagery | BLIP-2 + LoRA |
| 📝 **Image Captioning** | Generate detailed scene descriptions | BLIP-2 |
| 📍 **Region Grounding** | Locate objects/regions from text queries → bounding boxes | GroundingDINO |
| 🔄 **Change Detection** | Bi-temporal change maps + change-VQA + captioning | Siamese ResNet-18 + VLM |
| 🌐 **Optical-SAR Fusion** | Joint spectral/backscatter classification (water/flood/built-up) | CNN + NDVI/NDWI + VV/VH |
| 🤖 **Agentic Orchestrator** | Intent classification → tool routing → auditable trace | Rule-based + registry |

---

## 🏗️ Architecture

```mermaid
flowchart TB
    subgraph UI["🖥️ Clients"]
        Web["Next.js Frontend<br/>(React + Tailwind)"]
        Demo["Streamlit Demo App<br/>(app/ui.py)"]
    end

    subgraph API["🔌 API Layer"]
        FastAPI["FastAPI<br/>(api/main.py)"]
    end

    subgraph Core["⚙️ Core Engine"]
        Geo["GeoProcessor<br/>Band Norm · CRS · RGB"]
        Orch["Orchestrator<br/>Intent → Dispatch → Trace"]
        Reg["Tool Registry"]
    end

    subgraph Models["🧠 Specialist Models"]
        VQA["VQA & Captioner<br/>(BLIP-2 + LoRA)"]
        Gnd["Text Grounding<br/>(GroundingDINO)"]
        Chg["Change Detector<br/>(Siamese + VLM)"]
        Fus["Optical-SAR Fusion<br/>(CNN + Indices)"]
    end

    Web --> FastAPI
    Demo --> Orch
    FastAPI --> Geo --> Orch
    Orch <--> Reg
    Reg --> VQA & Gnd & Chg & Fus
    VQA & Gnd & Chg & Fus --> Orch
    Orch --> FastAPI
    FastAPI --> Web
```

---


## 📁 Repository Structure

```
sat-query/
├── README.md
├── requirements.txt
├── setup.py
├── api/
│ ├── init.py
│ └── main.py # FastAPI app entrypoint
├── app/
│ ├── ui.py # Streamlit demo app
│ └── utils.py # Visualization & report generation
├── config/
│ └── settings.yaml # Central configuration
├── core/
│ ├── init.py
│ ├── geoprocessor.py # Rasterio I/O, band normalization, CRS
│ ├── orchestrator.py # Agentic intent → dispatch → trace
│ └── registry.py # Tool & Model Registry (singleton)
├── data/
│ ├── sample_data/ # Synthetic sample images
│ └── download_benchmarks.py # BigEarthNet, RSVQA, VRSBench, CDVQA
├── models/
│ ├── init.py
│ ├── adapters.py # LoRA + BandProjection adapters
│ ├── vqa_caption.py # BLIP-2 VQA & captioning
│ ├── grounding.py # GroundingDINO visual grounding
│ ├── change_detector.py # Siamese change detection + VLM
│ └── optical_sar_fusion.py # Optical-SAR fusion classifier
├── training/
│ ├── train_bigearthnet_lora.py # LoRA fine-tuning on BigEarthNet-MM
│ └── eval_benchmarks.py # RSVQA / VRSBench / CDVQA evaluation
├── tests/
│ ├── test_agent.py # Orchestrator & registry tests
│ └── test_models.py # Model smoke tests
├── frontend/ # Next.js + Tailwind web client
│ ├── src/
│ ├── public/
│ ├── package.json
│ ├── next.config.mjs
│ └── tailwind.config.ts
└── .streamlit/ # Streamlit configuration
```


---

## 🛠️ Tech Stack

| Layer | Technology |
|:---|:---|
| **Frontend** | Next.js 14, React 18, TypeScript, Tailwind CSS, lucide-react |
| **Backend API** | FastAPI, Uvicorn |
| **Agentic Core** | Custom orchestrator (intent classification → tool routing → trace) |
| **Deep Learning** | PyTorch, Transformers, PEFT (LoRA), Accelerate, OpenCLIP |
| **Vision-Language Models** | BLIP-2, GroundingDINO |
| **Geospatial** | Rasterio, PyProj, Shapely, GeoPandas |
| **Vision & Image Processing** | OpenCV, Pillow, scikit-image |
| **Reports** | FPDF2, Jinja2 |
| **Internal Demo UI** | Streamlit |
| **Testing** | Pytest, pytest-cov |

---

## 🚀 Quickstart

### 1. Clone & set up the Python environment

```bash
git clone https://github.com/Divyansh9917/sat-query.git
cd sat-query

python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows

pip install -r requirements.txt
pip install -e .            # Editable install
```

### 2. Run the backend API

```bash
uvicorn api.main:app --reload --port 8000
```
- API: http://localhost:8000
- Docs: http://localhost:8000/docs

### 3. Run the frontend

```bash
cd frontend
npm install
npm run dev
```
- App: http://localhost:3000

### 4. (Optional) Run the Streamlit demo

```bash
streamlit run app/ui.py
```

### 5. Generate sample data

```bash
python data/sample_data/generate_samples.py
```

### 6. Run tests

```bash
python -m pytest tests/ -v
```

---

## 🔧 Configuration

All settings are in [`config/settings.yaml`](config/settings.yaml):

```yaml
device: "auto"              # "cuda" | "cpu" | "auto"

models:
  vqa_captioner:
    backbone: "Salesforce/blip2-opt-2.7b"
    lora_weights: null       # Path to LoRA checkpoint

  grounding:
    backbone: "IDEA-Research/grounding-dino-tiny"
    box_threshold: 0.25

geo:
  normalization_method: "percentile"
  default_crs: "EPSG:4326"
```

---

## 📊 Supported Benchmarks

| Benchmark | Task | Metrics |
|:---|:---|:---|
| **RSVQA** | Remote Sensing VQA | OA, AA, Kappa |
| **VRSBench** | Captioning + VQA + Grounding | BLEU, METEOR, CIDEr |
| **CDVQA** | Change Detection VQA | OA, F1 |
| **BigEarthNet-MM** | Multi-label Classification | LoRA domain adaptation |

Download benchmarks:
```bash
python data/download_benchmarks.py --list           # Show available datasets
python data/download_benchmarks.py --dataset rsvqa   # Download specific dataset
```

---

## 🎯 Usage Examples

### Single-Image VQA
```python
from core.geoprocessor import GeoProcessor
from core.orchestrator import Orchestrator
from models.vqa_caption import RSVQACaptioner

# Initialize
RSVQACaptioner()  # Self-registers into registry
orch = Orchestrator()
geo = GeoProcessor()

# Load and query
image = geo.load("path/to/satellite_image.tif")
response = orch.dispatch([image], "How many buildings are visible?")
print(response.answer)
print(response.trace.to_dict())
```

### Bi-temporal Change Detection
```python
pre = geo.load("before.png")
post = geo.load("after.png")
response = orch.dispatch([pre, post], "What has changed?")
# response.change_mask contains the spatial change map
```

### Optical-SAR Fusion
```python
optical = geo.load("optical.tif")
sar = geo.load("sar.tif")
response = orch.dispatch([optical, sar], "Identify flood zones")
# response.class_map contains pixel-wise classification
```

---

## 🏋️ Training

### LoRA Fine-Tuning on BigEarthNet
```bash
python training/train_bigearthnet_lora.py \
    --model_name Salesforce/blip2-opt-2.7b \
    --data_dir data/datasets/bigearthnet \
    --output_dir checkpoints/lora_bigearthnet \
    --epochs 5 \
    --lora_rank 16 \
    --batch_size 4
```

### Benchmark Evaluation
```bash
python training/eval_benchmarks.py \
    --benchmark rsvqa \
    --lora_path checkpoints/lora_bigearthnet \
    --data_dir data/datasets/rsvqa
```

---

## 📋 Execution Trace

Every query produces an auditable trace:

```json
{
  "task": "vqa",
  "selected_model": "rs_vqa_captioner",
  "inputs_summary": "1 image(s) [satellite.tif], config=single_optical",
  "confidence": 0.87,
  "latency_ms": 342.5,
  "timestamp": "2024-12-01T10:30:00Z",
  "status": "success"
}
```

---


