# 🚗 Automatic License Plate Recognition (ALPR)

<p align="center">
  <img src="output.png" alt="License Plate Detection Output" width="800"/>
</p>

A real-time **Automatic Number Plate Recognition (ANPR)** system built with **YOLOv8**, **EasyOCR**, and **SORT tracking**. This project detects vehicles, tracks them across frames, identifies license plates, and extracts text using OCR.

---

## ✨ Features

- 🚙 **Vehicle Detection** - Uses YOLOv8 to detect cars, trucks, buses, and motorcycles
- 🔍 **License Plate Detection** - Detects license plates using a custom-trained YOLOv8 model via Roboflow API
- 📝 **OCR Text Extraction** - Reads license plate text using EasyOCR
- 🎯 **Multi-Object Tracking** - Tracks vehicles across video frames using SORT algorithm
- 📊 **Data Export** - Exports results to CSV format for further analysis
- 🎬 **Video Visualization** - Generates annotated output videos with detected plates and text

---

## 📸 Sample Output

<p align="center">
  <img src="output.png" alt="Detection Results" width="700"/>
</p>

---

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/license-plate-detector.git
   cd license-plate-detector
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download required models**
   - The YOLOv8n model (`yolov8n.pt`) is included in the repository
   - License plate detection uses the Roboflow API (no local model needed)

---

## 📦 Dependencies

| Package | Version | Description |
|---------|---------|-------------|
| ultralytics | 8.0.114 | YOLOv8 object detection |
| opencv-python | 4.7.0.72 | Image and video processing |
| easyocr | 1.7.0 | Optical Character Recognition |
| numpy | 1.24.3 | Numerical computations |
| pandas | 2.0.2 | Data manipulation and CSV export |
| scipy | 1.10.1 | Scientific computing |
| filterpy | 1.4.5 | Kalman filtering for SORT tracker |

---

## 🚀 Usage

### 1. Basic Detection

Run the main detection script on your video:

```bash
python main.py
```

> **Note:** Edit `main.py` to specify your input video path (default: `./sample.mp4`)

### 2. Visualize Results

After running detection, visualize the results with annotated video:

```bash
python visualize.py
```

### 3. Process Missing Data

Interpolate missing detection data for smoother tracking:

```bash
python add_missing_data.py
```

---

## 📁 Project Structure

```
license-plate-detector/
├── main.py                 # Main detection script
├── util.py                 # Utility functions (OCR, CSV export)
├── visualize.py            # Video visualization with annotations
├── add_missing_data.py     # Data interpolation for missing frames
├── download_model.py       # Model download helper
├── requirements.txt        # Python dependencies
├── yolov8n.pt              # YOLOv8 nano model for vehicle detection
├── output.png              # Sample output image
├── sort/                   # SORT tracking algorithm
│   └── sort.py             # Multi-object tracker implementation
└── weights/                # Model weights directory
    ├── license_plate.pt    # License plate detection model
    └── yolov8n.pt          # YOLOv8 model backup
```

---

## ⚙️ How It Works

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Input     │────▶│   Vehicle   │────▶│   License   │────▶│    OCR      │
│   Video     │     │  Detection  │     │   Plate     │     │   Reading   │
│             │     │  (YOLOv8)   │     │  Detection  │     │  (EasyOCR)  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                           │                                       │
                           ▼                                       ▼
                    ┌─────────────┐                         ┌─────────────┐
                    │    SORT     │                         │   Output    │
                    │   Tracking  │────────────────────────▶│   CSV &     │
                    │             │                         │   Video     │
                    └─────────────┘                         └─────────────┘
```

1. **Vehicle Detection**: YOLOv8 detects vehicles (cars, trucks, buses, motorcycles)
2. **Object Tracking**: SORT algorithm assigns consistent IDs to tracked vehicles
3. **License Plate Detection**: Roboflow API detects license plates within vehicle bounding boxes
4. **OCR Processing**: EasyOCR extracts text from detected license plates
5. **Output Generation**: Results exported to CSV and annotated video

---

## 📊 Output Format

The detection results are saved in CSV format with the following columns:

| Column | Description |
|--------|-------------|
| `frame_nmr` | Frame number in the video |
| `car_id` | Unique tracking ID for each vehicle |
| `car_bbox` | Vehicle bounding box coordinates `[x1, y1, x2, y2]` |
| `license_plate_bbox` | License plate bounding box coordinates |
| `license_plate_bbox_score` | Detection confidence score |
| `license_number` | Extracted license plate text |
| `license_number_score` | OCR confidence score |


## 📚 Resources

- **Sample Video**: [Traffic Highway Video (Pexels)](https://www.pexels.com/video/traffic-flow-in-the-highway-2103099/)
- **License Plate Dataset**: [Roboflow Universe](https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4)
- **YOLOv8 Training Guide**: [Custom Dataset Training](https://github.com/computervisioneng/train-yolov8-custom-dataset-step-by-step-guide)
- **SORT Algorithm**: [Original Repository](https://github.com/abewley/sort)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- [Roboflow](https://roboflow.com/) for license plate detection API
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) for text recognition
- [SORT](https://github.com/abewley/sort) for multi-object tracking

---

<p align="center">
  Made with ❤️ for Computer Vision
</p>
