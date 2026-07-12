from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1800, 1200
BG_COLOR = (255, 255, 255)
BOX_FILL = (230, 243, 255)
BOX_STROKE = (30, 100, 170)
ARROW_COLOR = (0, 79, 134)
TEXT_COLOR = (20, 20, 20)
TITLE_COLOR = (10, 60, 110)
TITLE = "Repository Workflow Diagram: Tasks 1 - 8"

font = ImageFont.load_default()
font_title = ImageFont.load_default()

boxes = [
    {
        "title": "Task 1: Data Ingestion & EDA",
        "lines": [
            "data/heart_disease.csv",
            "src/data/",
            "src/features/",
            "notebooks/",
        ],
        "xy": (80, 140, 520, 340),
    },
    {
        "title": "Task 2: Model Training",
        "lines": [
            "src/models/",
            "src/features/preprocess.py",
            "models/model.joblib",
        ],
        "xy": (620, 140, 1060, 340),
    },
    {
        "title": "Task 3: Experiment Tracking",
        "lines": [
            "src/train.py",
            "mlruns/",
            "artifacts/mlflow/",
        ],
        "xy": (1160, 140, 1600, 340),
    },
    {
        "title": "Task 4: Packaging",
        "lines": [
            "src/api/main.py",
            "app/streamlit_app.py",
            "models/model.joblib",
        ],
        "xy": (80, 420, 520, 620),
    },
    {
        "title": "Task 5: CI/CD",
        "lines": [
            ".github/workflows/ci.yml",
            "tests/",
            "requirements-dev.txt",
        ],
        "xy": (620, 420, 1060, 620),
    },
    {
        "title": "Task 6: Containerization",
        "lines": [
            "docker/",
            "docker-compose.yml",
            "Dockerfile",
        ],
        "xy": (1160, 420, 1600, 620),
    },
    {
        "title": "Task 7: Deployment",
        "lines": [
            "deployment.yaml",
            "k8s-deployment.yaml",
            "config/",
        ],
        "xy": (350, 700, 790, 900),
    },
    {
        "title": "Task 8: Monitoring",
        "lines": [
            "monitoring/",
            "prometheus.yml",
            "Grafana dashboards",
        ],
        "xy": (1010, 700, 1450, 900),
    },
]

img = Image.new("RGB", (WIDTH, HEIGHT), BG_COLOR)
draw = ImageDraw.Draw(img)

# Title
w, h = draw.textsize(TITLE, font=font_title)
draw.text(((WIDTH - w) / 2, 30), TITLE, fill=TITLE_COLOR, font=font_title)

# Draw boxes
for box in boxes:
    x0, y0, x1, y1 = box["xy"]
    draw.rectangle([x0, y0, x1, y1], fill=BOX_FILL, outline=BOX_STROKE, width=4)
    draw.text((x0 + 18, y0 + 18), box["title"], fill=TEXT_COLOR, font=font)
    y = y0 + 48
    for line in box["lines"]:
        draw.text((x0 + 18, y), line, fill=TEXT_COLOR, font=font)
        y += 24

# Draw arrows
arrow_defs = [
    ((520, 240), (620, 240)),
    ((1060, 240), (1160, 240)),
    ((520, 520), (620, 520)),
    ((1060, 520), (1160, 520)),
    ((370, 620), (370, 700)),
    ((1010, 620), (1010, 700)),
    ((790, 800), (1010, 800)),
]

for start, end in arrow_defs:
    draw.line([start, end], fill=ARROW_COLOR, width=5)
    # arrowhead
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    if abs(dx) > abs(dy):
        sign = 1 if dx > 0 else -1
        arrow = [(end[0] - 16 * sign, end[1] - 10), (end[0], end[1]), (end[0] - 16 * sign, end[1] + 10)]
    else:
        sign = 1 if dy > 0 else -1
        arrow = [(end[0] - 10, end[1] - 16 * sign), (end[0], end[1]), (end[0] + 10, end[1] - 16 * sign)]
    draw.polygon(arrow, fill=ARROW_COLOR)

img.save("docs/architecture_diagram.png")
print("Generated docs/architecture_diagram.png")
