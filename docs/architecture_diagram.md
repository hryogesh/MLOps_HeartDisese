# Architecture Diagram

```mermaid
flowchart LR
    A[Task 1: Data Ingestion & EDA<br/>data/heart_disease.csv] --> B[Load & validate<br/>src/data/load_data.py<br/>src/data/validate.py]
    B --> C[Clean & preprocess<br/>src/features/preprocess.py]
    C --> D[Exploratory analysis<br/>notebooks/eda_training.ipynb<br/>artifacts/eda]

    D --> E[Task 2: Model Training & Selection<br/>src/models/model_comparison.py]
    E --> F[Best model artifact<br/>models/model.joblib]

    E --> G[Task 3: Experiment Tracking<br/>mlruns/]
    G --> H[Metrics & artifacts stored]

    F --> I[Task 4: Packaging & Reproducibility<br/>src/api/main.py<br/>app/streamlit_app.py]
    I --> J[Task 5: CI/CD verification<br/>.github/workflows/ci.yml<br/>tests/]

    J --> K[Task 6: Containerization<br/>docker/Dockerfile<br/>docker/docker-compose.yml]
    K --> L[Task 7: Deployment manifests<br/>deployment.yaml<br/>k8s-deployment.yaml]
    L --> M[Cluster deployment<br/>(local Kubernetes)]

    I --> N[Task 8: Monitoring & Logging<br/>monitoring/prometheus.yml<br/>monitoring/docker-compose.monitoring.yml]
    N --> O[Grafana dashboards & alerts]

    style A fill:#f3f8ff,stroke:#2c6fb5,stroke-width:2px
    style B fill:#f3f8ff,stroke:#2c6fb5,stroke-width:2px
    style C fill:#f3f8ff,stroke:#2c6fb5,stroke-width:2px
    style D fill:#f3f8ff,stroke:#2c6fb5,stroke-width:2px
    style E fill:#eff7f0,stroke:#2a7f41,stroke-width:2px
    style F fill:#eff7f0,stroke:#2a7f41,stroke-width:2px
    style G fill:#fff4e5,stroke:#c77d00,stroke-width:2px
    style H fill:#fff4e5,stroke:#c77d00,stroke-width:2px
    style I fill:#f7eef8,stroke:#8a4fa0,stroke-width:2px
    style J fill:#e8f3ff,stroke:#1464b3,stroke-width:2px
    style K fill:#f0f6f8,stroke:#236074,stroke-width:2px
    style L fill:#e8f5ef,stroke:#327a4c,stroke-width:2px
    style M fill:#e8f5ef,stroke:#327a4c,stroke-width:2px
    style N fill:#fff4f0,stroke:#c9372d,stroke-width:2px
    style O fill:#fff4f0,stroke:#c9372d,stroke-width:2px
```

This updated architecture diagram is structured for clarity and maps each major pipeline stage directly to Tasks 1 through 8, reflecting the project flow from data ingestion through deployment and monitoring.
