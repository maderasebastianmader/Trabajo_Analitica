# 💼 Income Predictor - ML Classification

Aplicación web full-stack para predecir si una persona gana más de $50K al año usando Machine Learning.

Compara dos modelos: **Regresión Logística** vs **Red Neuronal (MLP)** en el dataset **Adult** de UCI.

## 📊 Dataset

| Característica | Valor |
|---|---|
| **Fuente** | UCI Machine Learning Repository |
| **Nombre** | Adult |
| **Registros** | 32,561 |
| **Features** | 14 (edad, educación, ocupación, etc.) |
| **Clases** | <=50K (75.8%) / >50K (24.2%) |
| **Tarea** | Clasificación Binaria |

## 🤖 Modelos

### Regresión Logística
- **Configuración**: `max_iter=1000`, `solver='lbfgs'`
- **Accuracy**: ~81%
- **Ventaja**: Interpretable, rápido

### Red Neuronal (MLP)
- **Arquitectura**: [input] → [64] → [32] → [2]
- **Activación**: ReLU
- **Optimizador**: Adam
- **Accuracy**: ~85%
- **Ventaja**: Mayor capacidad predictiva

## 🏗️ Estructura del Proyecto

```
Trabajo-Final/
├── backend/
│   └── app.py                    # API Flask
├── frontend/
│   ├── index.html                # Interfaz principal
│   ├── css/
│   │   └── styles.css            # Estilos (diseño moderno)
│   └── js/
│       └── app.js                # Lógica JavaScript
├── data/
│   └── adult.csv                 # Dataset (32,561 registros)
├── models/
│   ├── logistic_regression.pkl   # Modelo entrenado LR
│   ├── neural_network.pkl        # Modelo entrenado NN
│   ├── preprocessor.pkl          # Preprocesador
│   ├── scaler.pkl                # StandardScaler
│   ├── lr_metrics.json           # Métricas LR
│   ├── nn_metrics.json           # Métricas NN
│   └── metadata.json             # Metadata del proyecto
├── notebooks/
│   ├── regresion_logistica.ipynb # Notebook LR
│   └── red_neuronal.ipynb        # Notebook NN
├── requirements.txt              # Dependencias Python
├── train_simple.py               # Script de entrenamiento
└── README.md                      # Este archivo
```

## 🚀 Instalación y Uso

### 1. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 2. Entrenar Modelos (Opcional)

Si deseas reentrenar los modelos:

```bash
python train_simple.py
```

Este script:
- Carga el dataset Adult
- Preprocesa los datos
- Entrena Regresión Logística
- Entrena Red Neuronal
- Guarda modelos, métricas y artefactos en `models/`

### 3. Iniciar el Backend

```bash
cd backend
python app.py
```

El servidor estará disponible en: **http://localhost:10000**

## 🌐 Interfaz Web

La aplicación web se compone de 4 secciones:

### 1. 🏠 Inicio
- Resumen del proyecto
- Tarjetas informativas (Dataset, Modelos, Predicción)
- Preview de métricas de ambos modelos

### 2. 🎯 Predicción Individual
- Formulario con 14 campos de entrada
- Selector de modelo (LR o NN)
- Resultado con emoji, probabilidades e interpretación visual

### 3. 📊 Predicción por Lote
- Carga de archivo CSV (drag & drop)
- Predicción masiva
- Tabla de resultados
- Descarga de CSV de ejemplo

### 4. 📈 Métricas
- Comparación detallada de ambos modelos
- Accuracy, Precision, Recall, F1-Score
- Visualización clara del desempeño

## 🔌 API REST

### Endpoints

#### GET `/api/info`
Retorna metadata del proyecto (features, labels, ejemplo)

```json
{
  "features": ["age", "workclass", ...],
  "target_labels": ["<=50K", ">50K"],
  "example": {...}
}
```

#### GET `/api/metrics`
Retorna métricas de ambos modelos

```json
{
  "logistic_regression": {
    "accuracy": 0.8123,
    "precision": 0.7856,
    "recall": 0.6234,
    "f1": 0.6954
  },
  "neural_network": {
    "accuracy": 0.8495,
    ...
  }
}
```

#### POST `/api/predict`
Predicción individual

**Request:**
```json
{
  "model": "logistic_regression",
  "features": {
    "age": 35,
    "workclass": "Private",
    ...
  }
}
```

**Response:**
```json
{
  "prediction": 1,
  "label": ">50K",
  "probability_class_0": 0.35,
  "probability_class_1": 0.65,
  "model": "logistic_regression"
}
```

#### POST `/api/predict-batch`
Predicción por lote

**Request:** FormData con archivo CSV y modelo

**Response:**
```json
{
  "predictions": [
    {
      "index": 0,
      "prediction": 1,
      "label": ">50K",
      "probability_class_0": 0.35,
      "probability_class_1": 0.65
    },
    ...
  ],
  "model": "neural_network",
  "total": 100
}
```

#### GET `/api/download-example`
Descarga CSV de ejemplo para predicción por lote

## 📓 Notebooks

### `regresion_logistica.ipynb`

Proceso paso a paso:
1. **Importar librerías**
2. **Cargar dataset** - Adult.csv
3. **Ver primeras filas**
4. **Dividir datos** - 70% entrenamiento, 30% prueba
5. **Entrenar modelo** - LogisticRegression
6. **Matriz de confusión**
7. **Calcular accuracy**

### `red_neuronal.ipynb`

Proceso paso a paso:
1. **Importar librerías**
2. **Cargar dataset**
3. **Dividir datos**
4. **Escalado de características** - StandardScaler
5. **Búsqueda de hiperparámetros** - RandomizedSearchCV
6. **Matriz de confusión**
7. **Imprimir accuracy**

## 🎨 Diseño UI/UX

- **Tema**: Moderno, limpio y profesional
- **Colores**: Gradiente púrpura/azul con acentos
- **Componentes**: Cards, formularios, tablas, progress bars
- **Responsivo**: Funciona en desktop y móvil
- **Interactividad**: Navegación fluida, predicciones en tiempo real

## ⚙️ Configuración

### Ambiente Python
- Python 3.9+
- Dependencias en `requirements.txt`

### Backend
- Flask 2.3+
- Flask-CORS habilitado
- Puerto por defecto: 10000

### Modelos
- Scikit-learn para LR y NN
- Joblib para serialización
- JSON para metadata y métricas

## 📈 Resultados Esperados

### Regresión Logística
```
Accuracy:  0.8123 (81.23%)
Precision: 0.7856
Recall:    0.6234
F1-Score:  0.6954
```

### Red Neuronal
```
Accuracy:  0.8495 (84.95%)
Precision: 0.8234
Recall:    0.7012
F1-Score:  0.7582
```

## 🔄 Flujo de Uso

1. **Usuario abre** http://localhost:10000
2. **Consulta** métricas e información
3. **Elige modelo** (LR o NN)
4. **Opción A**: Predicción individual
   - Ingresa datos
   - Obtiene resultado con probabilidades
5. **Opción B**: Predicción por lote
   - Carga CSV
   - Obtiene tabla de resultados

## 🛠️ Troubleshooting

### Error: "ModuleNotFoundError"
```bash
pip install -r requirements.txt
```

### Error: "Port already in use"
```bash
# Usa otro puerto
set PORT=8000
python app.py
```

### Modelo no entrena correctamente
Ejecuta:
```bash
python train_simple.py
```

## 📝 Notas

- El dataset Adult contiene datos reales y puede tener sesgos
- Los modelos son solo para fines educativos y de demostración
- Se recomienda validar predicciones en contextos reales
- Las métricas mostradas son en el conjunto de prueba

## 👤 Autor

Johan Hernandez - PROYECTO ANTIGRAVITY

## 📄 Licencia

Proyecto educativo - Uso libre

---

**Versión**: 1.0  
**Última actualización**: Mayo 2026
