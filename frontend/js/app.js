// ============= STATE =============
let features = [];
let featureLabels = {};
let exampleValues = {};
let metrics = null;

// ============= INITIALIZATION =============

document.addEventListener('DOMContentLoaded', () => {
    loadMetrics();
    setupEventListeners();
});

// ============= API CALLS =============

async function loadMetrics() {
    try {
        const response = await fetch('/api/info');
        const data = await response.json();
        features = data.features;
        featureLabels = createFeatureLabels(data.features);
        exampleValues = data.example;

        const metricsResponse = await fetch('/api/metrics');
        metrics = await metricsResponse.json();

        buildForm();
        renderMetrics();
    } catch (error) {
        console.error('Error loading metrics:', error);
        alert('Error cargando datos de la aplicación: ' + error.message);
    }
}

function createFeatureLabels(features) {
    const labels = {
        'age': 'Edad',
        'workclass': 'Clase de Trabajo',
        'fnlwgt': 'Peso Final',
        'education': 'Educación',
        'education-num': 'Años de Educación',
        'marital-status': 'Estado Civil',
        'occupation': 'Ocupación',
        'relationship': 'Relación',
        'race': 'Raza',
        'sex': 'Sexo',
        'capital-gain': 'Ganancia de Capital',
        'capital-loss': 'Pérdida de Capital',
        'hours-per-week': 'Horas por Semana',
        'native-country': 'País de Origen'
    };
    return labels;
}

// ============= RENDER FUNCTIONS =============

function buildForm() {
    const formGrid = document.querySelector('.form-grid');
    formGrid.innerHTML = '';

    features.forEach(feature => {
        const group = document.createElement('div');
        group.className = 'form-group';

        const label = document.createElement('label');
        label.textContent = featureLabels[feature] || feature;

        const input = document.createElement('input');
        input.type = 'text';
        input.name = feature;
        input.value = exampleValues[feature] || '';
        input.placeholder = feature;

        group.appendChild(label);
        group.appendChild(input);
        formGrid.appendChild(group);
    });
}

function renderMetrics() {
    if (!metrics) return;

    // Render detailed metrics
    renderDetailedMetrics('lr-detailed-metrics', metrics.logistic_regression);
    renderDetailedMetrics('nn-detailed-metrics', metrics.neural_network);
}

function renderDetailedMetrics(containerId, metricsData) {
    const container = document.getElementById(containerId);
    if (!container) {
        console.error(`Container ${containerId} not found`);
        return;
    }
    container.innerHTML = `
        <div class="metric-row">
            <span class="metric-row-label">Accuracy</span>
            <span class="metric-row-value">${(metricsData.accuracy * 100).toFixed(2)}%</span>
        </div>
        <div class="metric-row">
            <span class="metric-row-label">Precision</span>
            <span class="metric-row-value">${(metricsData.precision * 100).toFixed(2)}%</span>
        </div>
        <div class="metric-row">
            <span class="metric-row-label">Recall</span>
            <span class="metric-row-value">${(metricsData.recall * 100).toFixed(2)}%</span>
        </div>
        <div class="metric-row">
            <span class="metric-row-label">F1-Score</span>
            <span class="metric-row-value">${(metricsData.f1_score * 100).toFixed(2)}%</span>
        </div>
    `;
}

// ============= PREDICTION FUNCTIONS =============

async function predictIndividual(e) {
    e.preventDefault();

    const formData = new FormData(document.getElementById('predict-form'));
    const model = document.querySelector('input[name="model"]:checked').value;

    // Construir objeto de features
    const featuresObj = {};
    features.forEach(feature => {
        featuresObj[feature] = formData.get(feature);
    });

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: model,
                features: featuresObj
            })
        });

        const data = await response.json();

        if (response.ok) {
            showResult(data);
        } else {
            alert('Error en la predicción: ' + (data.error || 'Error desconocido'));
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Error de conexión');
    }
}

function showResult(data) {
    const resultContainer = document.getElementById('result');
    const resultLabel = document.getElementById('result-label');
    const resultEmoji = document.getElementById('result-emoji');
    const resultModel = document.getElementById('result-model');

    // Determinar clase y emoji
    const isHighIncome = data.prediction === 1;
    const emoji = isHighIncome ? '💰' : '💼';
    const labelClass = isHighIncome ? 'high-income' : 'low-income';

    resultEmoji.textContent = emoji;
    resultLabel.textContent = data.label;
    resultLabel.className = 'result-label ' + labelClass;
    resultModel.textContent = `Modelo: ${data.model === 'neural_network' ? 'Red Neuronal' : 'Regresión Logística'}`;

    // Actualizar barras de probabilidad
    const prob0Pct = (data.probability_class_0 * 100).toFixed(1);
    const prob1Pct = (data.probability_class_1 * 100).toFixed(1);

    document.getElementById('prob0').style.width = prob0Pct + '%';
    document.getElementById('prob1').style.width = prob1Pct + '%';
    document.getElementById('prob0-text').textContent = prob0Pct + '%';
    document.getElementById('prob1-text').textContent = prob1Pct + '%';

    resultContainer.style.display = 'block';
}

async function predictBatch() {
    const fileInput = document.getElementById('csv-input');
    const model = document.querySelector('input[name="batch-model"]:checked').value;

    if (!fileInput.files[0]) {
        alert('Por favor selecciona un archivo');
        return;
    }

    console.log('Iniciando predicción por lotes');
    console.log('Archivo:', fileInput.files[0].name);
    console.log('Modelo:', model);

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('model', model);

    try {
        const response = await fetch('/api/predict-batch', {
            method: 'POST',
            body: formData
        });

        console.log('Response status:', response.status);
        console.log('Response ok:', response.ok);

        const data = await response.json();
        console.log('Response data:', data);

        if (response.ok) {
            console.log('Predicción exitosa');
            showBatchResults(data);
        } else {
            const errorMsg = data.error || 'Error desconocido';
            console.error('Error del servidor:', errorMsg);
            alert('Error: ' + errorMsg);
        }
    } catch (error) {
        console.error('Error en fetch:', error);
        alert('Error de conexión: ' + error.message);
    }
}

function showBatchResults(data) {
    const batchResults = document.getElementById('batch-results');
    const rows = data.predictions.slice(0, 200);
    const model = data.model === 'neural_network' ? 'Red Neuronal' : 'Regresión Logística';
    const total = data.predictions.length;
    const highIncomeCount = data.predictions.filter(r => r.prediction === 1).length;
    const lowIncomeCount = total - highIncomeCount;
    
    // Construir HTML de métricas si están disponibles
    let metricsHTML = '';
    if (data.metrics) {
        const m = data.metrics;
        let confusionMatrixHTML = '';
        if (m.confusion_matrix) {
            confusionMatrixHTML = `
                <div style="background:white;padding:16px;border-radius:6px;border:1px solid #c8e6c9;margin-top:12px;">
                    <div style="font-size:14px;font-weight:600;margin-bottom:12px;color:#2e7d32;">Matriz de Confusión</div>
                    <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:8px;max-width:200px;margin:0 auto;">
                        <div style="text-align:center;padding:8px;background:#f1f8e9;border:1px solid #c8e6c9;border-radius:4px;">
                            <div style="font-size:18px;font-weight:700;color:#4caf50;">${m.confusion_matrix[0][0]}</div>
                            <div style="font-size:10px;color:#666;">Verdaderos <=50K</div>
                        </div>
                        <div style="text-align:center;padding:8px;background:#ffebee;border:1px solid #ffcdd2;border-radius:4px;">
                            <div style="font-size:18px;font-weight:700;color:#f44336;">${m.confusion_matrix[0][1]}</div>
                            <div style="font-size:10px;color:#666;">Falsos >50K</div>
                        </div>
                        <div style="text-align:center;padding:8px;background:#ffebee;border:1px solid #ffcdd2;border-radius:4px;">
                            <div style="font-size:18px;font-weight:700;color:#f44336;">${m.confusion_matrix[1][0]}</div>
                            <div style="font-size:10px;color:#666;">Falsos <=50K</div>
                        </div>
                        <div style="text-align:center;padding:8px;background:#f1f8e9;border:1px solid #c8e6c9;border-radius:4px;">
                            <div style="font-size:18px;font-weight:700;color:#4caf50;">${m.confusion_matrix[1][1]}</div>
                            <div style="font-size:10px;color:#666;">Verdaderos >50K</div>
                        </div>
                    </div>
                </div>
            `;
        }
        metricsHTML = `
            <div style="background:#e8f5e9;border-left:4px solid #4caf50;padding:16px;margin:16px 0;border-radius:8px;">
                <div style="font-weight:700;margin-bottom:12px;color:#2e7d32;">📊 Métricas de Desempeño</div>
                <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;">
                    <div style="background:white;padding:12px;border-radius:6px;border:1px solid #c8e6c9;">
                        <div style="font-size:12px;color:#666;">Accuracy</div>
                        <div style="font-size:24px;font-weight:700;color:#4caf50;">${(m.accuracy * 100).toFixed(2)}%</div>
                    </div>
                    <div style="background:white;padding:12px;border-radius:6px;border:1px solid #c8e6c9;">
                        <div style="font-size:12px;color:#666;">Precision</div>
                        <div style="font-size:24px;font-weight:700;color:#4caf50;">${(m.precision * 100).toFixed(2)}%</div>
                    </div>
                    <div style="background:white;padding:12px;border-radius:6px;border:1px solid #c8e6c9;">
                        <div style="font-size:12px;color:#666;">Recall</div>
                        <div style="font-size:24px;font-weight:700;color:#4caf50;">${(m.recall * 100).toFixed(2)}%</div>
                    </div>
                    <div style="background:white;padding:12px;border-radius:6px;border:1px solid #c8e6c9;">
                        <div style="font-size:12px;color:#666;">F1-Score</div>
                        <div style="font-size:24px;font-weight:700;color:#4caf50;">${(m.f1_score * 100).toFixed(2)}%</div>
                    </div>
                </div>
                ${confusionMatrixHTML}
            </div>
        `;
    }

    batchResults.innerHTML = `
        <div class="panel-title">Resultados</div>
        <div style="padding:12px 16px;background:#f8f9fa;border-bottom:1px solid #dee2e6;font-size:13px;display:flex;gap:20px;align-items:center;">
            <span>Modelo: <strong>${model}</strong></span>
            <span>Total: <strong>${total}</strong></span>
            <span style="color:#28a745"><=50K: <strong>${lowIncomeCount}</strong></span>
            <span style="color:#dc3545">>50K: <strong>${highIncomeCount}</strong></span>
            ${total > 200 ? '<span style="color:#6c757d;font-size:12px;">(Mostrando primeras 200 filas)</span>' : ''}
        </div>
        ${metricsHTML}
        <table class="results-table">
            <thead>
                <tr>
                    <th>#</th>
                    <th>Predicción</th>
                    <th>Prob <=50K</th>
                    <th>Prob >50K</th>
                </tr>
            </thead>
            <tbody>
                ${rows.map(r => `
                    <tr>
                        <td>${r.index + 1}</td>
                        <td><span class="badge ${r.prediction === 1 ? 'badge-high' : 'badge-low'}">${r.label}</span></td>
                        <td>${(r.probability_class_0 * 100).toFixed(1)}%</td>
                        <td>${(r.probability_class_1 * 100).toFixed(1)}%</td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;

    batchResults.style.display = 'block';
    batchResults.scrollIntoView({ behavior: 'smooth' });
}

async function downloadExample() {
    try {
        const response = await fetch('/api/download-example?t=' + new Date().getTime(), { cache: 'no-store' });
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'ejemplo_batch.csv';
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
    } catch (error) {
        console.error('Error:', error);
        alert('Error descargando archivo');
    }
}

// ============= EVENT LISTENERS =============

function setupEventListeners() {
    // Navigation
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const sectionId = link.dataset.section;
            navigateTo(sectionId);
        });
    });

    // Forms
    document.getElementById('predict-form').addEventListener('submit', predictIndividual);
    document.getElementById('btn-predict-batch').addEventListener('click', predictBatch);
    document.getElementById('btn-download-example').addEventListener('click', downloadExample);

    // File upload
    const uploadZone = document.getElementById('upload-zone');
    const csvInput = document.getElementById('csv-input');
    const btnPredictBatch = document.getElementById('btn-predict-batch');

    uploadZone.addEventListener('click', () => csvInput.click());

    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('dragover');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            csvInput.files = e.dataTransfer.files;
            btnPredictBatch.disabled = false;
            const contentDiv = uploadZone.querySelector('.upload-content');
            if (contentDiv) {
                contentDiv.innerHTML = `<p>✓ ${e.dataTransfer.files[0].name}</p>`;
            }
        }
    });

    csvInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            btnPredictBatch.disabled = false;
            const contentDiv = uploadZone.querySelector('.upload-content');
            if (contentDiv) {
                contentDiv.innerHTML = `<p>✓ ${e.target.files[0].name}</p>`;
            }
        }
    });
}

function navigateTo(sectionId) {
    // Update active nav
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
    });
    document.querySelector(`[data-section="${sectionId}"]`).classList.add('active');

    // Update active section
    document.querySelectorAll('.section').forEach(section => {
        section.classList.remove('active');
    });
    document.getElementById(sectionId).classList.add('active');
}
