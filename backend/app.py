from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import json
import os
import sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from io import BytesIO

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
frontend_dir = os.path.join(root_dir, 'frontend')
models_dir = os.path.join(root_dir, 'models')

app = Flask(__name__, template_folder=frontend_dir)
CORS(app)

print('Python ejecutable:', os.sys.executable)
print('sklearn versión:', sklearn.__version__)
print('Cargando modelos...')
preprocessor = joblib.load(os.path.join(models_dir, 'preprocessor.pkl'))
lr_model = joblib.load(os.path.join(models_dir, 'logistic_regression.pkl'))
if not hasattr(lr_model, 'multi_class'):
    lr_model.multi_class = 'auto'
if not hasattr(lr_model, 'multi_class_'):
    lr_model.multi_class_ = 'auto'
nn_model = joblib.load(os.path.join(models_dir, 'neural_network.pkl'))

with open(os.path.join(models_dir, 'metadata.json'), 'r', encoding='utf-8') as f:
    metadata = json.load(f)

with open(os.path.join(models_dir, 'lr_metrics.json'), 'r', encoding='utf-8') as f:
    lr_metrics = json.load(f)

with open(os.path.join(models_dir, 'nn_metrics.json'), 'r', encoding='utf-8') as f:
    nn_metrics = json.load(f)

print('Modelos cargados exitosamente')

numeric_features = metadata.get('numerical_features', [])


def prepare_features(features_dict):
    df = pd.DataFrame([features_dict])
    for col in numeric_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return preprocessor.transform(df)


@app.route('/css/<path:filename>')
def serve_css(filename):
    return send_from_directory(os.path.join(frontend_dir, 'css'), filename)


@app.route('/js/<path:filename>')
def serve_js(filename):
    return send_from_directory(os.path.join(frontend_dir, 'js'), filename)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/info', methods=['GET'])
def get_info():
    return jsonify({
        'features': metadata['features'],
        'target_labels': metadata['target_labels'],
        'example': metadata['example']
    })


@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    return jsonify({
        'logistic_regression': lr_metrics,
        'neural_network': nn_metrics
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        model_name = data.get('model', 'logistic_regression')
        features = data.get('features', {})
        X = prepare_features(features)

        if model_name == 'neural_network':
            prediction = nn_model.predict(X)[0]
            probabilities = nn_model.predict_proba(X)[0]
        else:
            prediction = lr_model.predict(X)[0]
            probabilities = lr_model.predict_proba(X)[0]

        return jsonify({
            'prediction': int(prediction),
            'label': metadata['target_labels'][prediction],
            'probability_class_0': float(probabilities[0]),
            'probability_class_1': float(probabilities[1]),
            'model': model_name
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/predict-batch', methods=['POST'])
def predict_batch():
    try:
        print('=== PREDICT BATCH ===')
        model_name = request.form.get('model', 'logistic_regression')
        print(f'Modelo: {model_name}')
        
        if 'file' not in request.files:
            print('Error: No file provided')
            return jsonify({'error': 'No file provided'}), 400
        file = request.files['file']
        if file.filename == '':
            print('Error: Empty filename')
            return jsonify({'error': 'Empty filename'}), 400

        print(f'Leyendo archivo: {file.filename}')
        df = pd.read_csv(file)
        
        # Limitar a un máximo de 200 registros
        if len(df) > 200:
            df = df.head(200)
            
        print(f'Archivo cargado (limitado): {df.shape[0]} filas x {df.shape[1]} columnas')
        print(f'Columnas del CSV: {list(df.columns)}')
        
        # Verificar si existe la columna 'income' para calcular métricas
        has_income = 'income' in df.columns
        y_true = None
        if has_income:
            print('Columna "income" detectada - Se calcularán métricas de desempeño')
            # Mapear los valores de income a etiquetas numéricas
            income_mapping = {label: idx for idx, label in enumerate(metadata['target_labels'])}
            # Limpiar espacios en blanco y hacer mapeo
            y_true = df['income'].str.strip().map(income_mapping).values
        
        # Validar que todas las features requeridas estén presentes
        required_features = set(metadata['features'])
        csv_features = set(df.columns)
        
        # Remover 'income' si está presente (no es requerido para predicción)
        csv_features.discard('income')
        
        missing_features = required_features - csv_features
        if missing_features:
            error_msg = f'Faltan las siguientes columnas en el CSV: {", ".join(missing_features)}'
            print(f'Error: {error_msg}')
            return jsonify({'error': error_msg}), 400
        
        extra_features = csv_features - required_features
        if extra_features:
            print(f'Advertencia: Columnas extra encontradas (serán ignoradas): {", ".join(extra_features)}')
        
        # Usar solo las features requeridas
        X_df = df[metadata['features']].copy()
        print(f'Después de seleccionar features: {X_df.shape[1]} columnas')
        print(f'Numeric features esperadas: {numeric_features}')
        
        # Procesar cada fila individualmente para evitar errores en lotes
        results = []
        y_pred = []
        for idx in range(len(X_df)):
            try:
                row = X_df.iloc[idx:idx+1].copy()
                for col in numeric_features:
                    if col in row.columns:
                        row[col] = pd.to_numeric(row[col], errors='coerce').fillna(0)
                
                X_row = preprocessor.transform(row)
                
                if model_name == 'neural_network':
                    prediction = nn_model.predict(X_row)[0]
                    probabilities = nn_model.predict_proba(X_row)[0]
                else:
                    prediction = lr_model.predict(X_row)[0]
                    probabilities = lr_model.predict_proba(X_row)[0]
                
                y_pred.append(int(prediction))
                results.append({
                    'index': idx,
                    'prediction': int(prediction),
                    'label': metadata['target_labels'][prediction],
                    'probability_class_0': float(probabilities[0]),
                    'probability_class_1': float(probabilities[1])
                })
            except Exception as row_error:
                print(f'Error procesando fila {idx}: {str(row_error)}')
                results.append({
                    'index': idx,
                    'prediction': -1,
                    'label': 'Error',
                    'probability_class_0': 0.0,
                    'probability_class_1': 0.0
                })
                y_pred.append(-1)
        
        print(f'Predicciones completadas: {len(results)} filas procesadas')

        # Calcular métricas si se proporcionaron etiquetas reales
        metrics = None
        if has_income and y_true is not None and len(y_pred) == len(y_true):
            try:
                # Filtrar solo las predicciones válidas (no -1)
                valid_mask = [p != -1 for p in y_pred]
                y_true_valid = y_true[valid_mask]
                y_pred_valid = [p for p, v in zip(y_pred, valid_mask) if v]
                
                if len(y_pred_valid) > 0:
                    accuracy = accuracy_score(y_true_valid, y_pred_valid)
                    precision = precision_score(y_true_valid, y_pred_valid, average='weighted', zero_division=0)
                    recall = recall_score(y_true_valid, y_pred_valid, average='weighted', zero_division=0)
                    f1 = f1_score(y_true_valid, y_pred_valid, average='weighted', zero_division=0)
                    cm = confusion_matrix(y_true_valid, y_pred_valid).tolist()
                    
                    metrics = {
                        'accuracy': float(accuracy),
                        'precision': float(precision),
                        'recall': float(recall),
                        'f1_score': float(f1),
                        'confusion_matrix': cm
                    }
                    print(f'Métricas calculadas - Accuracy: {accuracy:.4f}, F1: {f1:.4f}')
            except Exception as metrics_error:
                print(f'Error calculando métricas: {str(metrics_error)}')

        print(f'Retornando {len(results)} resultados')
        response = {
            'predictions': results,
            'model': model_name,
            'total': len(results)
        }
        if metrics:
            response['metrics'] = metrics
        
        return jsonify(response)
    except Exception as e:
        error_msg = f'Error en predict_batch: {str(e)}'
        print(error_msg)
        import traceback
        traceback.print_exc()
        return jsonify({'error': error_msg}), 400


@app.route('/api/download-example', methods=['GET'])
def download_example():
    try:
        csv_path = os.path.join(root_dir, 'data', 'adult.csv')
        if os.path.exists(csv_path):
            df_example = pd.read_csv(csv_path).head(200)
        else:
            df_example = pd.DataFrame([metadata['example']])
        
        output = BytesIO()
        df_example.to_csv(output, index=False)
        output.seek(0)
        response = send_file(
            output,
            mimetype='text/csv',
            as_attachment=True,
            download_name='ejemplo_batch.csv'
        )
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/static/training_summary.png')
def serve_training_summary():
    return send_file(os.path.join(models_dir, 'training_summary.png'), mimetype='image/png')


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Ruta no encontrada'}), 404


@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Error interno del servidor'}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=True)
