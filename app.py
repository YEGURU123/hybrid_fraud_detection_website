from flask import Flask, render_template, request, jsonify, session, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import pandas as pd
import numpy as np
import json
import time
import threading
from datetime import datetime
import os

from fraud_detection.models import HybridFraudDetector, TransactionGenerator
from fraud_detection.simulation import FraudSimulation

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hybrid_ai_fraud_secret_2026'
app.config['DEBUG'] = True

CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
simulation_instance = None
detector_instance = None
transaction_history = []
simulation_running = False
simulation_thread = None

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Dashboard route
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

# Simulation route
@app.route('/simulation')
def simulation():
    return render_template('simulation.html')

# API Routes
@app.route('/api/initialize', methods=['POST'])
def initialize_system():
    """Initialize the fraud detection system"""
    global detector_instance

    try:
        # Create and train the detector
        detector = HybridFraudDetector()

        # Generate training data
        generator = TransactionGenerator()
        transactions = []
        for i in range(10000):
            is_fraud = i < 10  # 0.1% fraud rate
            trans = generator.generate_transaction(is_fraud=is_fraud)
            transactions.append(trans)

        df = pd.DataFrame(transactions)
        X = df.drop('is_fraud', axis=1)
        y = df['is_fraud']

        # Train the model
        detector.fit(X, y)

        detector_instance = detector

        return jsonify({
            'status': 'success',
            'message': 'System initialized successfully',
            'model_trained': True
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/simulate', methods=['POST'])
def start_simulation():
    """Start a new simulation"""
    global simulation_instance, simulation_running, simulation_thread, transaction_history

    try:
        data = request.json
        n_transactions = data.get('n_transactions', 500)
        fraud_rate = data.get('fraud_rate', 0.001)

        if detector_instance is None:
            return jsonify({
                'status': 'error',
                'message': 'System not initialized. Please initialize first.'
            }), 400

        # Create simulation
        simulation_instance = FraudSimulation(
            n_transactions=n_transactions,
            fraud_rate=fraud_rate,
            detector=detector_instance
        )

        # Reset transaction history
        transaction_history = []
        simulation_running = True

        # Start simulation in background thread
        def run_simulation():
            global simulation_running, transaction_history
            try:
                results = simulation_instance.run_simulation_stream()
                for result in results:
                    if not simulation_running:
                        break
                    transaction_history.append(result)
                    socketio.emit('transaction_update', result)
                    time.sleep(0.01)  # Simulate real-time processing
            except Exception as e:
                socketio.emit('error', {'message': str(e)})
            finally:
                simulation_running = False
                socketio.emit('simulation_complete', {
                    'status': 'complete',
                    'total_transactions': len(transaction_history)
                })

        simulation_thread = threading.Thread(target=run_simulation)
        simulation_thread.daemon = True
        simulation_thread.start()

        return jsonify({
            'status': 'success',
            'message': f'Simulation started with {n_transactions} transactions'
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/stop_simulation', methods=['POST'])
def stop_simulation():
    """Stop the running simulation"""
    global simulation_running
    simulation_running = False
    return jsonify({
        'status': 'success',
        'message': 'Simulation stopped'
    })

@app.route('/api/get_metrics', methods=['GET'])
def get_metrics():
    """Get current performance metrics"""
    global simulation_instance, transaction_history

    if simulation_instance is None:
        return jsonify({
            'status': 'error',
            'message': 'No simulation running'
        }), 400

    try:
        metrics = simulation_instance.get_current_metrics()

        # Add real-time transaction data
        recent_transactions = transaction_history[-20:] if transaction_history else []

        return jsonify({
            'status': 'success',
            'metrics': metrics,
            'recent_transactions': recent_transactions,
            'total_processed': len(transaction_history)
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/get_confusion_matrix', methods=['GET'])
def get_confusion_matrix():
    """Get confusion matrix data"""
    global simulation_instance

    if simulation_instance is None:
        return jsonify({
            'status': 'error',
            'message': 'No simulation running'
        }), 400

    try:
        cm = simulation_instance.get_confusion_matrix()
        return jsonify({
            'status': 'success',
            'confusion_matrix': cm.tolist()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/get_q_table', methods=['GET'])
def get_q_table():
    """Get RL agent Q-table data"""
    global detector_instance

    if detector_instance is None:
        return jsonify({
            'status': 'error',
            'message': 'System not initialized'
        }), 400

    try:
        q_table = detector_instance.get_q_table()
        return jsonify({
            'status': 'success',
            'q_table': q_table
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/predict_single', methods=['POST'])
def predict_single():
    """Predict a single transaction"""
    global detector_instance

    if detector_instance is None:
        return jsonify({
            'status': 'error',
            'message': 'System not initialized'
        }), 400

    try:
        data = request.json
        transaction = {
            'amount': float(data.get('amount', 100)),
            'geo_distance': float(data.get('geo_distance', 10)),
            'velocity': int(data.get('velocity', 1)),
            'device_risk_score': float(data.get('device_risk', 0.2)),
            'time_delta': float(data.get('time_delta', 24))
        }

        result = detector_instance.predict_transaction(transaction, return_details=True)

        return jsonify({
            'status': 'success',
            'result': result
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# WebSocket events
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('connected', {'status': 'Connected to fraud detection system'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('start_simulation')
def handle_start_simulation(data):
    """Handle simulation start via WebSocket"""
    # This is handled by the REST API
    pass

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000)
