import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from collections import defaultdict
import random
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class TransactionGenerator:
    """Generates realistic transaction data"""

    def __init__(self):
        self.customer_profiles = self._generate_customer_profiles(100)

    def _generate_customer_profiles(self, n_customers):
        profiles = {}
        for i in range(n_customers):
            profiles[i] = {
                'avg_amount': np.random.gamma(2, 50),
                'std_amount': np.random.gamma(2, 20),
                'home_location': (np.random.uniform(-180, 180), np.random.uniform(-90, 90)),
                'avg_velocity': np.random.poisson(2),
                'device_risk': np.random.beta(1, 5),
                'preferred_merchant': random.choice(['retail', 'travel', 'online', 'dining'])
            }
        return profiles

    def generate_transaction(self, customer_id=None, is_fraud=False):
        """Generate a single transaction"""
        if customer_id is None:
            customer_id = random.choice(list(self.customer_profiles.keys()))

        profile = self.customer_profiles[customer_id]

        if is_fraud:
            amount = np.random.gamma(1, 300)
            geo_distance = np.random.exponential(1000)
            velocity = np.random.poisson(15)
            device_risk = np.random.beta(3, 1)
            time_delta = np.random.exponential(2)
        else:
            amount = abs(np.random.normal(profile['avg_amount'], profile['std_amount']))
            geo_distance = np.random.exponential(50)
            velocity = np.random.poisson(profile['avg_velocity'])
            device_risk = profile['device_risk'] + np.random.normal(0, 0.1)
            time_delta = np.random.exponential(24)

        return {
            'transaction_id': f"TXN_{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1000,9999)}",
            'timestamp': datetime.now().isoformat(),
            'customer_id': customer_id,
            'amount': max(0.01, amount),
            'geo_distance': max(0, geo_distance),
            'velocity': int(max(0, velocity)),
            'device_risk_score': max(0, min(1, device_risk)),
            'time_delta': max(0, time_delta),
            'is_fraud': int(is_fraud)
        }

class SimplifiedAutoencoder:
    """Simplified autoencoder for anomaly detection"""

    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.encoder = MLPRegressor(
            hidden_layer_sizes=(6, 3),
            activation='relu',
            max_iter=100,
            random_state=42
        )
        self.decoder = MLPRegressor(
            hidden_layer_sizes=(6, input_dim),
            activation='relu',
            max_iter=100,
            random_state=42
        )

    def fit(self, X):
        encoded = self.encoder.fit_transform(X, X)
        self.decoder.fit(encoded, X)

    def reconstruction_error(self, X):
        encoded = self.encoder.transform(X)
        reconstructed = self.decoder.predict(encoded)
        return np.mean((X - reconstructed) ** 2, axis=1)

class RLAgent:
    """Reinforcement Learning Agent"""

    def __init__(self):
        self.actions = ['approve', 'flag_2fa', 'block']
        self.q_table = defaultdict(float)
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.3

        self._initialize_q_table()

    def _initialize_q_table(self):
        """Initialize Q-table with domain knowledge"""
        initial_strategies = {
            'low_low': {'approve': 1.0, 'flag_2fa': 0.5, 'block': -0.5},
            'low_medium': {'approve': 0.5, 'flag_2fa': 1.0, 'block': -0.5},
            'low_high': {'approve': -0.5, 'flag_2fa': 1.0, 'block': 0.5},
            'medium_low': {'approve': 0.5, 'flag_2fa': 1.0, 'block': -0.5},
            'medium_medium': {'approve': -0.5, 'flag_2fa': 1.0, 'block': 0.5},
            'medium_high': {'approve': -1.0, 'flag_2fa': 0.5, 'block': 1.0},
            'high_low': {'approve': -0.5, 'flag_2fa': 0.5, 'block': 1.0},
            'high_medium': {'approve': -1.0, 'flag_2fa': 0.0, 'block': 1.0},
            'high_high': {'approve': -2.0, 'flag_2fa': -1.0, 'block': 2.0}
        }

        for state, actions in initial_strategies.items():
            for action, value in actions.items():
                self.q_table[f"{state}_{action}"] = value

    def get_state(self, anomaly_score, xgb_prob, ae_error):
        """Convert continuous values to discrete state"""
        anomaly_bin = 'low' if anomaly_score > -0.5 else 'medium' if anomaly_score > -1.0 else 'high'
        prob_bin = 'low' if xgb_prob < 0.3 else 'medium' if xgb_prob < 0.7 else 'high'
        return f"{anomaly_bin}_{prob_bin}"

    def decide_action(self, state, risk_score):
        """Choose action using epsilon-greedy policy"""
        if random.random() < self.exploration_rate:
            if risk_score > 0.7:
                action = random.choice(['flag_2fa', 'block'])
            elif risk_score < 0.3:
                action = random.choice(['approve', 'flag_2fa'])
            else:
                action = random.choice(self.actions)
            confidence = 0.3
        else:
            q_values = {action: self.q_table[f"{state}_{action}"] for action in self.actions}
            best_action = max(q_values, key=q_values.get)
            action = best_action
            confidence = 0.7 + random.random() * 0.3

        return action, confidence

    def update_q_value(self, state, action, reward, next_state):
        """Update Q-value using Q-learning"""
        current_q = self.q_table[f"{state}_{action}"]
        max_next_q = max([self.q_table[f"{next_state}_{a}"] for a in self.actions])
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[f"{state}_{action}"] = new_q
        self.exploration_rate = max(0.01, self.exploration_rate * 0.995)

class HybridFraudDetector:
    """Complete Hybrid AI Fraud Detection System"""

    def __init__(self):
        self.anomaly_detector = IsolationForest(contamination=0.01, random_state=42)
        self.autoencoder = None
        self.classifier = XGBClassifier(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.1,
            scale_pos_weight=1000,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        self.rl_agent = RLAgent()
        self.scaler = StandardScaler()
        self.feature_columns = ['amount', 'geo_distance', 'velocity', 'device_risk_score', 'time_delta']
        self.is_trained = False

    def preprocess_features(self, df):
        X = df[self.feature_columns].copy()
        X['amount'] = np.log1p(X['amount'])
        X['geo_distance'] = np.log1p(X['geo_distance'])
        return X

    def fit(self, X_train, y_train):
        X_train_scaled = self.preprocess_features(X_train)
        X_train_scaled = self.scaler.fit_transform(X_train_scaled)

        self.anomaly_detector.fit(X_train_scaled)

        self.autoencoder = SimplifiedAutoencoder(input_dim=X_train_scaled.shape[1])
        self.autoencoder.fit(X_train_scaled)

        self.classifier.fit(X_train_scaled, y_train, verbose=False)

        self.is_trained = True

    def predict_transaction(self, transaction, return_details=False):
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        if isinstance(transaction, dict):
            df = pd.DataFrame([transaction])
        else:
            df = transaction.copy()

        X_processed = self.preprocess_features(df)
        X_scaled = self.scaler.transform(X_processed)

        anomaly_score = self.anomaly_detector.decision_function(X_scaled)[0]
        is_anomaly = self.anomaly_detector.predict(X_scaled)[0] == -1

        ae_error = self.autoencoder.reconstruction_error(X_scaled)[0]

        xgb_prob = self.classifier.predict_proba(X_scaled)[0][1]

        risk_score = 0.4 * xgb_prob + 0.3 * (1 if is_anomaly else 0) + 0.3 * min(1, ae_error / 2)

        state = self.rl_agent.get_state(anomaly_score, xgb_prob, ae_error)
        action, confidence = self.rl_agent.decide_action(state, risk_score)

        result = {
            'anomaly_score': float(anomaly_score),
            'is_anomaly': bool(is_anomaly),
            'autoencoder_error': float(ae_error),
            'xgb_probability': float(xgb_prob),
            'risk_score': float(risk_score),
            'risk_level': self._get_risk_level(risk_score),
            'rl_state': state,
            'rl_action': action,
            'rl_confidence': float(confidence),
            'final_decision': self._get_final_decision(action)
        }

        if return_details:
            return result
        return result['final_decision']

    def _get_risk_level(self, risk_score):
        if risk_score < 0.3:
            return 'Low'
        elif risk_score < 0.6:
            return 'Medium'
        elif risk_score < 0.8:
            return 'High'
        return 'Critical'

    def _get_final_decision(self, rl_action):
        action_map = {
            'approve': 'APPROVE',
            'flag_2fa': 'FLAG - 2FA Required',
            'block': 'BLOCK - High Risk'
        }
        return action_map.get(rl_action, 'REVIEW')

    def process_transaction_with_feedback(self, transaction, true_label):
        result = self.predict_transaction(transaction, return_details=True)

        reward = self._calculate_reward(result['rl_action'], true_label)
        self.rl_agent.update_q_value(
            result['rl_state'],
            result['rl_action'],
            reward,
            result['rl_state']
        )

        return result, reward

    def _calculate_reward(self, action, true_label):
        if action == 'block' and true_label == 1:
            return 10
        elif action in ['approve', 'flag_2fa'] and true_label == 0:
            return 1
        elif action == 'block' and true_label == 0:
            return -20
        elif action in ['approve', 'flag_2fa'] and true_label == 1:
            return -30
        return 0

    def get_q_table(self):
        return dict(self.rl_agent.q_table)
