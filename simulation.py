import numpy as np
import pandas as pd
import random
from datetime import datetime
from .models import TransactionGenerator

class FraudSimulation:
    """Run fraud detection simulations"""

    def __init__(self, n_transactions=500, fraud_rate=0.001, detector=None):
        self.n_transactions = n_transactions
        self.fraud_rate = fraud_rate
        self.detector = detector
        self.generator = TransactionGenerator()
        self.results = []
        self.metrics = {
            'true_positives': 0,
            'true_negatives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'total_reward': 0,
            'actions': {'approve': 0, 'flag_2fa': 0, 'block': 0}
        }

    def run_simulation_stream(self):
        """Run simulation and yield results as they come"""
        # Generate transactions
        transactions = []
        fraud_count = int(self.n_transactions * self.fraud_rate)

        for i in range(self.n_transactions):
            is_fraud = i < fraud_count
            trans = self.generator.generate_transaction(is_fraud=is_fraud)
            transactions.append(trans)

        random.shuffle(transactions)

        # Process transactions
        for transaction in transactions:
            true_label = transaction.pop('is_fraud', 0)
            transaction['is_fraud'] = true_label

            # Process with detector
            if self.detector:
                result, reward = self.detector.process_transaction_with_feedback(
                    transaction, true_label
                )
            else:
                # Random baseline
                action = random.choice(['approve', 'flag_2fa', 'block'])
                reward = 1 if action == 'approve' and true_label == 0 else -1
                result = {
                    'final_decision': action,
                    'rl_action': action,
                    'risk_score': random.random(),
                    'risk_level': 'Medium'
                }

            # Update metrics
            self._update_metrics(result['rl_action'], true_label, reward)

            # Create result object
            transaction_result = {
                'transaction_id': transaction['transaction_id'],
                'amount': round(transaction['amount'], 2),
                'timestamp': transaction['timestamp'],
                'true_label': true_label,
                'prediction': result['final_decision'],
                'risk_score': round(result.get('risk_score', 0), 3),
                'risk_level': result.get('risk_level', 'Unknown'),
                'rl_action': result['rl_action'],
                'reward': reward,
                'anomaly_score': round(result.get('anomaly_score', 0), 3),
                'xgb_probability': round(result.get('xgb_probability', 0), 3)
            }

            self.results.append(transaction_result)
            yield transaction_result

    def _update_metrics(self, action, true_label, reward):
        """Update performance metrics"""
        if action == 'block' and true_label == 1:
            self.metrics['true_positives'] += 1
        elif action in ['approve', 'flag_2fa'] and true_label == 0:
            self.metrics['true_negatives'] += 1
        elif action == 'block' and true_label == 0:
            self.metrics['false_positives'] += 1
        elif action in ['approve', 'flag_2fa'] and true_label == 1:
            self.metrics['false_negatives'] += 1

        self.metrics['total_reward'] += reward
        self.metrics['actions'][action] = self.metrics['actions'].get(action, 0) + 1

    def get_current_metrics(self):
        """Get current performance metrics"""
        TP = self.metrics['true_positives']
        TN = self.metrics['true_negatives']
        FP = self.metrics['false_positives']
        FN = self.metrics['false_negatives']

        fraud_capture = TP / (TP + FN) if (TP + FN) > 0 else 0
        false_decline = FP / (FP + TN) if (FP + TN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        f1 = 2 * (precision * fraud_capture) / (precision + fraud_capture) if (precision + fraud_capture) > 0 else 0

        return {
            'fraud_capture_rate': round(fraud_capture, 4),
            'false_decline_rate': round(false_decline, 4),
            'precision': round(precision, 4),
            'accuracy': round(accuracy, 4),
            'f1_score': round(f1, 4),
            'total_reward': round(self.metrics['total_reward'], 2),
            'true_positives': TP,
            'false_positives': FP,
            'false_negatives': FN,
            'true_negatives': TN,
            'total_transactions': len(self.results),
            'action_distribution': self.metrics['actions']
        }

    def get_confusion_matrix(self):
        """Get confusion matrix"""
        return np.array([
            [self.metrics['true_positives'], self.metrics['false_negatives']],
            [self.metrics['false_positives'], self.metrics['true_negatives']]
        ])
