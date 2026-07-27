// static/js/main.js

/* ---------- Shared helpers ---------- */

async function postJSON(url, body) {
    const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body || {})
    });
    return res.json();
}

async function getJSON(url) {
    const res = await fetch(url);
    return res.json();
}

function riskBadgeClass(level) {
    switch (level) {
        case 'Low': return 'bg-success';
        case 'Medium': return 'bg-warning text-dark';
        case 'High': return 'bg-danger';
        case 'Critical': return 'bg-dark';
        default: return 'bg-secondary';
    }
}

/* ---------- Dashboard page ---------- */

let confusionChart = null;
let actionChart = null;

function initDashboardPage() {
    const initBtn = document.getElementById('initBtn');
    if (initBtn) {
        initBtn.addEventListener('click', async () => {
            initBtn.disabled = true;
            initBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Initializing...';
            const result = await postJSON('/api/initialize');
            if (result.status === 'success') {
                initBtn.innerHTML = '<i class="fas fa-check"></i> Initialized';
                initBtn.classList.remove('btn-primary');
                initBtn.classList.add('btn-success');
            } else {
                initBtn.disabled = false;
                initBtn.innerHTML = '<i class="fas fa-power-off"></i> Initialize System';
                alert('Initialization failed: ' + result.message);
            }
        });
    }

    refreshDashboardMetrics();
    setInterval(refreshDashboardMetrics, 3000);
}

async function refreshDashboardMetrics() {
    const metricsResp = await getJSON('/api/get_metrics');
    if (metricsResp.status === 'success') {
        const m = metricsResp.metrics;
        setText('fraudCaptureRate', (m.fraud_capture_rate * 100).toFixed(1) + '%');
        setText('falseDeclineRate', (m.false_decline_rate * 100).toFixed(1) + '%');
        setText('precision', (m.precision * 100).toFixed(1) + '%');
        setText('f1Score', m.f1_score.toFixed(3));

        updateActionChart(m.action_distribution);
    }

    const cmResp = await getJSON('/api/get_confusion_matrix');
    if (cmResp.status === 'success') {
        updateConfusionChart(cmResp.confusion_matrix);
    }

    const qResp = await getJSON('/api/get_q_table');
    if (qResp.status === 'success') {
        updateQTable(qResp.q_table);
    }
}

function setText(id, value) {
    const el = document.getElementById(id);
    if (el) el.textContent = value;
}

function updateConfusionChart(cm) {
    const ctx = document.getElementById('confusionMatrixChart');
    if (!ctx) return;

    const data = {
        labels: ['True Positive', 'False Negative', 'False Positive', 'True Negative'],
        datasets: [{
            label: 'Count',
            data: [cm[0][0], cm[0][1], cm[1][0], cm[1][1]],
            backgroundColor: ['#28a745', '#dc3545', '#ffc107', '#4f8cff']
        }]
    };

    if (confusionChart) {
        confusionChart.data = data;
        confusionChart.update();
    } else {
        confusionChart = new Chart(ctx, {
            type: 'bar',
            data,
            options: {
                plugins: { legend: { display: false } },
                scales: {
                    x: { ticks: { color: '#9aa4b8' } },
                    y: { ticks: { color: '#9aa4b8' } }
                }
            }
        });
    }
}

function updateActionChart(actions) {
    const ctx = document.getElementById('actionDistributionChart');
    if (!ctx) return;

    const labels = Object.keys(actions);
    const values = Object.values(actions);

    const data = {
        labels,
        datasets: [{
            data: values,
            backgroundColor: ['#28a745', '#ffc107', '#dc3545']
        }]
    };

    if (actionChart) {
        actionChart.data = data;
        actionChart.update();
    } else {
        actionChart = new Chart(ctx, {
            type: 'doughnut',
            data,
            options: {
                plugins: { legend: { labels: { color: '#e6e6e6' } } }
            }
        });
    }
}

function updateQTable(qTable) {
    const tbody = document.querySelector('#qTableBody tbody');
    if (!tbody) return;

    const states = {};
    Object.entries(qTable).forEach(([key, value]) => {
        const parts = key.split('_');
        const action = parts.pop();
        const state = parts.join('_');
        if (!states[state]) states[state] = {};
        states[state][action] = value;
    });

    tbody.innerHTML = '';
    Object.entries(states).forEach(([state, actions]) => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${state}</td>
            <td>${(actions.approve ?? 0).toFixed(2)}</td>
            <td>${(actions.flag_2fa ?? 0).toFixed(2)}</td>
            <td>${(actions.block ?? 0).toFixed(2)}</td>
        `;
        tbody.appendChild(row);
    });
}

/* ---------- Simulation page ---------- */

function initSimulationPage() {
    const socket = io();
    const feedBody = document.getElementById('liveFeedBody');
    const statusBadge = document.getElementById('simStatus');

    let counts = { processed: 0, approve: 0, flag_2fa: 0, block: 0 };

    socket.on('connect', () => {
        statusBadge.textContent = 'Connected';
        statusBadge.className = 'badge bg-info w-100 py-2';
    });

    socket.on('transaction_update', (txn) => {
        counts.processed += 1;
        if (txn.rl_action === 'approve') counts.approve += 1;
        else if (txn.rl_action === 'flag_2fa') counts.flag_2fa += 1;
        else if (txn.rl_action === 'block') counts.block += 1;

        setText('liveProcessed', counts.processed);
        setText('liveApproved', counts.approve);
        setText('liveFlagged', counts.flag_2fa);
        setText('liveBlocked', counts.block);

        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${txn.transaction_id}</td>
            <td>$${txn.amount.toFixed(2)}</td>
            <td><span class="badge ${riskBadgeClass(txn.risk_level)}">${txn.risk_level}</span></td>
            <td>${txn.risk_score.toFixed(3)}</td>
            <td>${txn.prediction}</td>
            <td>${txn.true_label === 1 ? 'Fraud' : 'Legit'}</td>
        `;
        feedBody.prepend(row);

        // Keep the feed to a reasonable length
        while (feedBody.rows.length > 200) {
            feedBody.deleteRow(feedBody.rows.length - 1);
        }
    });

    socket.on('simulation_complete', (data) => {
        statusBadge.textContent = 'Complete';
        statusBadge.className = 'badge bg-success w-100 py-2';
    });

    socket.on('error', (data) => {
        alert('Simulation error: ' + data.message);
    });

    document.getElementById('startSimBtn').addEventListener('click', async () => {
        counts = { processed: 0, approve: 0, flag_2fa: 0, block: 0 };
        feedBody.innerHTML = '';
        setText('liveProcessed', 0);
        setText('liveApproved', 0);
        setText('liveFlagged', 0);
        setText('liveBlocked', 0);

        const nTransactions = parseInt(document.getElementById('nTransactions').value, 10);
        const fraudRate = parseFloat(document.getElementById('fraudRate').value);

        statusBadge.textContent = 'Running...';
        statusBadge.className = 'badge bg-warning text-dark w-100 py-2';

        const result = await postJSON('/api/simulate', {
            n_transactions: nTransactions,
            fraud_rate: fraudRate
        });

        if (result.status !== 'success') {
            alert(result.message);
            statusBadge.textContent = 'Idle';
            statusBadge.className = 'badge bg-secondary w-100 py-2';
        }
    });

    document.getElementById('stopSimBtn').addEventListener('click', async () => {
        await postJSON('/api/stop_simulation');
        statusBadge.textContent = 'Stopped';
        statusBadge.className = 'badge bg-secondary w-100 py-2';
    });
}
