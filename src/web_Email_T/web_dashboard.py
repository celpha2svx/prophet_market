from flask import Flask, render_template_string, jsonify, request
import pandas as pd
import json
from datetime import datetime
import os
from src.utils.paths import RECOMMEND_JSON,RECOMMEND_CSV,STOCK_PREDICTABILY

app = Flask(__name__)

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Stock Prediction Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>

  <style>
    :root{
      --bg1: #0f172a;
      --accent1: #667eea;
      --accent2: #764ba2;
      --glass: rgba(255,255,255,0.06);
      --muted: #94a3b8;
      --card-shadow: 0 10px 30px rgba(2,6,23,0.6);
    }
    *{box-sizing:border-box}
    html,body,#root{height:100%}
    body{
      margin:0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
      background: radial-gradient(1200px 600px at 10% 15%, rgba(118,75,162,0.18), transparent),
                  radial-gradient(1000px 500px at 90% 85%, rgba(102,126,234,0.18), transparent),
                  linear-gradient(180deg, var(--bg1) 0%, #071028 100%);
      color: #e6eef8;
      padding:20px;
    }
    .container{max-width:1280px;margin:0 auto}
    header.app-header{
      background: linear-gradient(90deg, rgba(255,255,255,0.03), rgba(255,255,255,0.02));
      padding:18px 22px;border-radius:14px;box-shadow:var(--card-shadow);display:flex;gap:18px;align-items:center;justify-content:space-between;margin-bottom:22px;
      backdrop-filter: blur(6px);
    }
    .brand{display:flex;gap:12px;align-items:center}
    .logo {
      width:56px;height:56px;border-radius:12px;background:linear-gradient(135deg,var(--accent1),var(--accent2));
      display:flex;align-items:center;justify-content:center;font-weight:700;font-size:20px;color:white;box-shadow:0 8px 30px rgba(100,78,220,0.25)
    }
    .headline h1{font-size:20px;margin:0;color:#fff}
    .headline p{margin:0;color:var(--muted);font-size:13px}
    .header-actions{display:flex;gap:10px;align-items:center}
    .btn {
      background:transparent;border:1px solid rgba(255,255,255,0.08);padding:8px 12px;border-radius:10px;color:inherit;font-weight:600;cursor:pointer;
    }
    .btn.primary{background:linear-gradient(90deg,var(--accent1),var(--accent2));border:none;color:white;box-shadow:0 6px 18px rgba(102,126,234,0.18)}
    .grid {display:grid;grid-template-columns:repeat(12,1fr);gap:18px}
    .stats {grid-column:span 12;display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px}
    .card {
      background:var(--glass);padding:18px;border-radius:12px;box-shadow:0 8px 20px rgba(2,6,23,0.6);border:1px solid rgba(255,255,255,0.03)
    }
    .stat-label{font-size:12px;color:#c7d2fe;text-transform:uppercase;letter-spacing:0.6px}
    .stat-value{font-size:22px;font-weight:700;margin-top:8px;color:#fff}
    .stat-sub{font-size:12px;color:var(--muted);margin-top:6px}
    /* Table section */
    .section {grid-column:span 12;padding:0}
    .section .section-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px}
    .section h2{margin:0;color:#f1f5f9}
    .table-wrap{overflow:auto;border-radius:10px}
    table{width:100%;border-collapse:collapse;background:transparent}
    thead th{position:sticky;top:0;background:linear-gradient(90deg,var(--accent1),var(--accent2));color:white;padding:12px;text-align:left;font-weight:600}
    tbody td{padding:12px;border-bottom:1px dashed rgba(255,255,255,0.03);color:#e6eef8;font-size:14px}
    tbody tr:hover{background:linear-gradient(90deg, rgba(255,255,255,0.02), transparent)}
    .badge{padding:6px 10px;border-radius:999px;font-weight:700;font-size:12px;display:inline-block}
    .badge.buy{background:linear-gradient(90deg,#22c55e,#16a34a);color:#072014}
    .badge.sell{background:linear-gradient(90deg,#fb7185,#dc2626);color:#200005}
    .confidence-high{background:#a7f3d0;color:#064e3b;padding:6px 10px;border-radius:10px;font-weight:700}
    .confidence-med{background:#fde68a;color:#42340a;padding:6px 10px;border-radius:10px;font-weight:700}
    .confidence-low{background:#cbd5e1;color:#0f172a;padding:6px 10px;border-radius:10px;font-weight:700}

    /* Charts */
    .charts {display:grid;grid-template-columns:2fr 1fr;gap:16px}
    .chart-card{padding:12px;border-radius:10px;background:var(--glass)}
    .chart-container{height:360px}
    @media (max-width:900px){
      .charts{grid-template-columns:1fr}
      .stat-value{font-size:18px}
      .headline h1{font-size:16px}
    }

    /* Subscribe form */
    .subscribe-row{display:flex;gap:10px;align-items:center}
    input[type="email"]{padding:10px;border-radius:10px;border:1px solid rgba(255,255,255,0.06);background:transparent;color:inherit;min-width:220px}
    .toast{position:fixed;right:20px;bottom:20px;padding:12px 16px;border-radius:8px;background:#0b1220;color:#d1fae5;box-shadow:0 10px 30px rgba(2,6,23,0.4);display:none}
    .toast.show{display:block}

    /* tiny helper */
    .muted{color:var(--muted);font-size:13px}
    .small{font-size:12px;color:var(--muted)}
    .mono{font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, monospace}
  </style>
</head>
<body>
  <div class="container" id="root">
    <header class="app-header" role="banner" aria-label="Application header">
      <div class="brand">
        <div class="logo">SP</div>
        <div class="headline">
          <h1>Stock Prediction System</h1>
          <p class="small">AI signals · Portfolio insights · News sentiment</p>
        </div>
      </div>

      <div class="header-actions" role="navigation" aria-label="Top actions">
        <div class="small mono muted">Updated: <span id="updateTime">{{ update_time }}</span></div>
        <button class="btn" onclick="location.href='/signin'">Sign In</button>
        <button class="btn primary" id="refreshBtn">Refresh</button>
      </div>
    </header>

    <!-- Stats -->
    <div class="grid" style="margin-top:14px;">
      <div class="stats card" aria-live="polite" >
        <div class="card stat-card">
          <div class="stat-label">Total Positions</div>
          <div class="stat-value" id="totalPositions">0</div>
          <div class="stat-sub">Active recommendations</div>
        </div>
        <div class="card stat-card">
          <div class="stat-label">Capital Allocated</div>
          <div class="stat-value" id="capitalAllocated">$0</div>
          <div class="stat-sub" id="capPct">0% of portfolio</div>
        </div>
        <div class="card stat-card">
          <div class="stat-label">Cash Reserve</div>
          <div class="stat-value" id="cashReserve">$0</div>
          <div class="stat-sub" id="cashPct">0% available</div>
        </div>
        <div class="card stat-card">
          <div class="stat-label">Total Risk</div>
          <div class="stat-value" id="totalRisk">0%</div>
          <div class="stat-sub" id="riskVal">$0 at risk</div>
        </div>
        <div class="card stat-card">
          <div class="stat-label">Avg Confidence</div>
          <div class="stat-value" id="avgConfidence">0%</div>
          <div class="stat-sub small">System confidence level</div>
        </div>
        <div class="card stat-card">
          <div class="stat-label">Avg Accuracy</div>
          <div class="stat-value" id="avgAccuracy">0%</div>
          <div class="stat-sub small">Prophet directional accuracy</div>
        </div>
      </div>

      <!-- Recommendations -->
      <div class="section" style="margin-top:18px;">
        <div class="card">
          <div class="section-header">
            <h2>🎯 Today's Recommendations</h2>
            <div style="display:flex;gap:10px;align-items:center">
              <div class="small muted">Subscribe to signals:</div>
              <div class="subscribe-row">
                <input id="subscribeEmail" type="email" placeholder="you@domain.com" aria-label="subscribe email">
                <button class="btn primary" id="subscribeBtn">Subscribe</button>
              </div>
            </div>
          </div>

          <div class="table-wrap" id="tableWrap">
            {% if recommendations|length > 0 %}
            <table aria-describedby="recommendationsTable">
              <thead>
                <tr>
                  <th>Symbol</th>
                  <th>Action</th>
                  <th>Shares</th>
                  <th>Entry</th>
                  <th>Target</th>
                  <th>Stop</th>
                  <th>Position</th>
                  <th>Risk</th>
                  <th>Confidence</th>
                  <th>Accuracy</th>
                </tr>
              </thead>
              <tbody id="recBody">
                {% for rec in recommendations %}
                <tr>
                  <td><strong>{{ rec.symbol }}</strong><div class="small muted">{{ rec.sector if rec.sector is defined else '' }}</div></td>
                  <td><span class="badge {{ 'buy' if rec.action == 'BUY' else 'sell' }}">{{ rec.action }}</span></td>
                  <td>{{ rec.shares }}</td>
                  <td class="mono">${{ \"%.2f\"|format(rec.current_price) }}</td>
                  <td>${{ \"%.2f\"|format(rec.target_price) }}</td>
                  <td>${{ \"%.2f\"|format(rec.stop_loss) }}</td>
                  <td>${{ \"%.0f\"|format(rec.position_value) }}</td>
                  <td>{{ \"%.2f\"|format(rec.risk_pct) }}%</td>
                  <td>
                    {% set label = rec.confidence_label %}
                    {% if 'High' in label %}
                      <span class="confidence-high">{{ label }}</span>
                    {% elif 'Medium' in label %}
                      <span class="confidence-med">{{ label }}</span>
                    {% else %}
                      <span class="confidence-low">{{ label }}</span>
                    {% endif %}
                  </td>
                  <td>{{ \"%.1f\"|format(rec.prophet_accuracy) }}%</td>
                </tr>
                {% endfor %}
              </tbody>
            </table>
            {% else %}
            <div style="padding:40px;text-align:center;color:var(--muted)">
              <div style="font-size:18px;margin-bottom:8px">No recommendations today</div>
              <div class="small">Either your filters are strict or the model found no qualified signals.</div>
            </div>
            {% endif %}
          </div>
        </div>
      </div>

      <!-- Charts -->
      <div class="charts" style="grid-column:span 12;margin-top:16px;">
        <div class="chart-card card">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
            <h3 style="margin:0">📈 Prophet Model Accuracy</h3>
            <div class="small muted">Directional accuracy by ticker</div>
          </div>
          <div id="accuracyChart" class="chart-container"></div>
        </div>

        <div class="chart-card card">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
            <h3 style="margin:0">Allocation</h3>
            <div class="small muted">Current position allocation</div>
          </div>
          <div id="allocationChart" class="chart-container"></div>
        </div>
      </div>

      <!-- Disclaimer & footer -->
      <div style="grid-column:span 12;margin-top:18px">
        <div class="card" style="display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap">
          <div>
            <strong style="color:#fde68a">⚠ Important — Educational only</strong>
            <div class="small muted" style="margin-top:6px;max-width:80ch">
              This dashboard is an educational tool and demonstration of ML-based signals. This is not financial advice. Always do your own research and consult licensed professionals before trading. Past performance doesn't guarantee future returns.
            </div>
          </div>
          <div class="small muted">© 2025 Stock Prediction System</div>
        </div>
      </div>
    </div>
  </div>

  <div id="toast" class="toast" role="status" aria-live="polite"></div>

  <script>
    // Utility: show toast
    function showToast(msg, time=3500){
      const t = document.getElementById('toast');
      t.textContent = msg;
      t.classList.add('show');
      setTimeout(()=> t.classList.remove('show'), time);
    }

    // Animate numeric counters (simple)
    function animateValue(id, start, end, duration=800){
      const obj = document.getElementById(id);
      if(!obj) return;
      const range = end - start;
      let startTime = null;
      function step(timestamp){
        if (!startTime) startTime = timestamp;
        const progress = Math.min((timestamp - startTime) / duration, 1);
        const value = start + Math.round(range * progress);
        obj.textContent = value.toLocaleString();
        if(progress < 1) requestAnimationFrame(step);
      }
      requestAnimationFrame(step);
    }

    // Load initial summary values from server-rendered JSON (fallback replaced by AJAX on refresh)
    const initialSummary = {{ summary|tojson }};
    function setSummary(vals){
      animateValue('totalPositions', 0, vals.total_positions || 0);
      document.getElementById('capitalAllocated').textContent = '$' + (Math.round((vals.total_allocated || 0))).toLocaleString();
      document.getElementById('capPct').textContent = (vals.total_allocated_pct || 0).toFixed(1) + '% of portfolio';
      document.getElementById('cashReserve').textContent = '$' + Math.round((vals.cash_remaining || 0)).toLocaleString();
      document.getElementById('cashPct').textContent = (vals.cash_remaining_pct || 0).toFixed(1) + '% available';
      document.getElementById('totalRisk').textContent = (vals.total_risk_pct || 0).toFixed(2) + '%';
      document.getElementById('riskVal').textContent = '$' + Math.round((vals.total_risk || 0)).toLocaleString() + ' at risk';
      document.getElementById('avgConfidence').textContent = Math.round((vals.avg_confidence || 0) * 100) + '%';
      document.getElementById('avgAccuracy').textContent = (vals.avg_prophet_accuracy || 0).toFixed(1) + '%';
    }
    setSummary(initialSummary);

    // Plotly charts (render from server data initially)
    const accuracyData = {{ accuracy_data|tojson }};
    const allocationData = {{ allocation_data|tojson }};

    function renderAccuracyChart(data){
      const trace = {
        x: data.symbols || [],
        y: data.accuracies || [],
        type: 'bar',
        marker: {
          color: data.accuracies.map(v => v >= 60 ? '#16a34a' : v >= 45 ? '#f59e0b' : '#ef4444'),
        },
        hovertemplate: '%{x}: %{y:.1f}%<extra></extra>'
      };
      const layout = {
        margin:{l:40,r:10,t:40,b:60},
        yaxis:{title:'Directional accuracy (%)', range:[0,100]},
        xaxis:{tickangle:-45},
        height:360,
      };
      Plotly.newPlot('accuracyChart',[trace],layout,{responsive:true,displayModeBar:false});
    }

    function renderAllocationChart(data){
      const trace = {
        labels: data.symbols || [],
        values: data.allocations || [],
        type: 'pie',
        textinfo: 'label+percent',
        hoverinfo: 'label+value+percent'
      };
      const layout = {height:360,margin:{t:30,b:30}};
      Plotly.newPlot('allocationChart',[trace],layout,{responsive:true,displayModeBar:false});
    }

    renderAccuracyChart(accuracyData);
    renderAllocationChart(allocationData);

    // Refresh logic: fetch /api/summary and /api/recommendations every minute, manual refresh button
    const refreshBtn = document.getElementById('refreshBtn');
    async function refreshAll(showToastOnSuccess=true){
      try {
        const sRes = await fetch('/api/summary');
        if(sRes.ok){
          const s = await sRes.json();
          setSummary(s);
          const now = new Date().toLocaleString();
          document.getElementById('updateTime').textContent = now;
        }
        const rRes = await fetch('/api/recommendations');
        if(rRes.ok){
          const recs = await rRes.json();
          // update table
          const tbody = document.getElementById('recBody');
          if(tbody){
            tbody.innerHTML = '';
            recs.forEach(rec=>{
              const tr = document.createElement('tr');
              tr.innerHTML = `
                <td><strong>${rec.symbol}</strong><div class="small muted">${rec.sector || ''}</div></td>
                <td><span class="badge ${rec.action === 'BUY' ? 'buy' : 'sell'}">${rec.action}</span></td>
                <td>${rec.shares}</td>
                <td class="mono">$${(+rec.current_price).toFixed(2)}</td>
                <td>$${(+rec.target_price).toFixed(2)}</td>
                <td>$${(+rec.stop_loss).toFixed(2)}</td>
                <td>$${Math.round(rec.position_value || 0)}</td>
                <td>${(+rec.risk_pct).toFixed(2)}%</td>
                <td>${rec.confidence_label.includes('High') ? '<span class=\"confidence-high\">'+rec.confidence_label+'</span>' : rec.confidence_label.includes('Medium') ? '<span class=\"confidence-med\">'+rec.confidence_label+'</span>' : '<span class=\"confidence-low\">'+rec.confidence_label+'</span>'}</td>
                <td>${(+rec.prophet_accuracy).toFixed(1)}%</td>
              `;
              tbody.appendChild(tr);
            });
          }

          // update charts if leaderboard file exists
          const lbRes = await fetch('/filess_csv/stock_predictability_leaderboard.csv');
          // can't fetch CSV directly from static file reliably in this simple setup; instead call server endpoints if you add them
        }
        if(showToastOnSuccess) showToast('Dashboard refreshed');
      } catch(err){
        console.error('refresh error', err);
        showToast('Failed to refresh (server offline?)', 4000);
      }
    }

    refreshBtn.addEventListener('click', ()=> refreshAll(true));
    // periodic refresh every 60 seconds
    setInterval(()=> refreshAll(false), 60*1000);

    // Subscribe flow (simple)
    document.getElementById('subscribeBtn').addEventListener('click', async ()=>{
      const email = document.getElementById('subscribeEmail').value.trim();
      if(!email || !email.includes('@')) { showToast('Enter a valid email'); return; }
      try {
        const res = await fetch('/api/subscribe', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({email}) });
        const payload = await res.json();
        if(res.ok) {
          showToast(payload.message || 'Subscribed');
          document.getElementById('subscribeEmail').value = '';
        } else {
          showToast(payload.error || 'Subscription failed');
        }
      } catch(e){
        showToast('Subscription failed');
      }
    });

  </script>
</body>
</html>
"""

# Sign-in placeholder page (simple)
SIGNIN_HTML = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>Sign in</title>
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <style>
      body{display:flex;align-items:center;justify-content:center;height:100vh;margin:0;background:linear-gradient(180deg,#071028 0%, #0b1220 100%);color:#e6eef8;font-family:Inter,system-ui,Segoe UI,Roboto}
      .card{background:rgba(255,255,255,0.03);padding:26px;border-radius:12px;max-width:420px;width:90%;box-shadow:0 12px 40px rgba(2,6,23,0.6)}
      input{width:100%;padding:10px;border-radius:8px;border:1px solid rgba(255,255,255,0.06);margin-top:10px;background:transparent;color:inherit}
      .btn{margin-top:12px;padding:10px 12px;border-radius:8px;border:none;background:linear-gradient(90deg,#667eea,#764ba2);color:white;font-weight:700;cursor:pointer;width:100%}
      a{color:#9fb7ff;font-size:13px}
    </style>
  </head>
  <body>
    <div class="card" role="main">
      <h2 style="margin:0">Sign in (placeholder)</h2>
      <p class="small" style="color:#94a3b8;margin-top:6px">This page is a frontend placeholder. Hook to your auth endpoint /api/auth/login.</p>
      <form onsubmit="event.preventDefault();alert('Plug your auth here')">
        <input placeholder="you@domain.com" type="email" required>
        <input placeholder="password" type="password" required>
        <button class="btn" type="submit">Sign in</button>
      </form>
      <div style="margin-top:12px;display:flex;justify-content:space-between">
        <a href="/">Back to dashboard</a>
        <a href="/signup">Create account</a>
      </div>
    </div>
  </body>
</html>
"""

# Routes
@app.route('/')
def dashboard():
    try:
        # Load recommendations
        if os.path.exists(RECOMMEND_CSV):
            recommendations = pd.read_csv(RECOMMEND_CSV)
            # ensure keys exist and some fields have defaults
            recs = recommendations.fillna('').to_dict('records')
        else:
            recs = []

        # Load summary
        if os.path.exists(RECOMMEND_JSON):
            with open(RECOMMEND_JSON, 'r') as f:
                summary = json.load(f)
        else:
            summary = {
                'total_positions': 0,
                'total_allocated': 0,
                'total_allocated_pct': 0,
                'cash_remaining': 10000,
                'cash_remaining_pct': 100,
                'total_risk': 0,
                'total_risk_pct': 0,
                'avg_confidence': 0,
                'avg_prophet_accuracy': 0
            }

        # Load leaderboard for charts
        if os.path.exists(STOCK_PREDICTABILY):
            leaderboard = pd.read_csv(STOCK_PREDICTABILY)
            accuracy_data = {
                'symbols': leaderboard['symbol'].astype(str).tolist(),
                'accuracies': leaderboard['directional_accuracy'].fillna(0).astype(float).tolist()
            }
        else:
            accuracy_data = {'symbols': [], 'accuracies': []}

        # Allocation data (from recommendations)
        if recs:
            rec_df = pd.DataFrame(recs)
            if 'position_value' in rec_df.columns and len(rec_df) > 0:
                allocation_data = {
                    'symbols': rec_df['symbol'].astype(str).tolist(),
                    'allocations': rec_df['position_value'].fillna(0).astype(float).tolist()
                }
            else:
                allocation_data = {'symbols': [], 'allocations': []}
        else:
            allocation_data = {'symbols': [], 'allocations': []}

        return render_template_string(
            DASHBOARD_HTML,
            recommendations=recs,
            summary=summary,
            accuracy_data=accuracy_data,
            allocation_data=allocation_data,
            update_time=datetime.now().strftime("%B %d, %Y at %I:%M %p")
        )

    except Exception as e:
        return f"<h1>Error loading dashboard: {e}</h1><p>Make sure you've run recommendation_engine.py first!</p>"

@app.route('/signin')
def signin():
    return SIGNIN_HTML

@app.route('/api/recommendations')
def api_recommendations():
    try:
        if os.path.exists(RECOMMEND_CSV):
            recommendations = pd.read_csv(RECOMMEND_CSV)
            return jsonify(recommendations.fillna('').to_dict('records'))
        else:
            return jsonify([])
    except Exception as e:
        return jsonify([])

@app.route('/api/summary')
def api_summary():
    try:
        if os.path.exists(RECOMMEND_JSON):
            with open(RECOMMEND_JSON, 'r') as f:
                summary = json.load(f)
            return jsonify(summary)
        else:
            return jsonify({})
    except Exception as e:
        return jsonify({})

@app.route('/api/subscribe', methods=['POST'])
def api_subscribe():
    """Very small file-backed subscription stub for demo."""
    try:
        data = request.get_json() or {}
        email = (data.get('email') or '').strip()
        if not email or '@' not in email:
            return jsonify({'error': 'Invalid email'}), 400
        subs_file = 'subscribers.json'
        subs = []
        if os.path.exists(subs_file):
            with open(subs_file, 'r') as f:
                try:
                    subs = json.load(f)
                except:
                    subs = []
        if email in subs:
            return jsonify({'message': 'Already subscribed'}), 200
        subs.append(email)
        with open(subs_file, 'w') as f:
            json.dump(subs, f, indent=2)
        return jsonify({'message': 'Subscribed — check your inbox for confirmation (demo)'}), 200
    except Exception as e:
        return jsonify({'error': 'Failed to subscribe'}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("🌐 STARTING STOCK PREDICTION DASHBOARD (IMPROVED FRONTEND)")
    print("=" * 60)
    print("\n📊 Dashboard will be available at:")
    print("   http://127.0.0.1:5000")
    print("\n💡 Press Ctrl+C to stop the server")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)