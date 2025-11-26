from flask import Flask, render_template, jsonify, request
import pandas as pd
import json
from datetime import datetime
import os
import logging
from functools import lru_cache
from src.utils.paths import RECOMMEND_JSON, RECOMMEND_CSV, STOCK_PREDICTABILY

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)


# ============================================
# CONFIGURATION
# ============================================
class Config:
    PORTFOLIO_SIZE = 10000  # Default portfolio size
    REFRESH_INTERVAL = 60  # seconds
    SUBSCRIBERS_FILE = 'subscribers.json'
    DEBUG = True
    HOST = '0.0.0.0'
    PORT = 5000


config = Config()


# ============================================
# DATA LOADING WITH CACHING
# ============================================

@lru_cache(maxsize=32)
def get_file_mtime(filepath):
    """Get file modification time for cache invalidation"""
    try:
        return os.path.getmtime(filepath)
    except:
        return 0


def load_recommendations():
    """Load recommendations with error handling"""
    try:
        if not os.path.exists(RECOMMEND_CSV):
            logger.warning(f"Recommendations file not found: {RECOMMEND_CSV}")
            return []

        get_file_mtime(RECOMMEND_CSV)  # For cache invalidation
        df = pd.read_csv(RECOMMEND_CSV)
        return df.fillna('').to_dict('records')
    except Exception as e:
        logger.error(f"Error loading recommendations: {e}")
        return []


def load_summary():
    """Load summary statistics with defaults"""
    default_summary = {
        'total_positions': 0,
        'total_allocated': 0,
        'total_allocated_pct': 0,
        'cash_remaining': config.PORTFOLIO_SIZE,
        'cash_remaining_pct': 100,
        'total_risk': 0,
        'total_risk_pct': 0,
        'avg_confidence': 0,
        'avg_prophet_accuracy': 0
    }

    try:
        if not os.path.exists(RECOMMEND_JSON):
            logger.warning(f"Summary file not found: {RECOMMEND_JSON}")
            return default_summary

        get_file_mtime(RECOMMEND_JSON)
        with open(RECOMMEND_JSON, 'r') as f:
            summary = json.load(f)
        return {**default_summary, **summary}
    except Exception as e:
        logger.error(f"Error loading summary: {e}")
        return default_summary


def load_leaderboard():
    """Load stock predictability leaderboard"""
    try:
        if not os.path.exists(STOCK_PREDICTABILY):
            logger.warning(f"Leaderboard file not found: {STOCK_PREDICTABILY}")
            return pd.DataFrame()

        get_file_mtime(STOCK_PREDICTABILY)
        return pd.read_csv(STOCK_PREDICTABILY)
    except Exception as e:
        logger.error(f"Error loading leaderboard: {e}")
        return pd.DataFrame()


def get_accuracy_data():
    """Get accuracy data for charts"""
    leaderboard = load_leaderboard()
    if leaderboard.empty:
        return {'symbols': [], 'accuracies': []}

    return {
        'symbols': leaderboard['symbol'].astype(str).tolist(),
        'accuracies': leaderboard['directional_accuracy'].fillna(0).astype(float).tolist()
    }


def get_allocation_data():
    """Get allocation data for pie chart"""
    recs = load_recommendations()
    if not recs:
        return {'symbols': [], 'allocations': []}

    df = pd.DataFrame(recs)
    if 'position_value' not in df.columns:
        return {'symbols': [], 'allocations': []}

    return {
        'symbols': df['symbol'].astype(str).tolist(),
        'allocations': df['position_value'].fillna(0).astype(float).tolist()
    }


# ============================================
# ROUTES
# ============================================

@app.route('/')
def dashboard():
    """Main dashboard page"""
    try:
        return render_template(
            'dashboard.html',
            recommendations=load_recommendations(),
            summary=load_summary(),
            accuracy_data=get_accuracy_data(),
            allocation_data=get_allocation_data(),
            update_time=datetime.now().strftime("%B %d, %Y at %I:%M %p"),
            refresh_interval=config.REFRESH_INTERVAL
        )
    except Exception as e:
        logger.error(f"Error rendering dashboard: {e}")
        return render_template('error.html', error=str(e)), 500


@app.route('/signin')
def signin():
    """Sign-in placeholder page"""
    return render_template('signin.html')


# ============================================
# API ENDPOINTS
# ============================================

@app.route('/api/health')
def api_health():
    """Health check endpoint"""
    status = {
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'files': {
            'recommendations': os.path.exists(RECOMMEND_CSV),
            'summary': os.path.exists(RECOMMEND_JSON),
            'leaderboard': os.path.exists(STOCK_PREDICTABILY)
        }
    }
    return jsonify(status)


@app.route('/api/dashboard')
def api_dashboard():
    """Consolidated dashboard data endpoint"""
    try:
        data = {
            'recommendations': load_recommendations(),
            'summary': load_summary(),
            'accuracy_data': get_accuracy_data(),
            'allocation_data': get_allocation_data(),
            'timestamp': datetime.now().isoformat()
        }
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error in dashboard API: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/recommendations')
def api_recommendations():
    """Get recommendations only"""
    return jsonify(load_recommendations())


@app.route('/api/summary')
def api_summary():
    """Get summary statistics only"""
    return jsonify(load_summary())


@app.route('/api/subscribe', methods=['POST'])
def api_subscribe():
    """
    Email subscription endpoint - Now sends confirmation emails!
    """
    try:
        data = request.get_json() or {}
        email = (data.get('email') or '').strip().lower()

        # Validate email
        if not email or '@' not in email or '.' not in email:
            return jsonify({'error': 'Invalid email address'}), 400

        # Load existing subscribers
        subscribers = []
        if os.path.exists(config.SUBSCRIBERS_FILE):
            try:
                with open(config.SUBSCRIBERS_FILE, 'r') as f:
                    subscribers = json.load(f)
            except:
                subscribers = []

        # Check if already subscribed
        if email in subscribers:
            return jsonify({'message': 'You are already subscribed!'}), 200

        # Add new subscriber
        subscribers.append(email)

        # Save to file
        with open(config.SUBSCRIBERS_FILE, 'w') as f:
            json.dump(subscribers, f, indent=2)

        logger.info(f"New subscriber: {email}")

        # Send confirmation email using your EmailReporter!
        try:
            from email_sender import send_confirmation_email
            email_sent = send_confirmation_email(email)

            if email_sent:
                return jsonify({
                    'message': 'Successfully subscribed! Check your email for confirmation.',
                    'email': email
                }), 200
            else:
                # Still subscribed but email failed
                return jsonify({
                    'message': 'Subscribed! You will receive daily signals (confirmation email pending).',
                    'email': email
                }), 200
        except ImportError:
            logger.warning("email_sender module not found. Subscription saved but no confirmation sent.")
            return jsonify({
                'message': 'Successfully subscribed! You will receive daily signals.',
                'email': email
            }), 200

    except Exception as e:
        logger.error(f"Subscription error: {e}")
        return jsonify({'error': 'Failed to subscribe. Please try again.'}), 500


@app.route('/api/subscribers', methods=['GET'])
def api_get_subscribers():
    """
    Get list of subscribers (for your email engine)
    Consider adding authentication to this endpoint
    """
    try:
        if not os.path.exists(config.SUBSCRIBERS_FILE):
            return jsonify([])

        with open(config.SUBSCRIBERS_FILE, 'r') as f:
            subscribers = json.load(f)

        return jsonify(subscribers)
    except Exception as e:
        logger.error(f"Error getting subscribers: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================
# ERROR HANDLERS
# ============================================

@app.errorhandler(404)
def not_found(e):
    return render_template('error.html',
                           error='Page not found',
                           error_code=404), 404


@app.errorhandler(500)
def server_error(e):
    return render_template('error.html',
                           error='Internal server error',
                           error_code=500), 500


# ============================================
# MAIN
# ============================================

if __name__ == '__main__':
    print("=" * 60)
    print("🌐 STOCK PREDICTION DASHBOARD")
    print("=" * 60)
    print(f"\n📊 Dashboard: http://127.0.0.1:{config.PORT}")
    print(f"🔄 Auto-refresh: Every {config.REFRESH_INTERVAL}s")
    print(f"📧 Subscribers file: {config.SUBSCRIBERS_FILE}")
    print("\n💡 Press Ctrl+C to stop")
    print("=" * 60)

    app.run(
        debug=config.DEBUG,
        host=config.HOST,
        port=config.PORT
    )