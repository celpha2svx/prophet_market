"""
Email Integration Module - Now using your existing EmailReporter!

This module connects your EmailReporter class with the dashboard's
subscription system to send confirmation and daily signal emails.
"""

import logging
import json
import os
from datetime import datetime
from typing import List, Dict
import pandas as pd
from dotenv import load_dotenv
from src.utils.paths import RECOMMEND_JSON, RECOMMEND_CSV, STOCK_PREDICTABILY

# Import your existing EmailReporter
from email_engine import EmailReporter

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# ============================================
# CONFIGURATION
# ============================================

class EmailConfig:
    """Email configuration using your existing setup"""
    SENDER_EMAIL = os.getenv('EMAIL_ADDRESS')
    SENDER_PASSWORD = os.getenv('EMAIL_PASSWORD')
    SMTP_SERVER = "smtp.gmail.com"
    SMTP_PORT = 587



    # Subscribers file
    SUBSCRIBERS_FILE = 'subscribers.json'


    # Email subjects
    CONFIRMATION_SUBJECT = "✅ Welcome to Stock Prediction Signals!"
    DAILY_SIGNALS_SUBJECT = "📊 Daily Stock Recommendations - {date}"

# Initialize EmailReporter with your credentials
email_reporter = EmailReporter(
    smtp_server=EmailConfig.SMTP_SERVER,
    smtp_port=EmailConfig.SMTP_PORT,
    sender_email=EmailConfig.SENDER_EMAIL,
    sender_password=EmailConfig.SENDER_PASSWORD,

)

# ============================================
# CONFIRMATION EMAIL TEMPLATE
# ============================================

def get_confirmation_email_html(subscriber_email: str) -> str:
    """
    HTML template for subscription confirmation
    Matches the golden/black theme of your dashboard
    """
    return f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0a0a;
            color: #ffffff;
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 600px;
            margin: 0 auto;
            background: #121212;
            border: 1px solid rgba(212, 175, 55, 0.2);
            border-radius: 12px;
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #d4af37, #b8941f);
            padding: 40px 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            color: #0a0a0a;
            font-size: 28px;
            font-weight: 800;
        }}
        .header p {{
            margin: 10px 0 0 0;
            color: #0a0a0a;
            font-size: 14px;
            opacity: 0.8;
        }}
        .content {{
            padding: 40px 30px;
            line-height: 1.8;
        }}
        .content h2 {{
            color: #d4af37;
            margin-top: 0;
            font-size: 22px;
        }}
        .features {{
            background: #0a0a0a;
            padding: 20px;
            border-radius: 8px;
            margin: 25px 0;
        }}
        .features ul {{
            list-style: none;
            padding: 0;
            margin: 0;
        }}
        .features li {{
            padding: 10px 0;
            border-bottom: 1px solid rgba(212, 175, 55, 0.1);
        }}
        .features li:last-child {{
            border-bottom: none;
        }}
        .features li::before {{
            content: "✓";
            color: #d4af37;
            font-weight: bold;
            margin-right: 10px;
        }}
        .button {{
            display: inline-block;
            padding: 14px 30px;
            background: linear-gradient(135deg, #d4af37, #b8941f);
            color: #0a0a0a;
            text-decoration: none;
            border-radius: 8px;
            font-weight: bold;
            margin: 25px 0;
        }}
        .disclaimer {{
            background: rgba(212, 175, 55, 0.1);
            border-left: 3px solid #d4af37;
            padding: 15px;
            margin: 25px 0;
            font-size: 13px;
            color: #a0a0a0;
        }}
        .footer {{
            padding: 25px 30px;
            background: #0a0a0a;
            text-align: center;
            font-size: 12px;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎉 Welcome to Stock Prediction Signals</h1>
            <p>Your subscription is confirmed</p>
        </div>
        
        <div class="content">
            <h2>Thank You for Subscribing!</h2>
            
            <p>You're now part of our exclusive community receiving AI-powered trading signals directly to your inbox.</p>
            
            <div class="features">
                <ul>
                    <li><strong>📊 Daily Market Recommendations</strong> - Carefully analyzed stock picks</li>
                    <li><strong>🎯 Entry & Exit Levels</strong> - Clear target and stop-loss prices</li>
                    <li><strong>📈 Prophet ML Accuracy</strong> - Data-driven confidence metrics</li>
                    <li><strong>💰 Risk-Managed Positions</strong> - Smart position sizing included</li>
                    <li><strong>📉 Technical Analysis</strong> - RSI, MACD, SMA signals combined</li>
                </ul>
            </div>
            
            <p style="margin: 30px 0;">Ready to see today's signals?</p>
            
            <a href="http://127.0.0.1:5000" class="button">View Dashboard</a>
            
            <div class="disclaimer">
                <strong>⚠ Important:</strong> Our signals are for educational purposes only. 
                Always conduct your own research and consult licensed financial advisors before trading.
            </div>
            
            <p style="margin-top: 30px; color: #666; font-size: 13px;">
                Subscribed with: <strong>{subscriber_email}</strong>
            </p>
        </div>
        
        <div class="footer">
            <p>© 2025 Stock Prediction System</p>
            <p style="margin-top: 10px;">Powered by Prophet ML + Technical Analysis</p>
        </div>
    </div>
</body>
</html>
    """

# ============================================
# EMAIL SENDING FUNCTIONS
# ============================================

def send_confirmation_email(subscriber_email: str) -> bool:
    """
    Send subscription confirmation email using your EmailReporter

    Args:
        subscriber_email: The email address of the new subscriber

    Returns:
        bool: True if email sent successfully, False otherwise
    """
    try:
        if not EmailConfig.SENDER_EMAIL or not EmailConfig.SENDER_PASSWORD:
            logger.warning("Email credentials not configured. Skipping confirmation email.")
            return False

        logger.info(f"Sending confirmation email to: {subscriber_email}")

        html_content = get_confirmation_email_html(subscriber_email)

        success = email_reporter.send_email(
            to_email=subscriber_email,
            subject=EmailConfig.CONFIRMATION_SUBJECT,
            html_content=html_content
        )

        if success:
            logger.info(f"✅ Confirmation email sent to: {subscriber_email}")
        else:
            logger.error(f"❌ Failed to send confirmation to: {subscriber_email}")

        return success

    except Exception as e:
        logger.error(f"Error sending confirmation email to {subscriber_email}: {e}")
        return False

def send_daily_signals(
    subscribers: List[str],
    recommendations: pd.DataFrame,
    summary: Dict
) -> Dict[str, int]:
    """
    Send daily trading signals to all subscribers using your EmailReporter

    Args:
        subscribers: List of subscriber email addresses
        recommendations: DataFrame with recommendations
        summary: Summary statistics dictionary

    Returns:
        dict: {'sent': count, 'failed': count}
    """
    results = {'sent': 0, 'failed': 0}

    try:
        if not EmailConfig.SENDER_EMAIL or not EmailConfig.SENDER_PASSWORD:
            logger.warning("Email credentials not configured. Skipping daily signals.")
            return results

        logger.info(f"Sending daily signals to {len(subscribers)} subscribers...")

        for email in subscribers:
            try:
                # Use your existing EmailReporter.send_daily_report method!
                success = email_reporter.send_daily_report(
                    to_email=email,
                    recommendations_df=recommendations,
                    summary=summary,
                    attach_csv=False  # Set to True if you want to attach CSV
                )

                if success:
                    results['sent'] += 1
                    logger.info(f"✅ Daily signals sent to: {email}")
                else:
                    results['failed'] += 1
                    logger.error(f"❌ Failed to send to: {email}")

            except Exception as e:
                logger.error(f"Error sending to {email}: {e}")
                results['failed'] += 1

        logger.info(f"Daily signals complete: {results['sent']} sent, {results['failed']} failed")
        return results

    except Exception as e:
        logger.error(f"Error in send_daily_signals: {e}")
        return results

def get_subscribers_list(subscribers_file: str = None) -> List[str]:
    """
    Load subscribers from file

    Args:
        subscribers_file: Path to subscribers JSON file

    Returns:
        list: List of subscriber email addresses
    """
    if subscribers_file is None:
        subscribers_file = EmailConfig.SUBSCRIBERS_FILE

    try:
        if not os.path.exists(subscribers_file):
            logger.info(f"Subscribers file not found: {subscribers_file}")
            return []

        with open(subscribers_file, 'r') as f:
            subscribers = json.load(f)

        if not isinstance(subscribers, list):
            logger.warning("Subscribers file format invalid")
            return []

        logger.info(f"Loaded {len(subscribers)} subscribers from {subscribers_file}")
        return subscribers

    except Exception as e:
        logger.error(f"Failed to load subscribers: {e}")
        return []

# ============================================
# SCHEDULED EMAIL SENDING (DAILY CRON JOB)
# ============================================

def send_daily_emails_job():
    """
    Main function to send daily emails to all subscribers

    Run this as a cron job every day at your preferred time.
    Example crontab entry (daily at 8 AM):
    0 8 * * * cd /path/to/project && /usr/bin/python3 email_sender.py
    """
    logger.info("=" * 60)
    logger.info("Starting daily email job...")
    logger.info("=" * 60)

    # Check email credentials
    if not EmailConfig.SENDER_EMAIL or not EmailConfig.SENDER_PASSWORD:
        logger.error("❌ Email credentials not configured in .env file")
        logger.error("Set EMAIL_ADDRESS and EMAIL_PASSWORD in your .env file")
        return

    # Load subscribers
    subscribers = get_subscribers_list()
    if not subscribers:
        logger.info("No subscribers found. Skipping email send.")
        logger.info("Subscribers will be added when they sign up on the dashboard.")
        return

    logger.info(f"Found {len(subscribers)} subscribers")

    # Load recommendations
    try:
        if not os.path.exists(RECOMMEND_CSV):
            logger.warning(f"Recommendations file not found: {RECOMMEND_CSV}")
            logger.warning("Run your recommendation engine first!")
            recommendations = pd.DataFrame()
        else:
            recommendations = pd.read_csv(RECOMMEND_CSV)
            logger.info(f"Loaded {len(recommendations)} recommendations")
    except Exception as e:
        logger.error(f"Failed to load recommendations: {e}")
        recommendations = pd.DataFrame()

    # Load summary
    try:
        if not os.path.exists(RECOMMEND_JSON):
            logger.warning(f"Summary file not found: {RECOMMEND_JSON}")
            summary = {}
        else:
            with open(RECOMMEND_JSON, 'r') as f:
                summary = json.load(f)
            logger.info("Loaded summary statistics")
    except Exception as e:
        logger.error(f"Failed to load summary: {e}")
        summary = {}

    # Send emails
    logger.info("\n" + "=" * 60)
    logger.info("Sending emails...")
    logger.info("=" * 60)

    results = send_daily_signals(subscribers, recommendations, summary)

    logger.info("\n" + "=" * 60)
    logger.info("Daily email job completed!")
    logger.info(f"✅ Successfully sent: {results['sent']}")
    logger.info(f"❌ Failed: {results['failed']}")
    logger.info("=" * 60)

# ============================================
# COMMAND LINE INTERFACE
# ============================================

def main():
    """Command line interface for email sender"""
    import sys

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("\n" + "=" * 60)
    print("📧 STOCK PREDICTION EMAIL SENDER")
    print("=" * 60)

    # Check credentials
    if not EmailConfig.SENDER_EMAIL or not EmailConfig.SENDER_PASSWORD:
        print("\n⚠️  Email credentials not found!")
        print("\nPlease set these in your .env file:")
        print("  EMAIL_ADDRESS=your_email@gmail.com")
        print("  EMAIL_PASSWORD=your_app_password")
        print("\nFor Gmail, use an App Password:")
        print("  https://support.google.com/accounts/answer/185833")
        print("\n" + "=" * 60)
        return

    print(f"\n✅ Email configured: {EmailConfig.SENDER_EMAIL}")

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "send_daily":
            # Send daily emails to all subscribers
            print("\n📤 Sending daily signals to all subscribers...")
            send_daily_emails_job()

        elif command == "test_confirmation":
            # Test confirmation email
            if len(sys.argv) > 2:
                test_email = sys.argv[2]
                print(f"\n📧 Sending test confirmation email to: {test_email}")
                result = send_confirmation_email(test_email)
                if result:
                    print("✅ Test email sent successfully!")
                else:
                    print("❌ Failed to send test email")
            else:
                print("\n❌ Usage: python email_sender.py test_confirmation <email>")

        elif command == "test_daily":
            # Test daily signals to one address
            if len(sys.argv) > 2:
                test_email = sys.argv[2]
                print(f"\n📧 Sending test daily signals to: {test_email}")

                try:
                    recommendations = pd.read_csv(RECOMMEND_CSV)
                    with open(RECOMMEND_JSON, 'r') as f:
                        summary = json.load(f)

                    result = send_daily_signals([test_email], recommendations, summary)
                    if result['sent'] > 0:
                        print("✅ Test email sent successfully!")
                    else:
                        print("❌ Failed to send test email")
                except Exception as e:
                    print(f"❌ Error: {e}")
            else:
                print("\n❌ Usage: python email_sender.py test_daily <email>")

        elif command == "list_subscribers":
            # List all subscribers
            subscribers = get_subscribers_list()
            print(f"\n📋 Subscribers ({len(subscribers)}):")
            for i, email in enumerate(subscribers, 1):
                print(f"  {i}. {email}")

        else:
            print(f"\n❌ Unknown command: {command}")
            print_usage()

    else:
        print_usage()

    print("\n" + "=" * 60)

def print_usage():
    """Print usage instructions"""
    print("\n📖 Usage:")
    print("  python email_sender.py send_daily")
    print("    └─ Send daily signals to all subscribers")
    print("\n  python email_sender.py test_confirmation <email>")
    print("    └─ Test subscription confirmation email")
    print("\n  python email_sender.py test_daily <email>")
    print("    └─ Test daily signals email")
    print("\n  python email_sender.py list_subscribers")
    print("    └─ List all subscribers")
    print("\n⏰ Schedule with cron (daily at 8 AM):")
    print("  0 8 * * * cd /path/to/project && /usr/bin/python3 email_sender.py send_daily")

if __name__ == "__main__":
    main()