import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv
import logging
from src.utils.paths import RECOMMEND_JSON,RECOMMEND_CSV

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmailReporter:
    """
    Generates and sends beautiful HTML email reports with trade recommendations.
    """

    def __init__(self,
               smtp_server: str = "smtp.gmail.com",
               smtp_port: int = 587,
               sender_email: str = None,
               sender_password: str = None):
        """
        Initialize email reporter.

        Args:
            smtp_server: SMTP server address (default: Gmail)
            smtp_port: SMTP port (587 for TLS)
            sender_email: Your email address
            sender_password: Your email password or app password
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password

    def generate_html_report(self,
                             recommendations_df: pd.DataFrame,
                             summary: dict) -> str:
        """
        Generate beautiful HTML email report.
        """
        # Header styling
        html = f"""
        <html>
        <head>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background-color: #f5f5f5;
                    margin: 0;
                    padding: 20px;
                }}
                .container {{
                    max-width: 900px;
                    margin: 0 auto;
                    background-color: white;
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    overflow: hidden;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 30px;
                    text-align: center;
                }}
                .header h1 {{
                    margin: 0;
                    font-size: 32px;
                }}
                .header p {{
                    margin: 10px 0 0 0;
                    font-size: 16px;
                    opacity: 0.9;
                }}
                .summary {{
                    padding: 25px;
                    background-color: #f8f9fa;
                    border-bottom: 3px solid #e9ecef;
                }}
                .summary-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    margin-top: 15px;
                }}
                .summary-item {{
                    background: white;
                    padding: 15px;
                    border-radius: 8px;
                    border-left: 4px solid #667eea;
                }}
                .summary-item h3 {{
                    margin: 0 0 5px 0;
                    color: #666;
                    font-size: 13px;
                    text-transform: uppercase;
                }}
                .summary-item p {{
                    margin: 0;
                    font-size: 24px;
                    font-weight: bold;
                    color: #333;
                }}
                .content {{
                    padding: 25px;
                }}
                h2 {{
                    color: #333;
                    border-bottom: 2px solid #667eea;
                    padding-bottom: 10px;
                    margin-top: 30px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                    font-size: 14px;
                }}
                th {{
                    background-color: #667eea;
                    color: white;
                    padding: 12px;
                    text-align: left;
                    font-weight: 600;
                }}
                td {{
                    padding: 12px;
                    border-bottom: 1px solid #e9ecef;
                }}
                tr:hover {{
                    background-color: #f8f9fa;
                }}
                .badge {{
                    padding: 4px 8px;
                    border-radius: 4px;
                    font-size: 11px;
                    font-weight: bold;
                    text-transform: uppercase;
                }}
                .badge-buy {{
                    background-color: #28a745;
                    color: white;
                }}
                .badge-sell {{
                    background-color: #dc3545;
                    color: white;
                }}
                .badge-high {{
                    background-color: #28a745;
                    color: white;
                }}
                .badge-medium {{
                    background-color: #ffc107;
                    color: #333;
                }}
                .badge-low {{
                    background-color: #6c757d;
                    color: white;
                }}
                .footer {{
                    background-color: #333;
                    color: white;
                    padding: 20px;
                    text-align: center;
                    font-size: 12px;
                }}
                .disclaimer {{
                    background-color: #fff3cd;
                    border: 2px solid #ffc107;
                    border-radius: 8px;
                    padding: 15px;
                    margin: 20px 0;
                }}
                .disclaimer h3 {{
                    color: #856404;
                    margin: 0 0 10px 0;
                }}
                .disclaimer p {{
                    color: #856404;
                    margin: 5px 0;
                    font-size: 13px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📊 Daily Stock Recommendations</h1>
                    <p>{datetime.now().strftime("%B %d, %Y")}</p>
                </div>
        """
        # Summary section
        html += f"""
                <div class="summary">
                    <h2 style="margin-top: 0; border: none;">📈 Portfolio Summary</h2>
                    <div class="summary-grid">
                        <div class="summary-item">
                            <h3>Total Positions</h3>
                            <p>{summary.get('total_positions', 0)}</p>
                        </div>
                        <div class="summary-item">
                            <h3>Capital Allocated</h3>
                            <p>${summary.get('total_allocated', 0):,.0f}</p>
                        </div>
                        <div class="summary-item">
                            <h3>Cash Remaining</h3>
                            <p>${summary.get('cash_remaining', 0):,.0f}</p>
                        </div>
                        <div class="summary-item">
                            <h3>Total Risk</h3>
                            <p>{summary.get('total_risk_pct', 0):.2f}%</p>
                        </div>
                        <div class="summary-item">
                            <h3>Avg Confidence</h3>
                            <p>{summary.get('avg_confidence', 0):.1%}</p>
                        </div>
                        <div class="summary-item">
                            <h3>Avg Accuracy</h3>
                            <p>{summary.get('avg_prophet_accuracy', 0):.1f}%</p>
                        </div>
                    </div>
                </div>
        """

        # Recommendations table
        html += """
                <div class="content">
                    <h2>🎯 Trade Recommendations</h2>
        """

        if recommendations_df.empty:
            html += "<p>No recommendations meet the criteria today. Market conditions may not be favorable.</p>"
        else:
            html += "<table>"
            html += """
                <tr>
                    <th>Symbol</th>
                    <th>Action</th>
                    <th>Shares</th>
                    <th>Entry Price</th>
                    <th>Target</th>
                    <th>Stop Loss</th>
                    <th>Position</th>
                    <th>Confidence</th>
                    <th>Accuracy</th>
                </tr>
            """

            for _, row in recommendations_df.iterrows():
                action_class = 'badge-buy' if row['action'] == 'BUY' else 'badge-sell'

                conf_class = 'badge-high' if row['confidence_label'] in ['HIGH', 'VERY HIGH'] else 'badge-medium'

                html += f"""
                <tr>
                    <td><strong>{row['symbol']}</strong></td>
                    <td><span class="badge {action_class}">{row['action']}</span></td>
                    <td>{row['shares']}</td>
                    <td>${row['current_price']:.2f}</td>
                    <td>${row['target_price']:.2f}</td>
                    <td>${row['stop_loss']:.2f}</td>
                    <td>${row['position_value']:,.0f}</td>
                    <td><span class="badge {conf_class}">{row['confidence_label']}</span></td>
                    <td>{row['prophet_accuracy']:.1f}%</td>
                </tr>
                """

            html += "</table>"

            # Top recommendation highlight
            if len(recommendations_df) > 0:
                top = recommendations_df.iloc[0]
                html += f"""
                <div style="background-color: #e7f3ff; border-left: 4px solid #2196F3; padding: 15px; margin: 20px 0; border-radius: 4px;">
                    <h3 style="margin: 0 0 10px 0; color: #1976D2;">🌟 Top Recommendation: {top['symbol']}</h3>
                    <p style="margin: 5px 0;"><strong>Action:</strong> {top['action']} {top['shares']} shares at ${top['current_price']:.2f}</p>
                    <p style="margin: 5px 0;"><strong>Confidence:</strong> {top['confidence_label']} ({top['combined_confidence']:.1%})</p>
                    <p style="margin: 5px 0;"><strong>Prophet Accuracy:</strong> {top['prophet_accuracy']:.1f}%</p>
                    <p style="margin: 5px 0;"><strong>Signal Agreement:</strong> {top['signal_agreement']} ({top['rsi_signal']}, {top['macd_signal']}, {top['sma_signal']})</p>
                </div>
                """

        # Disclaimer
        html += """
                <div class="disclaimer">
                    <h3>⚠ Important Disclaimer</h3>
                    <p><strong>This is for educational purposes only.</strong></p>
                    <p>This report is not financial advice. The recommendations are generated by an algorithmic system for learning purposes.</p>
                    <p>Past performance does not guarantee future results. Trading stocks involves risk of loss.</p>
                    <p>Always do your own research and consult with a licensed financial advisor before making investment decisions.</p>
                </div>
                </div>
        """

        # Footer
        html += """
                <div class="footer">
                    <p>Generated by Stock Prediction System</p>
                    <p>Powered by Prophet ML + Technical Analysis</p>
                </div>
            </div>
        </body>
        </html>
        """

        return html

    def send_email(self,
                   to_email: str,
                   subject: str,
                   html_content: str,
                   attachments: list = None) -> bool:
        """
        Send HTML email with optional attachments.

        Args:
            to_email: Recipient email address
            subject: Email subject
            html_content: HTML email body
            attachments: List of file paths to attach

        Returns:
            True if sent successfully, False otherwise
        """
        if not self.sender_email or not self.sender_password:
            logger.error("Sender email or password not configured!")
            return False

        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['From'] = self.sender_email
            msg['To'] = to_email
            msg['Subject'] = subject

            # Attach HTML content
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)

            # Add attachments
            if attachments:
                for file_path in attachments:
                    if os.path.exists(file_path):
                        with open(file_path, 'rb') as f:
                            part = MIMEBase('application', 'octet-stream')
                            part.set_payload(f.read())
                            encoders.encode_base64(part)
                            part.add_header('Content-Disposition',
                                            f'attachment; filename={os.path.basename(file_path)}')
                            msg.attach(part)

            # Send email
            logger.info(f"Connecting to {self.smtp_server}:{self.smtp_port}...")
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()

            logger.info("Logging in...")
            server.login(self.sender_email, self.sender_password)

            logger.info(f"Sending email to {to_email}...")
            server.send_message(msg)
            server.quit()

            logger.info("✅ Email sent successfully!")
            return True

        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False

    def send_daily_report(self,
                          to_email: str,
                          recommendations_df: pd.DataFrame,
                          summary: dict,
                          attach_csv: bool = True) -> bool:
        """
        Send daily recommendations report.
        """
        # Generate HTML
        html_content = self.generate_html_report(recommendations_df, summary)

        # Subject line
        date_str = datetime.now().strftime("%B %d, %Y")
        num_recs = len(recommendations_df)
        subject = f"📊 Daily Stock Report - {num_recs} Recommendations ({date_str})"

        # Attachments
        attachments = []
        if attach_csv and os.path.exists(RECOMMEND_CSV):
            attachments.append('daily_recommendations.csv')

        # Send
        return self.send_email(to_email, subject, html_content, attachments)


# ============ EXAMPLE USAGE ============

def main():
    print("=" * 60)
    print("EMAIL REPORT GENERATOR")
    print("=" * 60)

    # Load recommendations and summary
    try:
        recommendations = pd.read_csv(RECOMMEND_CSV)

        with open(RECOMMEND_JSON, 'r') as f:
            import json

            summary = json.load(f)

        print(f"\nLoaded {len(recommendations)} recommendations")

    except FileNotFoundError:
        print("❌ Error: Run recommendation_engine.py first to generate recommendations!")
        exit(1)

    # Initialize email reporter
    print("\n" + "=" * 60)
    print("EMAIL CONFIGURATION")
    print("=" * 60)

    load_dotenv()
    sender_email = os.getenv('EMAIL_ADDRESS')
    sender_password = os.getenv('EMAIL_PASSWORD')

    if not sender_email or not sender_password:
        print("\n⚠ Email credentials not provided.")
        print("Generating HTML preview only...")

        reporter = EmailReporter()
        html = reporter.generate_html_report(recommendations, summary)

        # Save HTML preview
        with open('email_preview.html', 'w', encoding='utf-8') as f:
            f.write(html)

        print("✅ HTML preview saved to: email_preview.html")
        print("Open this file in your browser to see how the email looks!")

    else:
        reporter = EmailReporter(
            sender_email=sender_email,
            sender_password=sender_password
        )

        # Get recipient email
        to_email = input("\nSend report to (email address): ").strip() or sender_email

        print(f"\n📤 Sending report to {to_email}...")

        success = reporter.send_daily_report(
            to_email=to_email,
            recommendations_df=recommendations,
            summary=summary,
            attach_csv=True
        )

        if success:
            print("\n✅ Email sent successfully!")
            print("Check your inbox!")
        else:
            print("\n❌ Failed to send email.")
            print("Check your credentials and try again.")

    print("\n" + "=" * 60)

if __name__== "__main__":
    main()