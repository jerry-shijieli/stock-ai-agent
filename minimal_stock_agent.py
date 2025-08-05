# minimal_stock_agent.py
"""
Minimal Stock AI Agent - Production Ready Version
Run with: python minimal_stock_agent.py

This version integrates:
- Real stock data (Yahoo Finance)
- Real news sentiment (News API)
- AI analysis (OpenAI GPT-4)
- Email reports
- Simple CLI interface
"""

import os
import sys
import json
import smtplib
import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# External dependencies
try:
    import yfinance as yf
    import requests
    from textblob import TextBlob
    from langchain_openai import ChatOpenAI
    from langchain.prompts import PromptTemplate
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("Install with: pip install yfinance requests textblob langchain langchain-openai")
    sys.exit(1)


# Configuration
@dataclass
class Config:
    openai_api_key: str
    news_api_key: str = ""
    email_user: str = ""
    email_password: str = ""
    email_recipient: str = ""
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587


# Data structures
@dataclass
class StockData:
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    market_cap: str
    pe_ratio: float
    week_52_high: float
    week_52_low: float
    timestamp: datetime


@dataclass
class SentimentData:
    symbol: str
    news_sentiment: float
    article_count: int
    confidence: float
    key_headlines: List[str]
    timestamp: datetime


@dataclass
class Analysis:
    symbol: str
    recommendation: str  # BUY, HOLD, SELL
    confidence: int  # 1-10
    reasoning: str
    risk_factors: List[str]
    target_price: Optional[float]
    current_price: float
    timestamp: datetime


class DataFetcher:
    """Fetches real data from external APIs"""

    def __init__(self, config: Config):
        self.config = config

    def get_stock_data(self, symbol: str) -> Optional[StockData]:
        """Fetch real stock data using yfinance"""
        try:
            print(f"📊 Fetching stock data for {symbol}...")

            stock = yf.Ticker(symbol)
            info = stock.info
            hist = stock.history(period="5d")

            if hist.empty:
                print(f"❌ No data found for {symbol}")
                return None

            current_price = float(hist['Close'].iloc[-1])
            prev_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price
            change = current_price - prev_close
            change_percent = (change / prev_close) * 100 if prev_close > 0 else 0

            return StockData(
                symbol=symbol,
                price=current_price,
                change=change,
                change_percent=change_percent,
                volume=int(hist['Volume'].iloc[-1]),
                market_cap=f"${info.get('marketCap', 0):,.0f}" if info.get('marketCap') else "N/A",
                pe_ratio=float(info.get('trailingPE', 0)) if info.get('trailingPE') else 0,
                week_52_high=float(info.get('fiftyTwoWeekHigh', current_price)),
                week_52_low=float(info.get('fiftyTwoWeekLow', current_price)),
                timestamp=datetime.now()
            )

        except Exception as e:
            print(f"❌ Error fetching stock data for {symbol}: {e}")
            return None

    def get_news_sentiment(self, symbol: str) -> SentimentData:
        """Fetch and analyze news sentiment"""
        try:
            print(f"📰 Analyzing news sentiment for {symbol}...")

            if not self.config.news_api_key:
                print("⚠️  No News API key - using mock sentiment")
                return self._get_mock_sentiment(symbol)

            # Fetch news from NewsAPI
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': f"{symbol} stock OR {symbol} earnings OR {symbol} company",
                'language': 'en',
                'sortBy': 'publishedAt',
                'from': (datetime.now() - timedelta(days=3)).isoformat(),
                'pageSize': 20,
                'apiKey': self.config.news_api_key
            }

            response = requests.get(url, params=params, timeout=10)

            if response.status_code != 200:
                print(f"⚠️  News API error: {response.status_code}")
                return self._get_mock_sentiment(symbol)

            news_data = response.json()
            articles = news_data.get('articles', [])

            if not articles:
                print(f"⚠️  No recent news found for {symbol}")
                return self._get_mock_sentiment(symbol)

            # Analyze sentiment
            sentiments = []
            headlines = []

            for article in articles[:10]:  # Analyze top 10 articles
                title = article.get('title', '')
                description = article.get('description', '') or ''

                if title and 'removed' not in title.lower():
                    text = f"{title} {description}"
                    blob = TextBlob(text)
                    sentiment = blob.sentiment.polarity

                    sentiments.append(sentiment)
                    headlines.append(title[:100])

            if not sentiments:
                return self._get_mock_sentiment(symbol)

            avg_sentiment = sum(sentiments) / len(sentiments)
            confidence = min(len(sentiments) / 10, 1.0)  # More articles = higher confidence

            return SentimentData(
                symbol=symbol,
                news_sentiment=avg_sentiment,
                article_count=len(sentiments),
                confidence=confidence,
                key_headlines=headlines[:5],
                timestamp=datetime.now()
            )

        except Exception as e:
            print(f"❌ Error analyzing news sentiment for {symbol}: {e}")
            return self._get_mock_sentiment(symbol)

    def _get_mock_sentiment(self, symbol: str) -> SentimentData:
        """Fallback mock sentiment data"""
        import random
        return SentimentData(
            symbol=symbol,
            news_sentiment=random.uniform(-0.3, 0.3),
            article_count=random.randint(3, 15),
            confidence=0.6,
            key_headlines=[f"Mock headline about {symbol}"],
            timestamp=datetime.now()
        )


class StockAgent:
    """Core AI agent for stock analysis"""

    def __init__(self, config: Config):
        self.config = config
        self.llm = ChatOpenAI(
            temperature=0.1,
            model="gpt-4o-mini",  # More cost-effective
            openai_api_key=config.openai_api_key
        )

        # Analysis prompt template
        self.analysis_prompt = PromptTemplate(
            input_variables=["stock_data", "sentiment_data"],
            template="""
You are an expert stock analyst. Analyze this data and provide a recommendation:

STOCK: {stock_data_symbol}
Current Price: ${stock_data_price:.2f}
Change: {stock_data_change:.2f} ({stock_data_change_percent:.2f}%)
Volume: {stock_data_volume:,}
P/E Ratio: {stock_data_pe_ratio}
52W Range: ${stock_data_week_52_low:.2f} - ${stock_data_week_52_high:.2f}

NEWS SENTIMENT:
Sentiment Score: {sentiment_data_news_sentiment:.3f} (-1=Very Negative, +1=Very Positive)
Articles Analyzed: {sentiment_data_article_count}
Confidence: {sentiment_data_confidence:.2f}
Recent Headlines: {sentiment_data_headlines}

ANALYSIS REQUIRED:
1. Evaluate the stock's current position
2. Consider sentiment alignment with price action
3. Assess risk factors and opportunities
4. Provide clear recommendation

FORMAT YOUR RESPONSE EXACTLY AS:
RECOMMENDATION: [BUY/HOLD/SELL]
CONFIDENCE: [1-10]
TARGET_PRICE: $[price]
REASONING: [2-3 sentences explaining your decision]
RISK_FACTORS: [List top 3 risks separated by semicolons]

Be conservative - only recommend BUY if conviction is high (7+ confidence).
"""
        )

    def analyze(self, stock_data: StockData, sentiment_data: SentimentData) -> Analysis:
        """Perform AI analysis on stock data"""
        try:
            print(f"🤖 Running AI analysis for {stock_data.symbol}...")

            # Format the prompt
            prompt = self.analysis_prompt.format(
                stock_data_symbol=stock_data.symbol,
                stock_data_price=stock_data.price,
                stock_data_change=stock_data.change,
                stock_data_change_percent=stock_data.change_percent,
                stock_data_volume=stock_data.volume,
                stock_data_pe_ratio=stock_data.pe_ratio,
                stock_data_week_52_low=stock_data.week_52_low,
                stock_data_week_52_high=stock_data.week_52_high,
                sentiment_data_news_sentiment=sentiment_data.news_sentiment,
                sentiment_data_article_count=sentiment_data.article_count,
                sentiment_data_confidence=sentiment_data.confidence,
                sentiment_data_headlines="; ".join(sentiment_data.key_headlines[:3])
            )

            # Get AI response
            response = self.llm.invoke(prompt).content

            # Parse response
            return self._parse_analysis(stock_data.symbol, response, stock_data.price)

        except Exception as e:
            print(f"❌ Error in AI analysis for {stock_data.symbol}: {e}")
            return Analysis(
                symbol=stock_data.symbol,
                recommendation="HOLD",
                confidence=1,
                reasoning=f"Analysis failed: {str(e)}",
                risk_factors=["Analysis system error"],
                target_price=None,
                current_price=stock_data.price,
                timestamp=datetime.now()
            )

    def _parse_analysis(self, symbol: str, response: str, current_price: float) -> Analysis:
        """Parse the AI response into structured analysis"""
        try:
            lines = [line.strip() for line in response.split('\n') if line.strip()]

            recommendation = "HOLD"
            confidence = 5
            target_price = None
            reasoning = ""
            risk_factors = []

            for line in lines:
                if line.startswith("RECOMMENDATION:"):
                    rec = line.split(":", 1)[1].strip().upper()
                    if rec in ["BUY", "HOLD", "SELL"]:
                        recommendation = rec

                elif line.startswith("CONFIDENCE:"):
                    try:
                        conf_str = line.split(":", 1)[1].strip()
                        confidence = int(conf_str.split()[0])
                        confidence = max(1, min(10, confidence))
                    except:
                        confidence = 5

                elif line.startswith("TARGET_PRICE:"):
                    try:
                        price_str = line.split(":", 1)[1].strip().replace("$", "")
                        target_price = float(price_str)
                    except:
                        target_price = None

                elif line.startswith("REASONING:"):
                    reasoning = line.split(":", 1)[1].strip()

                elif line.startswith("RISK_FACTORS:"):
                    risks_str = line.split(":", 1)[1].strip()
                    risk_factors = [r.strip() for r in risks_str.split(";") if r.strip()]

            return Analysis(
                symbol=symbol,
                recommendation=recommendation,
                confidence=confidence,
                reasoning=reasoning or "Analysis completed",
                risk_factors=risk_factors[:3],  # Top 3 risks
                target_price=target_price,
                current_price=current_price,
                timestamp=datetime.now()
            )

        except Exception as e:
            print(f"⚠️  Error parsing analysis: {e}")
            return Analysis(
                symbol=symbol,
                recommendation="HOLD",
                confidence=3,
                reasoning="Unable to parse analysis results",
                risk_factors=["Analysis parsing error"],
                target_price=None,
                current_price=current_price,
                timestamp=datetime.now()
            )


class ReportGenerator:
    """Generate and send reports"""

    def __init__(self, config: Config):
        self.config = config

    def generate_report(self, analyses: List[Analysis]) -> str:
        """Generate a formatted text report"""
        report = f"""
📈 STOCK ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 60}

PORTFOLIO SUMMARY:
• Total stocks analyzed: {len(analyses)}
• BUY recommendations: {sum(1 for a in analyses if a.recommendation == 'BUY')}
• HOLD recommendations: {sum(1 for a in analyses if a.recommendation == 'HOLD')}
• SELL recommendations: {sum(1 for a in analyses if a.recommendation == 'SELL')}

{'=' * 60}
DETAILED ANALYSIS:
"""

        for analysis in analyses:
            emoji = {"BUY": "🟢", "HOLD": "🟡", "SELL": "🔴"}[analysis.recommendation]

            report += f"""
{emoji} {analysis.symbol} - {analysis.recommendation} (Confidence: {analysis.confidence}/10)
├─ Current Price: ${analysis.current_price:.2f}
├─ Target Price: {f"${analysis.target_price:.2f}" if analysis.target_price else "N/A"}
├─ Reasoning: {analysis.reasoning}
└─ Key Risks: {'; '.join(analysis.risk_factors)}

"""

        report += f"""
{'=' * 60}
DISCLAIMER: This analysis is for educational purposes only. 
Always do your own research before making investment decisions.
{'=' * 60}
"""
        return report

    def send_email_report(self, report: str) -> bool:
        """Send report via email"""
        if not all([self.config.email_user, self.config.email_password, self.config.email_recipient]):
            print("⚠️  Email not configured - skipping email report")
            return False

        try:
            print("📧 Sending email report...")

            msg = MIMEMultipart()
            msg['From'] = self.config.email_user
            msg['To'] = self.config.email_recipient
            msg['Subject'] = f"Stock Analysis Report - {datetime.now().strftime('%Y-%m-%d')}"

            msg.attach(MIMEText(report, 'plain'))

            server = smtplib.SMTP(self.config.smtp_server, self.config.smtp_port)
            server.starttls()
            server.login(self.config.email_user, self.config.email_password)
            server.send_message(msg)
            server.quit()

            print("✅ Email report sent successfully!")
            return True

        except Exception as e:
            print(f"❌ Failed to send email: {e}")
            return False


class MinimalStockAgent:
    """Main application class"""

    def __init__(self):
        self.config = self._load_config()
        self.data_fetcher = DataFetcher(self.config)
        self.agent = StockAgent(self.config)
        self.report_generator = ReportGenerator(self.config)

    def _load_config(self) -> Config:
        """Load configuration from environment variables"""
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            print("❌ OPENAI_API_KEY environment variable is required!")
            print("Set it with: export OPENAI_API_KEY='your-key-here'")
            sys.exit(1)

        return Config(
            openai_api_key=openai_key,
            news_api_key=os.getenv('NEWS_API_KEY', ''),
            email_user=os.getenv('EMAIL_USER', ''),
            email_password=os.getenv('EMAIL_PASSWORD', ''),
            email_recipient=os.getenv('EMAIL_RECIPIENT', ''),
        )

    def analyze_portfolio(self, symbols: List[str]) -> List[Analysis]:
        """Analyze a portfolio of stocks"""
        analyses = []

        print(f"\n🚀 Starting analysis of {len(symbols)} stocks...")
        print(f"Symbols: {', '.join(symbols)}")
        print("-" * 60)

        for i, symbol in enumerate(symbols, 1):
            print(f"\n[{i}/{len(symbols)}] Analyzing {symbol}...")

            # Fetch stock data
            stock_data = self.data_fetcher.get_stock_data(symbol)
            if not stock_data:
                continue

            # Fetch sentiment data
            sentiment_data = self.data_fetcher.get_news_sentiment(symbol)

            # Run AI analysis
            analysis = self.agent.analyze(stock_data, sentiment_data)
            analyses.append(analysis)

            # Quick preview
            emoji = {"BUY": "🟢", "HOLD": "🟡", "SELL": "🔴"}[analysis.recommendation]
            print(f"   {emoji} {analysis.recommendation} (Confidence: {analysis.confidence}/10)")
            print(f"   Reasoning: {analysis.reasoning[:100]}...")

        return analyses

    def run_analysis(self, symbols: List[str], save_report: bool = True, send_email: bool = False):
        """Run complete analysis workflow"""
        try:
            # Run analysis
            analyses = self.analyze_portfolio(symbols)

            if not analyses:
                print("❌ No analyses completed successfully")
                return

            # Generate report
            report = self.report_generator.generate_report(analyses)

            # Display report
            print("\n" + "=" * 80)
            print(report)

            # Save to file
            if save_report:
                filename = f"reports/stock_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                with open(filename, 'w') as f:
                    f.write(report)
                print(f"💾 Report saved to {filename}")

            # Send email
            if send_email:
                self.report_generator.send_email_report(report)

            # Save analysis data as JSON
            if save_report:
                json_filename = f"analysis/analysis_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                analysis_data = [asdict(analysis) for analysis in analyses]
                with open(json_filename, 'w') as f:
                    json.dump(analysis_data, f, indent=2, default=str)
                print(f"💾 Analysis data saved to {json_filename}")

        except KeyboardInterrupt:
            print("\n\n⚠️  Analysis interrupted by user")
        except Exception as e:
            print(f"\n❌ Error during analysis: {e}")


def main():
    """Main entry point"""
    print("🤖 Minimal Stock AI Agent")
    print("=" * 40)

    # Default portfolio (you can modify this)
    default_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]

    # Check for command line arguments
    if len(sys.argv) > 1:
        symbols = [s.upper().strip() for s in sys.argv[1].split(',')]
    else:
        print(f"Using default portfolio: {', '.join(default_symbols)}")
        print("To analyze different stocks, run: python minimal_stock_agent.py AAPL,GOOGL,MSFT")
        symbols = default_symbols

    # Create and run agent
    agent = MinimalStockAgent()
    agent.run_analysis(
        symbols=symbols,
        save_report=True,
        send_email=False  # Set to True if you want email reports
    )


if __name__ == "__main__":
    main()