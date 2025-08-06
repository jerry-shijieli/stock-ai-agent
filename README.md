# Stock AI Agent

## Overview
An intelligent stock analysis agent that combines real-time market data with AI-powered insights to provide investment recommendations. The system analyzes stocks using financial metrics, news sentiment, and machine learning to generate actionable trading signals.

## 🚀 Quick Setup (30 minutes)

### Step 1: Install Dependencies (5 minutes)

```bash
# Create a new directory
mkdir stock-ai-agent
cd stock-ai-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install yfinance requests textblob langchain-openai
```

### Step 2: Get API Keys (10 minutes)

**Required: OpenAI API Key**
- Go to [OpenAI API](https://platform.openai.com/api-keys)
- Create account and get API key
- Add $5 credit (should last for hundreds of analyses)

**Optional: News API Key** (for better sentiment analysis)
- Go to [NewsAPI](https://newsapi.org/)
- Free tier: 1000 requests/day
- Get your API key

### Step 3: Set Environment Variables (2 minutes)

```bash
# Required
export OPENAI_API_KEY="sk-your-openai-key-here"

# Optional (for news sentiment)
export NEWS_API_KEY="your-newsapi-key-here"

# Optional (for email reports)
export EMAIL_USER="your-gmail@gmail.com"
export EMAIL_PASSWORD="your-app-password"
export EMAIL_RECIPIENT="recipient@gmail.com"
```

### Step 4: Run the Agent (5 minutes)

```bash
# Download the code file (copy from the artifact above)
# Save it as: minimal_stock_agent.py

# Run with default stocks (AAPL, GOOGL, MSFT, TSLA, NVDA)
python minimal_stock_agent.py

# Or analyze specific stocks
python minimal_stock_agent.py AAPL,MSFT,AMZN
```

## 🎯 Sample Output

When you run the agent, you'll see:

```
🤖 Minimal Stock AI Agent
========================================
Using default portfolio: AAPL, GOOGL, MSFT, TSLA, NVDA

🚀 Starting analysis of 5 stocks...
Symbols: AAPL, GOOGL, MSFT, TSLA, NVDA
------------------------------------------------------------

[1/5] Analyzing AAPL...
📊 Fetching stock data for AAPL...
📰 Analyzing news sentiment for AAPL...
🤖 Running AI analysis for AAPL...
   🟢 BUY (Confidence: 7/10)
   Reasoning: Strong fundamentals with positive sentiment alignment...

[2/5] Analyzing GOOGL...
📊 Fetching stock data for GOOGL...
📰 Analyzing news sentiment for GOOGL...
🤖 Running AI analysis for GOOGL...
   🟡 HOLD (Confidence: 6/10)
   Reasoning: Mixed signals with moderate technical indicators...

... (continues for all stocks)

================================================================================
📈 STOCK ANALYSIS REPORT
Generated: 2025-08-04 15:30:22
============================================================================

PORTFOLIO SUMMARY:
- Total stocks analyzed: 5
- BUY recommendations: 2
- HOLD recommendations: 2
- SELL recommendations: 1

============================================================================
DETAILED ANALYSIS:

🟢 AAPL - BUY (Confidence: 7/10)
├─ Current Price: $186.40
├─ Target Price: $195.00
├─ Reasoning: Strong earnings momentum with positive news sentiment
└─ Key Risks: Market volatility; Competition from Android; Supply chain

🟡 GOOGL - HOLD (Confidence: 6/10)
├─ Current Price: $139.69
├─ Target Price: $145.00
├─ Reasoning: Solid fundamentals but facing AI competition concerns
└─ Key Risks: Regulatory pressure; AI disruption; Ad market slowdown

... (continues for all stocks)

💾 Report saved to stock_report_20250804_153022.txt
💾 Analysis data saved to analysis_data_20250804_153022.json
```

## 🔍 Key Features

### 1. Real Data Integration
- **Stock Data**: Live prices, volume, P/E ratios from Yahoo Finance
- **News Sentiment**: Analyzes recent news articles using TextBlob
- **Fallback Systems**: Works even if News API is unavailable

### 2. AI Reasoning Engine
- **Structured Prompts**: Clear analysis framework for the AI
- **Quantitative Validation**: Combines AI insights with data
- **Risk Assessment**: Identifies specific risk factors

### 3. Production Features
- **Error Handling**: Graceful failures and recovery
- **Data Persistence**: Saves reports and raw analysis data
- **Email Reports**: Optional automated notifications
- **CLI Interface**: Easy to use and automate

## 🎯 Testing Different Scenarios

**Bull Market Test**
```bash
python minimal_stock_agent.py AAPL,MSFT,GOOGL
```

**Volatile Stocks Test**
```bash
python minimal_stock_agent.py TSLA,GME,AMC
```

**Blue Chip Portfolio**
```bash
python minimal_stock_agent.py JNJ,PG,KO,WMT
```

**Tech Growth Stocks**
```bash
python minimal_stock_agent.py NVDA,AMD,CRM,SNOW
```

## 🚀 Next Steps

After getting the basic system running, consider these enhancements:
- Add technical indicators (RSI, MACD, Bollinger Bands)
- Implement portfolio optimization algorithms
- Create web dashboard for visualization
- Add backtesting capabilities
- Integrate with broker APIs for automated trading