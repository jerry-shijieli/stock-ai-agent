# multi_agent_stock_system.py
"""
Multi-Agent Stock Analysis System - Step by Step Implementation

This transforms your minimal agent into a collaborative team:
1. Data Collection Agent (gathers information)
2. Technical Analysis Agent (price/volume analysis)
3. Sentiment Analysis Agent (news/social analysis)
4. Risk Assessment Agent (risk evaluation)
5. Portfolio Manager Agent (final decisions)

Run with: python multi_agent_stock_system.py
"""

import os
import sys
import json
import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
import logging

# External dependencies
try:
    import yfinance as yf
    import requests
    from textblob import TextBlob
    from langchain_openai import ChatOpenAI
    from langchain.prompts import PromptTemplate
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("Install with: pip install yfinance requests textblob langchain-openai")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# ============================================================================
# STEP 1: Core Data Structures and Message Passing
# ============================================================================

class MessageType(Enum):
    """Types of messages agents can send"""
    DATA_REQUEST = "data_request"
    DATA_RESPONSE = "data_response"
    ANALYSIS_REQUEST = "analysis_request"
    ANALYSIS_RESPONSE = "analysis_response"
    FINAL_RECOMMENDATION = "final_recommendation"
    ERROR = "error"


@dataclass
class AgentMessage:
    """Message format for inter-agent communication"""
    id: str
    sender: str
    recipient: str
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: datetime
    priority: int = 5  # 1=highest, 10=lowest


@dataclass
class StockDataPackage:
    """Standardized stock data format"""
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    market_cap: str
    pe_ratio: float
    week_52_high: float
    week_52_low: float
    rsi: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    timestamp: datetime = None


@dataclass
class SentimentPackage:
    """Standardized sentiment data format"""
    symbol: str
    news_sentiment: float
    social_sentiment: float
    article_count: int
    confidence: float
    key_headlines: List[str]
    sentiment_trend: str  # "improving", "declining", "stable"
    timestamp: datetime = None


@dataclass
class AnalysisResult:
    """Analysis result from specialist agents"""
    agent_name: str
    symbol: str
    analysis_type: str
    score: float  # 0-10 scale
    confidence: float  # 0-1 scale
    key_insights: List[str]
    recommendations: List[str]
    warnings: List[str]
    raw_analysis: str
    timestamp: datetime


# ============================================================================
# STEP 2: Base Agent Class with Communication Framework
# ============================================================================

class BaseAgent:
    """Base class for all agents with communication capabilities"""

    def __init__(self, name: str, agent_type: str):
        self.name = name
        self.agent_type = agent_type
        self.logger = logging.getLogger(f"Agent.{name}")
        self.inbox = []
        self.outbox = []
        self.status = "initialized"

        # Agent memory for context
        self.memory = {
            "processed_symbols": [],
            "last_analysis": {},
            "performance_metrics": {}
        }

        self.logger.info(f"🤖 {name} agent initialized")

    async def send_message(self, recipient: str, message_type: MessageType, content: Dict, priority: int = 5) -> str:
        """Send message to another agent"""
        message_id = f"{self.name}_{datetime.now().strftime('%H%M%S_%f')}"

        message = AgentMessage(
            id=message_id,
            sender=self.name,
            recipient=recipient,
            message_type=message_type,
            content=content,
            timestamp=datetime.now(),
            priority=priority
        )

        self.outbox.append(message)
        self.logger.info(f"📤 Sent {message_type.value} to {recipient}")
        return message_id

    async def receive_message(self, message: AgentMessage):
        """Receive and queue message for processing"""
        self.inbox.append(message)
        self.logger.info(f"📥 Received {message.message_type.value} from {message.sender}")

    async def process_inbox(self):
        """Process all messages in inbox"""
        while self.inbox:
            message = self.inbox.pop(0)
            await self.handle_message(message)

    async def handle_message(self, message: AgentMessage):
        """Override in each agent to handle specific message types"""
        self.logger.warning(f"Unhandled message type: {message.message_type}")

    def update_memory(self, key: str, value: Any):
        """Update agent memory"""
        self.memory[key] = value
        self.logger.debug(f"Memory updated: {key}")


# ============================================================================
# STEP 3: Data Collection Agent (Replaces your data fetching)
# ============================================================================

class DataCollectionAgent(BaseAgent):
    """Specialist agent for collecting all external data"""

    def __init__(self, config: Dict):
        super().__init__("DataCollector", "data_collection")
        self.config = config
        self.cache = {}  # Simple caching to avoid duplicate API calls

    async def handle_message(self, message: AgentMessage):
        """Handle data requests from other agents"""
        if message.message_type == MessageType.DATA_REQUEST:
            symbol = message.content.get("symbol")
            data_types = message.content.get("data_types", ["stock", "sentiment"])

            # Collect requested data
            data_package = await self.collect_all_data(symbol, data_types)

            # Send response back
            await self.send_message(
                recipient=message.sender,
                message_type=MessageType.DATA_RESPONSE,
                content={
                    "request_id": message.id,
                    "symbol": symbol,
                    "data_package": asdict(data_package) if data_package else None,
                    "status": "success" if data_package else "failed"
                }
            )

    async def collect_all_data(self, symbol: str, data_types: List[str]) -> Optional[Dict]:
        """Collect all requested data for a symbol"""
        self.logger.info(f"🔍 Collecting data for {symbol}: {data_types}")

        result = {"symbol": symbol}

        try:
            # Collect stock data
            if "stock" in data_types:
                stock_data = await self._fetch_stock_data(symbol)
                if stock_data:
                    result["stock_data"] = asdict(stock_data)

            # Collect sentiment data
            if "sentiment" in data_types:
                sentiment_data = await self._fetch_sentiment_data(symbol)
                if sentiment_data:
                    result["sentiment_data"] = asdict(sentiment_data)

            # Update cache
            self.cache[symbol] = {
                "data": result,
                "timestamp": datetime.now()
            }

            return result

        except Exception as e:
            self.logger.error(f"Error collecting data for {symbol}: {e}")
            return None

    async def _fetch_stock_data(self, symbol: str) -> Optional[StockDataPackage]:
        """Fetch stock data with technical indicators"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            hist = stock.history(period="60d")  # More data for technical analysis

            if hist.empty:
                return None

            current_price = float(hist['Close'].iloc[-1])
            prev_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price

            # Calculate technical indicators
            rsi = self._calculate_rsi(hist['Close'])
            sma_20 = hist['Close'].rolling(20).mean().iloc[-1] if len(hist) >= 20 else None
            sma_50 = hist['Close'].rolling(50).mean().iloc[-1] if len(hist) >= 50 else None

            return StockDataPackage(
                symbol=symbol,
                price=current_price,
                change=current_price - prev_close,
                change_percent=((current_price - prev_close) / prev_close) * 100,
                volume=int(hist['Volume'].iloc[-1]),
                market_cap=f"${info.get('marketCap', 0):,.0f}",
                pe_ratio=float(info.get('trailingPE', 0)) if info.get('trailingPE') else 0,
                week_52_high=float(info.get('fiftyTwoWeekHigh', current_price)),
                week_52_low=float(info.get('fiftyTwoWeekLow', current_price)),
                rsi=rsi,
                sma_20=float(sma_20) if sma_20 else None,
                sma_50=float(sma_50) if sma_50 else None,
                timestamp=datetime.now()
            )

        except Exception as e:
            self.logger.error(f"Error fetching stock data: {e}")
            return None

    async def _fetch_sentiment_data(self, symbol: str) -> Optional[SentimentPackage]:
        """Fetch and analyze sentiment data"""
        try:
            news_api_key = self.config.get("news_api_key")
            if not news_api_key:
                return self._generate_mock_sentiment(symbol)

            # Fetch news
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': f"{symbol} stock OR {symbol} earnings",
                'language': 'en',
                'sortBy': 'publishedAt',
                'from': (datetime.now() - timedelta(days=3)).isoformat(),
                'pageSize': 30,
                'apiKey': news_api_key
            }

            response = requests.get(url, params=params, timeout=10)
            if response.status_code != 200:
                return self._generate_mock_sentiment(symbol)

            articles = response.json().get('articles', [])
            if not articles:
                return self._generate_mock_sentiment(symbol)

            # Analyze sentiment
            sentiments = []
            headlines = []

            for article in articles[:15]:
                title = article.get('title', '')
                description = article.get('description', '') or ''

                if title and 'removed' not in title.lower():
                    text = f"{title} {description}"
                    blob = TextBlob(text)
                    sentiment = blob.sentiment.polarity
                    sentiments.append(sentiment)
                    headlines.append(title[:80])

            if not sentiments:
                return self._generate_mock_sentiment(symbol)

            avg_sentiment = sum(sentiments) / len(sentiments)

            # Determine trend (simplified)
            recent_sentiments = sentiments[:5]
            older_sentiments = sentiments[5:10] if len(sentiments) > 5 else sentiments

            if len(recent_sentiments) > 0 and len(older_sentiments) > 0:
                recent_avg = sum(recent_sentiments) / len(recent_sentiments)
                older_avg = sum(older_sentiments) / len(older_sentiments)

                if recent_avg > older_avg + 0.1:
                    trend = "improving"
                elif recent_avg < older_avg - 0.1:
                    trend = "declining"
                else:
                    trend = "stable"
            else:
                trend = "stable"

            return SentimentPackage(
                symbol=symbol,
                news_sentiment=avg_sentiment,
                social_sentiment=avg_sentiment * 0.8,  # Mock social sentiment
                article_count=len(sentiments),
                confidence=min(len(sentiments) / 15, 1.0),
                key_headlines=headlines[:5],
                sentiment_trend=trend,
                timestamp=datetime.now()
            )

        except Exception as e:
            self.logger.error(f"Error fetching sentiment: {e}")
            return self._generate_mock_sentiment(symbol)

    def _generate_mock_sentiment(self, symbol: str) -> SentimentPackage:
        """Generate mock sentiment data as fallback"""
        import random
        return SentimentPackage(
            symbol=symbol,
            news_sentiment=random.uniform(-0.3, 0.3),
            social_sentiment=random.uniform(-0.2, 0.4),
            article_count=random.randint(5, 20),
            confidence=0.6,
            key_headlines=[f"Mock news about {symbol} performance"],
            sentiment_trend=random.choice(["improving", "stable", "declining"]),
            timestamp=datetime.now()
        )

    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi.iloc[-1]) if not rsi.iloc[-1] != rsi.iloc[-1] else 50.0  # Handle NaN
        except:
            return 50.0  # Neutral RSI if calculation fails


# ============================================================================
# STEP 4: Technical Analysis Agent (Specialist)
# ============================================================================

class TechnicalAnalysisAgent(BaseAgent):
    """Specialist agent for technical analysis"""

    def __init__(self, config: Dict):
        super().__init__("TechnicalAnalyst", "technical_analysis")
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            openai_api_key=config["openai_api_key"]
        )

        self.analysis_prompt = PromptTemplate(
            input_variables=["symbol", "price", "change_percent", "volume", "rsi", "sma_20", "sma_50", "week_52_high",
                             "week_52_low"],
            template="""
You are a technical analysis expert. Analyze this stock data:

STOCK: {symbol}
Current Price: ${price:.2f}
Change: {change_percent:.2f}%
Volume: {volume:,}
RSI: {rsi}
20-day SMA: ${sma_20}
50-day SMA: ${sma_50}
52W High: ${week_52_high:.2f}
52W Low: ${week_52_low:.2f}

Provide technical analysis focusing on:
1. Price momentum and trend direction
2. Support/resistance levels
3. Volume analysis and significance
4. Technical indicators interpretation
5. Entry/exit points

Rate the technical outlook 1-10 (10=very bullish, 1=very bearish)
Provide confidence level 0.0-1.0

Format response as:
TECHNICAL_SCORE: [1-10]
CONFIDENCE: [0.0-1.0]
TREND: [BULLISH/BEARISH/NEUTRAL]
KEY_INSIGHTS: [3 bullet points]
ENTRY_POINTS: [Price levels]
WARNINGS: [Risk factors]
"""
        )

    async def handle_message(self, message: AgentMessage):
        """Handle analysis requests"""
        if message.message_type == MessageType.ANALYSIS_REQUEST:
            stock_data = message.content.get("stock_data")
            if stock_data:
                analysis = await self.analyze_technical(stock_data)

                await self.send_message(
                    recipient=message.sender,
                    message_type=MessageType.ANALYSIS_RESPONSE,
                    content={
                        "request_id": message.id,
                        "analysis_result": asdict(analysis)
                    }
                )

    async def analyze_technical(self, stock_data: Dict) -> AnalysisResult:
        """Perform technical analysis"""
        self.logger.info(f"📊 Technical analysis for {stock_data['symbol']}")

        try:
            # Format prompt with stock data
            prompt = self.analysis_prompt.format(
                symbol=stock_data["symbol"],
                price=stock_data["price"],
                change_percent=stock_data["change_percent"],
                volume=stock_data["volume"],
                rsi=stock_data.get("rsi", "N/A"),
                sma_20=stock_data.get("sma_20", "N/A"),
                sma_50=stock_data.get("sma_50", "N/A"),
                week_52_high=stock_data["week_52_high"],
                week_52_low=stock_data["week_52_low"]
            )

            # Get AI analysis
            response = await self.llm.apredict(prompt)

            # Parse response
            score, confidence, insights, recommendations, warnings = self._parse_technical_response(response)

            return AnalysisResult(
                agent_name=self.name,
                symbol=stock_data["symbol"],
                analysis_type="technical",
                score=score,
                confidence=confidence,
                key_insights=insights,
                recommendations=recommendations,
                warnings=warnings,
                raw_analysis=response,
                timestamp=datetime.now()
            )

        except Exception as e:
            self.logger.error(f"Technical analysis error: {e}")
            return AnalysisResult(
                agent_name=self.name,
                symbol=stock_data["symbol"],
                analysis_type="technical",
                score=5.0,
                confidence=0.3,
                key_insights=["Analysis failed"],
                recommendations=["Hold until analysis complete"],
                warnings=[f"Technical analysis error: {str(e)}"],
                raw_analysis="",
                timestamp=datetime.now()
            )

    def _parse_technical_response(self, response: str):
        """Parse the AI response"""
        lines = response.strip().split('\n')

        score = 5.0
        confidence = 0.5
        insights = []
        recommendations = []
        warnings = []

        for line in lines:
            line = line.strip()
            if line.startswith("TECHNICAL_SCORE:"):
                try:
                    score = float(line.split(":")[1].strip())
                except:
                    pass
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(":")[1].strip())
                except:
                    pass
            elif line.startswith("KEY_INSIGHTS:"):
                insights_text = line.split(":", 1)[1].strip()
                insights = [i.strip("• -") for i in insights_text.split('\n') if i.strip()]
            elif line.startswith("ENTRY_POINTS:"):
                rec_text = line.split(":", 1)[1].strip()
                recommendations.append(f"Entry points: {rec_text}")
            elif line.startswith("WARNINGS:"):
                warn_text = line.split(":", 1)[1].strip()
                warnings = [w.strip("• -") for w in warn_text.split('\n') if w.strip()]

        return score, confidence, insights[:3], recommendations[:3], warnings[:3]


# ============================================================================
# STEP 5: Sentiment Analysis Agent (Specialist)
# ============================================================================

class SentimentAnalysisAgent(BaseAgent):
    """Specialist agent for sentiment analysis"""

    def __init__(self, config: Dict):
        super().__init__("SentimentAnalyst", "sentiment_analysis")
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            openai_api_key=config["openai_api_key"]
        )

    async def handle_message(self, message: AgentMessage):
        """Handle sentiment analysis requests"""
        if message.message_type == MessageType.ANALYSIS_REQUEST:
            sentiment_data = message.content.get("sentiment_data")
            if sentiment_data:
                analysis = await self.analyze_sentiment(sentiment_data)

                await self.send_message(
                    recipient=message.sender,
                    message_type=MessageType.ANALYSIS_RESPONSE,
                    content={
                        "request_id": message.id,
                        "analysis_result": asdict(analysis)
                    }
                )

    async def analyze_sentiment(self, sentiment_data: Dict) -> AnalysisResult:
        """Perform sentiment analysis"""
        self.logger.info(f"📰 Sentiment analysis for {sentiment_data['symbol']}")

        try:
            prompt = f"""
You are a sentiment analysis expert. Analyze this market sentiment data:

STOCK: {sentiment_data['symbol']}
News Sentiment: {sentiment_data['news_sentiment']:.3f} (-1=Very Negative, +1=Very Positive)
Social Sentiment: {sentiment_data['social_sentiment']:.3f}
Article Count: {sentiment_data['article_count']}
Confidence: {sentiment_data['confidence']:.2f}
Sentiment Trend: {sentiment_data['sentiment_trend']}
Key Headlines: {'; '.join(sentiment_data['key_headlines'][:3])}

Analyze:
1. Overall market mood and psychology
2. Sentiment momentum and trend changes
3. News impact on investor behavior
4. Potential sentiment catalysts
5. Crowd psychology indicators

Rate the sentiment outlook 1-10 (10=very positive, 1=very negative)
Provide confidence level 0.0-1.0

Format response as:
SENTIMENT_SCORE: [1-10]
CONFIDENCE: [0.0-1.0]
MOOD: [BULLISH/BEARISH/NEUTRAL]
KEY_INSIGHTS: [3 bullet points]
CATALYSTS: [Positive/negative drivers]
WARNINGS: [Sentiment risks]
"""

            response = await self.llm.apredict(prompt)
            score, confidence, insights, recommendations, warnings = self._parse_sentiment_response(response)

            return AnalysisResult(
                agent_name=self.name,
                symbol=sentiment_data["symbol"],
                analysis_type="sentiment",
                score=score,
                confidence=confidence,
                key_insights=insights,
                recommendations=recommendations,
                warnings=warnings,
                raw_analysis=response,
                timestamp=datetime.now()
            )

        except Exception as e:
            self.logger.error(f"Sentiment analysis error: {e}")
            return AnalysisResult(
                agent_name=self.name,
                symbol=sentiment_data["symbol"],
                analysis_type="sentiment",
                score=5.0,
                confidence=0.3,
                key_insights=["Sentiment analysis failed"],
                recommendations=["Monitor news carefully"],
                warnings=[f"Sentiment analysis error: {str(e)}"],
                raw_analysis="",
                timestamp=datetime.now()
            )

    def _parse_sentiment_response(self, response: str):
        """Parse sentiment analysis response"""
        lines = response.strip().split('\n')

        score = 5.0
        confidence = 0.5
        insights = []
        recommendations = []
        warnings = []

        for line in lines:
            line = line.strip()
            if line.startswith("SENTIMENT_SCORE:"):
                try:
                    score = float(line.split(":")[1].strip())
                except:
                    pass
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(":")[1].strip())
                except:
                    pass
            elif line.startswith("KEY_INSIGHTS:"):
                insights_text = line.split(":", 1)[1].strip()
                insights = [i.strip("• -") for i in insights_text.split('\n') if i.strip()]
            elif line.startswith("CATALYSTS:"):
                cat_text = line.split(":", 1)[1].strip()
                recommendations.append(f"Catalysts: {cat_text}")
            elif line.startswith("WARNINGS:"):
                warn_text = line.split(":", 1)[1].strip()
                warnings = [w.strip("• -") for w in warn_text.split('\n') if w.strip()]

        return score, confidence, insights[:3], recommendations[:3], warnings[:3]


# ============================================================================
# STEP 6: Portfolio Manager Agent (Coordinator & Decision Maker)
# ============================================================================

class PortfolioManagerAgent(BaseAgent):
    """Master agent that coordinates specialists and makes final decisions"""

    def __init__(self, config: Dict):
        super().__init__("PortfolioManager", "portfolio_management")
        self.llm = ChatOpenAI(
            model="gpt-4",  # Use GPT-4 for final decisions
            temperature=0.1,
            openai_api_key=config["openai_api_key"]
        )

        # Create specialist agents
        self.data_collector = DataCollectionAgent(config)
        self.technical_analyst = TechnicalAnalysisAgent(config)
        self.sentiment_analyst = SentimentAnalysisAgent(config)

        # Track pending analyses
        self.pending_analyses = {}

    async def analyze_stock(self, symbol: str) -> Dict:
        """Orchestrate complete multi-agent analysis"""
        self.logger.info(f"🎯 Starting multi-agent analysis for {symbol}")

        try:
            # Step 1: Request data collection
            data_package = await self.data_collector.collect_all_data(symbol, ["stock", "sentiment"])

            if not data_package:
                return self._create_error_result(symbol, "Data collection failed")

            # Step 2: Send analysis requests to specialists (in parallel)
            technical_task = self._request_technical_analysis(data_package.get("stock_data"))
            sentiment_task = self._request_sentiment_analysis(data_package.get("sentiment_data"))

            # Wait for both analyses
            technical_result, sentiment_result = await asyncio.gather(technical_task, sentiment_task)

            # Step 3: Synthesize results into final recommendation
            final_recommendation = await self._synthesize_analysis(
                symbol, technical_result, sentiment_result
            )

            # Step 4: Package complete result
            complete_result = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "data_package": data_package,
                "specialist_analyses": {
                    "technical": technical_result,
                    "sentiment": sentiment_result
                },
                "final_recommendation": final_recommendation,
                "status": "completed"
            }

            self.logger.info(f"✅ Multi-agent analysis completed for {symbol}")
            return complete_result

        except Exception as e:
            self.logger.error(f"Error in multi-agent analysis: {e}")
            return self._create_error_result(symbol, str(e))

    async def _request_technical_analysis(self, stock_data: Dict) -> AnalysisResult:
        """Request technical analysis from specialist"""
        if not stock_data:
            return None

        return await self.technical_analyst.analyze_technical(stock_data)

    async def _request_sentiment_analysis(self, sentiment_data: Dict) -> AnalysisResult:
        """Request sentiment analysis from specialist"""
        if not sentiment_data:
            return None

        return await self.sentiment_analyst.analyze_sentiment(sentiment_data)

    async def _synthesize_analysis(self, symbol: str, technical_result: AnalysisResult,
                                   sentiment_result: AnalysisResult) -> Dict:
        """Synthesize specialist analyses into final recommendation"""
        self.logger.info(f"🧠 Synthesizing analysis for {symbol}")

        try:
            # Create synthesis prompt
            prompt = f"""
You are a portfolio manager synthesizing specialist analyses for {symbol}:

TECHNICAL ANALYSIS ({technical_result.confidence:.2f} confidence):
Score: {technical_result.score}/10
Key Insights: {'; '.join(technical_result.key_insights)}
Recommendations: {'; '.join(technical_result.recommendations)}
Warnings: {'; '.join(technical_result.warnings)}

SENTIMENT ANALYSIS ({sentiment_result.confidence:.2f} confidence):
Score: {sentiment_result.score}/10  
Key Insights: {'; '.join(sentiment_result.key_insights)}
Recommendations: {'; '.join(sentiment_result.recommendations)}
Warnings: {'; '.join(sentiment_result.warnings)}

Synthesize these expert opinions considering:
- Agreement/disagreement between specialists
- Confidence levels and reliability
- Risk-adjusted expected returns
- Portfolio impact and position sizing

Provide final recommendation:
RECOMMENDATION: [BUY/HOLD/SELL]
CONFIDENCE: [1-10]
POSITION_SIZE: [SMALL/MEDIUM/LARGE]
TIME_HORIZON: [SHORT/MEDIUM/LONG]
REASONING: [Key synthesis points]
RISK_FACTORS: [Top 3 risks to monitor]
"""

            response = await self.llm.apredict(prompt)

            # Parse final recommendation
            recommendation = self._parse_final_recommendation(response, technical_result, sentiment_result)

            return recommendation

        except Exception as e:
            self.logger.error(f"Error synthesizing analysis: {e}")
            return {
                "recommendation": "HOLD",
                "confidence": 3,
                "reasoning": f"Synthesis failed: {str(e)}",
                "risk_factors": ["Analysis synthesis error"]
            }

    def _parse_final_recommendation(self, response: str, technical: AnalysisResult, sentiment: AnalysisResult) -> Dict:
        """Parse the final synthesis response"""
        lines = response.strip().split('\n')

        recommendation = "HOLD"
        confidence = 5
        position_size = "SMALL"
        time_horizon = "MEDIUM"
        reasoning = ""
        risk_factors = []

        for line in lines:
            line = line.strip()
            if line.startswith("RECOMMENDATION:"):
                rec = line.split(":", 1)[1].strip().upper()
                if rec in ["BUY", "HOLD", "SELL"]:
                    recommendation = rec
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = int(line.split(":", 1)[1].strip())
                except:
                    pass
            elif line.startswith("POSITION_SIZE:"):
                size = line.split(":", 1)[1].strip().upper()
                if size in ["SMALL", "MEDIUM", "LARGE"]:
                    position_size = size
            elif line.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()
            elif line.startswith("RISK_FACTORS:"):
                risk_text = line.split(":", 1)[1].strip()
                risk_factors = [r.strip("• -") for r in risk_text.split(';') if r.strip()]

        # Calculate combined score
        technical_weight = 0.6
        sentiment_weight = 0.4
        combined_score = (technical.score * technical_weight + sentiment.score * sentiment_weight)

        return {
            "recommendation": recommendation,
            "confidence": confidence,
            "position_size": position_size,
            "time_horizon": time_horizon,
            "reasoning": reasoning or response[:200],
            "risk_factors": risk_factors[:3],
            "combined_score": round(combined_score, 1),
            "specialist_agreement": abs(technical.score - sentiment.score) < 2.0,
            "synthesis_quality": "HIGH" if abs(technical.score - sentiment.score) < 2.0 else "MODERATE"
        }

    def _create_error_result(self, symbol: str, error_msg: str) -> Dict:
        """Create error result structure"""
        return {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "status": "error",
            "error": error_msg,
            "final_recommendation": {
                "recommendation": "HOLD",
                "confidence": 1,
                "reasoning": f"Analysis failed: {error_msg}",
                "risk_factors": ["System error", "Data unavailable"]
            }
        }


# ============================================================================
# STEP 7: Multi-Agent System Orchestrator
# ============================================================================

class MultiAgentStockSystem:
    """Main system that orchestrates the multi-agent analysis"""

    def __init__(self):
        self.config = self._load_config()
        self.portfolio_manager = PortfolioManagerAgent(self.config)
        self.logger = logging.getLogger("MultiAgentSystem")

        self.logger.info("🚀 Multi-Agent Stock Analysis System initialized")

    def _load_config(self) -> Dict:
        """Load configuration"""
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            print("❌ OPENAI_API_KEY environment variable required!")
            sys.exit(1)

        return {
            "openai_api_key": openai_key,
            "news_api_key": os.getenv('NEWS_API_KEY', ''),
        }

    async def analyze_portfolio(self, symbols: List[str]) -> List[Dict]:
        """Analyze multiple stocks using multi-agent system"""
        self.logger.info(f"📊 Starting multi-agent analysis of {len(symbols)} stocks")

        results = []

        for i, symbol in enumerate(symbols, 1):
            print(f"\n{'=' * 60}")
            print(f"[{i}/{len(symbols)}] MULTI-AGENT ANALYSIS: {symbol}")
            print(f"{'=' * 60}")

            # Perform multi-agent analysis
            result = await self.portfolio_manager.analyze_stock(symbol)
            results.append(result)

            # Display quick summary
            if result["status"] == "completed":
                final_rec = result["final_recommendation"]
                emoji = {"BUY": "🟢", "HOLD": "🟡", "SELL": "🔴"}[final_rec["recommendation"]]

                print(f"{emoji} FINAL: {final_rec['recommendation']} (Confidence: {final_rec['confidence']}/10)")
                print(f"📊 Technical Score: {result['specialist_analyses']['technical']['score']}/10")
                print(f"📰 Sentiment Score: {result['specialist_analyses']['sentiment']['score']}/10")
                print(f"🎯 Combined Score: {final_rec['combined_score']}/10")
                print(f"🤝 Specialist Agreement: {final_rec['specialist_agreement']}")
                print(f"💡 Reasoning: {final_rec['reasoning'][:100]}...")
            else:
                print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")

        return results

    def generate_report(self, results: List[Dict]) -> str:
        """Generate comprehensive multi-agent analysis report"""
        report = f"""
🤖 MULTI-AGENT STOCK ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 80}

PORTFOLIO SUMMARY:
• Total stocks analyzed: {len(results)}
• Successful analyses: {sum(1 for r in results if r['status'] == 'completed')}
• BUY recommendations: {sum(1 for r in results if r.get('final_recommendation', {}).get('recommendation') == 'BUY')}
• HOLD recommendations: {sum(1 for r in results if r.get('final_recommendation', {}).get('recommendation') == 'HOLD')}
• SELL recommendations: {sum(1 for r in results if r.get('final_recommendation', {}).get('recommendation') == 'SELL')}

{'=' * 80}
DETAILED MULTI-AGENT ANALYSIS:
"""

        for result in results:
            if result['status'] != 'completed':
                continue

            symbol = result['symbol']
            final_rec = result['final_recommendation']
            tech_analysis = result['specialist_analyses']['technical']
            sent_analysis = result['specialist_analyses']['sentiment']

            emoji = {"BUY": "🟢", "HOLD": "🟡", "SELL": "🔴"}[final_rec["recommendation"]]

            report += f"""
{emoji} {symbol} - {final_rec['recommendation']} (Final Confidence: {final_rec['confidence']}/10)
┌─ PORTFOLIO MANAGER SYNTHESIS:
│  ├─ Combined Score: {final_rec['combined_score']}/10
│  ├─ Position Size: {final_rec['position_size']}
│  ├─ Time Horizon: {final_rec['time_horizon']}
│  ├─ Specialist Agreement: {'✅ YES' if final_rec['specialist_agreement'] else '⚠️ NO'}
│  └─ Reasoning: {final_rec['reasoning']}
│
├─ TECHNICAL ANALYST FINDINGS:
│  ├─ Technical Score: {tech_analysis['score']}/10 (Confidence: {tech_analysis['confidence']:.2f})
│  ├─ Key Insights: {'; '.join(tech_analysis['key_insights'])}
│  └─ Technical Warnings: {'; '.join(tech_analysis['warnings'])}
│
├─ SENTIMENT ANALYST FINDINGS:
│  ├─ Sentiment Score: {sent_analysis['score']}/10 (Confidence: {sent_analysis['confidence']:.2f})
│  ├─ Key Insights: {'; '.join(sent_analysis['key_insights'])}
│  └─ Sentiment Warnings: {'; '.join(sent_analysis['warnings'])}
│
└─ KEY RISK FACTORS:
   {chr(10).join([f'   • {risk}' for risk in final_rec['risk_factors']])}

"""

        report += f"""
{'=' * 80}
MULTI-AGENT SYSTEM PERFORMANCE:
• Specialist agreement rate: {sum(1 for r in results if r.get('final_recommendation', {}).get('specialist_agreement', False)) / len([r for r in results if r['status'] == 'completed']) * 100:.1f}%
• Average technical confidence: {sum(r['specialist_analyses']['technical']['confidence'] for r in results if r['status'] == 'completed') / len([r for r in results if r['status'] == 'completed']):.2f}
• Average sentiment confidence: {sum(r['specialist_analyses']['sentiment']['confidence'] for r in results if r['status'] == 'completed') / len([r for r in results if r['status'] == 'completed']):.2f}

DISCLAIMER: Multi-agent analysis for educational purposes only.
Always conduct independent research before making investment decisions.
{'=' * 80}
"""
        return report


# ============================================================================
# STEP 8: Main Application
# ============================================================================

async def main():
    """Main application entry point"""
    print("🤖 Multi-Agent Stock Analysis System")
    print("=" * 60)

    # Default portfolio
    default_symbols = ["AAPL", "GOOGL", "MSFT"]

    # Get symbols from command line or use default
    if len(sys.argv) > 1:
        symbols = [s.upper().strip() for s in sys.argv[1].split(',')]
    else:
        print(f"Using default portfolio: {', '.join(default_symbols)}")
        print("To analyze different stocks, run: python multi_agent_stock_system.py AAPL,GOOGL,TSLA")
        symbols = default_symbols

    # Initialize multi-agent system
    system = MultiAgentStockSystem()

    # Run analysis
    results = await system.analyze_portfolio(symbols)

    # Generate and display report
    report = system.generate_report(results)
    print("\n" + "=" * 100)
    print(report)

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save detailed JSON results
    with open(f"multi_agent_analysis_{timestamp}.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Save report
    with open(f"multi_agent_report_{timestamp}.txt", 'w') as f:
        f.write(report)

    print(f"\n💾 Results saved:")
    print(f"   • multi_agent_analysis_{timestamp}.json")
    print(f"   • multi_agent_report_{timestamp}.txt")


if __name__ == "__main__":
    asyncio.run(main())