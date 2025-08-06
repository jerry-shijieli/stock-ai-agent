# multi_agent_system.py
"""
Next Level: Multi-Agent System
Each agent has specialized expertise and they collaborate
"""

from abc import ABC, abstractmethod
from typing import Dict, List
from dataclasses import dataclass
import asyncio


@dataclass
class AgentMessage:
    """Messages passed between agents"""
    sender: str
    recipient: str
    content: Dict
    message_type: str
    timestamp: str


class BaseAgent(ABC):
    """Base class for all specialist agents"""

    def __init__(self, name: str, expertise: str):
        self.name = name
        self.expertise = expertise
        self.inbox = []
        self.memory = {}

    @abstractmethod
    async def process(self, data: Dict) -> Dict:
        """Each agent implements their specialized processing"""
        pass

    async def send_message(self, recipient: str, content: Dict, msg_type: str):
        """Send message to another agent"""
        message = AgentMessage(
            sender=self.name,
            recipient=recipient,
            content=content,
            message_type=msg_type,
            timestamp=datetime.now().isoformat()
        )
        # In real system, this would go through message queue
        return message


class TechnicalAnalysisAgent(BaseAgent):
    """Specialist in technical indicators and price patterns"""

    def __init__(self):
        super().__init__("TechnicalAnalyst", "Technical Analysis")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    async def process(self, stock_data: Dict) -> Dict:
        """Analyze technical indicators"""
        prompt = f"""
        You are a technical analysis expert. Analyze these indicators:

        Price: ${stock_data['price']:.2f}
        RSI: {stock_data.get('rsi', 'N/A')}
        MACD: {stock_data.get('macd', 'N/A')}
        Volume: {stock_data['volume']:,}

        Provide technical analysis focusing on:
        - Price momentum and trends
        - Support/resistance levels
        - Volume analysis
        - Technical buy/sell signals

        Return: TECHNICAL_SCORE (1-10), KEY_SIGNALS, PRICE_TARGET
        """

        response = await self.llm.apredict(prompt)

        return {
            "agent": self.name,
            "analysis": response,
            "confidence": self._calculate_technical_confidence(stock_data)
        }


class SentimentAnalysisAgent(BaseAgent):
    """Specialist in news and social sentiment"""

    def __init__(self):
        super().__init__("SentimentAnalyst", "Sentiment Analysis")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    async def process(self, sentiment_data: Dict) -> Dict:
        """Analyze market sentiment"""
        prompt = f"""
        You are a sentiment analysis expert. Analyze this data:

        News Sentiment: {sentiment_data['news_sentiment']:.3f}
        Article Count: {sentiment_data['article_count']}
        Key Headlines: {sentiment_data['key_headlines']}

        Provide sentiment analysis focusing on:
        - Market psychology and mood
        - News impact assessment
        - Sentiment momentum and shifts
        - Crowd behavior indicators

        Return: SENTIMENT_SCORE (1-10), MARKET_MOOD, CATALYST_EVENTS
        """

        response = await self.llm.apredict(prompt)

        return {
            "agent": self.name,
            "analysis": response,
            "confidence": sentiment_data.get('confidence', 0.5)
        }


class RiskAnalysisAgent(BaseAgent):
    """Specialist in risk assessment and portfolio management"""

    def __init__(self):
        super().__init__("RiskAnalyst", "Risk Management")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    async def process(self, combined_data: Dict) -> Dict:
        """Analyze risk factors"""
        prompt = f"""
        You are a risk management expert. Assess risks for this investment:

        Stock: {combined_data['symbol']}
        Price: ${combined_data['price']:.2f}
        Technical Analysis: {combined_data.get('technical_summary', 'N/A')}
        Sentiment Analysis: {combined_data.get('sentiment_summary', 'N/A')}

        Provide risk analysis focusing on:
        - Downside risk assessment
        - Volatility indicators
        - Portfolio impact
        - Position sizing recommendations

        Return: RISK_SCORE (1-10), KEY_RISKS, MAX_POSITION_SIZE
        """

        response = await self.llm.apredict(prompt)

        return {
            "agent": self.name,
            "analysis": response,
            "risk_level": self._calculate_risk_level(combined_data)
        }


class CoordinatorAgent(BaseAgent):
    """Master agent that coordinates all specialists"""

    def __init__(self):
        super().__init__("Coordinator", "Analysis Coordination")
        self.technical_agent = TechnicalAnalysisAgent()
        self.sentiment_agent = SentimentAnalysisAgent()
        self.risk_agent = RiskAnalysisAgent()
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.1)  # Use GPT-4 for final decision

    async def analyze_stock(self, symbol: str, stock_data: Dict, sentiment_data: Dict) -> Dict:
        """Coordinate analysis across all specialist agents"""

        print(f"🎯 Coordinator: Starting multi-agent analysis for {symbol}")

        # Step 1: Run specialist analyses in parallel
        technical_task = self.technical_agent.process(stock_data)
        sentiment_task = self.sentiment_agent.process(sentiment_data)

        technical_result, sentiment_result = await asyncio.gather(
            technical_task, sentiment_task
        )

        # Step 2: Combine results for risk analysis
        combined_data = {
            "symbol": symbol,
            "price": stock_data["price"],
            "technical_summary": technical_result["analysis"][:200],
            "sentiment_summary": sentiment_result["analysis"][:200]
        }

        risk_result = await self.risk_agent.process(combined_data)

        # Step 3: Final synthesis and decision
        final_analysis = await self._synthesize_analysis(
            symbol, technical_result, sentiment_result, risk_result
        )

        return {
            "symbol": symbol,
            "recommendation": final_analysis["recommendation"],
            "confidence": final_analysis["confidence"],
            "reasoning": final_analysis["reasoning"],
            "specialist_insights": {
                "technical": technical_result,
                "sentiment": sentiment_result,
                "risk": risk_result
            }
        }

    async def _synthesize_analysis(self, symbol: str, technical: Dict, sentiment: Dict, risk: Dict) -> Dict:
        """Final synthesis using all specialist inputs"""

        prompt = f"""
        You are the master analyst coordinating specialist insights for {symbol}:

        TECHNICAL ANALYSIS:
        {technical['analysis']}

        SENTIMENT ANALYSIS:
        {sentiment['analysis']}

        RISK ANALYSIS:
        {risk['analysis']}

        Your job is to synthesize these expert opinions into a final recommendation.
        Consider:
        - Agreement/disagreement between specialists
        - Confidence levels of each analysis
        - Risk-adjusted expected returns

        Provide final decision in this format:
        RECOMMENDATION: [BUY/HOLD/SELL]
        CONFIDENCE: [1-10]
        REASONING: [Synthesis of all expert insights]
        KEY_FACTORS: [Top 3 decision factors]
        """

        response = await self.llm.apredict(prompt)

        # Parse response (simplified for example)
        return {
            "recommendation": "HOLD",  # Would parse from response
            "confidence": 7,  # Would parse from response
            "reasoning": response,
            "synthesis_quality": "HIGH"
        }


# Example usage
async def run_multi_agent_analysis():
    """Example of how to use the multi-agent system"""

    coordinator = CoordinatorAgent()

    # Sample data (in real system, would come from data fetchers)
    stock_data = {
        "symbol": "AAPL",
        "price": 186.40,
        "volume": 45000000,
        "rsi": 58.3,
        "macd": 1.24
    }

    sentiment_data = {
        "news_sentiment": 0.15,
        "article_count": 12,
        "key_headlines": ["Apple reports strong Q4", "New iPhone sales exceed expectations"],
        "confidence": 0.8
    }

    # Run multi-agent analysis
    result = await coordinator.analyze_stock("AAPL", stock_data, sentiment_data)

    print("🤖 MULTI-AGENT ANALYSIS RESULT:")
    print(f"Symbol: {result['symbol']}")
    print(f"Recommendation: {result['recommendation']}")
    print(f"Confidence: {result['confidence']}/10")
    print(f"Reasoning: {result['reasoning'][:200]}...")

    # You can also access individual specialist insights
    print("\n📊 SPECIALIST INSIGHTS:")
    for specialist, analysis in result['specialist_insights'].items():
        print(f"{specialist.title()}: {analysis['analysis'][:100]}...")


if __name__ == "__main__":
    asyncio.run(run_multi_agent_analysis())