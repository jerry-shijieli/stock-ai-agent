# memory_learning_system.py
"""
Memory & Learning System for AI Agents

This adds persistent memory and learning capabilities to your multi-agent system:
1. Vector Memory: Semantic search through historical analyses
2. SQL Memory: Structured performance tracking
3. Learning Engine: Improves predictions based on accuracy
4. Pattern Recognition: Identifies successful analysis patterns
5. Dynamic Prompts: Adjusts prompts based on performance

Run with: python memory_learning_system.py
"""

import os
import sys
import json
import sqlite3
import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
import logging
import pickle
from pathlib import Path

# External dependencies
try:
    import yfinance as yf
    import requests
    import numpy as np
    from textblob import TextBlob
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain.prompts import PromptTemplate
    from langchain.schema import Document
    from langchain.vectorstores import Chroma
    from sklearn.metrics import accuracy_score, mean_absolute_error
    import chromadb
    from chromadb.config import Settings
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("Install with: pip install yfinance requests textblob langchain-openai chromadb scikit-learn")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# ============================================================================
# STEP 1: Memory Data Structures
# ============================================================================

@dataclass
class AnalysisMemory:
    """Memory entry for storing analysis results"""
    id: str
    symbol: str
    analysis_date: datetime
    agent_name: str
    analysis_type: str
    prediction: Dict  # What the agent predicted
    actual_outcome: Optional[Dict] = None  # What actually happened
    accuracy_score: Optional[float] = None
    confidence: float = 0.0
    market_conditions: Dict = None
    raw_data: Dict = None


@dataclass
class PerformanceMetrics:
    """Performance tracking for agents"""
    agent_name: str
    total_predictions: int
    correct_predictions: int
    accuracy_rate: float
    avg_confidence: float
    recent_performance: List[float]  # Last 10 predictions
    improvement_trend: str  # "improving", "declining", "stable"
    last_updated: datetime


@dataclass
class MarketPattern:
    """Identified market patterns"""
    pattern_id: str
    pattern_type: str  # "bullish_momentum", "sentiment_reversal", etc.
    conditions: Dict  # Market conditions when pattern occurs
    success_rate: float
    sample_size: int
    confidence: float
    discovered_date: datetime


# ============================================================================
# STEP 2: Vector Memory System (Semantic Search)
# ============================================================================

class VectorMemorySystem:
    """Vector database for semantic search of historical analyses"""

    def __init__(self, persist_directory: str = "memory/vector_store"):
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings()
        self.logger = logging.getLogger("VectorMemory")

        # Create directory if it doesn't exist
        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="stock_analyses",
            metadata={"description": "Historical stock analysis memories"}
        )

        self.logger.info("🧠 Vector memory system initialized")

    async def store_analysis_memory(self, memory: AnalysisMemory):
        """Store analysis in vector database for semantic search"""
        try:
            # Create document content for embedding
            content = f"""
            Symbol: {memory.symbol}
            Analysis Type: {memory.analysis_type}
            Agent: {memory.agent_name}
            Prediction: {memory.prediction.get('recommendation', 'N/A')}
            Confidence: {memory.confidence}
            Market Conditions: {memory.market_conditions}
            Analysis Date: {memory.analysis_date.strftime('%Y-%m-%d')}
            """

            # Create metadata
            metadata = {
                "symbol": memory.symbol,
                "agent_name": memory.agent_name,
                "analysis_type": memory.analysis_type,
                "recommendation": memory.prediction.get('recommendation', 'HOLD'),
                "confidence": memory.confidence,
                "analysis_date": memory.analysis_date.isoformat(),
                "accuracy_score": memory.accuracy_score or 0.0
            }

            # Add to vector database
            self.collection.add(
                documents=[content],
                metadatas=[metadata],
                ids=[memory.id]
            )

            self.logger.info(f"💾 Stored analysis memory: {memory.id}")

        except Exception as e:
            self.logger.error(f"Error storing analysis memory: {e}")

    async def find_similar_analyses(self, query: str, symbol: str = None,
                                    agent_name: str = None, limit: int = 5) -> List[Dict]:
        """Find similar historical analyses using semantic search"""
        try:
            # Build where clause for filtering
            where_clause = {}
            if symbol:
                where_clause["symbol"] = symbol
            if agent_name:
                where_clause["agent_name"] = agent_name

            # Perform semantic search
            results = self.collection.query(
                query_texts=[query],
                n_results=limit,
                where=where_clause if where_clause else None
            )

            # Format results
            similar_analyses = []
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i]

                similar_analyses.append({
                    "content": doc,
                    "metadata": metadata,
                    "similarity_score": 1 - distance,  # Convert distance to similarity
                    "id": results['ids'][0][i]
                })

            self.logger.info(f"🔍 Found {len(similar_analyses)} similar analyses")
            return similar_analyses

        except Exception as e:
            self.logger.error(f"Error searching similar analyses: {e}")
            return []

    async def get_context_for_analysis(self, symbol: str, analysis_type: str) -> str:
        """Get relevant historical context for current analysis"""
        query = f"Analysis of {symbol} {analysis_type} market conditions predictions"

        similar = await self.find_similar_analyses(
            query=query,
            symbol=symbol,
            limit=3
        )

        if not similar:
            return "No similar historical analyses found."

        context = f"Historical Context for {symbol}:\n"
        for i, analysis in enumerate(similar, 1):
            metadata = analysis['metadata']
            context += f"""
{i}. Previous Analysis ({metadata['analysis_date'][:10]}):
   - Agent: {metadata['agent_name']}
   - Recommendation: {metadata['recommendation']}
   - Confidence: {metadata['confidence']:.2f}
   - Accuracy: {metadata['accuracy_score']:.2f}
   - Similarity: {analysis['similarity_score']:.2f}
"""

        return context


# ============================================================================
# STEP 3: SQL Memory System (Structured Data)
# ============================================================================

class SQLMemorySystem:
    """SQL database for structured performance tracking"""

    def __init__(self, db_path: str = "memory/agent_memory.db"):
        self.db_path = db_path
        self.logger = logging.getLogger("SQLMemory")

        # Create directory if it doesn't exist
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._initialize_database()
        self.logger.info("🗄️ SQL memory system initialized")

    def _initialize_database(self):
        """Create database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Analysis history table
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS analysis_history
                       (
                           id
                           TEXT
                           PRIMARY
                           KEY,
                           symbol
                           TEXT
                           NOT
                           NULL,
                           analysis_date
                           TEXT
                           NOT
                           NULL,
                           agent_name
                           TEXT
                           NOT
                           NULL,
                           analysis_type
                           TEXT
                           NOT
                           NULL,
                           prediction_data
                           TEXT
                           NOT
                           NULL, -- JSON
                           actual_outcome_data
                           TEXT, -- JSON
                           accuracy_score
                           REAL,
                           confidence
                           REAL
                           NOT
                           NULL,
                           market_conditions
                           TEXT, -- JSON
                           created_at
                           TIMESTAMP
                           DEFAULT
                           CURRENT_TIMESTAMP
                       )
                       ''')

        # Performance metrics table
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS agent_performance
                       (
                           agent_name
                           TEXT
                           PRIMARY
                           KEY,
                           total_predictions
                           INTEGER
                           DEFAULT
                           0,
                           correct_predictions
                           INTEGER
                           DEFAULT
                           0,
                           accuracy_rate
                           REAL
                           DEFAULT
                           0.0,
                           avg_confidence
                           REAL
                           DEFAULT
                           0.0,
                           recent_performance
                           TEXT, -- JSON array of recent scores
                           improvement_trend
                           TEXT
                           DEFAULT
                           'stable',
                           last_updated
                           TIMESTAMP
                           DEFAULT
                           CURRENT_TIMESTAMP
                       )
                       ''')

        # Market patterns table
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS market_patterns
                       (
                           pattern_id
                           TEXT
                           PRIMARY
                           KEY,
                           pattern_type
                           TEXT
                           NOT
                           NULL,
                           conditions
                           TEXT
                           NOT
                           NULL, -- JSON
                           success_rate
                           REAL
                           NOT
                           NULL,
                           sample_size
                           INTEGER
                           NOT
                           NULL,
                           confidence
                           REAL
                           NOT
                           NULL,
                           discovered_date
                           TEXT
                           NOT
                           NULL,
                           last_seen
                           TEXT
                           NOT
                           NULL
                       )
                       ''')

        # Price tracking table (for calculating accuracy)
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS price_tracking
                       (
                           symbol
                           TEXT
                           NOT
                           NULL,
                           date
                           TEXT
                           NOT
                           NULL,
                           price
                           REAL
                           NOT
                           NULL,
                           volume
                           INTEGER,
                           created_at
                           TIMESTAMP
                           DEFAULT
                           CURRENT_TIMESTAMP,
                           PRIMARY
                           KEY
                       (
                           symbol,
                           date
                       )
                           )
                       ''')

        conn.commit()
        conn.close()

    async def store_analysis(self, memory: AnalysisMemory):
        """Store analysis in SQL database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('''
            INSERT OR REPLACE INTO analysis_history 
            (id, symbol, analysis_date, agent_name, analysis_type, 
             prediction_data, actual_outcome_data, accuracy_score, 
             confidence, market_conditions)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                memory.id,
                memory.symbol,
                memory.analysis_date.isoformat(),
                memory.agent_name,
                memory.analysis_type,
                json.dumps(memory.prediction),
                json.dumps(memory.actual_outcome) if memory.actual_outcome else None,
                memory.accuracy_score,
                memory.confidence,
                json.dumps(memory.market_conditions) if memory.market_conditions else None
            ))

            conn.commit()
            self.logger.info(f"💾 Stored analysis in SQL: {memory.id}")

        except Exception as e:
            self.logger.error(f"Error storing analysis: {e}")
        finally:
            conn.close()

    async def update_performance_metrics(self, agent_name: str):
        """Update performance metrics for an agent"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Calculate current metrics
            cursor.execute('''
                           SELECT accuracy_score, confidence
                           FROM analysis_history
                           WHERE agent_name = ?
                             AND accuracy_score IS NOT NULL
                           ORDER BY analysis_date DESC LIMIT 50
                           ''', (agent_name,))

            results = cursor.fetchall()

            if not results:
                return

            # Calculate metrics
            total_predictions = len(results)
            accuracy_scores = [r[0] for r in results if r[0] is not None]
            confidences = [r[1] for r in results]

            avg_accuracy = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            # Recent performance (last 10)
            recent_performance = accuracy_scores[:10]

            # Determine trend
            if len(recent_performance) >= 5:
                recent_avg = sum(recent_performance[:5]) / 5
                older_avg = sum(recent_performance[5:10]) / len(recent_performance[5:10]) if len(
                    recent_performance) > 5 else recent_avg

                if recent_avg > older_avg + 0.1:
                    trend = "improving"
                elif recent_avg < older_avg - 0.1:
                    trend = "declining"
                else:
                    trend = "stable"
            else:
                trend = "stable"

            # Update performance table
            cursor.execute('''
            INSERT OR REPLACE INTO agent_performance 
            (agent_name, total_predictions, correct_predictions, accuracy_rate, 
             avg_confidence, recent_performance, improvement_trend, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                agent_name,
                total_predictions,
                len([a for a in accuracy_scores if a > 0.6]),  # Consider >0.6 as "correct"
                avg_accuracy,
                avg_confidence,
                json.dumps(recent_performance),
                trend,
                datetime.now().isoformat()
            ))

            conn.commit()
            self.logger.info(f"📊 Updated performance metrics for {agent_name}")

        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
        finally:
            conn.close()

    async def get_agent_performance(self, agent_name: str) -> Optional[PerformanceMetrics]:
        """Get performance metrics for an agent"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('''
                           SELECT *
                           FROM agent_performance
                           WHERE agent_name = ?
                           ''', (agent_name,))

            result = cursor.fetchone()

            if result:
                return PerformanceMetrics(
                    agent_name=result[0],
                    total_predictions=result[1],
                    correct_predictions=result[2],
                    accuracy_rate=result[3],
                    avg_confidence=result[4],
                    recent_performance=json.loads(result[5]) if result[5] else [],
                    improvement_trend=result[6],
                    last_updated=datetime.fromisoformat(result[7])
                )

            return None

        except Exception as e:
            self.logger.error(f"Error getting agent performance: {e}")
            return None
        finally:
            conn.close()

    async def store_price_data(self, symbol: str, price: float, volume: int = 0):
        """Store price data for accuracy calculations"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('''
            INSERT OR REPLACE INTO price_tracking (symbol, date, price, volume)
            VALUES (?, ?, ?, ?)
            ''', (symbol, datetime.now().date().isoformat(), price, volume))

            conn.commit()

        except Exception as e:
            self.logger.error(f"Error storing price data: {e}")
        finally:
            conn.close()


# ============================================================================
# STEP 4: Learning Engine
# ============================================================================

class LearningEngine:
    """Engine that learns from past performance and improves agents"""

    def __init__(self, vector_memory: VectorMemorySystem, sql_memory: SQLMemorySystem):
        self.vector_memory = vector_memory
        self.sql_memory = sql_memory
        self.logger = logging.getLogger("LearningEngine")

        # Learning parameters
        self.learning_config = {
            "min_samples_for_learning": 10,
            "accuracy_threshold": 0.6,
            "confidence_weight": 0.3,
            "recency_weight": 0.4
        }

        self.logger.info("🧠 Learning engine initialized")

    async def calculate_prediction_accuracy(self, analysis_id: str, days_to_wait: int = 7) -> float:
        """Calculate accuracy of a prediction after waiting period"""
        try:
            # Get the original analysis
            conn = sqlite3.connect(self.sql_memory.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                           SELECT symbol, analysis_date, prediction_data
                           FROM analysis_history
                           WHERE id = ?
                           ''', (analysis_id,))

            result = cursor.fetchone()
            conn.close()

            if not result:
                return 0.0

            symbol, analysis_date_str, prediction_json = result
            analysis_date = datetime.fromisoformat(analysis_date_str)
            prediction = json.loads(prediction_json)

            # Get price at prediction time and current price
            target_date = analysis_date + timedelta(days=days_to_wait)

            # Fetch current price (simplified - in production you'd use historical data)
            stock = yf.Ticker(symbol)
            hist = stock.history(period="1mo")

            if hist.empty:
                return 0.0

            # Calculate accuracy based on recommendation
            recommendation = prediction.get('recommendation', 'HOLD')
            predicted_direction = {'BUY': 1, 'HOLD': 0, 'SELL': -1}[recommendation]

            # Simple accuracy calculation (price change direction)
            price_at_analysis = hist['Close'].iloc[0]
            current_price = hist['Close'].iloc[-1]
            actual_direction = 1 if current_price > price_at_analysis else (
                -1 if current_price < price_at_analysis else 0)

            # Calculate accuracy score
            if predicted_direction == actual_direction:
                accuracy = 1.0
            elif predicted_direction == 0:  # HOLD
                accuracy = 0.7  # Neutral prediction gets partial credit
            else:
                accuracy = 0.0  # Wrong direction

            # Adjust for magnitude of confidence
            confidence = prediction.get('confidence', 5) / 10.0
            accuracy_score = accuracy * (0.5 + 0.5 * confidence)

            return accuracy_score

        except Exception as e:
            self.logger.error(f"Error calculating accuracy: {e}")
            return 0.0

    async def learn_and_improve_prompts(self, agent_name: str) -> Dict[str, Any]:
        """Learn from agent performance and suggest prompt improvements"""
        performance = await self.sql_memory.get_agent_performance(agent_name)

        if not performance or performance.total_predictions < self.learning_config["min_samples_for_learning"]:
            return {"status": "insufficient_data", "message": "Need more predictions to learn"}

        # Analyze performance patterns
        improvements = {
            "status": "learning_complete",
            "current_accuracy": performance.accuracy_rate,
            "trend": performance.improvement_trend,
            "suggestions": []
        }

        # Suggest improvements based on performance
        if performance.accuracy_rate < 0.6:
            improvements["suggestions"].append({
                "type": "prompt_adjustment",
                "recommendation": "Add more conservative language to prompts",
                "reason": f"Accuracy rate {performance.accuracy_rate:.2f} is below threshold"
            })

        if performance.improvement_trend == "declining":
            improvements["suggestions"].append({
                "type": "context_enhancement",
                "recommendation": "Include more historical context in analysis",
                "reason": "Performance is declining - may need more context"
            })

        if performance.avg_confidence > 0.8 and performance.accuracy_rate < 0.7:
            improvements["suggestions"].append({
                "type": "confidence_calibration",
                "recommendation": "Add uncertainty language to reduce overconfidence",
                "reason": "High confidence but lower accuracy suggests overconfidence"
            })

        return improvements

    async def get_adaptive_prompt(self, agent_name: str, base_prompt: str, context: str = "") -> str:
        """Generate adaptive prompt based on agent's performance"""
        performance = await self.sql_memory.get_agent_performance(agent_name)

        if not performance:
            return base_prompt

        # Get relevant historical context
        historical_context = context

        # Adjust prompt based on performance
        adaptive_additions = []

        if performance.accuracy_rate < 0.6:
            adaptive_additions.append("""
PERFORMANCE NOTE: Recent predictions have been less accurate. Be more conservative 
in your analysis and explicitly state uncertainties.""")

        if performance.improvement_trend == "declining":
            adaptive_additions.append(f"""
LEARNING CONTEXT: {historical_context}

Consider these historical patterns when making your analysis.""")

        if performance.avg_confidence > 0.8 and performance.accuracy_rate < 0.7:
            adaptive_additions.append("""
CONFIDENCE CALIBRATION: Recent predictions have been overconfident. 
Lower your confidence scores and highlight potential risks more prominently.""")

        # Combine base prompt with adaptive additions
        if adaptive_additions:
            adaptive_prompt = base_prompt + "\n\n" + "\n".join(adaptive_additions)
        else:
            adaptive_prompt = base_prompt

        return adaptive_prompt


# ============================================================================
# STEP 5: Enhanced Multi-Agent System with Memory
# ============================================================================

class MemoryEnhancedAgent:
    """Base agent class enhanced with memory and learning capabilities"""

    def __init__(self, name: str, agent_type: str, memory_systems: Dict):
        self.name = name
        self.agent_type = agent_type
        self.logger = logging.getLogger(f"MemoryAgent.{name}")

        # Memory systems
        self.vector_memory = memory_systems["vector"]
        self.sql_memory = memory_systems["sql"]
        self.learning_engine = memory_systems["learning"]

        # Performance tracking
        self.analysis_count = 0
        self.last_performance_update = None

        self.logger.info(f"🧠 Memory-enhanced {name} agent initialized")

    async def store_analysis_result(self, symbol: str, prediction: Dict,
                                    confidence: float, market_conditions: Dict = None):
        """Store analysis result in memory systems"""
        memory = AnalysisMemory(
            id=f"{self.name}_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            symbol=symbol,
            analysis_date=datetime.now(),
            agent_name=self.name,
            analysis_type=self.agent_type,
            prediction=prediction,
            confidence=confidence,
            market_conditions=market_conditions
        )

        # Store in both memory systems
        await self.vector_memory.store_analysis_memory(memory)
        await self.sql_memory.store_analysis(memory)

        self.analysis_count += 1

        # Update performance metrics periodically
        if self.analysis_count % 5 == 0:
            await self.sql_memory.update_performance_metrics(self.name)

    async def get_historical_context(self, symbol: str) -> str:
        """Get relevant historical context for analysis"""
        return await self.vector_memory.get_context_for_analysis(symbol, self.agent_type)

    async def get_adaptive_analysis_prompt(self, base_prompt: str, symbol: str) -> str:
        """Get adaptive prompt based on performance and context"""
        context = await self.get_historical_context(symbol)
        return await self.learning_engine.get_adaptive_prompt(self.name, base_prompt, context)


class SmartTechnicalAnalysisAgent(MemoryEnhancedAgent):
    """Technical analysis agent with memory and learning"""

    def __init__(self, config: Dict, memory_systems: Dict):
        super().__init__("SmartTechnicalAnalyst", "technical_analysis", memory_systems)
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            openai_api_key=config["openai_api_key"]
        )

        self.base_prompt = PromptTemplate(
            input_variables=["symbol", "price", "change_percent", "volume", "rsi",
                             "sma_20", "sma_50", "week_52_high", "week_52_low", "historical_context"],
            template="""
You are an expert technical analyst with access to historical performance data.

CURRENT ANALYSIS FOR {symbol}:
Current Price: ${price:.2f}
Change: {change_percent:.2f}%
Volume: {volume:,}
RSI: {rsi}
20-day SMA: ${sma_20}
50-day SMA: ${sma_50}
52W High: ${week_52_high:.2f}
52W Low: ${week_52_low:.2f}

HISTORICAL CONTEXT:
{historical_context}

Provide technical analysis considering:
1. Current technical indicators and their reliability
2. Historical patterns from similar market conditions
3. Volume analysis and confirmation signals
4. Risk factors and potential reversals

Rate technical outlook 1-10 and provide confidence 0.0-1.0

TECHNICAL_SCORE: [1-10]
CONFIDENCE: [0.0-1.0]
TREND: [BULLISH/BEARISH/NEUTRAL]
KEY_INSIGHTS: [3 key technical points]
HISTORICAL_LEARNINGS: [Insights from similar past conditions]
WARNINGS: [Key technical risks]
"""
        )

    async def analyze_with_memory(self, stock_data: Dict) -> Dict:
        """Perform technical analysis enhanced with memory"""
        symbol = stock_data["symbol"]

        # Get historical context
        historical_context = await self.get_historical_context(symbol)

        # Get adaptive prompt
        adaptive_prompt_text = await self.get_adaptive_analysis_prompt(
            self.base_prompt.template, symbol
        )

        # Create adaptive prompt
        adaptive_prompt = PromptTemplate(
            input_variables=self.base_prompt.input_variables,
            template=adaptive_prompt_text
        )

        # Format prompt
        prompt = adaptive_prompt.format(
            symbol=symbol,
            price=stock_data["price"],
            change_percent=stock_data["change_percent"],
            volume=stock_data["volume"],
            rsi=stock_data.get("rsi", "N/A"),
            sma_20=stock_data.get("sma_20", "N/A"),
            sma_50=stock_data.get("sma_50", "N/A"),
            week_52_high=stock_data["week_52_high"],
            week_52_low=stock_data["week_52_low"],
            historical_context=historical_context
        )

        # Get AI analysis
        response = await self.llm.apredict(prompt)

        # Parse response
        analysis_result = self._parse_analysis(response, stock_data)

        # Store in memory
        prediction = {
            "recommendation": self._score_to_recommendation(analysis_result["score"]),
            "score": analysis_result["score"],
            "confidence": analysis_result["confidence"]
        }

        await self.store_analysis_result(
            symbol=symbol,
            prediction=prediction,
            confidence=analysis_result["confidence"],
            market_conditions={
                "rsi": stock_data.get("rsi"),
                "price_position": (stock_data["price"] - stock_data["week_52_low"]) /
                                  (stock_data["week_52_high"] - stock_data["week_52_low"]),
                "volume_ratio": stock_data.get("volume", 0) / 10000000  # Normalized volume
            }
        )

        return analysis_result

    def _score_to_recommendation(self, score: float) -> str:
        """Convert numeric score to recommendation"""
        if score >= 7:
            return "BUY"
        elif score <= 3:
            return "SELL"
        else:
            return "HOLD"

    def _parse_analysis(self, response: str, stock_data: Dict) -> Dict:
        """Parse analysis response"""
        lines = response.strip().split('\n')

        score = 5.0
        confidence = 0.5
        insights = []
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
            elif line.startswith("WARNINGS:"):
                warn_text = line.split(":", 1)[1].strip()
                warnings = [w.strip("• -") for w in warn_text.split('\n') if w.strip()]

        return {
            "agent_name": self.name,
            "symbol": stock_data["symbol"],
            "analysis_type": "technical_memory_enhanced",
            "score": score,
            "confidence": confidence,
            "key_insights": insights[:3],
            "warnings": warnings[:3],
            "raw_analysis": response,
            "timestamp": datetime.now(),
            "used_historical_context": len(await self.get_historical_context(stock_data["symbol"])) > 100
        }


# ============================================================================
# STEP 6: Memory-Enhanced System Controller
# ============================================================================

class MemoryEnhancedStockSystem:
    """Stock analysis system with full memory and learning capabilities"""

    def __init__(self):
        self.config = self._load_config()
        self.logger = logging.getLogger("MemorySystem")

        # Initialize memory systems
        self.vector_memory = VectorMemorySystem()
        self.sql_memory = SQLMemorySystem()
        self.learning_engine = LearningEngine(self.vector_memory, self.sql_memory)

        memory_systems = {
            "vector": self.vector_memory,
            "sql": self.sql_memory,
            "learning": self.learning_engine
        }

        # Initialize memory-enhanced agents
        self.technical_agent = SmartTechnicalAnalysisAgent(self.config, memory_systems)

        self.logger.info("🧠 Memory-enhanced stock system initialized")

    def _load_config(self) -> Dict:
        """Load configuration"""
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            print("❌ OPENAI_API_KEY environment variable required!")
            sys.exit(1)

        return {
            "openai_api_key": openai_key,
        }

    async def analyze_with_memory(self, symbols: List[str]) -> List[Dict]:
        """Analyze stocks with full memory and learning capabilities"""
        self.logger.info(f"🧠 Starting memory-enhanced analysis of {len(symbols)} stocks")

        results = []

        for i, symbol in enumerate(symbols, 1):
            print(f"\n{'=' * 70}")
            print(f"[{i}/{len(symbols)}] MEMORY-ENHANCED ANALYSIS: {symbol}")
            print(f"{'=' * 70}")

            # Get stock data (simplified for example)
            stock_data = await self._fetch_stock_data(symbol)
            if not stock_data:
                continue

            # Perform memory-enhanced technical analysis
            technical_result = await self.technical_agent.analyze_with_memory(stock_data)

            # Get agent performance metrics
            performance = await self.sql_memory.get_agent_performance("SmartTechnicalAnalyst")

            # Show learning insights
            learning_insights = await self.learning_engine.learn_and_improve_prompts("SmartTechnicalAnalyst")

            result = {
                "symbol": symbol,
                "analysis": technical_result,
                "agent_performance": asdict(performance) if performance else None,
                "learning_insights": learning_insights,
                "timestamp": datetime.now().isoformat()
            }

            results.append(result)

            # Display enhanced results
            print(f"🎯 ANALYSIS RESULT:")
            print(f"   Score: {technical_result['score']}/10")
            print(f"   Confidence: {technical_result['confidence']:.2f}")
            print(f"   Used Historical Context: {technical_result['used_historical_context']}")

            if performance:
                print(f"🧠 AGENT LEARNING:")
                print(f"   Total Predictions: {performance.total_predictions}")
                print(f"   Accuracy Rate: {performance.accuracy_rate:.2f}")
                print(f"   Trend: {performance.improvement_trend}")

            if learning_insights.get("suggestions"):
                print(f"💡 LEARNING SUGGESTIONS:")
                for suggestion in learning_insights["suggestions"]:
                    print(f"   • {suggestion['recommendation']}")

        return results

    async def _fetch_stock_data(self, symbol: str) -> Optional[Dict]:
        """Fetch stock data (simplified version)"""
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="60d")
            info = stock.info

            if hist.empty:
                return None

            current_price = float(hist['Close'].iloc[-1])
            prev_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price

            # Store price data for future accuracy calculations
            await self.sql_memory.store_price_data(symbol, current_price, int(hist['Volume'].iloc[-1]))

            return {
                "symbol": symbol,
                "price": current_price,
                "change_percent": ((current_price - prev_close) / prev_close) * 100,
                "volume": int(hist['Volume'].iloc[-1]),
                "week_52_high": float(info.get('fiftyTwoWeekHigh', current_price)),
                "week_52_low": float(info.get('fiftyTwoWeekLow', current_price)),
                "rsi": self._calculate_rsi(hist['Close']),
                "sma_20": hist['Close'].rolling(20).mean().iloc[-1] if len(hist) >= 20 else None,
                "sma_50": hist['Close'].rolling(50).mean().iloc[-1] if len(hist) >= 50 else None
            }

        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            return None

    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi.iloc[-1]) if not rsi.iloc[-1] != rsi.iloc[-1] else 50.0
        except:
            return 50.0


# ============================================================================
# STEP 7: Main Application
# ============================================================================

async def main():
    """Main application demonstrating memory and learning"""
    print("🧠 Memory & Learning Enhanced Stock Analysis System")
    print("=" * 70)

    # Default stocks
    default_symbols = ["AAPL", "GOOGL", "MSFT"]

    if len(sys.argv) > 1:
        symbols = [s.upper().strip() for s in sys.argv[1].split(',')]
    else:
        print(f"Using default portfolio: {', '.join(default_symbols)}")
        symbols = default_symbols

    # Initialize memory-enhanced system
    system = MemoryEnhancedStockSystem()

    # Run analysis multiple times to demonstrate learning
    print("\n🔄 Running analysis to demonstrate learning...")

    for round_num in range(1, 4):  # Run 3 rounds
        print(f"\n🎯 ANALYSIS ROUND {round_num}")
        print("=" * 50)

        results = await system.analyze_with_memory(symbols)

        # Show learning progress
        performance = await system.sql_memory.get_agent_performance("SmartTechnicalAnalyst")
        if performance and performance.total_predictions > 0:
            print(f"\n📈 LEARNING PROGRESS AFTER ROUND {round_num}:")
            print(f"   Total Predictions: {performance.total_predictions}")
            print(f"   Current Accuracy: {performance.accuracy_rate:.2f}")
            print(f"   Trend: {performance.improvement_trend}")

        # Small delay between rounds
        if round_num < 3:
            print("\n⏳ Waiting 30 seconds before next round...")
            await asyncio.sleep(30)

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f"memory_enhanced_results_{timestamp}.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n💾 Results saved to memory_enhanced_results_{timestamp}.json")
    print(f"📊 Memory databases stored in memory/ directory")


if __name__ == "__main__":
    asyncio.run(main())