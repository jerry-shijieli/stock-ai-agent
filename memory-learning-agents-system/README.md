# Memory & Learning Enhanced Stock Analysis System

This system adds persistent memory and learning capabilities to a multi-agent stock analysis system, enabling agents to learn from past performance and improve predictions over time.

## Features

- **Vector Memory**: Semantic search through historical analyses using ChromaDB
- **SQL Memory**: Structured performance tracking in SQLite database
- **Learning Engine**: Improves predictions based on accuracy feedback
- **Pattern Recognition**: Identifies successful analysis patterns
- **Dynamic Prompts**: Adjusts prompts based on performance trends
- **Smart Technical Analysis**: Memory-enhanced technical analysis agent

## Prerequisites

- Python 3.8+
- OpenAI API key
- Internet connection for stock data

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install yfinance requests textblob langchain-openai chromadb scikit-learn
   ```

2. **Set up OpenAI API key:**
   ```bash
   export OPENAI_API_KEY="your_openai_api_key_here"
   ```
   
   Or create a `.env` file:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. **Create memory directory:**
   ```bash
   mkdir -p memory
   ```

## Usage

### Basic Usage

Run with default stocks (AAPL, GOOGL, MSFT):
```bash
python memory_learning_system.py
```

### Custom Stock Analysis

Analyze specific stocks:
```bash
python memory_learning_system.py "TSLA,NVDA,AMD"
```

### System Components

The system consists of several key components:

1. **Memory Systems**:
   - `VectorMemorySystem`: ChromaDB for semantic search
   - `SQLMemorySystem`: SQLite for structured data and performance tracking

2. **Learning Engine**:
   - Calculates prediction accuracy
   - Identifies performance patterns
   - Generates adaptive prompts

3. **Memory-Enhanced Agents**:
   - `SmartTechnicalAnalysisAgent`: Technical analysis with historical context

## Output

The system generates:

- **Console Output**: Real-time analysis results, learning progress, and suggestions
- **JSON Results**: Detailed analysis saved to `memory_enhanced_results_[timestamp].json`
- **Memory Databases**: Stored in `memory/` directory
  - `memory/vector_store/`: ChromaDB vector database
  - `memory/agent_memory.db`: SQLite database with performance metrics

## Understanding the Learning Process

The system learns through multiple rounds of analysis:

1. **Round 1**: Initial baseline predictions
2. **Round 2**: Begins incorporating historical context
3. **Round 3+**: Adaptive prompts based on performance trends

### Performance Metrics

- **Accuracy Rate**: Percentage of correct predictions
- **Confidence Calibration**: Alignment between confidence and accuracy
- **Improvement Trend**: "improving", "declining", or "stable"
- **Recent Performance**: Last 10 predictions for trend analysis

### Learning Adaptations

Based on performance, the system makes automatic adjustments:

- **Low Accuracy (<60%)**: Adds conservative language to prompts
- **Declining Performance**: Includes more historical context
- **Overconfidence**: Adds uncertainty language and risk highlights

## File Structure

```
memory-learning-agents-system/
├── memory_learning_system.py    # Main system
├── README.md                    # This file
└── memory/                      # Created at runtime
    ├── vector_store/            # ChromaDB vector database
    └── agent_memory.db          # SQLite performance database
```

## Configuration

The system uses environment variables for configuration:

- `OPENAI_API_KEY`: Required for AI analysis
- Logging level can be modified in the script (default: INFO)

## Troubleshooting

### Common Issues

1. **Missing Dependencies**:
   ```bash
   pip install yfinance requests textblob langchain-openai chromadb scikit-learn
   ```

2. **Missing OpenAI API Key**:
   ```bash
   export OPENAI_API_KEY="your_key_here"
   ```

3. **Permission Errors**:
   Ensure the script has write permissions for the `memory/` directory

4. **Stock Data Issues**:
   - Check internet connection
   - Verify stock symbols are valid
   - Some stocks may not have sufficient historical data

### Debug Mode

Enable debug logging by modifying the logging level in the script:
```python
logging.basicConfig(level=logging.DEBUG, ...)
```

## Data Privacy

- All data is stored locally in the `memory/` directory
- No data is shared with external services except for:
  - OpenAI API calls for analysis
  - Yahoo Finance for stock data

## Performance Considerations

- Initial runs create memory databases (slower)
- Subsequent runs benefit from historical context (faster and more accurate)
- Vector similarity searches scale well with data volume
- Consider periodic cleanup of very old analysis data

## 🏗️ System Architecture & Design

### **Architecture Overview**

This is a **multi-layered AI agent system** with persistent memory and learning capabilities:

```
┌─────────────────────────────────────────────────┐
│                 Main Controller                 │
│           (MemoryEnhancedStockSystem)          │
└─────────────────┬───────────────────────────────┘
                  │
┌─────────────────┴───────────────────────────────┐
│              Memory Systems                     │
│  ┌─────────────┬─────────────┬─────────────┐    │
│  │   Vector    │     SQL     │  Learning   │    │
│  │   Memory    │   Memory    │   Engine    │    │
│  │ (ChromaDB)  │ (SQLite)    │             │    │
│  └─────────────┴─────────────┴─────────────┘    │
└─────────────────┬───────────────────────────────┘
                  │
┌─────────────────┴───────────────────────────────┐
│              AI Agents                          │
│        (SmartTechnicalAnalysisAgent)           │
└─────────────────────────────────────────────────┘
```

## 📋 Code Structure Walkthrough

### **STEP 1: Data Structures (Lines 52-91)**

The system defines core data structures that act like "memory cards":

```python
@dataclass
class AnalysisMemory:
    """Stores each analysis with prediction and actual outcome"""
    id: str                    # Unique identifier
    symbol: str               # Stock symbol (AAPL, GOOGL)
    analysis_date: datetime   # When analysis was done
    agent_name: str          # Which agent made the analysis
    prediction: Dict         # What the agent predicted
    actual_outcome: Optional[Dict]  # What actually happened
    accuracy_score: Optional[float] # How accurate was it
```

**Purpose**: These structures store every analysis the system makes, enabling learning from past performance.

### **STEP 2: Vector Memory System (Lines 97-218)**

This is the **"semantic brain"** of the system:

```python
class VectorMemorySystem:
    """Uses ChromaDB for semantic search of past analyses"""
    
    async def store_analysis_memory(self, memory):
        # Converts analysis to text and stores as vector
        # Allows finding similar past situations
    
    async def find_similar_analyses(self, query, symbol=None):
        # "Find me analyses similar to current market conditions"
        # Returns historically similar situations
```

**How it works**: 
- Takes analysis text and converts it to numerical vectors (embeddings)
- When you ask "what happened before with AAPL in similar conditions?", it finds semantically similar past analyses
- Like having a super-smart search that understands meaning, not just keywords

### **STEP 3: SQL Memory System (Lines 224-553)**

This is the **"structured database brain"**:

```python
class SQLMemorySystem:
    """SQLite database for performance tracking and structured data"""
    
    async def store_analysis(self, memory):
        # Saves analysis in structured database table
    
    async def update_performance_metrics(self, agent_name):
        # Calculates accuracy rates, trends for each agent
    
    async def get_agent_performance(self, agent_name):
        # Returns how well an agent has been performing
```

**Database Tables**:
1. **analysis_history**: Every prediction made
2. **agent_performance**: How accurate each agent is
3. **price_tracking**: Stock prices for accuracy calculation

### **STEP 4: Learning Engine (Lines 559-713)**

This is the **"improvement brain"**:

```python
class LearningEngine:
    """Learns from past performance and adapts"""
    
    async def calculate_prediction_accuracy(self, analysis_id):
        # Compares prediction vs actual stock price movement
        # Returns accuracy score (0.0 to 1.0)
    
    async def learn_and_improve_prompts(self, agent_name):
        # Analyzes agent performance and suggests improvements
        # "Agent X is overconfident, add uncertainty language"
    
    async def get_adaptive_prompt(self, agent_name, base_prompt):
        # Modifies prompts based on performance
        # Better performing agents get more confidence
        # Poor performers get more conservative language
```

**Learning Process**:
1. Agent makes prediction → Store in memory
2. Wait for actual outcome → Calculate accuracy
3. Analyze patterns → Identify improvement areas
4. Adapt prompts → Better future performance

### **STEP 5: Memory-Enhanced Agent (Lines 719-927)**

This is where the **"AI analyst"** lives:

```python
class SmartTechnicalAnalysisAgent(MemoryEnhancedAgent):
    """Technical analyst that learns from experience"""
    
    async def analyze_with_memory(self, stock_data):
        # 1. Get historical context from vector memory
        historical_context = await self.get_historical_context(symbol)
        
        # 2. Get adaptive prompt based on past performance
        adaptive_prompt = await self.get_adaptive_analysis_prompt(...)
        
        # 3. Run AI analysis with enhanced prompt
        response = await self.llm.ainvoke(prompt)
        
        # 4. Store results in memory for future learning
        await self.store_analysis_result(...)
        
        return analysis_result
```

**Analysis Workflow**:
```
Stock Data Input → Get Historical Context → Adapt Prompt → 
AI Analysis → Parse Results → Store in Memory → Return Analysis
```

## 🔄 Complete Workflow Example

Let's trace what happens when you run the system:

### **Round 1 (First Run)**
1. **Input**: `["AAPL", "GOOGL", "MSFT"]`
2. **For each stock**:
   - Fetch stock data (price, volume, technical indicators)
   - Check vector memory → "No similar analyses found" (first run)
   - Use base prompt (no adaptations yet)
   - AI analyzes → "BUY AAPL, confidence 0.8"
   - Store analysis in both vector and SQL memory
3. **Result**: Initial predictions stored, no learning yet

### **Round 2 (Learning Begins)**
1. **For each stock**:
   - Fetch new stock data
   - Check vector memory → "Found 3 similar past analyses"
   - Historical context: *"Previous analysis showed strong momentum..."*
   - Use enhanced prompt with historical context
   - AI analyzes with better context
   - Calculate accuracy from Round 1 predictions
   - Update performance metrics
2. **Result**: Better informed predictions, performance tracking begins

### **Round 3 (Adaptive Learning)**
1. **For each stock**:
   - Performance analysis: "Agent accuracy: 65%, trend: improving"
   - Learning engine suggests: "Add conservative language"
   - Adaptive prompt: *"Be more cautious given recent market volatility..."*
   - AI analyzes with adapted prompt
   - Even better predictions based on learned patterns
2. **Result**: Continuously improving analysis quality

## 🧠 Key Learning Mechanisms

### **1. Semantic Memory (Vector Store)**
```python
# Stores: "AAPL technical analysis showing RSI oversold, volume spike, breaking resistance"
# Later finds: Similar patterns in other analyses
# Provides: Relevant historical context for current analysis
```

### **2. Performance Tracking (SQL Database)**
```python
# Tracks: Agent made 47 predictions, 31 correct (66% accuracy)
# Identifies: "Agent overconfident - high confidence but lower accuracy"
# Adapts: Reduces confidence levels in future prompts
```

### **3. Dynamic Prompt Engineering**
```python
# Base prompt: "Analyze this stock..."
# If low accuracy: + "Be more conservative and state uncertainties"
# If declining trend: + "Consider these historical patterns: [context]"
# If overconfident: + "Lower confidence and highlight risks"
```

## 📊 Output Structure

The system produces rich, multi-layered results:

```json
{
  "symbol": "AAPL",
  "analysis": {
    "score": 7.2,
    "confidence": 0.75,
    "key_insights": ["Strong momentum", "Volume confirmation"],
    "used_historical_context": true
  },
  "agent_performance": {
    "total_predictions": 23,
    "accuracy_rate": 0.67,
    "improvement_trend": "improving"
  },
  "learning_insights": {
    "suggestions": ["Add more conservative language"]
  }
}
```

## 🎯 Why This Architecture is Powerful

1. **Memory**: Never forgets past analyses - builds institutional knowledge
2. **Learning**: Gets better over time - self-improving system
3. **Context**: Uses relevant historical patterns - informed decisions  
4. **Adaptation**: Adjusts strategy based on performance - dynamic optimization
5. **Transparency**: Shows learning progress - explainable AI

This isn't just a stock analyzer - it's a **learning AI system** that mimics how human experts develop intuition and improve their skills over time through experience and reflection.

## Extending the System

To add new analysis agents:

1. Inherit from `MemoryEnhancedAgent`
2. Implement the analysis method
3. Register with the memory systems
4. Add to the main system controller

Example:
```python
class CustomAgent(MemoryEnhancedAgent):
    def __init__(self, config, memory_systems):
        super().__init__("CustomAgent", "custom_analysis", memory_systems)
    
    async def analyze_with_memory(self, data):
        # Your custom analysis logic here
        pass
```