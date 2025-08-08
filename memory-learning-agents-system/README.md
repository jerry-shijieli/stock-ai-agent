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