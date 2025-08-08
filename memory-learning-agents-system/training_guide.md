# Training Guide: Memory Enhanced Stock System

This guide explains how to train the MemoryEnhancedStockSystem to improve price predictions using historical data.

## 🎯 Training Overview

The system learns through **backtesting** - running analyses on historical data where we already know the outcomes. This allows us to:

1. **Build Historical Memory**: Store past analyses with known results
2. **Calculate Real Accuracy**: Compare predictions vs actual price movements
3. **Optimize Prompts**: Adapt based on performance patterns
4. **Validate Patterns**: Identify what works and what doesn't

## 📋 Training Steps

### **Step 1: Prepare Historical Dataset**

First, we'll create a training script that fetches historical data and simulates real-time analysis:

```python
# training_script.py
import asyncio
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from memory_learning_system import MemoryEnhancedStockSystem
import json

class StockTrainer:
    def __init__(self, symbol: str, months_back: int = 3):
        self.symbol = symbol.upper()
        self.months_back = months_back
        self.system = MemoryEnhancedStockSystem()
        
    def prepare_training_dataset(self):
        """Prepare historical dataset for training"""
        print(f"📊 Preparing training dataset for {self.symbol}")
        
        # Fetch extended historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.months_back * 30 + 60)  # Extra buffer
        
        ticker = yf.Ticker(self.symbol)
        hist = ticker.history(start=start_date, end=end_date)
        
        if hist.empty:
            raise ValueError(f"No data available for {self.symbol}")
        
        # Create training data points (daily snapshots)
        training_points = []
        
        for i in range(60, len(hist) - 7):  # Need 60 days for indicators, 7 for prediction window
            current_date = hist.index[i]
            current_data = hist.iloc[:i+1]  # Data available up to current_date
            future_data = hist.iloc[i+1:i+8]  # Next 7 days for accuracy calculation
            
            # Calculate technical indicators
            stock_data = self._calculate_indicators(current_data, current_date)
            
            # Calculate actual outcome (what happened in next 7 days)
            actual_outcome = self._calculate_actual_outcome(current_data, future_data)
            
            training_points.append({
                'date': current_date,
                'stock_data': stock_data,
                'actual_outcome': actual_outcome
            })
        
        print(f"✅ Created {len(training_points)} training data points")
        return training_points
    
    def _calculate_indicators(self, hist_data, current_date):
        """Calculate technical indicators for a specific date"""
        current_price = float(hist_data['Close'].iloc[-1])
        prev_close = float(hist_data['Close'].iloc[-2]) if len(hist_data) > 1 else current_price
        
        # RSI calculation
        def calculate_rsi(prices, period=14):
            if len(prices) < period + 1:
                return 50.0
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            if loss.iloc[-1] == 0:
                return 100.0
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
        
        return {
            "symbol": self.symbol,
            "price": current_price,
            "change_percent": ((current_price - prev_close) / prev_close) * 100,
            "volume": int(hist_data['Volume'].iloc[-1]),
            "week_52_high": float(hist_data['High'].rolling(252).max().iloc[-1]),
            "week_52_low": float(hist_data['Low'].rolling(252).min().iloc[-1]),
            "rsi": calculate_rsi(hist_data['Close']),
            "sma_20": float(hist_data['Close'].rolling(20).mean().iloc[-1]) if len(hist_data) >= 20 else current_price,
            "sma_50": float(hist_data['Close'].rolling(50).mean().iloc[-1]) if len(hist_data) >= 50 else current_price
        }
    
    def _calculate_actual_outcome(self, current_data, future_data):
        """Calculate what actually happened in the prediction window"""
        if future_data.empty:
            return None
            
        current_price = current_data['Close'].iloc[-1]
        final_price = future_data['Close'].iloc[-1]
        
        price_change_pct = ((final_price - current_price) / current_price) * 100
        max_price = future_data['High'].max()
        min_price = future_data['Low'].min()
        
        # Determine actual direction
        if price_change_pct > 2.0:
            direction = "BUY"  # Strong upward movement
        elif price_change_pct < -2.0:
            direction = "SELL"  # Strong downward movement
        else:
            direction = "HOLD"  # Sideways movement
            
        return {
            "price_change_pct": price_change_pct,
            "direction": direction,
            "max_gain_pct": ((max_price - current_price) / current_price) * 100,
            "max_loss_pct": ((min_price - current_price) / current_price) * 100,
            "volatility": future_data['Close'].std() / current_price * 100
        }
```

### **Step 2: Training Execution Script**

```python
    async def run_training(self, training_points):
        """Execute training on historical data points"""
        print(f"🚀 Starting training on {len(training_points)} data points...")
        
        successful_analyses = 0
        
        for i, point in enumerate(training_points):
            try:
                print(f"📈 Training point {i+1}/{len(training_points)} - {point['date'].strftime('%Y-%m-%d')}")
                
                # Run analysis on historical data point
                analysis = await self.system.technical_agent.analyze_with_memory(point['stock_data'])
                
                # Calculate accuracy based on actual outcome
                if point['actual_outcome']:
                    accuracy = self._calculate_training_accuracy(analysis, point['actual_outcome'])
                    
                    # Create memory with actual outcome
                    memory_id = f"training_{self.symbol}_{point['date'].strftime('%Y%m%d')}"
                    
                    # Update the analysis memory with actual results
                    await self._store_training_result(memory_id, point, analysis, accuracy)
                    
                    successful_analyses += 1
                
                # Update performance metrics every 10 analyses
                if i % 10 == 0:
                    await self.system.sql_memory.update_performance_metrics("SmartTechnicalAnalyst")
                    
            except Exception as e:
                print(f"⚠️ Error in training point {i+1}: {e}")
                continue
        
        print(f"✅ Training completed: {successful_analyses}/{len(training_points)} successful analyses")
        
        # Final performance update
        await self.system.sql_memory.update_performance_metrics("SmartTechnicalAnalyst")
        
        # Generate training report
        await self._generate_training_report()
    
    def _calculate_training_accuracy(self, analysis, actual_outcome):
        """Calculate accuracy score for training data"""
        predicted_direction = self._score_to_direction(analysis['score'])
        actual_direction = actual_outcome['direction']
        
        # Direction accuracy
        if predicted_direction == actual_direction:
            direction_score = 1.0
        elif predicted_direction == "HOLD":
            direction_score = 0.7  # Neutral gets partial credit
        else:
            direction_score = 0.0
        
        # Confidence calibration (penalize overconfidence)
        confidence = analysis['confidence']
        actual_volatility = actual_outcome['volatility']
        
        # High volatility should reduce confidence score
        volatility_penalty = min(actual_volatility / 10.0, 0.3)  # Max 30% penalty
        calibrated_confidence = max(0.1, confidence - volatility_penalty)
        
        # Final accuracy score
        accuracy = direction_score * (0.7 + 0.3 * calibrated_confidence)
        
        return min(1.0, max(0.0, accuracy))
    
    def _score_to_direction(self, score):
        """Convert analysis score to direction"""
        if score >= 7:
            return "BUY"
        elif score <= 3:
            return "SELL"
        else:
            return "HOLD"
    
    async def _store_training_result(self, memory_id, point, analysis, accuracy):
        """Store training result in memory systems"""
        from memory_learning_system import AnalysisMemory
        
        # Create memory object with training results
        memory = AnalysisMemory(
            id=memory_id,
            symbol=self.symbol,
            analysis_date=point['date'],
            agent_name="SmartTechnicalAnalyst",
            analysis_type="technical_training",
            prediction={
                "recommendation": self._score_to_direction(analysis['score']),
                "score": analysis['score'],
                "confidence": analysis['confidence']
            },
            actual_outcome=point['actual_outcome'],
            accuracy_score=accuracy,
            confidence=analysis['confidence'],
            market_conditions={
                "rsi": point['stock_data'].get("rsi"),
                "price_position": (point['stock_data']["price"] - point['stock_data']["week_52_low"]) /
                                  (point['stock_data']["week_52_high"] - point['stock_data']["week_52_low"]),
                "volume_ratio": point['stock_data'].get("volume", 0) / 10000000
            }
        )
        
        # Store in both memory systems
        await self.system.vector_memory.store_analysis_memory(memory)
        await self.system.sql_memory.store_analysis(memory)
    
    async def _generate_training_report(self):
        """Generate comprehensive training report"""
        performance = await self.system.sql_memory.get_agent_performance("SmartTechnicalAnalyst")
        learning_insights = await self.system.learning_engine.learn_and_improve_prompts("SmartTechnicalAnalyst")
        
        report = {
            "training_summary": {
                "symbol": self.symbol,
                "training_period": f"{self.months_back} months",
                "completed_at": datetime.now().isoformat()
            },
            "performance_metrics": {
                "total_predictions": performance.total_predictions if performance else 0,
                "accuracy_rate": performance.accuracy_rate if performance else 0,
                "improvement_trend": performance.improvement_trend if performance else "unknown"
            },
            "learning_insights": learning_insights
        }
        
        # Save training report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"training_report_{self.symbol}_{timestamp}.json"
        
        with open(f"results/{filename}", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📊 Training report saved: results/{filename}")
        
        # Display summary
        if performance:
            print(f"""
🎯 TRAINING SUMMARY FOR {self.symbol}:
   Total Analyses: {performance.total_predictions}
   Accuracy Rate: {performance.accuracy_rate:.1%}
   Trend: {performance.improvement_trend}
   Suggestions: {len(learning_insights.get('suggestions', []))}
""")


# Main training execution
async def main():
    # Configuration
    SYMBOL = "AAPL"  # Change this to train on different stocks
    MONTHS_BACK = 3   # How many months of historical data to use
    
    try:
        # Create trainer
        trainer = StockTrainer(SYMBOL, MONTHS_BACK)
        
        # Prepare dataset
        training_points = trainer.prepare_training_dataset()
        
        # Run training
        await trainer.run_training(training_points)
        
        print("🎉 Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
```

## 🚀 **How to Execute Training**

### **Step 1: Set Up Environment**

```bash
# Make sure you have the OpenAI API key
export OPENAI_API_KEY="your_openai_api_key_here"

# Create results directory
mkdir -p results

# Install additional dependencies if needed
pip install pandas
```

### **Step 2: Create Training Script**

Save the training code above as `train_stock_system.py` in your project directory.

### **Step 3: Run Training**

```bash
# Train on AAPL with 3 months of data
python train_stock_system.py

# Or modify the script to train on different stocks:
# SYMBOL = "TSLA"  # Tesla
# SYMBOL = "GOOGL"  # Google
# SYMBOL = "MSFT"  # Microsoft
```

## 📊 **Training Process Explained**

### **Dataset Preparation**
1. **Historical Data Fetch**: Downloads 3+ months of daily stock data
2. **Feature Engineering**: Calculates RSI, moving averages, volume ratios
3. **Outcome Labeling**: Determines what actually happened 7 days later
4. **Training Points**: Creates ~90 data points (3 months × 30 days)

### **Training Execution**
1. **Sequential Analysis**: Runs analysis on each historical point
2. **Accuracy Calculation**: Compares prediction vs actual outcome
3. **Memory Storage**: Stores each analysis with known results
4. **Performance Updates**: Updates agent metrics every 10 analyses
5. **Adaptive Learning**: System starts adapting prompts based on performance

### **Learning Outcomes**
- **Vector Memory**: Builds database of similar market patterns
- **Performance Metrics**: Tracks accuracy trends and improvements
- **Prompt Adaptation**: Automatically adjusts analysis strategy
- **Pattern Recognition**: Identifies successful prediction patterns

## 🎯 **Training Results**

After training, you'll see improvements in:

1. **Historical Context Usage**: System finds relevant past patterns
2. **Confidence Calibration**: Better alignment between confidence and accuracy
3. **Adaptive Prompts**: Context-aware analysis instructions
4. **Performance Metrics**: Clear tracking of improvement over time

### **Expected Training Output**

```
📊 Preparing training dataset for AAPL
✅ Created 87 training data points
🚀 Starting training on 87 data points...
📈 Training point 1/87 - 2024-08-15
📈 Training point 10/87 - 2024-08-26
...
✅ Training completed: 85/87 successful analyses

🎯 TRAINING SUMMARY FOR AAPL:
   Total Analyses: 85
   Accuracy Rate: 72.3%
   Trend: improving
   Suggestions: 2

📊 Training report saved: results/training_report_AAPL_20250108_143022.json
🎉 Training completed successfully!
```

## 🔧 **Advanced Training Options**

### **Multi-Stock Training**

```python
# Train on multiple stocks
SYMBOLS = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]

for symbol in SYMBOLS:
    trainer = StockTrainer(symbol, 3)
    training_points = trainer.prepare_training_dataset()
    await trainer.run_training(training_points)
```

### **Different Time Horizons**

```python
# Train for different prediction windows
PREDICTION_WINDOWS = [3, 7, 14]  # 3 days, 1 week, 2 weeks

for window in PREDICTION_WINDOWS:
    trainer = StockTrainer("AAPL", 3)
    trainer.prediction_window = window
    # ... run training
```

### **Market Condition Filtering**

```python
# Train only on specific market conditions
def filter_training_points(points, condition="high_volatility"):
    if condition == "high_volatility":
        return [p for p in points if p['actual_outcome']['volatility'] > 5.0]
    elif condition == "trending_up":
        return [p for p in points if p['actual_outcome']['price_change_pct'] > 1.0]
    return points
```

This training approach transforms your system from a basic analyzer into a **learned expert** with deep historical knowledge and adaptive strategies specific to each stock's behavior patterns.