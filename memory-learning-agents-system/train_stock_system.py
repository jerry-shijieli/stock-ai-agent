#!/usr/bin/env python3
"""
Training Script for Memory Enhanced Stock System

This script trains the system using historical data to improve prediction accuracy.
Usage: python train_stock_system.py
"""

import asyncio
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import sys
import os

# Add current directory to path to import our system
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from memory_learning_system import MemoryEnhancedStockSystem, AnalysisMemory
except ImportError as e:
    print(f"❌ Error importing memory_learning_system: {e}")
    print("Make sure memory_learning_system.py is in the same directory")
    sys.exit(1)


class StockTrainer:
    def __init__(self, symbol: str, months_back: int = 3):
        self.symbol = symbol.upper()
        self.months_back = months_back
        self.prediction_window = 7  # Days to look ahead for accuracy calculation
        self.system = None
        
    async def initialize_system(self):
        """Initialize the memory enhanced system"""
        try:
            self.system = MemoryEnhancedStockSystem()
            print("🧠 Memory Enhanced Stock System initialized")
        except Exception as e:
            raise Exception(f"Failed to initialize system: {e}")
        
    def prepare_training_dataset(self):
        """Prepare historical dataset for training"""
        print(f"📊 Preparing training dataset for {self.symbol}")
        
        # Fetch extended historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.months_back * 30 + 60)  # Extra buffer
        
        try:
            ticker = yf.Ticker(self.symbol)
            hist = ticker.history(start=start_date, end=end_date)
        except Exception as e:
            raise ValueError(f"Failed to fetch data for {self.symbol}: {e}")
        
        if hist.empty:
            raise ValueError(f"No data available for {self.symbol}")
        
        print(f"📈 Fetched {len(hist)} days of historical data")
        
        # Create training data points (daily snapshots)
        training_points = []
        min_data_points = 60  # Need minimum data for technical indicators
        
        for i in range(min_data_points, len(hist) - self.prediction_window):
            current_date = hist.index[i]
            current_data = hist.iloc[:i+1]  # Data available up to current_date
            future_data = hist.iloc[i+1:i+1+self.prediction_window]  # Next N days
            
            try:
                # Calculate technical indicators
                stock_data = self._calculate_indicators(current_data, current_date)
                
                # Calculate actual outcome (what happened in prediction window)
                actual_outcome = self._calculate_actual_outcome(current_data, future_data)
                
                if stock_data and actual_outcome:
                    training_points.append({
                        'date': current_date,
                        'stock_data': stock_data,
                        'actual_outcome': actual_outcome
                    })
            except Exception as e:
                print(f"⚠️ Skipping data point {current_date}: {e}")
                continue
        
        print(f"✅ Created {len(training_points)} training data points")
        return training_points
    
    def _calculate_indicators(self, hist_data, current_date):
        """Calculate technical indicators for a specific date"""
        try:
            current_price = float(hist_data['Close'].iloc[-1])
            prev_close = float(hist_data['Close'].iloc[-2]) if len(hist_data) > 1 else current_price
            current_volume = int(hist_data['Volume'].iloc[-1])
            
            # Validate data
            if current_price <= 0 or np.isnan(current_price):
                return None
            
            # RSI calculation
            rsi = self._calculate_rsi(hist_data['Close'])
            
            # Moving averages
            sma_20 = float(hist_data['Close'].rolling(20).mean().iloc[-1]) if len(hist_data) >= 20 else current_price
            sma_50 = float(hist_data['Close'].rolling(50).mean().iloc[-1]) if len(hist_data) >= 50 else current_price
            
            # 52-week high/low
            week_52_high = float(hist_data['High'].rolling(min(252, len(hist_data))).max().iloc[-1])
            week_52_low = float(hist_data['Low'].rolling(min(252, len(hist_data))).min().iloc[-1])
            
            return {
                "symbol": self.symbol,
                "price": current_price,
                "change_percent": ((current_price - prev_close) / prev_close) * 100 if prev_close > 0 else 0,
                "volume": current_volume,
                "week_52_high": week_52_high,
                "week_52_low": week_52_low,
                "rsi": rsi,
                "sma_20": sma_20,
                "sma_50": sma_50
            }
        except Exception as e:
            print(f"⚠️ Error calculating indicators: {e}")
            return None
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        try:
            if len(prices) < period + 1:
                return 50.0
                
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            if loss.iloc[-1] == 0:
                return 100.0
                
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            rsi_value = rsi.iloc[-1]
            
            if np.isnan(rsi_value):
                return 50.0
                
            return float(rsi_value)
        except Exception:
            return 50.0
    
    def _calculate_actual_outcome(self, current_data, future_data):
        """Calculate what actually happened in the prediction window"""
        if future_data.empty:
            return None
            
        try:
            current_price = current_data['Close'].iloc[-1]
            final_price = future_data['Close'].iloc[-1]
            
            price_change_pct = ((final_price - current_price) / current_price) * 100
            max_price = future_data['High'].max()
            min_price = future_data['Low'].min()
            
            # Determine actual direction with thresholds
            threshold = 1.5  # 1.5% threshold to avoid noise
            if price_change_pct > threshold:
                direction = "BUY"
            elif price_change_pct < -threshold:
                direction = "SELL"
            else:
                direction = "HOLD"
                
            return {
                "price_change_pct": float(price_change_pct),
                "direction": direction,
                "max_gain_pct": float(((max_price - current_price) / current_price) * 100),
                "max_loss_pct": float(((min_price - current_price) / current_price) * 100),
                "volatility": float(future_data['Close'].std() / current_price * 100)
            }
        except Exception as e:
            print(f"⚠️ Error calculating outcome: {e}")
            return None

    async def run_training(self, training_points):
        """Execute training on historical data points"""
        if not self.system:
            raise Exception("System not initialized. Call initialize_system() first.")
            
        print(f"🚀 Starting training on {len(training_points)} data points...")
        
        successful_analyses = 0
        total_accuracy = 0.0
        
        for i, point in enumerate(training_points):
            try:
                if (i + 1) % 10 == 0:
                    print(f"📈 Training point {i+1}/{len(training_points)} - {point['date'].strftime('%Y-%m-%d')}")
                
                # Run analysis on historical data point
                analysis = await self.system.technical_agent.analyze_with_memory(point['stock_data'])
                
                # Calculate accuracy based on actual outcome
                accuracy = self._calculate_training_accuracy(analysis, point['actual_outcome'])
                total_accuracy += accuracy
                
                # Create memory with actual outcome
                memory_id = f"training_{self.symbol}_{point['date'].strftime('%Y%m%d_%H%M%S')}"
                
                # Store training result
                await self._store_training_result(memory_id, point, analysis, accuracy)
                
                successful_analyses += 1
                
                # Update performance metrics every 20 analyses
                if successful_analyses % 20 == 0:
                    await self.system.sql_memory.update_performance_metrics("SmartTechnicalAnalyst")
                    
            except Exception as e:
                print(f"⚠️ Error in training point {i+1}: {e}")
                continue
        
        avg_accuracy = total_accuracy / successful_analyses if successful_analyses > 0 else 0
        
        print(f"✅ Training completed: {successful_analyses}/{len(training_points)} successful analyses")
        print(f"📊 Average training accuracy: {avg_accuracy:.1%}")
        
        # Final performance update
        if successful_analyses > 0:
            await self.system.sql_memory.update_performance_metrics("SmartTechnicalAnalyst")
            
            # Generate training report
            await self._generate_training_report(successful_analyses, avg_accuracy)
    
    def _calculate_training_accuracy(self, analysis, actual_outcome):
        """Calculate accuracy score for training data"""
        try:
            predicted_direction = self._score_to_direction(analysis['score'])
            actual_direction = actual_outcome['direction']
            
            # Direction accuracy
            if predicted_direction == actual_direction:
                direction_score = 1.0
            elif predicted_direction == "HOLD":
                direction_score = 0.6  # Neutral gets partial credit
            else:
                direction_score = 0.0
            
            # Confidence calibration
            confidence = analysis.get('confidence', 0.5)
            actual_volatility = actual_outcome.get('volatility', 5.0)
            
            # Penalize overconfidence in volatile conditions
            volatility_penalty = min(actual_volatility / 15.0, 0.3)
            if confidence > 0.8 and direction_score < 0.5:
                volatility_penalty *= 2  # Double penalty for overconfident wrong predictions
            
            calibrated_score = direction_score * (1.0 - volatility_penalty)
            
            # Final accuracy score
            accuracy = max(0.0, min(1.0, calibrated_score))
            
            return accuracy
            
        except Exception as e:
            print(f"⚠️ Error calculating accuracy: {e}")
            return 0.0
    
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
        try:
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
                    "confidence": analysis.get('confidence', 0.5)
                },
                actual_outcome=point['actual_outcome'],
                accuracy_score=accuracy,
                confidence=analysis.get('confidence', 0.5),
                market_conditions={
                    "rsi": point['stock_data'].get("rsi"),
                    "price_position": (point['stock_data']["price"] - point['stock_data']["week_52_low"]) /
                                      max(1, point['stock_data']["week_52_high"] - point['stock_data']["week_52_low"]),
                    "volume_ratio": point['stock_data'].get("volume", 0) / 10000000
                }
            )
            
            # Store in both memory systems
            await self.system.vector_memory.store_analysis_memory(memory)
            await self.system.sql_memory.store_analysis(memory)
            
        except Exception as e:
            print(f"⚠️ Error storing training result: {e}")
    
    async def _generate_training_report(self, successful_analyses, avg_accuracy):
        """Generate comprehensive training report"""
        try:
            performance = await self.system.sql_memory.get_agent_performance("SmartTechnicalAnalyst")
            learning_insights = await self.system.learning_engine.learn_and_improve_prompts("SmartTechnicalAnalyst")
            
            report = {
                "training_summary": {
                    "symbol": self.symbol,
                    "training_period_months": self.months_back,
                    "prediction_window_days": self.prediction_window,
                    "successful_analyses": successful_analyses,
                    "average_training_accuracy": avg_accuracy,
                    "completed_at": datetime.now().isoformat()
                },
                "performance_metrics": {
                    "total_predictions": performance.total_predictions if performance else 0,
                    "accuracy_rate": performance.accuracy_rate if performance else 0,
                    "improvement_trend": performance.improvement_trend if performance else "unknown",
                    "recent_performance": performance.recent_performance if performance else []
                },
                "learning_insights": learning_insights
            }
            
            # Ensure results directory exists
            Path("results").mkdir(exist_ok=True)
            
            # Save training report
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"training_report_{self.symbol}_{timestamp}.json"
            filepath = Path("results") / filename
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            print(f"📊 Training report saved: {filepath}")
            
            # Display summary
            if performance:
                print(f"""
🎯 TRAINING SUMMARY FOR {self.symbol}:
   Training Period: {self.months_back} months
   Total Analyses: {performance.total_predictions}
   System Accuracy Rate: {performance.accuracy_rate:.1%}
   Training Accuracy: {avg_accuracy:.1%}
   Improvement Trend: {performance.improvement_trend}
   Learning Suggestions: {len(learning_insights.get('suggestions', []))}
""")
                
                if learning_insights.get('suggestions'):
                    print("💡 KEY LEARNING INSIGHTS:")
                    for suggestion in learning_insights['suggestions'][:3]:
                        print(f"   • {suggestion.get('recommendation', 'N/A')}")
                        
        except Exception as e:
            print(f"⚠️ Error generating training report: {e}")


async def main():
    """Main training execution function"""
    print("🧠 Memory Enhanced Stock System - Training Mode")
    print("=" * 60)
    
    # Configuration - Modify these as needed
    SYMBOL = "GOOG"      # Stock symbol to train on
    MONTHS_BACK = 3      # Months of historical data to use
    
    # You can also train on multiple stocks
    # SYMBOLS = ["AAPL", "GOOGL", "MSFT", "TSLA"]
    
    try:
        # Single stock training
        trainer = StockTrainer(SYMBOL, MONTHS_BACK)
        
        # Initialize system
        await trainer.initialize_system()
        
        # Prepare dataset
        training_points = trainer.prepare_training_dataset()
        
        if len(training_points) < 10:
            print(f"⚠️ Warning: Only {len(training_points)} training points available. Need at least 10 for meaningful training.")
            return
        
        # Run training
        await trainer.run_training(training_points)
        
        print("\n🎉 Training completed successfully!")
        print(f"📂 Check the 'results/' folder for detailed training reports")
        print(f"🧠 Memory databases updated in 'memory/' folder")
        
        # Optional: Test the trained system
        print(f"\n🔬 Testing trained system on current {SYMBOL} data...")
        current_result = await trainer.system.analyze_with_memory([SYMBOL])
        
        if current_result:
            analysis = current_result[0]['analysis']
            print(f"🎯 Current analysis: Score {analysis['score']:.1f}/10, Confidence {analysis['confidence']:.1%}")
            print(f"📈 Historical context used: {analysis['used_historical_context']}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())