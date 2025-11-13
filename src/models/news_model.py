from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

# Global model cache for efficiency
_model = None
_pipe = None
_advanced_model = None

def init_news_model(model_name='yiyanghkust/finbert-tone'):
    """Initialize basic news model (legacy)"""
    global _model, _pipe
    _pipe = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)

def score_text(text):
    """Score single text (legacy function)"""
    global _pipe
    if _pipe is None:
        init_news_model()
    res = _pipe(text[:512])
    label = res[0]['label']
    score = res[0]['score']
    if label.lower().startswith('positive'):
        return score
    if label.lower().startswith('negative'):
        return -score
    return 0.0


class AdvancedNewsModel:
    """
    Enhanced news sentiment with:
    - Multi-source aggregation
    - Temporal decay
    - Sentiment momentum
    - News volume tracking
    """
    def __init__(self, model_name="ProsusAI/finbert"):
        print("Initializing Advanced News Model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
    def fetch_news_yfinance(self, ticker, lookback_days=7):
        """Fetch news from Yahoo Finance"""
        try:
            stock = yf.Ticker(ticker)
            news = stock.news
            
            headlines = []
            cutoff_date = datetime.now() - timedelta(days=lookback_days)
            
            for item in news:
                pub_date = datetime.fromtimestamp(item.get('providerPublishTime', 0))
                if pub_date >= cutoff_date:
                    headlines.append({
                        'title': item.get('title', ''),
                        'date': pub_date,
                        'source': 'yfinance'
                    })
            
            return headlines
        except Exception as e:
            print(f"Warning: Could not fetch news: {e}")
            return []
    
    def score_sentiment(self, text):
        """Score sentiment of single text"""
        if not text or len(text.strip()) == 0:
            return 0.0
        
        inputs = self.tokenizer(
            text[:512],
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            # FinBERT: [positive, negative, neutral]
            score = probs[0][0].item() - probs[0][1].item()
        
        return score
    
    def score_with_temporal_decay(self, headlines_with_dates, half_life_days=2):
        """
        Score news with exponential decay for recency
        Recent news matters more than old news
        """
        if not headlines_with_dates:
            return {
                'score': 0.0,
                'momentum': 0.0,
                'volume': 0,
                'controversy': 0.0
            }
        
        scores = []
        weights = []
        dates = []
        
        now = datetime.now()
        decay_constant = np.log(2) / half_life_days
        
        for item in headlines_with_dates:
            headline = item['title']
            date = item['date']
            source = item.get('source', 'unknown')
            
            # Get sentiment score
            score = self.score_sentiment(headline)
            scores.append(score)
            dates.append(date)
            
            # Calculate recency weight
            days_ago = (now - date).total_seconds() / 86400
            recency_weight = np.exp(-decay_constant * days_ago)
            
            # Source reliability weight
            source_weights = {
                'yfinance': 1.0,
                'newsapi': 1.0,
                'finnhub': 0.9,
                'reddit': 0.6
            }
            source_weight = source_weights.get(source, 0.7)
            
            final_weight = recency_weight * source_weight
            weights.append(final_weight)
        
        # Weighted average sentiment
        if len(scores) > 0:
            weighted_score = np.average(scores, weights=weights)
            
            # Calculate momentum (recent vs old)
            sorted_data = sorted(zip(dates, scores), key=lambda x: x[0])
            scores_sorted = [s for d, s in sorted_data]
            
            split_idx = len(scores_sorted) // 2
            if split_idx > 0:
                old_sentiment = np.mean(scores_sorted[:split_idx])
                recent_sentiment = np.mean(scores_sorted[split_idx:])
                momentum = recent_sentiment - old_sentiment
            else:
                momentum = 0.0
            
            # Controversy = standard deviation
            controversy = np.std(scores)
            
            return {
                'score': weighted_score,
                'momentum': momentum,
                'volume': len(scores),
                'controversy': controversy
            }
        
        return {
            'score': 0.0,
            'momentum': 0.0,
            'volume': 0,
            'controversy': 0.0
        }
    
    def get_news_features(self, ticker, lookback_days=7):
        """
        Get comprehensive news features for a ticker
        Returns dict that can be added to DataFrame
        """
        headlines = self.fetch_news_yfinance(ticker, lookback_days)
        
        if not headlines:
            return {
                'news_score': 0.0,
                'news_momentum': 0.0,
                'news_volume': 0,
                'news_controversy': 0.0
            }
        
        news_data = self.score_with_temporal_decay(headlines)
        
        return {
            'news_score': news_data['score'],
            'news_momentum': news_data['momentum'],
            'news_volume': news_data['volume'],
            'news_controversy': news_data['controversy']
        }


def get_advanced_model():
    """Get singleton instance of advanced model"""
    global _advanced_model
    if _advanced_model is None:
        _advanced_model = AdvancedNewsModel()
    return _advanced_model