from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

_model = None
_pipe = None

def init_news_model(model_name='yiyanghkust/finbert-tone'):
    global _model, _pipe
    _pipe = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)

def score_text(text):
    # returns numeric score approx in [-1, 1]
    global _pipe
    if _pipe is None:
        init_news_model()
    res = _pipe(text[:512]) # truncate
    label = res[0]['label']
    score = res[0]['score']
    if label.lower().startswith('positive'):
        return score
    if label.lower().startswith('negative'):
        return -score
    return 0.0