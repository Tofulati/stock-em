from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Configuration - use your actual results directory
RESULTS_DIR = "data/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Track running backtests
running_backtests = {}


def run_backtest_async(ticker, period, use_ensemble, epochs, request_id):
    """
    Runs the backtest in a separate thread and updates status
    """
    try:
        running_backtests[request_id] = {
            'status': 'running',
            'progress': 0,
            'step': 'Initializing pipeline...',
            'ticker': ticker,
            'started_at': datetime.now().isoformat()
        }
        
        # Update progress
        running_backtests[request_id]['step'] = 'Downloading price data...'
        running_backtests[request_id]['progress'] = 10
        
        # Import your toy_run module
        from toy_run import improved_pipeline_with_ensemble
        
        running_backtests[request_id]['step'] = 'Training model (this may take 10-30 minutes)...'
        running_backtests[request_id]['progress'] = 20
        
        # Run the pipeline
        results = improved_pipeline_with_ensemble(
            ticker=ticker,
            period=period,
            seq_len=30,
            batch_size=64,
            epochs=epochs,
            use_ensemble=use_ensemble,
            test_live=False,  # Disable live testing for speed
            enable_shorting=True,
            export_results=True
        )
        
        # Mark as complete
        running_backtests[request_id]['status'] = 'completed'
        running_backtests[request_id]['progress'] = 100
        running_backtests[request_id]['step'] = 'Backtest complete!'
        running_backtests[request_id]['result_file'] = f'model_results_{ticker}.json'
        
    except Exception as e:
        print(f"Error in backtest thread: {e}")
        import traceback
        traceback.print_exc()
        running_backtests[request_id]['status'] = 'error'
        running_backtests[request_id]['error'] = str(e)
        running_backtests[request_id]['step'] = f'Error: {str(e)[:100]}'


@app.route('/api/run_backtest', methods=['POST'])
def run_backtest():
    """
    Endpoint to trigger a new backtest
    
    Expected JSON payload:
    {
        "ticker": "AAPL",
        "period": "5y",
        "use_ensemble": true,
        "epochs": 150
    }
    """
    try:
        data = request.json
        ticker = data.get('ticker', 'AAPL').upper()
        period = data.get('period', '5y')
        use_ensemble = data.get('use_ensemble', True)
        epochs = data.get('epochs', 150)
        
        # Validate ticker
        if not ticker or len(ticker) > 5:
            return jsonify({'error': 'Invalid ticker symbol'}), 400
        
        # Generate request ID
        request_id = f"{ticker}_{int(time.time())}"
        
        # Start backtest in background thread
        thread = threading.Thread(
            target=run_backtest_async,
            args=(ticker, period, use_ensemble, epochs, request_id)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'request_id': request_id,
            'message': f'Backtest started for {ticker}',
            'status': 'running',
            'estimated_time': '10-30 minutes'
        }), 202
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/backtest_status/<request_id>', methods=['GET'])
def backtest_status(request_id):
    """
    Check the status of a running backtest
    """
    if request_id not in running_backtests:
        return jsonify({'error': 'Request ID not found'}), 404
    
    status_data = running_backtests[request_id].copy()
    
    # If completed, load the result file
    if status_data['status'] == 'completed':
        result_file = status_data.get('result_file')
        if result_file:
            result_path = os.path.join(RESULTS_DIR, result_file)
            if os.path.exists(result_path):
                try:
                    with open(result_path, 'r') as f:
                        status_data['results'] = json.load(f)
                except Exception as e:
                    status_data['error'] = f'Failed to load results: {str(e)}'
    
    return jsonify(status_data)


@app.route('/api/results/<ticker>', methods=['GET'])
def get_results(ticker):
    """
    Retrieve results for a specific ticker
    """
    ticker = ticker.upper()
    result_file = f'model_results_{ticker}.json'
    result_path = os.path.join(RESULTS_DIR, result_file)
    
    if not os.path.exists(result_path):
        return jsonify({'error': f'No results found for {ticker}'}), 404
    
    try:
        with open(result_path, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/list_results', methods=['GET'])
def list_results():
    """
    List all available backtest results
    """
    try:
        files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('.json')]
        results = []
        
        for filename in files:
            filepath = os.path.join(RESULTS_DIR, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    results.append({
                        'filename': filename,
                        'ticker': data.get('ticker'),
                        'period': data.get('period'),
                        'use_ensemble': data.get('use_ensemble'),
                        'total_return': data.get('metrics', {}).get('Total Return (%)', 0),
                        'modified': datetime.fromtimestamp(
                            os.path.getmtime(filepath)
                        ).isoformat()
                    })
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue
        
        return jsonify({'results': results, 'count': len(results)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/delete_result/<ticker>', methods=['DELETE'])
def delete_result(ticker):
    """
    Delete results for a specific ticker
    """
    ticker = ticker.upper()
    result_file = f'model_results_{ticker}.json'
    result_path = os.path.join(RESULTS_DIR, result_file)
    
    if not os.path.exists(result_path):
        return jsonify({'error': f'No results found for {ticker}'}), 404
    
    try:
        os.remove(result_path)
        return jsonify({'message': f'Results for {ticker} deleted successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'active_backtests': len([b for b in running_backtests.values() if b['status'] == 'running']),
        'results_directory': os.path.abspath(RESULTS_DIR)
    })


@app.route('/')
def index():
    """
    Serve info page
    """
    active_count = len([b for b in running_backtests.values() if b['status'] == 'running'])
    
    # Count result files
    try:
        result_files = len([f for f in os.listdir(RESULTS_DIR) if f.endswith('.json')])
    except:
        result_files = 0
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Trading Model Backend</title>
        <style>
            body {{ 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background: #0f172a;
                color: #e5e7eb;
            }}
            h1 {{ color: #60a5fa; }}
            h2 {{ color: #34d399; margin-top: 30px; }}
            .status {{ 
                background: #1e293b;
                padding: 15px;
                border-radius: 8px;
                margin: 20px 0;
            }}
            .endpoint {{ 
                background: #374151;
                padding: 10px;
                margin: 5px 0;
                border-radius: 5px;
                font-family: monospace;
            }}
            .method {{ 
                display: inline-block;
                padding: 2px 8px;
                border-radius: 3px;
                font-weight: bold;
                margin-right: 10px;
            }}
            .post {{ background: #10b981; color: white; }}
            .get {{ background: #3b82f6; color: white; }}
            .delete {{ background: #ef4444; color: white; }}
        </style>
    </head>
    <body>
        <h1>Trading Model Backend Server</h1>
        <div class="status">
            <strong>Status:</strong> Running<br>
            <strong>Active Backtests:</strong> {active_count}<br>
            <strong>Saved Results:</strong> {result_files}<br>
            <strong>Results Directory:</strong> {os.path.abspath(RESULTS_DIR)}
        </div>

        <h2>Available Endpoints</h2>
        
        <div class="endpoint">
            <span class="method post">POST</span>
            <strong>/api/run_backtest</strong> - Start a new backtest
        </div>
        
        <div class="endpoint">
            <span class="method get">GET</span>
            <strong>/api/backtest_status/&lt;request_id&gt;</strong> - Check backtest status
        </div>
        
        <div class="endpoint">
            <span class="method get">GET</span>
            <strong>/api/results/&lt;ticker&gt;</strong> - Get results for a ticker
        </div>
        
        <div class="endpoint">
            <span class="method get">GET</span>
            <strong>/api/list_results</strong> - List all available results
        </div>
        
        <div class="endpoint">
            <span class="method delete">DELETE</span>
            <strong>/api/delete_result/&lt;ticker&gt;</strong> - Delete results
        </div>
        
        <div class="endpoint">
            <span class="method get">GET</span>
            <strong>/health</strong> - Health check
        </div>

        <h2>Quick Test</h2>
        <p>Test the API with curl:</p>
        <pre style="background: #1e293b; padding: 15px; border-radius: 5px; overflow-x: auto;">
curl -X POST http://localhost:5000/api/run_backtest \\
  -H "Content-Type: application/json" \\
  -d '{{"ticker":"AAPL","period":"1y","use_ensemble":false,"epochs":50}}'
        </pre>
    </body>
    </html>
    """


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Trading Model Backend Server")
    print("=" * 60)
    print(f"Server:    http://localhost:5000")
    print(f"Results:   {os.path.abspath(RESULTS_DIR)}")
    print(f"Status:    Ready")
    print("\nAvailable endpoints:")
    print("  - POST   /api/run_backtest")
    print("  - GET    /api/backtest_status/<request_id>")
    print("  - GET    /api/results/<ticker>")
    print("  - GET    /api/list_results")
    print("  - DELETE /api/delete_result/<ticker>")
    print("  - GET    /health")
    print("\nOpen http://localhost:5000 in browser for details")
    print("Press CTRL+C to stop")
    print("=" * 60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)