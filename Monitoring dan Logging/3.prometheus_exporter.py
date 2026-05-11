from prometheus_client import start_http_server, Counter, Histogram, Gauge
import time
import random
import psutil

REQUEST_COUNT = Counter('model_requests_total', 'Total number of requests to the model', ['endpoint'])
ERROR_COUNT = Counter('model_errors_total', 'Total number of errors', ['endpoint'])
LATENCY = Histogram('model_response_time_seconds', 'Response time in seconds', ['endpoint'])
PREDICTION_SCORE = Gauge('model_prediction_score_average', 'Average prediction score output by model')
CPU_USAGE = Gauge('system_cpu_usage_percent', 'Current CPU usage percentage')
MEMORY_USAGE = Gauge('system_memory_usage_percent', 'Current Memory usage percentage')
ACTIVE_REQUESTS = Gauge('model_active_requests', 'Number of currently active requests')
MODEL_CONFIDENCE = Histogram('model_confidence_score', 'Confidence score of predictions', buckets=[0.1, 0.5, 0.8, 0.9, 0.95, 0.99])
DATA_DRIFT_SCORE = Gauge('model_data_drift_score', 'Calculated data drift score')
OUTLIER_DETECTED = Counter('model_outliers_total', 'Total number of outliers detected in input data')
BATCH_SIZE = Histogram('model_inference_batch_size', 'Number of instances per request batch')

def process_request():
    ACTIVE_REQUESTS.inc()
    start_time = time.time()
    
    time.sleep(random.uniform(0.1, 0.5))
    
    endpoint = "/predict"
    REQUEST_COUNT.labels(endpoint=endpoint).inc()
    
    confidence = random.uniform(0.5, 1.0)
    MODEL_CONFIDENCE.observe(confidence)
    PREDICTION_SCORE.set(confidence)
    
    batch = random.randint(1, 32)
    BATCH_SIZE.observe(batch)
    
    if random.random() < 0.05:
        ERROR_COUNT.labels(endpoint=endpoint).inc()
        
    if random.random() < 0.10:
        OUTLIER_DETECTED.inc()
        
    CPU_USAGE.set(psutil.cpu_percent())
    MEMORY_USAGE.set(psutil.virtual_memory().percent)
    
    DATA_DRIFT_SCORE.set(random.uniform(0.01, 0.15))

    LATENCY.labels(endpoint=endpoint).observe(time.time() - start_time)
    ACTIVE_REQUESTS.dec()

if __name__ == '__main__':
    print("Memulai Prometheus Exporter di port 8000...")

    start_http_server(8000)
    
    print("Exporter siap menerima data dari inference.py.")
    while True:
        process_request()
        time.sleep(random.uniform(0.5, 2.0))
