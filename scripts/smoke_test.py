import json
import time
import urllib.request
from http.client import HTTPException

def main():
    base = 'http://127.0.0.1:8000'
    for attempt in range(90):
        try:
            with urllib.request.urlopen(base+'/health', timeout=2) as response:
                assert json.load(response)['model_loaded']
            break
        except (OSError, HTTPException):
            time.sleep(1)
    else:
        raise RuntimeError('API did not become ready')
    body = dict(hours_studied=7, previous_scores=99, extracurricular_activities='Yes', sleep_hours=9, sample_question_papers_practiced=1)
    req = urllib.request.Request(base+'/predict', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=10) as response:
        result = json.load(response)
    assert 80 < result['predicted_performance_index'] < 105, result
    print(json.dumps(result, indent=2))
    with urllib.request.urlopen(base+'/model-info', timeout=10) as response:
        assert json.load(response)['metrics']['r2'] > 0.9
    with urllib.request.urlopen(base+'/openapi.json', timeout=10) as response:
        assert '/predict' in json.load(response)['paths']
    print('Docker endpoint checks passed')

if __name__ == '__main__':
    main()
