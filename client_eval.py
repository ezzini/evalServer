import requests
import json

# الرابط الذي حصلت عليه من ngrok
API_URL = "https://4cae70ec10a7.ngrok-free.app/predict"

def evaluate_submission(file_path):
    # فرضاً أن ملف التقديم يحتوي على نصوص (سطر بسطر)
    with open(file_path, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f.readlines() if line.strip()]

    # إرسال البيانات إلى جهازك المحلي عبر ngrok
    payload = {"sentences": sentences}
    
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        results = response.json()["predictions"]
        
        # معالجة النتائج (مثلاً طباعة أول 5 توقعات)
        for i, res in enumerate(results[:5]):
            print(f"Sentence: {sentences[i]} -> Label: {res['label']} ({res['score']:.2f})")
            
        return results
    except Exception as e:
        print(f"Error connecting to local API: {e}")
        return None

# تشغيل التقييم
evaluate_submission("submission.txt")
