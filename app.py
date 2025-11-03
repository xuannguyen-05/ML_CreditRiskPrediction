import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# Load pipeline (chỉ 1 file duy nhất)
try:
    pipeline = joblib.load('credit_risk_pipeline.pkl')
    print("Đã load pipeline thành công!")
except Exception as e:
    print(f"Lỗi khi load pipeline: {e}")
    pipeline = None

# Định nghĩa route cho trang chủ
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if pipeline is None:
        return jsonify({'error': 'Pipeline chưa được load.'}), 500

    try:
        data = request.get_json(force=True)
        
        # 1. Tạo DataFrame (Giữ nguyên)
        feature_cols = [
            "Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed",
            "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio",
            "Education", "EmploymentType", "HasMortgage", "HasDependents", "HasCoSigner",
            "MaritalStatus", "LoanPurpose"
        ]
        input_data = pd.DataFrame([data], columns=feature_cols)

        # 2. LẤY XÁC SUẤT (THAY ĐỔI LỚN NHẤT)
        # Chúng ta chỉ lấy xác suất của class "1" (Default/Vỡ nợ)
        prediction_proba = pipeline.predict_proba(input_data)
        risk_score = prediction_proba[0][1] # Lấy xác suất của class '1'

        # 3. LOGIC "ĐÈN GIAO THÔNG" THỰC TẾ
        recommendation = ""
        recommendation_class = "" # Dùng cho CSS
        reason = ""

        if risk_score > 0.70: # 70%
            # Đèn Đỏ: Rủi ro quá cao
            recommendation = "Tự động Từ chối"
            recommendation_class = "status-reject"
            reason = "Điểm rủi ro (70%) vượt ngưỡng an toàn."
        elif risk_score < 0.20: # 20%
            # Đèn Xanh: Rủi ro rất thấp
            recommendation = "Tự động Duyệt"
            recommendation_class = "status-approve"
            reason = "Điểm rủi ro (20%) nằm trong vùng an toàn."
        else:
            # Đèn Vàng: Vùng xám (từ 20% - 70%)
            recommendation = "Chuyển duyệt thủ công"
            recommendation_class = "status-review"
            reason = "Điểm rủi ro nằm trong vùng 'cần xem xét'. Yêu cầu chuyên viên tín dụng đánh giá thêm."

        # 4. Trả kết quả về cho frontend
        return jsonify({
            'risk_score': f'{(risk_score * 100):.2f}%', # Ví dụ: "45.12%"
            'recommendation': recommendation,
            'recommendation_class': recommendation_class,
            'reason': reason
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Chạy ứng dụng
if __name__ == '__main__':
    app.run(debug=True)