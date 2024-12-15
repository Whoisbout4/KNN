import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from flask import Flask, request, jsonify, render_template
import matplotlib.pyplot as plt
import seaborn as sns
import os
import io
import base64

# Tải dữ liệu
data_path = '/Users/HUY BAO/Desktop/KNN/teleCust1000t.csv'
data = pd.read_csv(data_path)

# Tiền xử lý dữ liệu
def tien_xu_ly_du_lieu(data):
    # Loại bỏ các hàng không có giá trị mục tiêu
    data = data.dropna(subset=['custcat'])

    # Tách dữ liệu thành đặc trưng (X) và mục tiêu (y)
    X = data.drop(columns=['custcat'])
    y = data['custcat']

    # Chuẩn hóa các đặc trưng
    scaler = StandardScaler()
    X_chuan_hoa = scaler.fit_transform(X)

    return X_chuan_hoa, y

X, y = tien_xu_ly_du_lieu(data)

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình KNN
def huan_luyen_mo_hinh_knn(X_train, y_train, so_lan_can_ke):
    """
    Huấn luyện mô hình KNN với số lượng láng giềng gần nhất `so_lan_can_ke`.
    Giá trị `so_lan_can_ke` quyết định số lượng điểm lân cận để xác định nhãn dự đoán.
    """
    knn = KNeighborsClassifier(n_neighbors=so_lan_can_ke)
    knn.fit(X_train, y_train)
    return knn

# Cài đặt Flask
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train_model():
    try:
        # Lấy giá trị k từ yêu cầu của người dùng
        data = request.json
        k = data.get('k', 5)  # Giá trị mặc định của k là 5 nếu không được truyền

        if not isinstance(k, int) or k <= 0:
            return jsonify({"error": "Giá trị k phải là số nguyên dương"}), 400

        # Huấn luyện mô hình với k được chọn
        global knn_model
        knn_model = huan_luyen_mo_hinh_knn(X_train, y_train, k)

        # Đánh giá mô hình
        y_du_doan = knn_model.predict(X_test)
        do_chinh_xac = accuracy_score(y_test, y_du_doan)
        return jsonify({"message": "Mô hình đã được huấn luyện", "do_chinh_xac": do_chinh_xac})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def du_doan():
    try:
        # Kiểm tra xem có file được tải lên không
        if 'file' not in request.files:
            return jsonify({"error": "Không có file được tải lên"}), 400
        file = request.files['file']

        # Kiểm tra file có hợp lệ hay không
        if file.filename == '':
            return jsonify({"error": "Tên file không hợp lệ"}), 400
        if not file.filename.endswith('.csv'):
            return jsonify({"error": "Vui lòng tải lên file CSV"}), 400

        # Đọc dữ liệu từ file CSV
        du_lieu_df = pd.read_csv(file)

        # Kiểm tra xem các cột trong file có đúng không
        if 'custcat' not in du_lieu_df.columns:
            return jsonify({"error": "File không chứa cột 'custcat'"}), 400

        # Tiền xử lý dữ liệu
        X_du_lieu = du_lieu_df.drop(columns=['custcat'])
        scaler = StandardScaler()
        X_chuan_hoa = scaler.fit_transform(X_du_lieu)

        # Dự đoán với mô hình đã huấn luyện
        du_doan_ket_qua = knn_model.predict(X_chuan_hoa)
        
        # Trả về kết quả dự đoán
        du_lieu_df['predicted_custcat'] = du_doan_ket_qua.tolist()
        return jsonify(du_lieu_df.to_dict(orient="records"))
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/analysis')
def analysis():
    try:
        # Trả về thông tin phân tích
        loai_khach = data['custcat'].value_counts()
        dac_trung_nhom = data.groupby('custcat').mean()
        return jsonify({
            "loai_khach": loai_khach.to_dict(),
            "dac_trung_nhom": dac_trung_nhom.to_dict()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/k_analysis', methods=['GET'])
def k_analysis():
    try:
        # Thử nghiệm với các giá trị k khác nhau
        k_values = range(1, 21)
        accuracies = []

        for k in k_values:
            knn = huan_luyen_mo_hinh_knn(X_train, y_train, k)
            y_pred = knn.predict(X_test)
            accuracies.append(accuracy_score(y_test, y_pred))

        # Vẽ đồ thị
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, accuracies, marker='o', linestyle='-', color='b')
        plt.title('Ảnh hưởng của k đến độ chính xác của mô hình')
        plt.xlabel('Số lượng láng giềng gần nhất (k)')
        plt.ylabel('Độ chính xác')
        plt.grid(True)

        # Lưu đồ thị vào bộ nhớ dưới dạng hình ảnh
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        img_base64 = base64.b64encode(img.getvalue()).decode()
        plt.close()

        return jsonify({"image": img_base64})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Chạy Flask
    app.run(debug=True, host='0.0.0.0', port=5000)
