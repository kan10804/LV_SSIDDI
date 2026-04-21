Phan Hoàng Khang 
B2205943
Giới thiệu:
Dự án này tập trung vào bài toán dự đoán tương tác thuốc – thuốc (DDI) bằng các mô hình học sâu trên dữ liệu đồ thị. Mô hình đề xuất kết hợp Graph Isomorphism Network (GIN) và Multi-Co-Attention nhằm khai thác hiệu quả đặc trưng cấu trúc phân tử và mối quan hệ tương tác giữa các cặp thuốc.

Sử dụng hai tập dữ liệu:
   DrugBank: dữ liệu có cấu trúc rõ ràng, dựa trên cơ chế sinh học
   TWOSIDES: dữ liệu thực tế, nhiều nhiễu và phức tạp
   Dữ liệu được tiền xử lý và áp dụng negative sampling để giảm mất cân bằng.
Phương pháp
   Biểu diễn mỗi phân tử dưới dạng đồ thị
   Sử dụng GIN để học đặc trưng cấu trúc
   Áp dụng Multi-Co-Attention để mô hình hóa tương tác giữa hai thuốc
   Kết hợp đặc trưng để dự đoán tương tác
