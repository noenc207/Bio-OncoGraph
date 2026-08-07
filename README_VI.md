# BioOncoGraph v2.0

BioOncoGraph là một khuôn khổ mã nguồn mở toàn diện để xây dựng và phân tích các đồ thị tri thức y sinh học với trọng tâm cụ thể vào ung thư học.

## Tính năng
- Xây dựng đồ thị tri thức có thể mở rộng
- Khả năng Mạng nơ-ron đồ thị (GNN) tiên tiến
- Quy trình end-to-end cho huấn luyện, đánh giá và dự đoán
- Các mô hình được huấn luyện sẵn cho nhận dạng thực thể có tên trong ung thư học và trích xuất mối quan hệ

## Bắt đầu nhanh
```bash
pip install bio-oncograph
bio-oncograph train --config configs/train.yaml
bio-oncograph evaluate --model-path models/best_model
```

## Tài liệu
Để xem tài liệu đầy đủ, vui lòng truy cập [Trang tài liệu](https://bio-oncograph.readthedocs.io/) của chúng tôi.

## Giấy phép
Apache License 2.0. Xem [LICENSE](LICENSE) để biết chi tiết.
