

Dưới đây là bản kế hoạch chi tiết và đầy đủ nhất cho bài tập lớn môn Deep Learning của bạn. Kế hoạch này được thiết kế để cân bằng giữa tính tham vọng (8 models x 4 datasets) và tính khả thi thông qua chiến lược tối ưu hóa tài nguyên thông minh.

---

# PROJECT PLAN: COMPREHENSIVE BENCHMARK OF CNN ARCHITECTURES (FROM SCRATCH)

## 1. Đối tượng Nghiên cứu (Scope & Objects)

### 1.1 Danh sách Mô hình (8 Models)
Chia thành 3 nhóm theo đặc điểm kiến trúc để phân tích hành vi:

*   **Group 1: Legacy (Cổ điển)**
    *   **AlexNet:** Kiến trúc tiên phong, dùng để thấy sự tiến hóa của độ sâu.
    *   **VGG16:** Phiên bản `vgg16_bn` (có Batch Norm) - yêu cầu bắt buộc để train from scratch hội tụ tốt hơn.
*   **Group 2: Residual (Tiêu chuẩn)**
    *   **ResNet18 & ResNet34:** Mạng mỏng nhẹ, phổ biến cho Edge devices.
    *   **ResNet50:** Mạng tiêu chuẩn cho bài toán cân bằng hiệu năng/tài nguyên.
*   **Group 3: Modern/Efficient (Hiện đại & Tối ưu)**
    *   **MobileNetV3 (Large):** Tối ưu cho mobile (NAS-based).
    *   **EfficientNetV2 (B0):** Tối ưu hóa FLOPs và tốc độ train.
    *   **ConvNeXtV2 (Nano):** Kiến trúc CNN hiện đại lấy cảm hứng từ ViT, dùng Fused Layer Norm và Global Response Norm (GRN).

### 1.2 Danh sách Dữ liệu (4 Datasets)
*   **CIFAR100:** Dataset nhỏ (50k train), 100 lớp. Dùng làm **Proxy Search** cho HPO và Benchmark độ nhiễu.
*   **MIT Indoor 67:** Phân loại cảnh trong nhà. Độ phức tạp cao về texture và bố cục.
*   **Intel Image Classification:** Dữ liệu cảnh tự nhiên (Buildings, Forest, Glacier...). Dữ liệu tương đối dễ, cân bằng.
*   **Caltech101:** Dữ liệu object, đa dạng về số lượng ảnh/lớp (imbalance), thử thách khả năng xử lý dữ liệu không đều.

---

## 2. Tiêu chí Đánh giá (Benchmark Criteria)

Đánh giá đa chiều để phục vụ mục đích triển khai thực tế:

1.  **Performance (Chất lượng):**
    *   Top-1 Accuracy & Top-5 Accuracy.
    *   Generalization Gap: $|Train_{Acc} - Val_{Acc}|$ (Càng thấp càng tốt).
2.  **Efficiency (Hiệu năng tính toán):**
    *   **Throughput:** Số lượng ảnh xử lý được mỗi giây (Images/sec) trên GPU (Batch size = 32/64).
    *   **Latency (Inference Time):** Thời gian suy luận trung bình cho 1 ảnh (Batch size = 1) trên GPU và CPU (mô phỏng môi trường thực tế).
3.  **Resources (Tài nguyên):**
    *   **Peak VRAM Usage:** Bộ nhớ GPU tối đa khi train/inference.
    *   **Model Size:** Dung lượng đĩa (MB) và số lượng tham số (Params).
4.  **Deployability (Khả năng triển khai):**
    *   Ước lượng **Minimum Device Requirement** (Ví dụ: Chạy được trên Jetson Nano, hay cần V100; Chạy mượt trên CPU hay không).

---

## 3. Model Selection: Geometric-Aware Bayesian Optimization

Thay vì dùng Grid Search tốn kém, ta sử dụng **Bayesian Optimization** (thư viện `Optuna`) kết hợp phân tích hình học Loss Landscape.

### 3.1 Chiến lược Tối ưu hóa (Optimization Strategy)
Để giảm thời gian, ta không search riêng cho từng cặp (Model, Dataset).
*   **Bước 1 (Proxy Search):** Chạy HPO trên **CIFAR100** cho từng nhóm kiến trúc (Legacy, ResNet, Modern).
*   **Bước 2 (Transfer Hyperparameters):** Áp dụng bộ tham số tốt nhất tìm được để train trên các dataset còn lại (MIT, Intel, Caltech).

### 3.2 Không gian Tìm kiếm (Search Space)
*   **Optimizer:** Categorical `{SGD, AdamW, RMSprop}`.
*   **Learning Rate (LR):** Log-uniform distribution $[1e-5, 1e-3]$.
*   **Weight Decay (WD):** Log-uniform $[1e-5, 1e-1]$.
*   **Scheduler:** Categorical `{CosineAnnealing, OneCycleLR, StepLR}`.
*   **Dropout / Stochastic Depth:** Uniform $[0.0, 0.5]$ (tùy model hỗ trợ).
*   **Label Smoothing:** Uniform $[0.0, 0.2]$.

---

## 4. Thiết lập Thí nghiệm (Experimental Setup)

### 4.1 Tổng quan Quy trình Huấn luyện
*   **Train from Scratch:** Không dùng pre-trained weights. Random initialization theo phương pháp của từng paper (Kaiming He cho ResNet/VGG, Truncated Normal cho MobileNet/EfficientNet).
*   **Hardware:** Chạy trên GPU (NVIDIA V100/A100/RTX 3090 ưu tiên, nếu không đủ thì Colab Pro).
*   **Reproducibility:** Fix `RANDOM_SEED=42` cho Numpy, PyTorch và CUDA.

### 4.2 Preprocessing & Augmentation
Để train from scratch hiệu quả trên dữ liệu nhỏ, cần Augmentation mạnh:
*   **Resize:** Tất cả về $224 \times 224$ (Bicubic interpolation).
*   **Training Augmentation:**
    *   Random Resized Crop (Scale 0.8~1.0).
    *   Random Horizontal Flip (p=0.5).
    *   **Mixup & CutMix:** Kích hoạt với alpha xác định từ bước HPO (thường $\alpha \in [0.2, 1.0]$).
    *   Color Jitter (độ biến đổi nhẹ).
*   **Validation/Test:** Chỉ Resize và Center Crop.

### 4.3 Kỹ thuật Chống Overfitting & Adaptive Learning
Thay vì dùng cấu hình tĩnh, implement một **Smart Trainer** có cơ chế thích ứng:

*   **Theo dõi:** Generalization Gap $G = Loss_{val} - Loss_{train}$.
*   **Cơ chế Adaptive Logic (Kiểm tra mỗi 5 epochs):**
    *   **Trường hợp 1: Overfitting nghiêm trọng** ($G$ tăng nhanh liên tục):
        *   Tăng Weight Decay hiện tại lên 1.5x.
        *   Tăng xác suất áp dụng CutMix (nếu đang dùng).
    *   **Trường hợp 2: Underfitting** ($Loss_{train}$ giảm rất chậm, vẫn cao):
        *   Giảm cường độ Augmentation (tắt CutMix/Mixup tạm thời).
        *   Kích hoạt **Cyclical Learning Rate** để nhảy ra khỏi vùng cực tiểu cục bộ.
    *   **Trường hợp 3: Plateau (Loss đi ngang):**
        *   Kích hoạt **SAM (Sharpness-Aware Minimization)** cho 10 epoch tiếp theo để tìm vùng phẳng hơn.

---

## 5. Ablation Study (Nghiên cứu Phân rã)

Thiết lập các thí nghiệm riêng biệt trên tập CIFAR100 (Train 50 epochs) để trả lời các câu hỏi "Tại sao?":

1.  **Impact of Modern Recipe:**
    *   So sánh: ResNet50 (SGD + basic aug) vs. ResNet50 (AdamW + Mixup + CutMix + Cosine).
    *   *Mục đích:* Định lượng phần trăm improvement đến từ kiến trúc hay công thức train (recipe).
2.  **Impact of SAM (Sharpness-Aware Minimization):**
    *   So sánh: Model train với SGD tiêu chuẩn vs. SGD + SAM.
    *   *Mục đích:* Chứng minh SAM giúp giảm Top Eigenvalue và cải thiện Accuracy trên tập Test (đặc biệt khi thêm nhiễu).
3.  **Impact of Resolution (CIFAR Only):**
    *   So sánh: Train CIFAR100 trên size 32x32 (native) vs. 224x224 (resized).
    *   *Mục đích:* Xem việc resize ảnh nhỏ lên lớn có thực sự mang lại lợi ích thông tin cho các model hiện đại (như ConvNeXt) hay chỉ gây nhiễu.
4.  **Impact of Batch Norm (VGG):**
    *   So sánh: VGG16 (no BN) vs. VGG16 (with BN).
    *   *Mục đích:* Minh họa vai trò sống còn của BN trong việc train mạng sâu from scratch.

---

## 6. Đánh giá Độ tin cậy & Robustness (Evaluation Protocol)

### 6.1 Robustness Benchmark (OOD & Noise)
Thay vì chỉ test trên dữ liệu sạch (Clean Data), ta đánh giá độ bền vững:
*   **Noise Injection:** Tạo 3 bản sao của Test Set:
    *   Gaussian Noise ($\sigma=0.15$).
    *   Salt & Pepper Noise (Density=0.05).
    *   Gaussian Blur (Kernel=5).
*   **Metric:** Tính % sụt giảm Accuracy ($\Delta Acc = Acc_{clean} - Acc_{noisy}$). Model có $\Delta Acc$ nhỏ nhất là model Robust nhất.

### 6.2 Calibration Evaluation (Reliability)
*   Metric: **Expected Calibration Error (ECE)**.
*   Phương pháp: Chia độ tự tin (0-1) thành 15 bin (khoảng). So sánh Accuracy trung bình và Confidence trung bình trong mỗi bin.
*   *Mục đích:* Một model "đáng tin cậy" không chỉ đúng mà còn phải biết mình đúng bao nhiêu phần trăm (quan trọng trong y tế, tự lái).

### 6.3 Statistical Significance
*   **Method:** Sử dụng **Bootstrap Sampling**. Lấy mẫu ngẫu nhiên có hoàn lại từ Test Set (1000 lần) để tính khoảng tin cậy 95% (95% CI) cho Accuracy.
*   **McNemar’s Test:** So sánh cặp trực tiếp giữa ResNet50 và ConvNeXtV2 dựa trên số lượng mẫu chúng dự đoán đúng/sai. Chấp nhận mô hình tốt hơn nếu p-value < 0.05.

---

## 7. Lộ trình Thực hiện (Timeline Gợi ý)

1.  **Tuần 1:** Setup codebase, Data Loader, implement 8 models (dùng `timm` hoặc custom). Viết hàm tính Hessian.
2.  **Tuần 2:** Chạy HPO trên CIFAR100 (Proxy) + Phân tích Loss Landscape -> Chọn bộ tham số tối ưu.
3.  **Tuần 3-4:** Chạy Full Training (8 models x 4 datasets = 32 runs) với cơ chế Adaptive Learning + SAM. Đây là giai đoạn chạy dài (có thể chạy song song trên nhiều máy/Colab).
4.  **Tuần 5:** Chạy Evaluation Robustness & Calibration. Thực hiện Ablation Study.
5.  **Tuần 6:** Tổng hợp dữ liệu, vẽ biểu đồ, viết báo cáo và so sánh theo Ma trận Khuyến nghị.

