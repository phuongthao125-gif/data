Chào bạn, với kinh nghiệm triển khai ứng dụng Python trên Streamlit, tôi sẽ giúp bạn tích hợp một **khung chat AI hỏi đáp tài chính** sử dụng mô hình Gemini, cho phép người dùng tương tác trực tiếp với dữ liệu đã được tải lên.

Để làm được điều này, chúng ta cần:

1.  **Khởi tạo lịch sử chat** trong `st.session_state`.
2.  Tạo **hàm mới** để gọi API Gemini cho chế độ chat.
3.  **Vòng lặp hiển thị lịch sử** và **khung nhập liệu chat**.
4.  Gắn dữ liệu đã phân tích vào **ngữ cảnh (context)** của mỗi câu hỏi chat.

Đây là đoạn mã đã được chỉnh sửa:

```python
import streamlit as st
import pandas as pd
from google import genai
from google.genai.errors import APIError

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Phân Tích Báo Cáo Tài chính",
    layout="wide"
)

st.title("Ứng dụng Phân Tích Báo Cáo Tài chính 📊")

# --- Hàm tính toán chính (Sử dụng Caching để Tối ưu hiệu suất) ---
@st.cache_data
def process_financial_data(df):
    """Thực hiện các phép tính Tăng trưởng và Tỷ trọng."""
    
    # Đảm bảo các giá trị là số để tính toán
    numeric_cols = ['Năm trước', 'Năm sau']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 1. Tính Tốc độ Tăng trưởng
    # Dùng .replace(0, 1e-9) cho Series Pandas để tránh lỗi chia cho 0
    df['Tốc độ tăng trưởng (%)'] = (
        (df['Năm sau'] - df['Năm trước']) / df['Năm trước'].replace(0, 1e-9)
    ) * 100

    # 2. Tính Tỷ trọng theo Tổng Tài sản
    # Lọc chỉ tiêu "TỔNG CỘNG TÀI SẢN"
    tong_tai_san_row = df[df['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN', case=False, na=False)]
    
    if tong_tai_san_row.empty:
        raise ValueError("Không tìm thấy chỉ tiêu 'TỔNG CỘNG TÀI SẢN'.")

    tong_tai_san_N_1 = tong_tai_san_row['Năm trước'].iloc[0]
    tong_tai_san_N = tong_tai_san_row['Năm sau'].iloc[0]

    # ******************************* PHẦN SỬA LỖI BẮT ĐẦU *******************************
    # Lỗi xảy ra khi dùng .replace() trên giá trị đơn lẻ (numpy.int64).
    # Sử dụng điều kiện ternary để xử lý giá trị 0 thủ công cho mẫu số.
    
    divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else 1e-9
    divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else 1e-9

    # Tính tỷ trọng với mẫu số đã được xử lý
    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100
    df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100
    # ******************************* PHẦN SỬA LỖI KẾT THÚC *******************************
    
    return df

# --- Hàm gọi API Gemini cho Nhận xét tổng quan ---
def get_ai_analysis(data_for_ai, api_key):
    """Gửi dữ liệu phân tích đến Gemini API và nhận nhận xét."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash' 

        prompt = f"""
        Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Dựa trên các chỉ số tài chính sau, hãy đưa ra một nhận xét khách quan, ngắn gọn (khoảng 3-4 đoạn) về tình hình tài chính của doanh nghiệp. Đánh giá tập trung vào tốc độ tăng trưởng, thay đổi cơ cấu tài sản và khả năng thanh toán hiện hành.
        
        Dữ liệu thô và chỉ số:
        {data_for_ai}
        """

        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except KeyError:
        return "Lỗi: Không tìm thấy Khóa API 'GEMINI_API_KEY'. Vui lòng kiểm tra cấu hình Secrets trên Streamlit Cloud."
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"

# --- HÀM MỚI: Gọi API Gemini cho CHAT ---
def chat_with_gemini(full_context, user_prompt, api_key):
    """Tương tác với Gemini trong chế độ chat với ngữ cảnh (context) được cung cấp."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash'
        
        # Thiết lập System Instruction để giữ vai trò chuyên gia tài chính và sử dụng ngữ cảnh
        system_instruction = f"""
        Bạn là một chuyên gia phân tích tài chính Python. Hãy trả lời các câu hỏi của người dùng dựa trên dữ liệu phân tích tài chính sau (nếu có liên quan). 
        Dữ liệu phân tích: 
        {full_context}
        Nếu câu hỏi không liên quan đến tài chính hoặc dữ liệu, hãy trả lời một cách lịch sự nhưng tập trung lại vào chủ đề tài chính.
        """
        
        # Tải lịch sử chat từ session_state và thêm system instruction
        history = [{"role": "user", "parts": [system_instruction]}] 
        # Thêm lịch sử chat đã có (trừ system instruction giả định)
        for msg in st.session_state.messages:
            history.append({"role": msg["role"], "parts": [msg["content"]]})

        # Thêm câu hỏi mới nhất của người dùng
        history.append({"role": "user", "parts": [user_prompt]})
        
        # Gọi API
        response = client.models.generate_content(
            model=model_name,
            contents=history
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định trong Chat: {e}"
    
# --- Khởi tạo State (Quan trọng cho Chat) ---
if "messages" not in st.session_state:
    st.session_state.messages = []
# Biến để lưu trữ dữ liệu phân tích cho Chat Context
if "df_processed_for_chat" not in st.session_state:
    st.session_state.df_processed_for_chat = None

# --- Chức năng 1: Tải File ---
uploaded_file = st.file_uploader(
    "1. Tải file Excel Báo cáo Tài chính (Chỉ tiêu | Năm trước | Năm sau)",
    type=['xlsx', 'xls']
)

# Thẻ giữ chỗ cho Khung chat
chat_placeholder = st.empty()


if uploaded_file is not None:
    try:
        df_raw = pd.read_excel(uploaded_file)
        
        # Tiền xử lý: Đảm bảo chỉ có 3 cột quan trọng
        df_raw.columns = ['Chỉ tiêu', 'Năm trước', 'Năm sau']
        
        # Xử lý dữ liệu
        df_processed = process_financial_data(df_raw.copy())
        
        # LƯU TRỮ DỮ LIỆU ĐÃ XỬ LÝ VÀO SESSION STATE CHO CHAT
        st.session_state.df_processed_for_chat = df_processed.to_markdown(index=False)

        if df_processed is not None:
            
            # --- Chức năng 2 & 3: Hiển thị Kết quả ---
            st.subheader("2. Tốc độ Tăng trưởng & 3. Tỷ trọng Cơ cấu Tài sản")
            st.dataframe(df_processed.style.format({
                'Năm trước': '{:,.0f}',
                'Năm sau': '{:,.0f}',
                'Tốc độ tăng trưởng (%)': '{:.2f}%',
                'Tỷ trọng Năm trước (%)': '{:.2f}%',
                'Tỷ trọng Năm sau (%)': '{:.2f}%'
            }), use_container_width=True)
            
            # --- Chức năng 4: Tính Chỉ số Tài chính ---
            st.subheader("4. Các Chỉ số Tài chính Cơ bản")
            
            try:
                # Lọc giá trị cho Chỉ số Thanh toán Hiện hành (Ví dụ)
                
                # Lấy Tài sản ngắn hạn
                tsnh_n = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                tsnh_n_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Lấy Nợ ngắn hạn 
                no_ngan_han_N = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]  
                no_ngan_han_N_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Tính toán
                thanh_toan_hien_hanh_N = tsnh_n / no_ngan_han_N
                thanh_toan_hien_hanh_N_1 = tsnh_n_1 / no_ngan_han_N_1
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm trước)",
                        value=f"{thanh_toan_hien_hanh_N_1:.2f} lần"
                    )
                with col2:
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm sau)",
                        value=f"{thanh_toan_hien_hanh_N:.2f} lần",
                        delta=f"{thanh_toan_hien_hanh_N - thanh_toan_hien_hanh_N_1:.2f}"
                    )
                    
            except IndexError:
                 st.warning("Thiếu chỉ tiêu 'TÀI SẢN NGẮN HẠN' hoặc 'NỢ NGẮN HẠN' để tính chỉ số.")
                 thanh_toan_hien_hanh_N = "N/A" # Dùng để tránh lỗi ở Chức năng 5
                 thanh_toan_hien_hanh_N_1 = "N/A"
            
            # --- Chức năng 5: Nhận xét AI ---
            st.subheader("5. Nhận xét Tình hình Tài chính (AI)")
            
            # Chuẩn bị dữ liệu để gửi cho AI
            data_for_ai = pd.DataFrame({
                'Chỉ tiêu': [
                    'Toàn bộ Bảng phân tích (dữ liệu thô)', 
                    'Tăng trưởng Tài sản ngắn hạn (%)', 
                    'Thanh toán hiện hành (N-1)', 
                    'Thanh toán hiện hành (N)'
                ],
                'Giá trị': [
                    st.session_state.df_processed_for_chat, # Dùng dữ liệu markdown đã lưu
                    f"{df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Tốc độ tăng trưởng (%)'].iloc[0]:.2f}%", 
                    f"{thanh_toan_hien_hanh_N_1}", 
                    f"{thanh_toan_hien_hanh_N}"
                ]
            }).to_markdown(index=False) 

            if st.button("Yêu cầu AI Phân tích"):
                api_key = st.secrets.get("GEMINI_API_KEY") 
                
                if api_key:
                    with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                        ai_result = get_ai_analysis(data_for_ai, api_key)
                        st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                        st.info(ai_result)
                else:
                     st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.")
            
            # --- PHẦN MỚI: CHỨC NĂNG 6: KHUNG CHAT HỎI ĐÁP ---
            
            with chat_placeholder.container():
                st.markdown("---")
                st.subheader("6. Chat Hỏi đáp Tài chính với Gemini AI 💬")
                st.info("Bây giờ bạn có thể hỏi AI bất kỳ câu hỏi nào về dữ liệu tài chính đã tải lên (ví dụ: 'Khoản mục nào có tốc độ tăng trưởng cao nhất?').")

                # 1. Hiển thị lịch sử chat
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                # 2. Khung nhập liệu chat mới
                if prompt := st.chat_input("Hỏi AI một câu hỏi về dữ liệu đã tải..."):
                    
                    # Thêm câu hỏi người dùng vào lịch sử
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    # Chuẩn bị dữ liệu và gọi API
                    api_key = st.secrets.get("GEMINI_API_KEY")
                    if api_key:
                        with st.chat_message("assistant"):
                            with st.spinner("Gemini đang phân tích dữ liệu và trả lời..."):
                                # Gửi toàn bộ dữ liệu phân tích làm context cho mỗi lần chat
                                full_context = st.session_state.df_processed_for_chat 
                                response = chat_with_gemini(full_context, prompt, api_key)
                                st.markdown(response)
                        
                        # Thêm câu trả lời của AI vào lịch sử
                        st.session_state.messages.append({"role": "assistant", "content": response})

                    else:
                        st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY'.")


    except ValueError as ve:
        st.error(f"Lỗi cấu trúc dữ liệu: {ve}")
    except Exception as e:
        st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}. Vui lòng kiểm tra định dạng file.")

else:
    st.info("Vui lòng tải lên file Excel để bắt đầu phân tích.")
    # Xóa lịch sử chat khi chưa có file để tránh nhầm lẫn
    st.session_state.messages = [] 
    st.session_state.df_processed_for_chat = None
```

-----

## Chi tiết các thay đổi chính

### 1\. Khởi tạo `st.session_state`

Thêm các dòng này ở đầu file (sau các hàm định nghĩa) để duy trì lịch sử chat giữa các lần tương tác:

```python
# --- Khởi tạo State (Quan trọng cho Chat) ---
if "messages" not in st.session_state:
    st.session_state.messages = []
# Biến để lưu trữ dữ liệu phân tích cho Chat Context
if "df_processed_for_chat" not in st.session_state:
    st.session_state.df_processed_for_chat = None
```

### 2\. Hàm Chat API Mới: `chat_with_gemini`

Hàm này sử dụng `client.models.generate_content` nhưng xây dựng một **lịch sử hội thoại (history)** bao gồm cả **System Instruction** và **dữ liệu phân tích** (`full_context`) để đảm bảo Gemini luôn có ngữ cảnh của bảng phân tích khi trả lời.

### 3\. Lưu trữ Dữ liệu Phân tích cho Chat Context

Ngay sau khi `df_processed` được tính toán thành công, ta lưu nó dưới dạng Markdown vào `session_state` để tái sử dụng làm ngữ cảnh (context) cho hàm chat:

```python
# LƯU TRỮ DỮ LIỆU ĐÃ XỬ LÝ VÀO SESSION STATE CHO CHAT
st.session_state.df_processed_for_chat = df_processed.to_markdown(index=False)
```

### 4\. Khung Chat Hỏi đáp (Chức năng 6)

Đây là phần giao diện chính:

  * Sử dụng `st.chat_message` để hiển thị lịch sử chat đã lưu trong `st.session_state.messages`.
  * Sử dụng `st.chat_input` để lấy câu hỏi mới từ người dùng.
  * Khi người dùng nhập câu hỏi (`if prompt := st.chat_input(...)`), câu hỏi được thêm vào lịch sử và gửi đến hàm `chat_with_gemini` cùng với `full_context` (dữ liệu phân tích).
  * Câu trả lời của AI được hiển thị và lưu lại vào lịch sử chat.
