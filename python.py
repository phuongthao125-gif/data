import streamlit as st
import pandas as pd
from google import genai
from google.genai.errors import APIError

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Phân Tích Báo Cáo Tài Chính",
    layout="wide"
)

st.title("Ứng dụng Phân Tích Báo Cáo Tài chính 📊")

# --- Khởi tạo State (Quan trọng cho Lịch sử Chat) ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_session" not in st.session_state:
    st.session_state.chat_session = None
if "gemini_context" not in st.session_state:
    st.session_state.gemini_context = ""
    
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

# --- Hàm gọi API Gemini cho Nhận xét (Chức năng 5) ---
# Hàm này giữ nguyên logic ban đầu cho phần phân tích tự động
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
        
# ----------------------------------------------------------------------
# --- BỔ SUNG: Hàm Xử lý Chat Gemini (Cho Khung Chat) ---
# ----------------------------------------------------------------------
def initialize_chat_session(api_key, context_data):
    """Khởi tạo hoặc đặt lại phiên chat với bối cảnh dữ liệu đã phân tích."""
    try:
        # 1. Khởi tạo Client
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash'
        
        # 2. Xây dựng System Instruction (Giúp AI hiểu vai trò và bối cảnh)
        system_instruction = f"""
        Bạn là một Trợ lý phân tích tài chính thông minh dựa trên mô hình Gemini.
        Nhiệm vụ của bạn là trả lời các câu hỏi của người dùng dựa trên dữ liệu Báo cáo Tài chính đã được phân tích sau đây.
        Hãy sử dụng dữ liệu này làm bối cảnh chính cho mọi câu trả lời. Trả lời bằng tiếng Việt.
        
        Dữ liệu Báo cáo Tài chính đã phân tích:
        {context_data}
        """

        # 3. Khởi tạo Chat Session
        st.session_state.chat_session = client.chats.create(
            model=model_name,
            config={"system_instruction": system_instruction}
        )
        st.session_state.messages = [] # Reset lịch sử tin nhắn cũ
        st.success("🤖 Trợ lý Phân tích AI đã sẵn sàng. Hãy bắt đầu hỏi về dữ liệu!")

    except APIError as e:
        st.error(f"Lỗi khởi tạo Chat: Vui lòng kiểm tra Khóa API. Chi tiết: {e}")
        st.session_state.chat_session = None
    except Exception as e:
        st.error(f"Lỗi không xác định khi khởi tạo Chat: {e}")
        st.session_state.chat_session = None

# --- Chức năng 1: Tải File ---
uploaded_file = st.file_uploader(
    "1. Tải file Excel Báo cáo Tài chính (Chỉ tiêu | Năm trước | Năm sau)",
    type=['xlsx', 'xls']
)

if uploaded_file is not None:
    try:
        df_raw = pd.read_excel(uploaded_file)
        
        # Tiền xử lý: Đ
