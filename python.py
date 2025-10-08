import streamlit as st
import pandas as pd
from google import genai
from google.genai.errors import APIError

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Phân Tích Báo Cáo Tài Chính",
    layout="wide"
)

st.title("Ứng dụng Phân Tích Báo Cáo Tài Chính 📊")

# --- Cấu hình Khung Chat ---
# Khởi tạo lịch sử chat
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# --- Hàm helper: Khởi tạo Gemini Client ---
def get_gemini_client():
    """Lấy Gemini Client và kiểm tra API Key từ Streamlit Secrets."""
    api_key = st.secrets.get("GEMINI_API_KEY") 
    if not api_key:
        # Không dùng st.error() ở đây để tránh lặp lại, mà dùng thông báo trong phần gọi
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception:
        # Lỗi thường là do key không hợp lệ
        return "ERROR: Khóa API không hợp lệ hoặc cấu hình Client bị lỗi."

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

    # Xử lý giá trị 0 cho mẫu số
    divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else 1e-9
    divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else 1e-9

    # Tính tỷ trọng với mẫu số đã được xử lý
    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100
    df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100
    
    return df

# --- Hàm gọi API Gemini (Chức năng Phân tích cấu trúc) ---
def get_ai_analysis(data_for_ai, client):
    """Gửi dữ liệu phân tích đến Gemini API và nhận nhận xét."""
    if not client:
        return "Lỗi: Không tìm thấy Khóa API 'GEMINI_API_KEY'. Vui lòng kiểm tra cấu hình Secrets."
    
    try:
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
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"


# --- Chức năng 1: Tải File ---
uploaded_file = st.file_uploader(
    "1. Tải file Excel Báo cáo Tài chính (Chỉ tiêu | Năm trước | Năm sau)",
    type=['xlsx', 'xls']
)

if uploaded_file is not None:
    try:
        df_raw = pd.read_excel(uploaded_file)
        
        # Tiền xử lý: Đảm bảo chỉ có 3 cột quan trọng
        df_raw.columns = ['Chỉ tiêu', 'Năm trước', 'Năm sau']
        
        # Xử lý dữ liệu
        df_processed = process_financial_data(df_raw.copy())

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
                # Lấy Tài sản ngắn hạn
                tsnh_n = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                tsnh_n_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Lấy Nợ ngắn hạn
                no_ngan_han_N = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]  
                no_ngan_han_N_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Tính toán
                thanh_toan_hien_hanh_N = tsnh_n / no_ngan_han_N if no_ngan_han_N else 0
                thanh_toan_hien_hanh_N_1 = tsnh_n_1 / no_ngan_han_N_1 if no_ngan_han_N_1 else 0
                
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
                    df_processed.to_markdown(index=False),
                    f"{df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Tốc độ tăng trưởng (%)'].iloc[0]:.2f}%" if any(df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)) else "N/A", 
                    f"{thanh_toan_hien_hanh_N_1}", 
                    f"{thanh_toan_hien_hanh_N}"
                ]
            }).to_markdown(index=False) 

            if st.button("Yêu cầu AI Phân tích"):
                client = get_gemini_client() 
                
                if isinstance(client, str) and client.startswith("ERROR"):
                     st.error(client.split(": ")[1])
                elif client:
                    with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                        ai_result = get_ai_analysis(data_for_ai, client)
                        st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                        st.info(ai_result)
                else:
                     st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.")

    except ValueError as ve:
        st.error(f"Lỗi cấu trúc dữ liệu: {ve}")
    except Exception as e:
        st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}. Vui lòng kiểm tra định dạng file và console để biết thêm chi tiết.")

else:
    st.info("Vui lòng tải lên file Excel để bắt đầu phân tích.")


# --- PHẦN KHUNG CHAT TƯ VẤN TÀI CHÍNH VỚI GEMINI (Chức năng mới) ---
st.markdown("---")
st.subheader("6. Khung Chat Tư Vấn Tài Chính AI (Gemini)")
st.markdown("Sử dụng khung chat này để hỏi các câu hỏi chung về tài chính, kinh tế hoặc đầu tư.")

# Khởi tạo hoặc lấy Gemini Client
client = get_gemini_client()

if isinstance(client, str) and client.startswith("ERROR"):
    st.error(f"Lỗi Chatbot: {client.split(': ')[1]}")
elif client:
    # Lấy đối tượng chat và cấu hình hệ thống
    if "chat" not in st.session_state:
        # Sử dụng System Instruction để định hướng vai trò của AI
        system_instruction = (
            "Bạn là một chuyên gia tư vấn và hỗ trợ tài chính xuất sắc. "
            "Hãy trả lời các câu hỏi của người dùng về tài chính, kinh tế, "
            "và đầu tư một cách chuyên nghiệp, dễ hiểu, và chính xác. "
            "Nếu người dùng muốn phân tích dữ liệu đã tải lên, hãy yêu cầu họ sử dụng Chức năng 5."
        )
        try:
            st.session_state.chat = client.chats.create(
                model='gemini-2.5-flash',
                system_instruction=system_instruction
            )
            # Thêm tin nhắn chào mừng (Chỉ lần đầu)
            st.session_state.chat_messages.append({"role": "assistant", "content": "Chào bạn! Tôi là trợ lý tài chính AI của bạn. Bạn muốn hỏi gì về tài chính, kinh tế, hay các chỉ số đầu tư không?"})
        except Exception as e:
            st.error(f"Không thể khởi tạo dịch vụ Chatbot: {e}")
            client = None # Vô hiệu hóa chat nếu khởi tạo lỗi

    if client:
        # Hiển thị lịch sử chat
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Xử lý input của người dùng
        if prompt := st.chat_input("Hỏi Gemini về tài chính, đầu tư..."):
            
            # 1. Thêm tin nhắn người dùng vào lịch sử
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # 2. Gửi tin nhắn đến Gemini và nhận phản hồi
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                try:
                    # Gửi tin nhắn đến đối tượng chat đã có
                    response = st.session_state.chat.send_message(prompt, stream=True)
                    
                    # Xử lý streaming response
                    for chunk in response:
                        full_response += chunk.text
                        message_placeholder.markdown(full_response + "▌") # Dùng hiệu ứng gõ
                    
                    message_placeholder.markdown(full_response)
                except APIError as e:
                    full_response = f"Lỗi Gemini API: {e}"
                    st.error(full_response)
                except Exception as e:
                    full_response = f"Lỗi không xác định: {e}"
                    st.error(full_response)

            # 3. Thêm tin nhắn AI vào lịch sử
            st.session_state.chat_messages.append({"role": "assistant", "content": full_response})
