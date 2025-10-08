# python.py

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
# Chỉ lưu lịch sử tin nhắn và bối cảnh dữ liệu, KHÔNG lưu đối tượng chat_session
if "messages" not in st.session_state:
    st.session_state.messages = []
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
    df['Tốc độ tăng trưởng (%)'] = (
        (df['Năm sau'] - df['Năm trước']) / df['Năm trước'].replace(0, 1e-9)
    ) * 100

    # 2. Tính Tỷ trọng theo Tổng Tài sản
    tong_tai_san_row = df[df['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN', case=False, na=False)]
    
    if tong_tai_san_row.empty:
        raise ValueError("Không tìm thấy chỉ tiêu 'TỔNG CỘNG TÀI SẢN'.")

    tong_tai_san_N_1 = tong_tai_san_row['Năm trước'].iloc[0]
    tong_tai_san_N = tong_tai_san_row['Năm sau'].iloc[0]

    divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else 1e-9
    divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else 1e-9

    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100
    df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100
    
    return df

# --- Hàm gọi API Gemini cho Nhận xét Tự động (Chức năng 5) ---
# Hàm này khởi tạo client, sử dụng và kết thúc, tránh lỗi client đóng
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
        return "Lỗi: Không tìm thấy Khóa API 'GEMINI_API_KEY'. Vui lòng kiểm tra cấu hình Secrets."
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
            
            # --- CẬP NHẬT: Gán dữ liệu phân tích vào session state cho Chat ---
            current_data_for_chat = df_processed.to_markdown(index=False)
            if st.session_state.gemini_context != current_data_for_chat:
                st.session_state.gemini_context = current_data_for_chat
                # Reset chat history khi dữ liệu mới được tải lên
                st.session_state.messages = [] 

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
            
            thanh_toan_hien_hanh_N = "N/A"
            thanh_toan_hien_hanh_N_1 = "N/A"
            try:
                # Lọc giá trị cho Chỉ số Thanh toán Hiện hành
                tsnh_n = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                tsnh_n_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

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
            except ZeroDivisionError:
                 st.error("Không thể tính chỉ số thanh toán hiện hành do mẫu số (Nợ Ngắn Hạn) bằng 0.")
                 
            # --- Chức năng 5: Nhận xét AI (Tự động) ---
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
                    f"{df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Tốc độ tăng trưởng (%)'].iloc[0]:.2f}%" if not isinstance(thanh_toan_hien_hanh_N, str) else "N/A", 
                    f"{thanh_toan_hien_hanh_N_1:.2f}" if isinstance(thanh_toan_hien_hanh_N_1, float) else "N/A", 
                    f"{thanh_toan_hien_hanh_N:.2f}" if isinstance(thanh_toan_hien_hanh_N, float) else "N/A"
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

            # ----------------------------------------------------------------------
            # --- Chức năng 6: Khung Chat Hỏi Đáp (Interactive) ---
            # --- Đã sửa lỗi "client has been closed" bằng cách tái tạo session ---
            # ----------------------------------------------------------------------
            st.markdown("---")
            st.subheader("6. Chat Hỏi Đáp chuyên sâu về Báo cáo Tài chính (Gemini)")
            
            api_key = st.secrets.get("GEMINI_API_KEY")

            if not api_key:
                st.error("Lỗi: Không tìm thấy Khóa API 'GEMINI_API_KEY'. Vui lòng kiểm tra Secrets để sử dụng Chat.")
            else:
                # 1. Hiển thị lịch sử tin nhắn
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                # 2. Xử lý input của người dùng
                if prompt := st.chat_input("Hỏi về Tốc độ tăng trưởng, tỷ trọng cơ cấu, hoặc bất kỳ chỉ tiêu nào..."):
                    # Thêm tin nhắn của người dùng vào lịch sử
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    # 3. Gọi API để nhận phản hồi (Tái tạo Client và Session trước mỗi lần gửi)
                    with st.chat_message("assistant"):
                        with st.spinner("Gemini đang phân tích và trả lời..."):
                            try:
                                # **Khắc phục lỗi Client Closed:** Tái tạo Client và Session trước mỗi lần sử dụng
                                client = genai.Client(api_key=api_key)
                                
                                system_instruction = f"""
                                Bạn là một Trợ lý phân tích tài chính thông minh dựa trên mô hình Gemini.
                                Nhiệm vụ của bạn là trả lời các câu hỏi của người dùng dựa trên dữ liệu Báo cáo Tài chính đã được phân tích sau đây.
                                Hãy sử dụng dữ liệu này làm bối cảnh chính cho mọi câu trả lời. Trả lời bằng tiếng Việt.
                                
                                Dữ liệu Báo cáo Tài chính đã phân tích:
                                {st.session_state.gemini_context}
                                """
                                
                                # Tái tạo Chat Session, truyền lịch sử cũ để duy trì ngữ cảnh
                                chat_session = client.chats.create(
                                    model='gemini-2.5-flash',
                                    history=st.session_state.messages, 
                                    config={"system_instruction": system_instruction}
                                )
                                
                                # Gửi tin nhắn mới nhất
                                response = chat_session.send_message(prompt) 
                                
                                st.markdown(response.text)
                                # Thêm phản hồi của AI vào lịch sử
                                st.session_state.messages.append({"role": "assistant", "content": response.text})
                                
                            except APIError as e:
                                error_msg = f"Lỗi gọi Chat API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
                                st.error(error_msg)
                                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                            except Exception as e:
                                error_msg = f"Lỗi không xác định: {e}"
                                st.error(error_msg)
                                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                                
                    # Quan trọng: Kích hoạt rerun để cập nhật khung chat ngay lập tức
                    st.rerun()

    except ValueError as ve:
        st.error(f"Lỗi cấu trúc dữ liệu: {ve}")
        st.session_state.messages = []
    except Exception as e:
        st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}. Vui lòng kiểm tra định dạng file.")
        st.session_state.messages = []

else:
    st.info("Vui lòng tải lên file Excel để bắt đầu phân tích.")
