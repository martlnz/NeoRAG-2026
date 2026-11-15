"""Chatbot cho phần Demo Product NeoRAG - S2B Community (PTIT).
Chạy: python -m streamlit run app/chat.py
"""

from __future__ import annotations

import os
import time

import streamlit as st

from neorag.data import Document, load_corpus
from neorag.pipelines import RAGPipeline, load_pipeline

# ---------------------------------------------------------
# Fast Intent Router (Điều hướng câu hỏi xã giao không cần RAG)
# ---------------------------------------------------------
FAST_INTENT_PATTERNS = {
    "greeting": [
        r"^\s*(xin\s+)?chào(\s+(bạn|ad|bot|cậu|mọi người|anh|chị|e|em))?\s*[!.,~?]*$",
        r"^\s*(hi|hello|hey|alo|halo|helo)(\s+(bạn|ad|bot|cậu|there))?\s*[!.,~?]*$",
    ],
    "identity": [
        r"^\s*(bạn|cậu|bot)\s+(là\s+ai|tên\s+(gì|là\s+gì)|có\s+thể\s+làm\s+(được\s+)?gì)\s*[?.,!]*$",
        r"^\s*(giới\s+thiệu\s+(về\s+)?(bản\s+thân|bạn|mình))\s*[?.,!]*$",
        r"^\s*who\s+are\s+you\s*[?.,!]*$",
    ],
    "thanks": [
        r"^\s*(cảm\s+ơn|cam\s+on|thank(s)?(\s+you)?|tks|tkss)(\s+(bạn|ad|bot|nhiều|nhe|nhé))?\s*[!.,~]*$",
        r"^\s*(ok|oki|oke)(\s+cảm\s+ơn|\s+thanks|\s+cảm\s+ơn\s+bạn)?\s*[!.,~]*$",
    ],
    "farewell": [
        r"^\s*(tạm\s+biệt|tam\s+biet|bye(\s+bye)?|goodbye|hẹn\s+gặp\s+lại)\s*[!.,~]*$",
    ],
}

FAST_INTENT_RESPONSES = {
    "greeting": (
        "Xin chào bạn! 👋 Mình là trợ lý hỏi đáp về cộng đồng **S2B PTIT** "
        "(gồm **CLB Lập Trình ProPTIT**, **CLB Multimedia PTIT** và **CLB Nhà Sáng Tạo Game PTIT**).\n\n"
        "Bạn cần tìm hiểu thông tin gì về tuyển thành viên, ban chuyên môn, sự kiện hay quyền lợi cứ nhắn cho mình nhé!"
    ),
    "identity": (
        "Mình là trợ lý ảo hỗ trợ thông tin cộng đồng sinh viên **S2B PTIT** 🤖.\n\n"
        "Mình có thể giải đáp cho bạn:\n"
        "- Lịch sử, cơ cấu ban, phương châm hoạt động của 3 CLB (ProPTIT, Multimedia, Game GCC).\n"
        "- Kế hoạch tuyển Gen mới, tiêu chí và vòng thi tuyển.\n"
        "- Các giải thưởng, dự án và hoạt động đào tạo nổi bật."
    ),
    "thanks": (
        "Rất vui được hỗ trợ bạn! 😊 Nếu bạn còn câu hỏi nào về các CLB, đừng ngần ngại nhắn tiếp nhé."
    ),
    "farewell": (
        "Tạm biệt bạn! 👋 Chúc bạn học tập tốt và gặt hái nhiều thành công tại PTIT!"
    ),
}


def check_fast_intent(query: str) -> str | None:
    """Trả về câu trả lời định sẵn nếu câu hỏi chỉ là chào hỏi/xã giao, không cần truy xuất RAG."""
    import re

    clean = query.strip().lower()
    for intent, regexes in FAST_INTENT_PATTERNS.items():
        for r in regexes:
            if re.match(r, clean):
                return FAST_INTENT_RESPONSES[intent]
    return None


def stream_response(text: str, delay: float = 0.012):
    """Mô phỏng hiệu ứng streaming hiển thị từng từ mượt mà."""
    import re

    tokens = re.split(r"(\s+)", text)
    for token in tokens:
        yield token
        if token.strip():
            time.sleep(delay)


# ---------------------------------------------------------
# Cấu hình Pipeline (Mặc định bản tối ưu nhất)
# ---------------------------------------------------------
PIPELINE_SPEC = os.getenv("NEORAG_PIPELINE", "neorag.pipelines.pipeline:MyPipelineV11Audit")

st.set_page_config(
    page_title="NeoRAG Chatbot - S2B Community PTIT",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Custom Styling (CSS)
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .stChatInputContainer {
        border-radius: 12px !important;
        padding-bottom: 10px;
    }
    .club-tag-proptit {
        background-color: #eff6ff;
        color: #1d4ed8;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
        border: 1px solid #bfdbfe;
    }
    .club-tag-mpc {
        background-color: #faf5ff;
        color: #7e22ce;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
        border: 1px solid #e9d5ff;
    }
    .club-tag-gcc {
        background-color: #f0fdf4;
        color: #15803d;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
        border: 1px solid #bbf7d0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Đang nạp mô hình và lập chỉ mục văn bản (vui lòng đợi vài giây)...")
def get_pipeline() -> tuple[RAGPipeline, dict[int, Document]]:
    corpus = load_corpus()
    pipeline = load_pipeline(PIPELINE_SPEC)
    pipeline.prepare()
    pipeline.index(corpus)
    return pipeline, {d.id: d for d in corpus}


# ---------------------------------------------------------
# Quản lý Đa Phiên Chat (Multi-Session Chat History)
# ---------------------------------------------------------
if "sessions" not in st.session_state:
    st.session_state.sessions = {}

if "current_session_id" not in st.session_state:
    init_sid = str(int(time.time() * 1000))
    st.session_state.sessions[init_sid] = {
        "id": init_sid,
        "title": "Cuộc trò chuyện mới",
        "messages": [],
        "created_at": time.time(),
    }
    st.session_state.current_session_id = init_sid

if st.session_state.current_session_id not in st.session_state.sessions:
    first_sid = next(iter(st.session_state.sessions.keys()))
    st.session_state.current_session_id = first_sid

current_session = st.session_state.sessions[st.session_state.current_session_id]

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.title("🤖 NeoRAG Assistant")
    st.caption("Cộng đồng sinh viên S2B · PTIT")

    # Nút tạo phiên chat mới
    if st.button("➕ Phiên trò chuyện mới", use_container_width=True, type="primary"):
        new_sid = str(int(time.time() * 1000))
        st.session_state.sessions[new_sid] = {
            "id": new_sid,
            "title": "Cuộc trò chuyện mới",
            "messages": [],
            "created_at": time.time(),
        }
        st.session_state.current_session_id = new_sid
        st.rerun()

    st.markdown("---")
    st.subheader("🗂️ Lịch sử phiên chat")

    sessions_sorted = sorted(
        st.session_state.sessions.values(),
        key=lambda s: s["created_at"],
        reverse=True,
    )

    for sess in sessions_sorted:
        sid = sess["id"]
        is_active = sid == st.session_state.current_session_id
        col_btn, col_del = st.columns([5, 1])

        label_prefix = "💬 " if is_active else "🗨️ "
        title_text = sess["title"]
        if len(title_text) > 24:
            title_text = title_text[:22] + "..."

        with col_btn:
            if st.button(
                f"{label_prefix}{title_text}",
                key=f"sess_btn_{sid}",
                use_container_width=True,
                type="secondary" if not is_active else "primary",
            ):
                st.session_state.current_session_id = sid
                st.rerun()

        with col_del:
            if len(st.session_state.sessions) > 1 and st.button("✕", key=f"del_sess_{sid}", help="Xóa phiên này"):
                del st.session_state.sessions[sid]
                if st.session_state.current_session_id == sid:
                    st.session_state.current_session_id = next(iter(st.session_state.sessions.keys()))
                st.rerun()

    st.markdown("---")
    st.caption("NeoRAG 2026 · S2B: ProPTIT, Multimedia MPC, Game GCC")


# ---------------------------------------------------------
# Main Chat Area
# ---------------------------------------------------------
st.title("💬 Hỏi đáp Cộng đồng S2B PTIT")
st.caption("CLB Lập Trình PTIT · CLB Multimedia PTIT · CLB Nhà Sáng Tạo Game PTIT")

# Nạp pipeline tối ưu
pipeline, docs = get_pipeline()

# Hiển thị lịch sử tin nhắn của phiên chat hiện tại
messages = current_session["messages"]

# Giao diện chào mừng nếu phiên mới chưa có tin nhắn
if not messages:
    st.markdown(
        """
        <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; margin-top: 10px; margin-bottom: 20px;">
            <h4 style="margin-top: 0; color: #1e293b;">👋 Chào mừng bạn đến với Hệ thống Hỏi đáp S2B PTIT!</h4>
            <p style="color: #475569; font-size: 0.95rem; margin-bottom: 10px;">
                Tôi có thể giúp bạn giải đáp mọi thông tin chính xác về <b>CLB Lập Trình PTIT (ProPTIT)</b>,
                <b>CLB Multimedia PTIT (MPC)</b> và <b>CLB Nhà Sáng Tạo Game PTIT (GCC)</b>.
            </p>
            <p style="color: #64748b; font-size: 0.85rem; margin: 0;">
                💡 <i>Mẹo: Bạn có thể nhập câu hỏi trực tiếp vào ô chat bên dưới để bắt đầu.</i>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("retrieved_ids"):
            r_ids = msg["retrieved_ids"]
            primary_ids = r_ids[:3]
            extra_ids = r_ids[3:]
            elapsed_info = f" · {msg['elapsed']:.2f}s" if "elapsed" in msg else ""
            with st.expander(f"📚 Nguồn trích dẫn chính ({len(primary_ids)} tài liệu){elapsed_info}"):
                for doc_id in primary_ids:
                    if doc_id in docs:
                        d = docs[doc_id]
                        tag_class = "club-tag-proptit" if "Lập Trình" in d.club else ("club-tag-mpc" if "Multimedia" in d.club else "club-tag-gcc")
                        st.markdown(
                            f"<span class='{tag_class}'>{d.club}</span> <b>[{d.id}]</b> · <i>{d.topic}</i>\n\n{d.text}\n",
                            unsafe_allow_html=True,
                        )
                if extra_ids:
                    st.caption(f"ℹ️ Tài liệu liên quan bổ sung: {', '.join(f'[{i}]' for i in extra_ids)}")

# ---------------------------------------------------------
# Khung Chat Input (LUÔN CỐ ĐỊNH Ở ĐÁY MÀN HÌNH)
# ---------------------------------------------------------
user_query = st.chat_input("Hỏi gì đó về ProPTIT, Multimedia hoặc GCC...")

# ---------------------------------------------------------
# Xử lý câu hỏi
# ---------------------------------------------------------
if user_query:
    # Cập nhật tiêu đề phiên chat tự động theo câu hỏi đầu tiên
    if len(messages) == 0 or current_session["title"] == "Cuộc trò chuyện mới":
        clean_title = user_query.strip().replace("\n", " ")
        current_session["title"] = clean_title[:32] + ("..." if len(clean_title) > 32 else "")

    # Thêm câu hỏi vào session
    messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        start_time = time.perf_counter()
        fast_answer = check_fast_intent(user_query)

        if fast_answer is not None:
            answer = fast_answer
            retrieved_ids = []
            elapsed = time.perf_counter() - start_time
            st.write_stream(stream_response(answer, delay=0.01))
        else:
            with st.spinner("Đang truy xuất tài liệu và tổng hợp câu trả lời..."):
                try:
                    out = pipeline.run(user_query)
                    answer = out.answer
                    retrieved_ids = out.retrieved_ids
                except Exception as exc:
                    answer = f"⚠️ **Đã xảy ra lỗi khi gọi pipeline:** {exc}\n\n*Gợi ý: Kiểm tra kết nối API gateway hoặc thử các câu hỏi mẫu đã được lưu trong cache.*"
                    retrieved_ids = []
                elapsed = time.perf_counter() - start_time

            st.write_stream(stream_response(answer, delay=0.012))

        if retrieved_ids:
            primary_ids = retrieved_ids[:3]
            extra_ids = retrieved_ids[3:]
            with st.expander(f"📚 Nguồn trích dẫn chính ({len(primary_ids)} tài liệu) · {elapsed:.2f}s"):
                for doc_id in primary_ids:
                    if doc_id in docs:
                        d = docs[doc_id]
                        tag_class = "club-tag-proptit" if "Lập Trình" in d.club else ("club-tag-mpc" if "Multimedia" in d.club else "club-tag-gcc")
                        st.markdown(
                            f"<span class='{tag_class}'>{d.club}</span> <b>[{d.id}]</b> · <i>{d.topic}</i>\n\n{d.text}\n",
                            unsafe_allow_html=True,
                        )
                if extra_ids:
                    st.caption(f"ℹ️ Tài liệu liên quan bổ sung: {', '.join(f'[{i}]' for i in extra_ids)}")

    # Lưu câu trả lời vào lịch sử session
    messages.append(
        {
            "role": "assistant",
            "content": answer,
            "retrieved_ids": retrieved_ids,
            "elapsed": elapsed,
        }
    )
