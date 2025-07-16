import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from collections import Counter
import re

# ✅ 모델 로딩 (들여쓰기 오류 수정)
@st.cache_resource
def load_model():
    return SentenceTransformer("jhgan/ko-sroberta-multitask")

model = load_model()  # ← 반드시 함수 밖에 있어야 정상 작동함!

# ✅ Streamlit 초기 설정
st.set_page_config(layout="wide")
st.title("📚 생기부 문장 유사도 검사기")

st.markdown("### 📂 예시 파일 다운로드")
with open("example.xlsx", "rb") as f:
    st.download_button(
        label="📥 엑셀 예시 파일 받기",
        data=f,
        file_name="세특_예시.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

uploaded_file = st.file_uploader("📎 엑셀 파일 업로드 (학생 이름, 세특 전체)", type="xlsx")


# ✅ 공통 단어 하이라이트 함수
def highlight_common_phrases(sentences):
    words_list = [re.findall(r'\b\w+\b', s) for s in sentences]
    flat_words = [word for words in words_list for word in set(words)]
    common_words = [word for word, count in Counter(flat_words).items() if count > 1 and len(word) > 1]
    highlighted = []
    for s in sentences:
        for word in common_words:
            s = re.sub(rf'\b({re.escape(word)})\b', r"<span style='color:red; font-weight:bold;'>\1</span>", s)
        highlighted.append(s)
    return highlighted

# ✅ 업로드 후 처리
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    expected_cols = {"학생 이름", "세특 전체"}
    if not expected_cols.issubset(set(df.columns)):
        st.error(f"엑셀의 열 이름이 정확한지 확인해주세요: {expected_cols}")
        st.write("업로드된 열 목록:", list(df.columns))
        st.stop()

    df["세특 전체"] = df["세특 전체"].astype(str)

    # 문장 분리
    sentences, meta = [], []
    for _, row in df.iterrows():
        name = row["학생 이름"]
        text = row["세특 전체"]
        split_sents = [s.strip() for s in text.replace("\n", " ").split('.') if s.strip()]
        for idx, sent in enumerate(split_sents):
            sentences.append(sent)
            meta.append({"학생": name, "문장": sent, "전체": text, "순번": idx})

    embeddings = model.encode(sentences, convert_to_tensor=True)
    sim_matrix = util.cos_sim(embeddings, embeddings).cpu().numpy()
    sentence_index_map = {m["문장"]: i for i, m in enumerate(meta)}

    # 탭 구성
    tab1, tab2 = st.tabs(["👤 학생별 세특 보기", "📋 전체 유사 문장 보기"])

    with tab1:
        st.header("👤 학생별 세특 보기")
        student_names = sorted(set(str(m["학생"]) for m in meta if pd.notnull(m["학생"])))
        selected_student = st.selectbox("학생 선택", student_names)
        indices = [i for i, m in enumerate(meta) if m["학생"] == selected_student]
        clicked_sentence = None

        st.markdown("### 세특 내용:")
        highlighted = []
        for i in indices:
            m = meta[i]
            s = m["문장"]
            scores = sim_matrix[i]
            has_similar = any(scores[j] >= 0.95 and meta[j]["학생"] != selected_student for j in range(len(meta)))
            if has_similar:
                if st.button(f"⭐ {s}", key=f"{i}"):
                    clicked_sentence = m["문장"]
                highlighted.append(f"<span style='color:red; font-weight:bold;'>{s}</span>")
            else:
                highlighted.append(s)
        st.markdown(" ".join(highlighted), unsafe_allow_html=True)

        if clicked_sentence:
            st.markdown("---")
            st.subheader(f"🔍 유사 문장: '{clicked_sentence}'")
            clicked_idx = sentence_index_map[clicked_sentence]
            sims = sim_matrix[clicked_idx]
            similar_list = []
            for i, score in enumerate(sims):
                if meta[i]["학생"] != selected_student and score >= 0.95:
                    similar_list.append({
                        "학생": meta[i]["학생"],
                        "문장": meta[i]["문장"],
                        "유사도": round(float(score), 3)
                    })
            sim_df = pd.DataFrame(similar_list).sort_values(by="유사도", ascending=False)
            st.dataframe(sim_df, use_container_width=True)

    with tab2:
        st.header("👥 유사 학생 그룹(0.95 이상)")

        # 학생 단위 전체 세특 유사도
        full_texts = [row["세특 전체"] for _, row in df.iterrows()]
        stu_embeddings = model.encode(full_texts, convert_to_tensor=True)
        stu_sim = util.cos_sim(stu_embeddings, stu_embeddings).cpu().numpy()
        stu_names = df["학생 이름"].tolist()

        threshold = 0.95
        adj = {name: set() for name in stu_names}
        for i in range(len(stu_names)):
            for j in range(i + 1, len(stu_names)):
                if stu_sim[i][j] >= threshold:
                    adj[stu_names[i]].add(stu_names[j])
                    adj[stu_names[j]].add(stu_names[i])

        from collections import defaultdict, deque

        visited = set()
        groups = []
        for name in stu_names:
            if name not in visited:
                stack = [name]
                comp = []
                while stack:
                    cur = stack.pop()
                    if cur in visited:
                        continue
                    visited.add(cur)
                    comp.append(cur)
                    stack.extend(adj[cur] - visited)
                if len(comp) > 1:
                    groups.append(sorted(comp))

        if not groups:
            st.info("유사 학생 그룹(두 명 이상)이 없습니다.")
            st.stop()

        group_labels = [f"그룹 {idx+1}: " + ", ".join(g) for idx, g in enumerate(groups)]
        selected_idx = st.selectbox("🔽 그룹 선택", range(len(groups)), format_func=lambda i: group_labels[i])
        selected_group = groups[selected_idx]

        st.markdown("---")
        st.subheader("📄 선택한 그룹의 유사 문장 모음 (0.95 이상)")

        sent_graph = defaultdict(set)
        for i in range(len(meta)):
            for j in range(i + 1, len(meta)):
                if meta[i]["학생"] in selected_group and meta[j]["학생"] in selected_group:
                    if sim_matrix[i][j] >= 0.95:
                        sent_graph[i].add(j)
                        sent_graph[j].add(i)

        seen = set()
        components = []
        for i in sent_graph:
            if i not in seen:
                q = deque([i])
                comp = []
                while q:
                    cur = q.popleft()
                    if cur in seen:
                        continue
                    seen.add(cur)
                    comp.append(cur)
                    q.extend(sent_graph[cur] - seen)
                if len(comp) > 1:
                    components.append(comp)

        if not components:
            st.info("0.95 이상의 유사 문장 그룹이 없습니다.")
        else:
            for idx, comp in enumerate(components, 1):
                st.markdown(f"### 🔹 문장 그룹 {idx}")
                group_sentences = [meta[i]["문장"] for i in comp]
                highlighted_sents = highlight_common_phrases(group_sentences)
                for sent, i in zip(highlighted_sents, sorted(comp, key=lambda x: meta[x]["학생"])):
                    student_name = meta[i]["학생"]
                    st.markdown(f"- **{student_name}**: {sent}", unsafe_allow_html=True)
