"""長照居家照顧 AI 派單系統 - 網頁儀表板

執行方式：
    streamlit run app.py

工作流程：
    1. 側邊欄調整核心派單政策參數（含車程、優先級、照護連續性、財務拆帳比例）。
    2. 區塊①：即時 BA 服務代碼併報法規防呆健檢（Phase 0），資料一載入／編輯
       即顯示，不需按鈕即可檢視。
    3. 區塊②：上傳／檢視／直接編輯居服員、案家、任務三張表格（可新增列與欄位）。
       任務工作表可為單日的「Today_Pending_Tasks」或含日期／星期欄位的月批次
       「Monthly_Pending_Tasks」，兩者皆可直接使用。
    4. 區塊③：按下「執行 AI 最佳化派單」後，依序呼叫 caregiver_engine.py 的
       Phase 1～3（若任務含「日期」欄位，可另選單日或全月一鍵批次排程），並顯示
       KPI（含財務試算）、四合一分析儀表板與指派明細表。

核心運算邏輯（適配度評分、OR-Tools 最佳化、財務點數試算、BA 代碼防呆、DiD 效益
回溯）皆定義於 caregiver_engine.py，本檔僅負責資料編輯介面與結果呈現，不重複
實作演算法。
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ortools.linear_solver import pywraplp
from plotly.subplots import make_subplots

from calendar_view import apply_overrides_to_result, render_calendar_overview, render_task_override_picker
from caregiver_engine import (
    DEFAULT_EXCEL_PATH,
    PipelineConfig,
    check_reassignment_conflict,
    get_weekday_name,
    load_override_log,
    rank_candidates_by_availability,
    run_monthly_batch_dispatch,
    run_phase1_matching,
    run_phase2_optimization,
    run_phase3_did,
    save_override_log,
    validate_ba_codes,
)

# ==========================================
# 色彩定義（來源：dataviz 色票，固定順序、非隨機挑色）
# ==========================================
BLUE = "#2a78d6"
ORANGE = "#eb6834"
MUTED = "#898781"
GOOD = "#0ca30c"

FIXED_REQUIRED_SHEETS = ["Caregiver_Profiles", "Client_Profiles", "Historical_Service_Logs"]

# 任務工作表名稱彈性相容：優先偵測月批次格式，其次退回現行單日格式。
TASKS_SHEET_CANDIDATES = ["Monthly_Pending_Tasks", "Today_Pending_Tasks"]

REQUIRED_COLUMNS = {
    "Caregiver_Profiles": [
        "居服員ID", "性別", "服務起點_經度(家)", "服務起點_緯度(家)", "核心專長證照",
        "每日工時上限(小時)", "當月累計服務時數(疲勞度)", "歷史滿意度均值",
        "具備重度移位體力(0/1)", "特殊排斥條件",
    ],
    "Client_Profiles": [
        "案家ID", "指定居服員性別", "服務地點_經度", "服務地點_緯度", "特殊照護需求",
        "案家環境特徵", "需重度移位協助(0/1)", "歷史首選居服員ID",
    ],
    "Historical_Service_Logs": [
        "居服員ID", "案家ID", "歷史媒合機制(Treatment)", "不滿意導致提早結案(0/1)", "案家滿意度(1-5)",
    ],
}
# 任務工作表的必要欄位為兩種格式共通的欄位；「今日既定行程」「可排班星期」等
# 各版本專屬欄位皆為選填——caregiver_engine 已能在欄位缺席時安全跳過對應檢查。
REQUIRED_TASKS_COLUMNS = ["任務ID", "案家ID", "時間窗_開始", "時間窗_結束", "服務歷時(分鐘)", "任務優先級"]

st.set_page_config(page_title="長照居家照顧 AI 派單系統", page_icon="🏠", layout="wide")

# ==========================================
# 全域視覺樣式（卡片化 KPI、章節橫幅）
# ==========================================
# 只加樣式、不改版面結構：KPI 卡片用半透明疊色（而非依賴 Streamlit 主題 CSS
# 變數），在淺色／深色主題下都不會因背景色對比不足而讀不清楚。
st.markdown(
    f"""
    <style>
    [data-testid="stMetric"] {{
        background: rgba(127, 127, 127, 0.07);
        border: 1px solid rgba(127, 127, 127, 0.18);
        border-radius: 14px;
        padding: 0.9rem 1.1rem 0.7rem 1.1rem;
        position: relative;
        overflow: hidden;
    }}
    [data-testid="stMetric"]::before {{
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 4px;
        background: {BLUE};
    }}
    [data-testid="stMetricValue"] {{
        font-weight: 700;
    }}
    .section-banner {{
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin: 1.6rem 0 0.15rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid {BLUE};
    }}
    .section-banner .section-badge {{
        flex-shrink: 0;
        width: 2rem;
        height: 2rem;
        border-radius: 50%;
        background: {BLUE};
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 1rem;
    }}
    .section-banner .section-title {{
        font-size: 1.35rem;
        font-weight: 700;
    }}
    .empty-state {{
        text-align: center;
        padding: 2.4rem 1rem;
        margin: 0.5rem 0 1rem 0;
        border: 1px dashed rgba(127, 127, 127, 0.4);
        border-radius: 16px;
        background: rgba(127, 127, 127, 0.04);
    }}
    .empty-state .empty-icon {{
        font-size: 2.2rem;
        line-height: 1;
    }}
    .empty-state .empty-title {{
        font-size: 1.05rem;
        font-weight: 700;
        margin-top: 0.5rem;
    }}
    .empty-state .empty-subtitle {{
        color: rgba(127, 127, 127, 1);
        margin-top: 0.3rem;
        font-size: 0.92rem;
    }}
    .sheet-summary-card {{
        background: rgba(127, 127, 127, 0.06);
        border: 1px solid rgba(127, 127, 127, 0.16);
        border-radius: 12px;
        padding: 0.6rem 1rem;
        margin-bottom: 0.6rem;
        font-size: 0.9rem;
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem 1.6rem;
    }}
    .sheet-summary-card b {{
        color: {BLUE};
    }}
    .sheet-summary-card .alert {{
        color: {ORANGE};
        font-weight: 600;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def section_banner(number: str, title: str):
    """章節橫幅：取代 st.header，統一①②③④區塊的視覺樣式（圓形編號徽章＋底線）。"""
    st.markdown(
        f"""
        <div class="section-banner">
            <span class="section-badge">{number}</span>
            <span class="section-title">{title}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def empty_state(icon: str, title: str, subtitle: str = ""):
    """統一的「尚無資料／結果」空狀態樣式，取代裸露的 st.info。"""
    subtitle_html = f'<div class="empty-subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="empty-state">
            <div class="empty-icon">{icon}</div>
            <div class="empty-title">{title}</div>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


st.title("🏠 長照居家照顧 AI 派單系統")
st.caption("居督與營運團隊可上傳／編輯排班資料、調整派單政策參數，並一鍵執行 AI 最佳化派單")

# ==========================================
# 側邊欄：核心派單政策參數
# ==========================================
defaults = PipelineConfig()

with st.sidebar:
    st.header("⚙️ 核心派單政策參數")

    if st.button("🔄 還原預設值", width="stretch"):
        for key in [
            "cfg_buffer_mins", "cfg_travel_penalty_weight", "cfg_urgent_priority_bonus",
            "cfg_continuity_bonus", "cfg_skill_bonus", "cfg_salary_rate_pct",
        ]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    buffer_mins = st.slider(
        "轉場與交接緩衝時間（分鐘）", 0.0, 60.0, defaults.buffer_mins, 1.0,
        key="cfg_buffer_mins",
        help="居服員完成一項任務後，考量停車、上下樓、門禁核對與交接紀錄所需的緩衝時間。",
    )
    travel_penalty_weight = st.slider(
        "車程扣分權重（分／分鐘）", 0.0, 5.0, defaults.travel_penalty_weight, 0.1,
        key="cfg_travel_penalty_weight",
        help="每一分鐘車程對適配度分數與派單目標函數的扣分幅度，數值越高代表越優先就近派案。"
        "（此權重不影響案家歷史首選居服員：連續性優先於車程限制，詳見下方照護連續性加分說明。）",
    )
    urgent_priority_bonus = st.slider(
        "緊急任務優先加分（分）", 0.0, 100.0, defaults.urgent_priority_bonus, 5.0,
        key="cfg_urgent_priority_bonus",
        help="任務優先級標示為「緊急」時，在派單目標函數中額外獲得的加分，數值越高代表緊急任務越優先被指派。",
    )
    continuity_bonus = st.slider(
        "照護連續性加分（分）", 0.0, 100.0, defaults.preferred_caregiver_bonus, 1.0,
        key="cfg_continuity_bonus",
        help="當案家的歷史首選居服員恰為候選居服員時的加分，鼓勵維持既有照護關係與默契。"
        "表現優良（滿意度達門檻）的首選居服員另有動態加成，避免因車程等微幅優化而被替換。",
    )
    skill_bonus = st.slider(
        "核心專長匹配加分（分）", 0.0, 50.0, defaults.cert_bonus_dementia, 1.0,
        key="cfg_skill_bonus",
        help="居服員持有之核心專長證照符合案家特殊照護需求時的加分。",
    )
    salary_rate_pct = st.slider(
        "居服員拆帳比例（%）", 0, 100, int(defaults.caregiver_salary_rate_per_point * 100), 5,
        key="cfg_salary_rate_pct",
        help="居服員實領薪資佔該任務長照申報點數（機構營收）的比例，用於試算「預估居服員拆帳薪資」。",
    )

config = PipelineConfig(
    buffer_mins=buffer_mins,
    travel_penalty_weight=travel_penalty_weight,
    objective_travel_weight=travel_penalty_weight,
    urgent_priority_bonus=urgent_priority_bonus,
    preferred_caregiver_bonus=continuity_bonus,
    cert_bonus_dementia=skill_bonus,
    cert_bonus_other=skill_bonus,
    caregiver_salary_rate_per_point=salary_rate_pct / 100.0,
)

# ==========================================
# 資料載入（先於區塊①②，供 Phase 0 健檢與編輯區共用）
# ==========================================
uploaded_file = st.file_uploader("上傳派單資料庫（.xlsx，未上傳則使用預設檔案）", type=["xlsx"])


@st.cache_data(show_spinner="讀取 Excel 資料中...")
def _read_raw_sheets(file_or_path):
    xl = pd.ExcelFile(file_or_path)
    missing_sheets = [s for s in FIXED_REQUIRED_SHEETS if s not in xl.sheet_names]
    if missing_sheets:
        raise ValueError(f"檔案缺少必要工作表：{'、'.join(missing_sheets)}")

    tasks_sheet_name = next((s for s in TASKS_SHEET_CANDIDATES if s in xl.sheet_names), None)
    if tasks_sheet_name is None:
        raise ValueError(f"檔案缺少任務工作表，需具備「{'」或「'.join(TASKS_SHEET_CANDIDATES)}」其中之一。")

    sheets = {name: pd.read_excel(xl, sheet_name=name) for name in FIXED_REQUIRED_SHEETS}
    sheets["Tasks"] = pd.read_excel(xl, sheet_name=tasks_sheet_name)

    column_checks = dict(REQUIRED_COLUMNS)
    column_checks["Tasks"] = REQUIRED_TASKS_COLUMNS
    for name, required_cols in column_checks.items():
        missing_cols = [c for c in required_cols if c not in sheets[name].columns]
        if missing_cols:
            label = tasks_sheet_name if name == "Tasks" else name
            raise ValueError(f"工作表「{label}」缺少必要欄位：{'、'.join(missing_cols)}")

    return sheets, tasks_sheet_name


try:
    if uploaded_file is not None:
        source_key = f"upload::{uploaded_file.name}::{uploaded_file.size}"
        sheets, tasks_sheet_name = _read_raw_sheets(uploaded_file)
    else:
        import os

        if not os.path.exists(DEFAULT_EXCEL_PATH):
            st.error(f"找不到預設檔案：{DEFAULT_EXCEL_PATH}，請改用上方上傳功能。")
            st.stop()
        source_key = f"default::{os.path.getmtime(DEFAULT_EXCEL_PATH)}"
        sheets, tasks_sheet_name = _read_raw_sheets(DEFAULT_EXCEL_PATH)
except ValueError as e:
    st.error(f"⚠️ 資料格式錯誤：{e}")
    st.stop()
except Exception as e:
    st.error(f"⚠️ 無法讀取上傳的檔案，請確認上傳的是有效的 Excel（.xlsx）檔案。錯誤訊息：{e}")
    st.stop()

# 僅在資料來源變更時（首次載入／換檔案／原檔被覆寫）重設編輯區，避免使用者的編輯內容被覆蓋
if st.session_state.get("_data_source_key") != source_key:
    st.session_state["_data_source_key"] = source_key
    st.session_state["edit_cg"] = sheets["Caregiver_Profiles"].copy()
    st.session_state["edit_cl"] = sheets["Client_Profiles"].copy()
    st.session_state["edit_tasks"] = sheets["Tasks"].copy()
    st.session_state["data_hist"] = sheets["Historical_Service_Logs"].copy()
    st.session_state["tasks_sheet_name"] = tasks_sheet_name

# ==========================================
# 區塊①：Phase 0 申報法規防呆健檢
# ==========================================
section_banner("①", "Phase 0：申報法規防呆健檢")
st.caption(
    "依目前編輯中的任務與案家資料，即時檢核 BA 服務代碼併報合規性（依長照給付支付基準併報規則）。"
    "本健檢僅檢核與提示、不會排除任何任務，亦不影響下方③區塊的派單運算。"
)
try:
    _merged_preview = st.session_state["edit_tasks"].merge(
        st.session_state["edit_cl"], on="案家ID", how="left"
    )
    _validated_preview = validate_ba_codes(_merged_preview)
    _total_tasks_preview = len(_validated_preview)
    _violation_count = int(_validated_preview["含違規代碼"].sum())
    _qualified_count = _total_tasks_preview - _violation_count

    h1, h2, h3 = st.columns(3)
    h1.metric("總任務數", f"{_total_tasks_preview}")
    h2.metric("合格任務數", f"{_qualified_count}")
    h3.metric(
        "⚠️ 偵測到違規申報數", f"{_violation_count}",
        delta="需複核" if _violation_count > 0 else "正常",
        delta_color="inverse" if _violation_count > 0 else "off",
    )

    if _violation_count > 0:
        st.warning(
            f"偵測到 {_violation_count} 筆任務的 BA 服務代碼併報疑似違反長照給付支付基準規則，"
            "建議於下方②區塊修正服務代碼，或由居督複核後仍可略過警告繼續派單。"
        )
        st.dataframe(
            _validated_preview.loc[_validated_preview["含違規代碼"], ["任務ID", "案家ID", "BA代碼檢核異常"]],
            width="stretch",
            hide_index=True,
        )
    else:
        st.success("目前資料未偵測到已知的 BA 服務代碼併報違規。")
except Exception as e:
    st.info(f"暫無法執行 BA 代碼健檢（請確認任務與案家資料的「案家ID」欄位可正常對應）：{e}")

# ==========================================
# 區塊②：Excel 載入與現況檢視／動態編輯
# ==========================================
section_banner("②", "資料載入與編輯")


def render_editable_sheet(session_key: str):
    with st.expander("➕ 新增欄位"):
        c1, c2 = st.columns([3, 1])
        new_col_name = c1.text_input("欄位名稱", key=f"newcol_input_{session_key}", label_visibility="collapsed",
                                      placeholder="輸入新欄位名稱")
        if c2.button("新增欄位", key=f"newcol_btn_{session_key}", width="stretch"):
            df_current = st.session_state[session_key]
            if not new_col_name:
                st.warning("請先輸入欄位名稱。")
            elif new_col_name in df_current.columns:
                st.warning(f"欄位「{new_col_name}」已存在。")
            else:
                df_current[new_col_name] = None
                st.session_state[session_key] = df_current
                st.rerun()

    df = st.session_state[session_key]
    edited = st.data_editor(
        df,
        num_rows="dynamic",
        width="stretch",
        key=f"editor_{session_key}_{len(df.columns)}",
    )
    st.session_state[session_key] = edited


def render_sheet_summary(items: list[tuple[str, str, bool]]):
    """分頁摘要卡：不展開表格也能一眼掌握資料健康度（列數、需留意項目等）。"""
    parts = [
        f'<span><b>{label}</b>：<span class="{"alert" if alert else ""}">{value}</span></span>'
        for label, value, alert in items
    ]
    st.markdown(f'<div class="sheet-summary-card">{"".join(parts)}</div>', unsafe_allow_html=True)


tab_cg, tab_cl, tab_tasks = st.tabs([
    "居服員資料 (Caregiver_Profiles)",
    "案家資料 (Client_Profiles)",
    f"任務資料 ({st.session_state['tasks_sheet_name']})",
])

with tab_cg:
    _df_cg_preview = st.session_state["edit_cg"]
    _cg_items = [("居服員人數", f"{len(_df_cg_preview)} 人", False)]
    if "當月累計服務時數(疲勞度)" in _df_cg_preview.columns:
        _fatigue_high = int(
            (pd.to_numeric(_df_cg_preview["當月累計服務時數(疲勞度)"], errors="coerce")
             >= config.fatigue_reference_hours).sum()
        )
        _cg_items.append((
            f"疲勞度偏高（≥{config.fatigue_reference_hours:.0f}小時）", f"{_fatigue_high} 人", _fatigue_high > 0,
        ))
    if "具備重度移位體力(0/1)" in _df_cg_preview.columns:
        _heavy_capable = int(pd.to_numeric(_df_cg_preview["具備重度移位體力(0/1)"], errors="coerce").eq(1).sum())
        _cg_items.append(("具重度移位體力", f"{_heavy_capable} 人", False))
    render_sheet_summary(_cg_items)
    st.caption("可直接編輯儲存格、新增／刪除列（勾選列號後按 Delete），或透過「新增欄位」新增自訂欄位。")
    render_editable_sheet("edit_cg")

with tab_cl:
    _df_cl_preview = st.session_state["edit_cl"]
    _cl_items = [("案家戶數", f"{len(_df_cl_preview)} 戶", False)]
    if "需重度移位協助(0/1)" in _df_cl_preview.columns:
        _heavy_needed = int(pd.to_numeric(_df_cl_preview["需重度移位協助(0/1)"], errors="coerce").eq(1).sum())
        _cl_items.append(("需重度移位協助", f"{_heavy_needed} 戶", False))
    if "指定居服員性別" in _df_cl_preview.columns:
        _gender_req = int(_df_cl_preview["指定居服員性別"].isin(["限女性", "限男性"]).sum())
        _cl_items.append(("有性別限定", f"{_gender_req} 戶", False))
    render_sheet_summary(_cl_items)
    st.caption("可直接編輯儲存格、新增／刪除列，或透過「新增欄位」新增自訂欄位。")
    render_editable_sheet("edit_cl")

with tab_tasks:
    _df_tasks_preview = st.session_state["edit_tasks"]
    _tasks_items = [("任務筆數", f"{len(_df_tasks_preview)} 筆", False)]
    if "任務優先級" in _df_tasks_preview.columns:
        _urgent_count = int(_df_tasks_preview["任務優先級"].astype(str).str.contains("緊急", na=False).sum())
        _tasks_items.append(("緊急任務", f"{_urgent_count} 筆", False))
    if "日期" in _df_tasks_preview.columns:
        _date_span = _df_tasks_preview["日期"].dropna().nunique()
        _tasks_items.append(("涵蓋日期數", f"{_date_span} 天", False))
    render_sheet_summary(_tasks_items)
    st.caption("可直接編輯儲存格、新增／刪除列，或透過「新增欄位」新增自訂欄位。")
    render_editable_sheet("edit_tasks")

# ==========================================
# 區塊③：一鍵執行與成果儀表板
# ==========================================
section_banner("③", "執行最佳化派單與成果儀表板")

_tasks_preview_df = st.session_state["edit_tasks"]
has_date_column = "日期" in _tasks_preview_df.columns

schedule_mode = "單日排程"
selected_schedule_date = None
if has_date_column:
    schedule_mode = st.radio(
        "排程範圍",
        ["單日排程", "全月一鍵排程"],
        horizontal=True,
        key="schedule_mode",
        help="偵測到任務資料含「日期」欄位：可選擇僅排單一天的班表，或一次排完整個月"
        "（依日期逐日各自求解 Phase 1 + Phase 2 後彙整為全期間派單總表）。",
    )
    if schedule_mode == "單日排程":
        available_dates = sorted(_tasks_preview_df["日期"].dropna().unique())
        if available_dates:
            selected_schedule_date = st.selectbox("選擇排程日期", available_dates, key="schedule_selected_date")

if st.button("🚀 執行 AI 最佳化派單", type="primary"):
    df_cg = st.session_state["edit_cg"].copy()
    df_cl = st.session_state["edit_cl"].copy()
    df_hist = st.session_state["data_hist"].copy()

    df_tasks = st.session_state["edit_tasks"].copy()
    if has_date_column and schedule_mode == "單日排程" and selected_schedule_date is not None:
        df_tasks = df_tasks[df_tasks["日期"] == selected_schedule_date].copy()

    with st.status("🚀 AI 派單運算中...", expanded=True) as status:
        try:
            status.write("🔍 Phase 1／候選配對評分中（含 OSRM 車程查詢，視資料量可能需要數秒）...")
            tasks = df_tasks.merge(df_cl, on="案家ID", how="left")
            df_matches = run_phase1_matching(tasks, df_cg, config)

            if has_date_column and schedule_mode == "全月一鍵排程":
                status.write("🧮 Phase 2／全月批次最佳化求解中（逐日求解後彙整）...")
                batch = run_monthly_batch_dispatch(tasks, df_cg, config, date_column="日期")
                failed_dates = [
                    d for d, r in batch["daily_results"].items() if r["status"] != pywraplp.Solver.OPTIMAL
                ]
                phase2 = {
                    "df_valid": pd.DataFrame(),
                    "status": pywraplp.Solver.OPTIMAL if not failed_dates else None,
                    "df_result": batch["df_result_all"],
                    "assigned_count": batch["total_assigned_count"],
                }
            else:
                status.write("🧮 Phase 2／OR-Tools 最佳化求解中...")
                phase2 = run_phase2_optimization(df_matches, tasks, df_cg, config)
                failed_dates = []

            status.write("📈 Phase 3／AI vs 人工歷史成效回溯分析中...")
            did = run_phase3_did(df_hist)

            status.update(label="✅ AI 派單運算完成", state="complete", expanded=False)
        except Exception as e:
            status.update(label="❌ 派單運算失敗", state="error", expanded=True)
            st.error(
                "⚠️ 派單運算過程發生錯誤，請確認表格中的 ID 對應是否存在、數值欄位是否填寫正確"
                f"（例如經緯度、工時、時間格式「HH:MM」）。錯誤訊息：{e}"
            )
            st.stop()

    st.session_state["last_result"] = {
        "df_tasks": df_tasks,
        "df_cg": df_cg,
        "df_cl": df_cl,
        "df_matches": df_matches,
        "phase2": phase2,
        "did": did,
        "schedule_failed_dates": failed_dates,
    }
    # 每次重新執行最佳化後，AI 建議已改變，先前的居督覆寫不再對應同一份建議，故一併清空。
    st.session_state["overrides"] = {}

if "last_result" in st.session_state:
    res = st.session_state["last_result"]
    df_tasks = res["df_tasks"]
    df_matches = res["df_matches"]
    phase2 = res["phase2"]
    did = res["did"]
    schedule_failed_dates = res.get("schedule_failed_dates") or []

    df_result = phase2["df_result"]
    assigned_count = phase2["assigned_count"]
    solver_ok = phase2["status"] == pywraplp.Solver.OPTIMAL

    # overrides／assigned_map 提前到此處初始化（原僅存在於④區塊），因為月曆視角
    # 「一鍵調班」與④區塊「居督人工覆寫」共用同一份 overrides 狀態與同一套
    # save_override_log 稽核紀錄，月曆總覽（KPI 之後即會渲染）需要在此之前就能
    # 讀寫這份狀態，兩處才不會各自維護一份互不同步的覆寫紀錄。
    st.session_state.setdefault("overrides", {})
    overrides = st.session_state["overrides"]
    assigned_map = dict(zip(df_result["任務ID"], df_result["派單居服員"])) if not df_result.empty else {}
    task_locations = (
        df_matches.drop_duplicates("任務ID").set_index("任務ID")[["地點緯度", "地點經度"]].apply(tuple, axis=1).to_dict()
        if not df_matches.empty
        else {}
    )

    def _quick_reassign(task_id, new_cg_id, reason):
        """統一的居督覆寫寫入入口：即時衝突檢查通過後，寫入 overrides 狀態與稽核
        日誌；回傳 None 表示成功，否則回傳供呼叫端 Modal 顯示的錯誤訊息。

        供月曆「居服員 x 日期」改派 Modal、月曆「未派單案件」處置，以及④區塊
        「居督人工覆寫」任務搜尋器共用（calendar_view._task_override_dialog／
        _quick_reassign_dialog）：三者差異只在 task_id 原本是否已有指派、觸發
        來源不同，apply_overrides_to_result 本就支援替未指派任務新增覆寫列，
        不需要另外實作一套指派邏輯；reason 由呼叫端的原因選擇器收集（必填）。
        """
        df_result_effective = apply_overrides_to_result(df_result, overrides)
        conflict_msg = check_reassignment_conflict(
            task_id, new_cg_id, df_tasks, df_result_effective, res.get("df_cg", pd.DataFrame()),
            config, task_locations=task_locations, date_column="日期",
        )
        if conflict_msg:
            return conflict_msg

        task_rows = df_tasks[df_tasks["任務ID"] == task_id]
        cl_id = task_rows.iloc[0]["案家ID"] if not task_rows.empty else ""
        ai_cg = assigned_map.get(task_id)
        ai_label = str(ai_cg) if ai_cg is not None else "未指派"
        save_override_log(task_id, cl_id, ai_label, new_cg_id if new_cg_id is not None else "未指派", reason)
        overrides[task_id] = {"cg_id": new_cg_id, "reason": reason}
        return None

    def _clear_override(task_id):
        """清除單一任務的居督覆寫，還原為 AI 建議；與原④區塊「清除覆寫」行為
        相同，不寫入稽核日誌（稽核日誌只記錄實際發生過的覆寫變更）。"""
        overrides.pop(task_id, None)

    def _list_candidates(task_id, candidate_cg_ids):
        """供月曆快速改派下拉選單使用：把候選居服員依「該時段是否有空檔」排序＋標籤。

        與 _quick_reassign 共用同一套 caregiver_engine.rank_candidates_by_availability／
        check_reassignment_conflict 判定邏輯（同源於 _evaluate_reassignment），確保
        選單顯示「可派單」的候選人在按下確認改派時不會被判定衝突而拒絕。

        另外傳入 df_cl，讓排序結果一併標記 Phase 1 硬性資格條件（性別、重度移位、
        環境排斥、可排班星期、可服務時段、請假、專長認證）不符者，而不是讓這些人
        因為候選名單只取全體居服員（見 calendar_view._quick_reassign_dialog）而
        「看似可派但其實從未檢查資格」——這裡刻意不影響 _quick_reassign 的
        check_reassignment_conflict（未傳 df_cl，維持原本只擋時間衝突／工時超額的
        權威判定範圍），因為本系統無強制派單權限機制，居督看到標記後仍可自行覆寫。
        """
        df_result_effective = apply_overrides_to_result(df_result, overrides)
        return rank_candidates_by_availability(
            task_id, candidate_cg_ids, df_tasks, df_result_effective, res.get("df_cg", pd.DataFrame()),
            config, task_locations=task_locations, date_column="日期",
            df_cl=res.get("df_cl", pd.DataFrame()),
        )

    if not solver_ok:
        if schedule_failed_dates:
            st.warning(
                f"以下 {len(schedule_failed_dates)} 個日期無法求得最佳解（其餘日期仍正常派單）："
                f"{', '.join(str(d) for d in schedule_failed_dates)}"
            )
        else:
            st.warning("OR-Tools 無法在目前資料與參數下找到最佳解，請檢查資料是否有效或調整參數。")

    total_tasks = len(df_tasks)
    assign_rate = (assigned_count / total_tasks * 100) if total_tasks else 0.0
    avg_transition = (
        (df_result["預估車程(分)"] + config.buffer_mins).mean() if not df_result.empty else 0.0
    )
    avg_score = df_result["適配分數"].mean() if not df_result.empty else 0.0
    total_revenue = df_result["預估長照申報點數(營收)"].sum() if not df_result.empty else 0.0
    total_salary = df_result["預估居服員拆帳薪資"].sum() if not df_result.empty else 0.0

    st.subheader("📌 KPI 指標")
    if assign_rate >= 90:
        assign_rate_delta, assign_rate_delta_color = "良好", "normal"
    elif assign_rate >= 70:
        assign_rate_delta, assign_rate_delta_color = "普通", "off"
    else:
        assign_rate_delta, assign_rate_delta_color = "偏低", "inverse"

    k1, k2, k3 = st.columns(3)
    k1.metric(
        "派單成功率", f"{assign_rate:.1f}%",
        delta=assign_rate_delta, delta_color=assign_rate_delta_color,
        help=f"{assigned_count} / {total_tasks} 筆任務成功指派",
    )
    k2.metric("平均轉場時間", f"{avg_transition:.1f} 分", help="已派單任務的平均車程時間＋轉場緩衝時間")
    k3.metric("平均適配得分", f"{avg_score:.1f} 分", help="已派單配對的平均適配度分數")

    k4, k5 = st.columns(2)
    k4.metric(
        "預估長照申報總點數（營收）", f"{total_revenue:,.0f} 點",
        help="已派單任務的長照申報點數總和，依 BA 服務代碼點值試算（無代碼者以服務歷時概算）。",
    )
    k5.metric(
        "預估居服員拆帳總薪資", f"{total_salary:,.0f} 元",
        help=f"申報點數 × 側邊欄設定的拆帳比例（目前 {salary_rate_pct}%）加總。",
    )

    # 月曆班表總覽：彙總既有派單結果與居服員資料表（並套用 overrides 顯示目前實際
    # 生效班表），不重呼叫任何排班演算法（見 calendar_view.py）；作為整個結果區的
    # 「首頁總覽」，置於 KPI 之後、派單分析儀表板之前。overrides／on_reassign／
    # on_list_candidates 供「月曆視角一鍵調班」（含空檔優先排序）使用。
    render_calendar_overview(
        df_result, df_tasks, res.get("df_cg", pd.DataFrame()),
        overrides=overrides,
        on_reassign=_quick_reassign, on_list_candidates=_list_candidates,
    )

    st.subheader("📊 派單分析儀表板")
    if df_matches.empty:
        empty_state("🧩", "目前無候選配對可供分析", "可能所有配對皆被硬性條件過濾，請檢查②區塊的居服員／案家資料是否有衝突條件。")
    else:
        mean_score = df_matches["適配度分數"].mean()
        unassigned = total_tasks - assigned_count

        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "domain"}, {"type": "xy"}]],
            subplot_titles=[
                "適配度分數分布（全部候選配對）",
                "各居服員派單量",
                "任務派單狀態占比",
                f"歷史平均滿意度比較（淨提升 +{did['uplift_sat']:.2f} 分）",
            ],
            vertical_spacing=0.14, horizontal_spacing=0.1,
        )

        # (1) 適配度分數分布
        fig.add_trace(
            go.Histogram(x=df_matches["適配度分數"], nbinsx=20, marker_color=BLUE, showlegend=False),
            row=1, col=1,
        )
        fig.add_vline(
            x=mean_score, line_dash="dash", line_color=MUTED,
            annotation_text=f"平均 {mean_score:.1f}", annotation_position="top",
            row=1, col=1,
        )
        fig.update_xaxes(title_text="適配度分數", row=1, col=1)
        fig.update_yaxes(title_text="候選配對數", row=1, col=1)

        # (2) 各居服員派單量
        if not df_result.empty:
            load_counts = df_result["派單居服員"].value_counts().sort_values(ascending=True)
            fig.add_trace(
                go.Bar(x=load_counts.values, y=load_counts.index.astype(str), orientation="h",
                       marker_color=BLUE, showlegend=False),
                row=1, col=2,
            )
        else:
            fig.add_trace(
                go.Bar(x=[0], y=["尚無派單結果"], orientation="h", marker_color=MUTED, showlegend=False),
                row=1, col=2,
            )
        fig.update_xaxes(title_text="派單任務數", row=1, col=2)

        # (3) 派單狀態占比
        if total_tasks > 0:
            fig.add_trace(
                go.Pie(
                    labels=["已派單", "未派單"], values=[assigned_count, unassigned],
                    marker_colors=[BLUE, MUTED], textinfo="percent+label", showlegend=False,
                ),
                row=2, col=1,
            )

        # (4) AI vs 人工歷史成效比較（DiD）
        fig.add_trace(
            go.Bar(
                x=["AI 派單(Treatment)", "人工派單(Control)"],
                y=[did["ai_sat"], did["human_sat"]],
                marker_color=[BLUE, ORANGE],
                text=[f"{v:.2f}" for v in [did["ai_sat"], did["human_sat"]]],
                textposition="outside", showlegend=False,
            ),
            row=2, col=2,
        )
        fig.update_yaxes(title_text="案家滿意度（1-5分）", range=[0, 5], row=2, col=2)

        fig.update_layout(height=760, margin=dict(t=60, b=20, l=10, r=10))
        st.plotly_chart(fig, width="stretch")

    st.subheader("📋 指派明細表")
    if df_result.empty:
        empty_state("📭", "目前無派單結果")
    else:
        df_display = df_result.copy()
        has_date_for_display = "日期" in df_tasks.columns
        if has_date_for_display:
            date_map = df_tasks.set_index("任務ID")["日期"]
            df_display["日期"] = df_display["任務ID"].map(date_map)
            if "星期" in df_tasks.columns:
                weekday_map = df_tasks.set_index("任務ID")["星期"]
                df_display["星期"] = df_display["任務ID"].map(weekday_map)
            else:
                df_display["星期"] = df_display["日期"].apply(get_weekday_name)

        display_cols = [
            "任務ID", "案家ID", "派單居服員", "適配分數", "預估車程(分)", "服務時段", "任務優先級",
            "預估長照申報點數(營收)", "預估居服員拆帳薪資", "原首選替換原因",
        ]
        if has_date_for_display:
            display_cols[1:1] = ["日期", "星期"]
        st.dataframe(df_display[display_cols], width="stretch", hide_index=True)

        csv = df_display[display_cols].to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ 匯出指派結果 (assigned_results.csv)",
            csv,
            file_name="assigned_results.csv",
            mime="text/csv",
        )

    # ==========================================
    # 區塊④：居督人工覆寫（Supervisor Override）與稽核日誌
    # ==========================================
    section_banner("④", "居督人工覆寫（Supervisor Override）")
    st.caption(
        "當 AI 建議排單不符合實際場域狀況時（例如居服員臨時請假、案家臨時改期），"
        "居督可搜尋／選擇任務後於彈出視窗手動重新指定居服員（與月曆視角「一鍵調班」"
        "共用同一套覆寫互動介面）；每一筆變更皆會記錄原因並寫入稽核日誌，供後續演算法迭代分析。"
    )

    # overrides 已於本區塊之前（月曆總覽渲染前）初始化，此處沿用同一份 session_state。

    OVERRIDE_TRAVEL_ALERT_THRESHOLD = 3

    override_log_df = load_override_log()
    travel_reason_count = (
        (override_log_df["變更原因"] == "車程太遠").sum() if not override_log_df.empty else 0
    )
    if travel_reason_count >= OVERRIDE_TRAVEL_ALERT_THRESHOLD:
        st.warning(
            f"📈 稽核日誌累計已有 {travel_reason_count} 筆覆寫原因為「車程太遠」，"
            "建議提高側邊欄的『車程扣分權重』，讓 AI 派單更優先考量就近指派。"
        )

    # assigned_map 已於本區塊之前初始化並供月曆快速改派共用，此處沿用同一份。
    render_task_override_picker(
        df_tasks, df_result, res.get("df_cg", pd.DataFrame()), overrides,
        on_reassign=_quick_reassign, on_clear_override=_clear_override, on_list_candidates=_list_candidates,
    )

    st.subheader("📄 最終派單結果（含居督覆寫）")
    has_date_for_final = "日期" in df_tasks.columns
    has_weekday_col_for_final = "星期" in df_tasks.columns
    final_rows = []
    for _, task_row in df_tasks.iterrows():
        t_id = task_row["任務ID"]
        cl_id = task_row["案家ID"]
        base = df_result[df_result["任務ID"] == t_id] if not df_result.empty else pd.DataFrame()
        ai_row = base.iloc[0] if not base.empty else None
        ai_cg = ai_row["派單居服員"] if ai_row is not None else None

        override = overrides.get(t_id)
        if override:
            final_cg = override["cg_id"]
            change_note = f"居督覆寫：{override['reason']}"
        else:
            final_cg = ai_cg
            change_note = ai_row["原首選替換原因"] if ai_row is not None else ""

        row_dict = {"任務ID": t_id, "案家ID": cl_id}
        if has_date_for_final:
            date_val = task_row.get("日期")
            row_dict["日期"] = date_val
            row_dict["星期"] = (
                task_row.get("星期")
                if has_weekday_col_for_final and pd.notna(task_row.get("星期"))
                else get_weekday_name(date_val)
            )
        row_dict.update({
            "AI建議居服員": ai_cg if ai_cg is not None else "未指派",
            "最終派單居服員": final_cg if final_cg is not None else "未指派",
            "適配分數": ai_row["適配分數"] if ai_row is not None else None,
            "預估車程(分)": ai_row["預估車程(分)"] if ai_row is not None else None,
            "服務時段": ai_row["服務時段"] if ai_row is not None else "",
            "任務優先級": ai_row["任務優先級"] if ai_row is not None else task_row.get("任務優先級", ""),
            "預估長照申報點數(營收)": ai_row["預估長照申報點數(營收)"] if ai_row is not None else None,
            "預估居服員拆帳薪資": ai_row["預估居服員拆帳薪資"] if ai_row is not None else None,
            "備註": change_note,
        })
        final_rows.append(row_dict)

    df_final = pd.DataFrame(final_rows)
    st.dataframe(df_final, width="stretch", hide_index=True)

    csv_final = df_final.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇️ 匯出最終派單結果（含覆寫） (final_assignment_with_overrides.csv)",
        csv_final,
        file_name="final_assignment_with_overrides.csv",
        mime="text/csv",
    )

    with st.expander("🗂️ 檢視完整稽核日誌 (supervisor_override_log.csv)"):
        if override_log_df.empty:
            empty_state("🗂️", "尚無居督覆寫紀錄")
        else:
            st.dataframe(override_log_df, width="stretch", hide_index=True)
else:
    empty_state("🚀", "尚未執行派單", "請先於上方②區塊確認／編輯資料，再按下「🚀 執行 AI 最佳化派單」開始運算。")
