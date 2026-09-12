"""月曆式班表總覽 - Streamlit 前端呈現模組。

讀取 app.py 既有的 st.session_state["last_result"]（df_tasks、phase2["df_result"]）
與 st.session_state["edit_cg"] 做彙總與呈現，並套用居督覆寫
（st.session_state["overrides"]，與④區塊「居督人工覆寫」共用同一份狀態）以顯示
「目前實際生效」的班表，不重複實作、也不修改 caregiver_engine.py 的任何排班演算法
邏輯。僅在任務資料具備「日期」欄位（月批次排班模式）時才顯示；單日排程模式下由
呼叫端（app.py）略過本區塊。

本模組的「寫入」動作有三種入口，皆委派給同一個 `on_reassign` callback（app.py 中
實作，會呼叫 caregiver_engine.check_reassignment_conflict 與 save_override_log）：
1. 「月曆視角一鍵調班」：點擊週矩陣中「居服員 x 日期」格位，開啟 Modal 檢視當日
   任務並快速改派給其他居服員（見 _quick_reassign_dialog）。
2. 「未派單案件快速指派」：點擊月曆格位中的「🚩 N 筆未派單」，於週矩陣下方列出
   當日未派單任務，逐筆開啟單一任務覆寫 Modal（見 _task_override_dialog）。
3. 「④居督人工覆寫」：搜尋／選擇任一任務（不論是否已指派），開啟與上述相同的
   單一任務覆寫 Modal（見 render_task_override_picker），取代逐筆列出全部任務的
   舊版清單。

三者都不直接呼叫任何排班演算法函式，僅呈現 callback 回傳的成功/錯誤結果；
「原本已指派」與「原本未指派」的任務走同一套覆寫邏輯，因為
apply_overrides_to_result 本就支援替未指派任務新增覆寫列。

Dialog 開啟狀態統一收斂到單一 session_state 鍵 `calendar_active_dialog`
（None｜{"kind": "reassign", "cg_id", "date"}｜{"kind": "task", "task_id"}），
不使用各自獨立的旗標。原因：st.dialog 的原生右上角關閉鈕（× icon）沒有
on_close callback 可掛勾，居督若透過它關閉而非本模組的自訂「✕ 關閉」按鈕，
觸發開啟的旗標不會被清除；下次任何 rerun（例如點選月曆上完全無關的日期或
未派單徽章）只要仍呼叫到同一段「if 旗標為真就開 Dialog」的程式碼，就會讓
Dialog 意外重新彈出。改用單一鍵有兩個好處：(1) 開啟一種 Dialog 必然覆蓋掉另一
種的待開啟狀態，兩者不可能同時為真，從結構上就不會撞上 Streamlit 的
「Only one dialog is allowed to be opened at the same time」例外；(2) 「日期」與
「未派單」徽章的點擊處理常式會明確把這個鍵重設為 None，即使先前有殘留的
待開啟狀態，點下這兩個按鈕後也保證不會意外彈出任何 Dialog。

kind="task" 的 Dialog 統一由 render_task_override_picker（於 app.py④區塊呼叫，
每次 rerun 都會執行）集中派送，不論觸發來源是月曆「未派單」按鈕還是④區塊的
任務搜尋器：因為 st.dialog 是浮動於畫面中央的 Modal，不受呼叫點在腳本中的
位置影響，集中派送可讓「成功後彈出 Toast 並關閉 Dialog」只需實作一次
（_task_override_dialog 開頭的 success_key 檢查），不必為「觸發來源不同、
成功後任務是否還留在候選清單中」個別寫收尾判斷。

點擊「日期」／「未派單」徽章另外會設定 session_state["calendar_scroll_to"]
（None｜"matrix"｜"unassigned"），驅動頁面平滑捲動到對應錨點（見
_inject_scroll_js）。與 calendar_active_dialog 相同的「單一鍵、用完即清」原則：
_render_week_matrix 一開始就把這個鍵 pop 出來（而非只是讀取），因此同一次
Dialog 內互動觸發的後續 rerun 一定讀到 None，不會重複觸發捲動。
"""

from typing import Optional
import calendar as pycal
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from caregiver_engine import WEEKDAY_NUM_TO_NAME

# 狀態色票：與月曆總覽格的文字警示、細節矩陣的儲存格底色共用同一套色彩語意，
# 避免居督在「月曆格」與「矩陣細節」之間看到兩套不同顏色定義而混淆。
STATUS_COLORS = {
    "normal": ("#e6f4ea", "#1e7d34"),   # 淡綠：正常工時
    "warning": ("#fff6d6", "#8a6100"),  # 淡黃：接近工時上限
    "danger": ("#fdecea", "#c62828"),   # 淡紅：超過工時上限
    "off": ("#eeeeee", "#6b6b6b"),      # 淡灰：請假或當日不可排班
}
STATUS_LABELS = {
    "normal": "正常工時（< 80% 上限）",
    "warning": "接近工時上限（80%~100%）",
    "danger": "超過工時上限",
    "off": "請假或不可排班",
}

# 居督覆寫原因分類：月曆快速改派、未派單／單一任務覆寫 Modal 共用同一套分類，
# 確保無論從哪個入口寫入，稽核日誌（supervisor_override_log.csv）的「變更原因」
# 欄位格式一致，可直接彙總統計（例如④區塊的「車程太遠」累計筆數警示）。
OVERRIDE_UNASSIGN = "撤銷指派（不指派）"
OVERRIDE_REASONS = ["車程／交通因素",
                    "居服員臨時請假",
                    "案家臨時改期",
                    "長者情緒或接受度",
                    "案家／家屬服務偏好",
                    "服務連續性／熟悉度考量",
                    "特殊照護需求", "其他",
]


def _to_date(value):
    """將日期欄位值（可能是字串／Timestamp／date）正規化為 date；無法解析回傳 None。"""
    if pd.isna(value):
        return None
    try:
        return pd.to_datetime(value).date()
    except (ValueError, TypeError):
        return None


def apply_overrides_to_result(df_result: pd.DataFrame, overrides: dict) -> pd.DataFrame:
    """將居督覆寫（④區塊與月曆快速改派共用的 overrides dict，任務ID -> {"cg_id", "reason"}）
    疊加到 AI 原始 df_result，回傳反映『目前實際生效』指派的 df_result。

    純資料整併：只操作 任務ID／派單居服員 兩欄，不重新計算適配分數、車程等其他欄位
    （其餘欄位對新增列一律留空，因為月曆彙總函式僅需這兩欄）。cg_id 為 None
    表示該任務已被撤銷指派，會從結果中移除。
    """
    if not overrides:
        return df_result

    df = df_result.copy() if not df_result.empty else pd.DataFrame(columns=["任務ID", "派單居服員"])
    override_cg = {tid: ov["cg_id"] for tid, ov in overrides.items()}

    if not df.empty:
        mask = df["任務ID"].isin(override_cg)
        df.loc[mask, "派單居服員"] = df.loc[mask, "任務ID"].map(override_cg)
        unassigned_ids = {tid for tid, cg in override_cg.items() if cg is None}
        df = df[~df["任務ID"].isin(unassigned_ids)]

    existing_ids = set(df["任務ID"]) if not df.empty else set()
    new_rows = [
        {"任務ID": tid, "派單居服員": cg}
        for tid, cg in override_cg.items()
        if cg is not None and tid not in existing_ids
    ]
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    return df


# ==========================================
# 資料層：彙總函式（純讀取既有排班結果，不涉及派單演算法）
# ==========================================
@st.cache_data(show_spinner=False)
def build_daily_summary(
    df_result: pd.DataFrame,
    df_tasks: pd.DataFrame,
    df_cg: pd.DataFrame,
    date_column: str = "日期",
) -> dict:
    """依日期彙總每日派單概況，供月曆總覽格顯示用。

    回傳 {date: {"total_tasks", "assigned_count", "unassigned_count",
    "over_capacity_cg_count", "has_unassigned"}}。date_column 不存在於
    df_tasks 時回傳空 dict（由呼叫端另行顯示「不適用」提示）。
    """
    summary: dict = {}
    if date_column not in df_tasks.columns:
        return summary

    duration_map = df_tasks.set_index("任務ID")["服務歷時(分鐘)"]
    cap_map = (
        df_cg.assign(居服員ID=df_cg["居服員ID"].astype(str)).set_index("居服員ID")["每日工時上限(小時)"]
        if "居服員ID" in df_cg.columns and not df_cg.empty
        else pd.Series(dtype=float)
    )

    for date_val, day_tasks in df_tasks.groupby(date_column):
        d = _to_date(date_val)
        if d is None:
            continue

        task_ids = set(day_tasks["任務ID"])
        total_tasks = len(task_ids)

        day_result = df_result[df_result["任務ID"].isin(task_ids)] if not df_result.empty else pd.DataFrame()
        assigned_count = int(day_result["任務ID"].nunique()) if not day_result.empty else 0
        unassigned_count = total_tasks - assigned_count

        over_capacity_cg_count = 0
        if not day_result.empty:
            hours_by_cg = day_result.assign(
                _工時=day_result["任務ID"].map(duration_map).fillna(0.0) / 60.0
            ).groupby("派單居服員")["_工時"].sum()
            for cg_id, hrs in hours_by_cg.items():
                cap = cap_map.get(str(cg_id))
                if pd.notna(cap) and hrs > cap:
                    over_capacity_cg_count += 1

        summary[d] = {
            "total_tasks": total_tasks,
            "assigned_count": assigned_count,
            "unassigned_count": unassigned_count,
            "over_capacity_cg_count": over_capacity_cg_count,
            "has_unassigned": unassigned_count > 0,
        }

    return summary


@st.cache_data(show_spinner=False)
def build_caregiver_day_matrix(
    df_result: pd.DataFrame,
    df_tasks: pd.DataFrame,
    df_cg: pd.DataFrame,
    week_dates: list,
    date_column: str = "日期",
) -> pd.DataFrame:
    """建立「居服員 x 日期」矩陣，供點日期展開的細節檢視使用。

    每格內容為 {"hours": 當日總工時(小時，四捨五入至小數點一位), "status": 狀態}，
    狀態為 'off'（請假或不在可排班星期名單）／'normal'（<80% 上限）／
    'warning'（80%~100%）／'danger'（超過 100%）之一。df_cg 缺少
    「請假或不排班日期」「可排班星期」欄位時，一律視為可排班（不判定為 off）。

    week_dates 為 7 個 date（週一到週日）。若居服員總數 > 15，僅列出當週有
    出現在 df_result 派單結果中的居服員，避免矩陣過長；15 位以下則列出全部。
    """
    if date_column not in df_tasks.columns or df_cg.empty or "居服員ID" not in df_cg.columns:
        return pd.DataFrame()

    week_dates_set = set(week_dates)
    task_date_map = {
        row["任務ID"]: _to_date(row[date_column])
        for _, row in df_tasks.iterrows()
        if _to_date(row[date_column]) in week_dates_set
    }
    week_task_ids = set(task_date_map.keys())

    if df_result.empty:
        week_result = pd.DataFrame(columns=["任務ID", "派單居服員"])
    else:
        week_result = df_result[df_result["任務ID"].isin(week_task_ids)].copy()
        week_result["_日期"] = week_result["任務ID"].map(task_date_map)
        week_result["派單居服員"] = week_result["派單居服員"].astype(str)

    all_cg_ids = sorted(df_cg["居服員ID"].astype(str).unique().tolist())
    if len(all_cg_ids) > 15:
        cg_ids = sorted(week_result["派單居服員"].unique().tolist()) if not week_result.empty else []
    else:
        cg_ids = all_cg_ids

    duration_map = df_tasks.set_index("任務ID")["服務歷時(分鐘)"]
    cg_indexed = df_cg.assign(居服員ID=df_cg["居服員ID"].astype(str)).set_index("居服員ID")

    rows = {}
    for cg_id in cg_ids:
        cg_row = cg_indexed.loc[cg_id] if cg_id in cg_indexed.index else None
        cap = cg_row.get("每日工時上限(小時)") if cg_row is not None else None

        leave_str = cg_row.get("請假或不排班日期") if cg_row is not None else None
        leave_list = []
        if pd.notna(leave_str) and str(leave_str).strip() not in ("", "無"):
            leave_list = [x.strip() for x in str(leave_str).split(",") if x.strip()]

        allowed_days_str = cg_row.get("可排班星期") if cg_row is not None else None
        allowed_days = None
        if pd.notna(allowed_days_str) and str(allowed_days_str).strip():
            allowed_days = [int(x.strip()) for x in str(allowed_days_str).split(",") if x.strip().isdigit()]

        row_cells = {}
        for d in week_dates:
            is_off = d.isoformat() in leave_list
            if not is_off and allowed_days is not None and d.isoweekday() not in allowed_days:
                is_off = True

            hours = 0.0
            if not week_result.empty:
                cg_day_rows = week_result[
                    (week_result["派單居服員"] == cg_id) & (week_result["_日期"] == d)
                ]
                hours = sum(duration_map.get(tid, 0.0) for tid in cg_day_rows["任務ID"]) / 60.0

            if is_off:
                status = "off"
            elif cap is None or pd.isna(cap) or cap <= 0:
                status = "normal"
            else:
                ratio = hours / cap
                status = "danger" if ratio > 1.0 else ("warning" if ratio >= 0.8 else "normal")

            row_cells[d] = {"hours": round(hours, 1), "status": status}
        rows[cg_id] = row_cells

    if not rows:
        return pd.DataFrame()

    matrix = pd.DataFrame.from_dict(rows, orient="index")
    matrix.index.name = "居服員ID"
    return matrix[week_dates]


# ==========================================
# UI 層
# ==========================================
def _month_grid(year: int, month: int) -> list:
    """回傳該月份的週列表（每週 7 格，星期一到星期日），月份外的格子以 None 補齊。"""
    first_day = date(year, month, 1)
    days_in_month = pycal.monthrange(year, month)[1]
    lead_blanks = first_day.isoweekday() - 1  # 星期一=1 -> 0 個前導空白
    cells = [None] * lead_blanks + [date(year, month, d) for d in range(1, days_in_month + 1)]
    while len(cells) % 7 != 0:
        cells.append(None)
    return [cells[i : i + 7] for i in range(0, len(cells), 7)]


def _cell_badge_html(cell: dict, bold: bool) -> str:
    bg, fg = STATUS_COLORS.get(cell["status"], ("", ""))
    # 「off」原意是「當日不排班／請假」，正常情況下這樣的格子 hours 必為 0。但月曆
    # 快速改派／未派單指派刻意不檢查可排班星期／請假等硬性資格（見 on_reassign＝
    # check_reassignment_conflict 的說明，只擋時間衝突與工時上限，其餘留給居督
    # 自行判斷），所以居督仍可能把任務強制指派給當天「off」的居服員。此時若仍顯示
    # 「請假」二字會讓這筆已存在的任務在畫面上完全消失，牴觸「指派後必須正確顯示
    # 在對應格位」的需求，故 hours > 0 時一律顯示實際工時（off 用 ⚠ 前綴另外提示
    # 這是排班日以外的強制指派，而不是隱藏掉）。
    if cell["status"] == "off" and cell["hours"] <= 0:
        text = "請假"
    elif cell["status"] == "off":
        text = f"⚠{cell['hours']:.1f}h"
    else:
        text = f"{cell['hours']:.1f}h"
    weight = "font-weight:bold;" if bold else ""
    return (
        f'<div style="background-color:{bg};color:{fg};padding:3px 6px;border-radius:4px;'
        f'text-align:center;font-size:0.85em;{weight}">{text}</div>'
    )


def _inject_scroll_js(anchor_id: str) -> None:
    """插入一次性 JS，平滑捲動到指定錨點。st.iframe 傳入的 HTML 字串一樣渲染在
    獨立 iframe 中（st.components.v1.html 已棄用，1.56 版起改用 st.iframe 支援
    直接嵌入 HTML 字串），故必須透過 window.parent.document 存取外層 Streamlit
    頁面的 DOM，而非 document（iframe 自身文件裡沒有這個錨點）。height=1 讓這個
    iframe 幾乎不佔用頁面版面；setTimeout 給錨點所在區塊一點時間完成渲染再捲動。
    """
    st.iframe(
        f"""
        <script>
        setTimeout(function() {{
            var el = window.parent.document.getElementById('{anchor_id}');
            if (el) {{ el.scrollIntoView({{behavior: 'smooth', block: 'start'}}); }}
        }}, 150);
        </script>
        """,
        height=1,
    )


def _render_week_matrix(
    df_result: pd.DataFrame,
    df_tasks: pd.DataFrame,
    df_cg: pd.DataFrame,
    selected_date: date,
    date_column: str,
    on_reassign,
    on_list_candidates=None,
) -> None:
    # 用完即清：見模組頂端說明，確保 Dialog 內互動觸發的後續 rerun 不會重複捲動。
    scroll_target = st.session_state.pop("calendar_scroll_to", None)

    week_start = selected_date - timedelta(days=selected_date.isoweekday() - 1)
    week_dates = [week_start + timedelta(days=i) for i in range(7)]

    st.markdown('<div id="shift-matrix-anchor"></div>', unsafe_allow_html=True)
    st.subheader(f"📋 {selected_date.isoformat()} 所在週：居服員排班矩陣")
    st.caption("點選格位中的 🔍 可檢視該居服員當日任務並快速改派給其他居服員。")
    if scroll_target == "matrix":
        _inject_scroll_js("shift-matrix-anchor")

    matrix = build_caregiver_day_matrix(df_result, df_tasks, df_cg, week_dates, date_column=date_column)

    if matrix.empty:
        st.info("本週查無可顯示的居服員排班資料。")
    else:
        col_labels = {
            d: f"{d.strftime('%m/%d')}({WEEKDAY_NUM_TO_NAME[d.isoweekday()][-1]})"
            for d in week_dates
        }

        header_cols = st.columns([1.1] + [1] * 7)
        header_cols[0].markdown("**居服員**")
        for i, d in enumerate(week_dates, start=1):
            marker = "▶ " if d == selected_date else ""
            header_cols[i].markdown(f"**{marker}{col_labels[d]}**")

        for cg_id in matrix.index:
            row_cols = st.columns([1.1] + [1] * 7)
            row_cols[0].markdown(f"**{cg_id}**")
            for i, d in enumerate(week_dates, start=1):
                cell = matrix.loc[cg_id, d]
                with row_cols[i]:
                    st.markdown(_cell_badge_html(cell, bold=(d == selected_date)), unsafe_allow_html=True)
                    # 見 _cell_badge_html 同一段說明：即使當天對該居服員是 off（休假／
                    # 不排班），只要 hours > 0（人工強制指派造成），仍要留著 🔍 按鈕，
                    # 否則這筆已存在的任務會完全無法從矩陣點入檢視／再次改派。
                    if cell["hours"] > 0:
                        if st.button(
                            "🔍", key=f"cal_cell_{cg_id}_{d.isoformat()}", width="stretch",
                            help=f"檢視／改派居服員 {cg_id} 於 {d.isoformat()} 的任務",
                        ):
                            st.session_state["calendar_active_dialog"] = {
                                "kind": "reassign", "cg_id": str(cg_id), "date": d.isoformat(),
                            }
                            st.rerun()

    st.caption("圖例：")
    legend_html = "&nbsp;&nbsp;&nbsp;".join(
        f'<span style="background-color:{STATUS_COLORS[s][0]};color:{STATUS_COLORS[s][1]};'
        f'padding:2px 10px;border-radius:4px;border:1px solid rgba(0,0,0,0.08);font-size:0.85em;">'
        f'{STATUS_LABELS[s]}</span>'
        for s in ("normal", "warning", "danger", "off")
    )
    st.markdown(legend_html, unsafe_allow_html=True)

    # 未派單案件處置：僅列出「已選取日期」（selected_date）當天的未派單任務，與
    # 居服員快速改派 Modal 的顆粒度一致（一次處理一天）。df_result 為呼叫端傳入的
    # 目前實際生效派單結果（已套用居督覆寫），故任務一旦透過本區塊指派成功，下次
    # rerun 即會從此清單消失。
    if date_column in df_tasks.columns:
        day_task_ids = df_tasks.loc[
            df_tasks[date_column].apply(_to_date) == selected_date, "任務ID"
        ].tolist()
    else:
        day_task_ids = df_tasks["任務ID"].tolist()
    assigned_ids = set(df_result["任務ID"]) if not df_result.empty else set()
    unassigned_task_ids = sorted(t for t in day_task_ids if t not in assigned_ids)

    if unassigned_task_ids:
        st.markdown('<div id="unassigned-cases-anchor"></div>', unsafe_allow_html=True)
        st.markdown("---")
        st.subheader(f"🚩 {selected_date.isoformat()} 未派單案件（{len(unassigned_task_ids)} 筆）")
        st.caption("點選「選擇居服員」可檢視全體候選人與未派原因，並快速完成指派。")
        if scroll_target == "unassigned":
            _inject_scroll_js("unassigned-cases-anchor")
        task_info = df_tasks.set_index("任務ID")
        has_service_type = "派單要求服務類型" in df_tasks.columns
        for t_id in unassigned_task_ids:
            if t_id not in task_info.index:
                continue
            t_row = task_info.loc[t_id]
            service_type = t_row.get("派單要求服務類型", "") if has_service_type else ""
            with st.container(border=True):
                c1, c2 = st.columns([4, 1])
                c1.markdown(
                    f"**任務 {t_id}**　⏰ {t_row['時間窗_開始']}–{t_row['時間窗_結束']}　"
                    f"👤 案家 {t_row['案家ID']}"
                    + (f"　🩺 {service_type}" if service_type else "")
                )
                if c2.button("🔍 選擇居服員", key=f"cal_unassigned_open_{t_id}", width="stretch"):
                    st.session_state["calendar_active_dialog"] = {"kind": "task", "task_id": t_id}
                    st.rerun()

    if st.button("✕ 關閉細節", key="calendar_close_detail"):
        st.session_state["calendar_selected_date"] = None
        st.session_state["calendar_active_dialog"] = None
        st.rerun()

    # 單一 Dialog 分派點（僅處理本函式自己觸發的「reassign」種類）：
    # kind="task"（未派單／④區塊任務覆寫）改由 render_task_override_picker
    # 集中派送，見該函式與模組頂端說明；calendar_active_dialog 一次只會是其中
    # 一種（或 None），從結構上就不可能在同一次 rerun 呼叫超過一個 @st.dialog
    # 函式（修正 StreamlitAPIException: Only one dialog is allowed to be
    # opened at the same time）。
    active_dialog = st.session_state.get("calendar_active_dialog")
    if active_dialog and active_dialog.get("kind") == "reassign":
        cell_cg_id = active_dialog.get("cg_id")
        cell_date_str = active_dialog.get("date")
        cell_date = date.fromisoformat(cell_date_str) if cell_date_str else None
        if cell_date in week_dates:
            _quick_reassign_dialog(
                cell_cg_id, cell_date, df_result, df_tasks, df_cg, date_column,
                on_reassign, on_list_candidates,
            )


def _resolve_reason(key_prefix: str) -> str:
    """從 _reason_picker 寫入的 session_state 讀回目前選定的覆寫原因（在 on_click
    callback 內呼叫，而非直接沿用渲染當下的區域變數，確保讀到的是使用者最後
    互動後的最終值，與本檔其餘 callback 讀 sel_key 的既有作法一致）。
    """
    reason = st.session_state.get(f"{key_prefix}_reason", "")
    if reason == "其他":
        reason = st.session_state.get(f"{key_prefix}_reason_detail", "")
    return (reason or "").strip()


def _reason_picker(key_prefix: str) -> str:
    """居督覆寫原因選擇器：下拉常見原因＋「其他」搭配自由文字輸入。原④區塊
    「居督人工覆寫」逐筆列表的原因欄位規格，現由所有覆寫入口共用。"""
    st.selectbox("換人原因（必填）", OVERRIDE_REASONS, key=f"{key_prefix}_reason")
    if st.session_state.get(f"{key_prefix}_reason") == "其他":
        st.text_input("請說明其他原因（必填）", key=f"{key_prefix}_reason_detail")
    return _resolve_reason(key_prefix)


def _reassign_confirm_click(task_id, current_cg_id, sel_key, error_key, success_key, on_reassign) -> None:
    chosen = st.session_state.get(sel_key, current_cg_id)
    reason = _resolve_reason(sel_key)
    err = on_reassign(task_id, chosen, reason)
    if err:
        st.session_state[error_key] = err
        st.session_state[sel_key] = current_cg_id
    else:
        st.session_state.pop(error_key, None)
        st.session_state[success_key] = True
        st.session_state[f"{success_key}_detail"] = (task_id, chosen)


def _reassign_cancel_click(current_cg_id, sel_key, error_key) -> None:
    st.session_state[sel_key] = current_cg_id
    st.session_state.pop(error_key, None)


def _task_override_confirm_click(task_id, current_cg_id, sel_key, error_key, success_key, on_reassign) -> None:
    chosen = st.session_state.get(sel_key, current_cg_id)
    new_cg = None if chosen == OVERRIDE_UNASSIGN else chosen
    reason = _resolve_reason(sel_key)
    err = on_reassign(task_id, new_cg, reason)
    if err:
        st.session_state[error_key] = err
    else:
        st.session_state.pop(error_key, None)
        st.session_state[success_key] = True
        st.session_state[f"{success_key}_detail"] = (task_id, new_cg if new_cg is not None else "未指派")


def _candidate_option_label(cg_id: str, ranked_by_id: dict, current_cg_id: Optional[str] = None) -> str:
    """把候選居服員的可派狀態轉成下拉選單的文字標籤（Streamlit selectbox 只能顯示
    純文字選項，無法對個別選項套用顏色／HTML，故以 ✅／⚠ 前綴＋具體衝突時段取代
    色塊標籤；下方的可派狀態摘要另以色塊呈現整體分布，兩者互補。

    current_cg_id 為選填：僅「居服員 x 日期」改派 Modal（_quick_reassign_dialog，
    有現任指派對象）與單一任務覆寫 Modal（_task_override_dialog，任務已指派時）
    會傳入；任務原本未指派時傳 None，「（目前指派）」這個分支從結構上就不可能
    被觸發，而不是仰賴傳入一個不會匹配到任何真實居服員 ID 的哨兵值。
    """
    if cg_id == OVERRIDE_UNASSIGN:
        return f"{OVERRIDE_UNASSIGN}（目前狀態）" if current_cg_id is None else OVERRIDE_UNASSIGN
    if current_cg_id is not None and cg_id == current_cg_id:
        return f"{cg_id}（目前指派）"
    info = ranked_by_id.get(cg_id)
    if info is None:
        return cg_id
    if info["available"]:
        return f"{cg_id} ✅ 可派單"
    if info["reason"] == "time_conflict":
        return f"{cg_id} ⚠ {info['conflict_start']}-{info['conflict_end']} 服務中"
    if info["reason"] == "hour_cap":
        return f"{cg_id} ⚠ 工時將超過上限"
    return f"{cg_id} ⚠ {info['detail'] or '無法派單'}"


@st.dialog("📋 任務詳情與快速改派", width="large")
def _quick_reassign_dialog(
    cg_id: str,
    day: date,
    df_result: pd.DataFrame,
    df_tasks: pd.DataFrame,
    df_cg: pd.DataFrame,
    date_column: str,
    on_reassign,
    on_list_candidates=None,
) -> None:
    """月曆快速改派 Modal：列出 cg_id 於 day 的任務，逐筆提供改派下拉選單。

    每次改派皆委派給呼叫端注入的 on_reassign(task_id, new_cg_id, reason) -> Optional[str]
    （app.py 中實作，內部呼叫 caregiver_engine.check_reassignment_conflict 做即時
    衝突檢查，通過後才寫入 overrides 與稽核日誌）；本函式只負責顯示成功/錯誤結果，
    不自行判斷是否可改派。reason 由 _reason_picker 收集（必填，與原④區塊逐筆列表
    的原因欄位規格相同），確認按鈕在原因未填妥前停用。

    候選名單一律取「機構內全體居服員」（df_cg，排除 cg_id 本人），不會用 Phase 1
    的候選配對名單事先篩過一輪——那是給 AI 自動媒合找最佳解用的，會把性別不符、
    缺乏證照、環境排斥、休假等人整批排除，若拿來源限縮改派選單，會讓「看起來可以
    配合」的人完全不出現在下拉選單中。改派是居督人工判斷的場景，所有人都該留著，
    不合資格者改用標記呈現（見 on_list_candidates）。

    on_list_candidates(task_id, candidate_cg_ids) -> list[dict]（app.py 中實作，內部
    呼叫 caregiver_engine.rank_candidates_by_availability，與 on_reassign 走同一套
    時間衝突／工時判定邏輯，並額外標記 Phase 1 硬性資格條件）用於把下拉選單依
    「該時段是否有空檔、是否符合資格」排序＋標籤，避免居督逐一試錯；未提供時退回
    原始未排序清單，不影響改派功能本身。
    """
    success_key = "calendar_quick_reassign_success"
    if st.session_state.pop(success_key, False):
        # apply_overrides_to_result（非快取函式）與 build_daily_summary／
        # build_caregiver_day_matrix（@st.cache_data，依內容雜湊）會在下一次 rerun
        # 讀到 overrides 已更新的 df_result_effective，自然產生新的快取鍵、
        # 觸發重新計算，故此處不需另外呼叫 st.cache_data.clear()。
        detail = st.session_state.pop(f"{success_key}_detail", None)
        if detail:
            st.toast(f"✅ 已將任務 {detail[0]} 改派給 {detail[1]}", icon="✅")
        else:
            st.toast("✅ 改派成功！", icon="✅")
        st.session_state["calendar_active_dialog"] = None
        st.rerun()

    st.caption(f"居服員 **{cg_id}** — {day.isoformat()}（{WEEKDAY_NUM_TO_NAME[day.isoweekday()]}）")

    if on_reassign is None:
        st.info("快速改派功能目前未啟用（呼叫端尚未提供 on_reassign callback）。")
        if st.button("✕ 關閉", key=f"close_quick_reassign_noop_{cg_id}_{day.isoformat()}"):
            st.session_state["calendar_active_dialog"] = None
            st.rerun()
        return

    if date_column in df_tasks.columns:
        day_task_ids = set(
            df_tasks.loc[df_tasks[date_column].apply(_to_date) == day, "任務ID"]
        )
    else:
        day_task_ids = set(df_tasks["任務ID"])

    cg_tasks = (
        df_result[
            (df_result["派單居服員"].astype(str) == str(cg_id))
            & (df_result["任務ID"].isin(day_task_ids))
        ]
        if not df_result.empty
        else pd.DataFrame(columns=["任務ID", "派單居服員"])
    )

    if cg_tasks.empty:
        st.info("此格已無任務（可能剛被改派至其他居服員）。")
    else:
        avail_only_key = f"quick_reassign_avail_only_{cg_id}_{day.isoformat()}"
        show_available_only = st.toggle(
            "僅顯示有空檔人員", value=False, key=avail_only_key,
            help="開啟後，下拉選單只列出該時段無其他任務衝突的居服員；"
            "關閉時仍會列出全部候選人，但有空檔者排在前面並標示 ✅，衝突者標示 ⚠ 與衝突時段。",
        )

        task_info = df_tasks.set_index("任務ID")
        has_service_type = "派單要求服務類型" in df_tasks.columns

        for t_id in sorted(cg_tasks["任務ID"].tolist()):
            if t_id not in task_info.index:
                continue
            t_row = task_info.loc[t_id]
            service_type = t_row.get("派單要求服務類型", "") if has_service_type else ""

            with st.container(border=True):
                st.markdown(
                    f"**任務 {t_id}**　⏰ {t_row['時間窗_開始']}–{t_row['時間窗_結束']}　"
                    f"👤 案家 {t_row['案家ID']}"
                    + (f"　🩺 {service_type}" if service_type else "")
                )

                # 候選名單一律取「機構內全體居服員」，不可只取 df_matches（Phase 1
                # 硬性條件倖存者）——df_matches 已把性別不符、缺乏證照、環境排斥、
                # 休假、工時超額等人整批剔除，導致下拉選單「看似可以配合的人員完全
                # 沒有出現」。改派選單與 Phase 1 自動媒合的目的不同：後者要找最佳解，
                # 前者要讓居督自行判斷，不合資格的人也該留著並標註原因（見
                # on_list_candidates → rank_candidates_by_availability 的
                # reason="hard_constraint" 標籤），而不是直接消失。
                other_cg_ids = (
                    sorted(c for c in df_cg["居服員ID"].astype(str).unique().tolist() if c != str(cg_id))
                    if not df_cg.empty
                    else []
                )

                sel_key = f"quick_reassign_{cg_id}_{day.isoformat()}_{t_id}"
                error_key = f"{sel_key}_error"

                ranked_by_id = {}
                available_ids, conflicted_ids = other_cg_ids, []
                if on_list_candidates is not None and other_cg_ids:
                    ranked = on_list_candidates(t_id, other_cg_ids)
                    ranked_by_id = {r["cg_id"]: r for r in ranked}
                    available_ids = [r["cg_id"] for r in ranked if r["available"]]
                    conflicted_ids = [r["cg_id"] for r in ranked if not r["available"]]

                candidate_ids = available_ids if show_available_only else available_ids + conflicted_ids
                options = [str(cg_id)] + candidate_ids

                if not available_ids:
                    st.caption("⚠ 目前查無此時段有空檔的候選居服員。")
                elif ranked_by_id:
                    st.caption(f"🟢 {len(available_ids)} 位有空檔　🔴 {len(conflicted_ids)} 位時間衝突")

                if show_available_only and not available_ids:
                    st.info("關閉「僅顯示有空檔人員」可查看所有候選人（含時間衝突者）。")

                # 切換「僅顯示有空檔人員」可能讓選單先前選到的候選人從 options 中消失
                # （Streamlit 要求 key 對應的 session_state 值須為目前 options 之一，
                # 否則會擲例外），故在建立 widget 前（widget 尚未實例化，仍可安全寫入
                # session_state）先復原為目前指派居服員。
                if st.session_state.get(sel_key) not in options:
                    st.session_state[sel_key] = str(cg_id)

                chosen = st.selectbox(
                    "快速改派", options, key=sel_key, label_visibility="collapsed",
                    format_func=lambda v: _candidate_option_label(v, ranked_by_id, current_cg_id=str(cg_id)),
                )

                if st.session_state.get(error_key):
                    st.error(f"⚠️ 改派失敗：{st.session_state[error_key]}")

                if chosen != str(cg_id):
                    st.warning(f"確定要將任務 {t_id} 由 **{cg_id}** 改派給 **{chosen}** 嗎？")
                    reason = _reason_picker(sel_key)
                    c1, c2 = st.columns(2)
                    c1.button(
                        "✅ 確認改派", key=f"confirm_{sel_key}", width="stretch",
                        disabled=not reason.strip(),
                        on_click=_reassign_confirm_click,
                        args=(t_id, str(cg_id), sel_key, error_key, success_key, on_reassign),
                    )
                    c2.button(
                        "✕ 取消", key=f"cancel_{sel_key}", width="stretch",
                        on_click=_reassign_cancel_click,
                        args=(str(cg_id), sel_key, error_key),
                    )

    if st.button("✕ 關閉", key=f"close_quick_reassign_{cg_id}_{day.isoformat()}"):
        st.session_state["calendar_active_dialog"] = None
        st.rerun()


@st.dialog("🔍 任務覆寫", width="large")
def _task_override_dialog(
    task_id: str,
    df_tasks: pd.DataFrame,
    df_cg: pd.DataFrame,
    df_result: pd.DataFrame,
    overrides: dict,
    on_reassign,
    on_clear_override=None,
    on_list_candidates=None,
) -> None:
    """單一任務覆寫 Modal：涵蓋「已指派任務改派／撤銷」與「未派單任務快速指派」
    兩種情境，取代原④區塊「居督人工覆寫」逐筆列出全部任務的 expander 清單，
    也是月曆視角「未派單案件」的處置入口——兩者本來就走同一套
    on_reassign(task_id, new_cg_id, reason) -> Optional[str] 覆寫邏輯（app.py 的
    _quick_reassign，內部呼叫 caregiver_engine.check_reassignment_conflict 做即時
    衝突檢查，通過後才寫入 overrides 與稽核日誌；apply_overrides_to_result 本就
    支援替未指派任務新增覆寫列），合併為一套介面可少維護一份重複邏輯。

    df_result 為呼叫端傳入的『AI 原始結果』（未套用 overrides），本函式自行比對
    overrides dict 算出「AI 建議」與「目前實際生效」兩個狀態分開顯示，讓居督看得
    出這筆任務是否已被覆寫過、覆寫成什麼、原因為何；existing_override 存在時另外
    提供「清除覆寫」按鈕（委派給 on_clear_override，不寫入稽核日誌，與原④區塊
    「清除覆寫，還原 AI 建議」行為一致）。

    候選名單一律取「機構內全體居服員」＋一個「撤銷指派」選項，並在提供
    on_list_candidates 時（app.py 的 _list_candidates，內部呼叫
    caregiver_engine.rank_candidates_by_availability）依「是否符合派單資格」標籤
    ＋排序，讓居督不必逐一試錯即可判斷；不合資格者仍可選取（本系統無強制派單
    權限機制，最終判斷權在居督）。選擇與目前生效指派不同的對象後才需要填寫覆寫
    原因（必填，見 _reason_picker），與原④區塊規格相同。
    """
    success_key = "task_override_success"
    if st.session_state.pop(success_key, False):
        detail = st.session_state.pop(f"{success_key}_detail", None)
        if detail:
            st.toast(f"✅ 任務 {detail[0]} 已更新為：{detail[1]}", icon="✅")
        else:
            st.toast("✅ 已更新！", icon="✅")
        st.session_state["calendar_active_dialog"] = None
        st.rerun()

    task_info = df_tasks.set_index("任務ID")
    if task_id not in task_info.index:
        st.info("找不到此任務資料（可能資料已更新）。")
        if st.button("✕ 關閉", key=f"close_task_override_missing_{task_id}"):
            st.session_state["calendar_active_dialog"] = None
            st.rerun()
        return

    t_row = task_info.loc[task_id]
    has_service_type = "派單要求服務類型" in df_tasks.columns
    service_type = t_row.get("派單要求服務類型", "") if has_service_type else ""
    priority = t_row.get("任務優先級", "")

    st.markdown(
        f"**任務 {task_id}**　⏰ {t_row['時間窗_開始']}–{t_row['時間窗_結束']}　"
        f"👤 案家 {t_row['案家ID']}"
        + (f"　🩺 {service_type}" if service_type else "")
        + (f"　🔥 {priority}" if priority else "")
    )

    ai_row = df_result[df_result["任務ID"] == task_id] if not df_result.empty else pd.DataFrame()
    ai_cg = str(ai_row.iloc[0]["派單居服員"]) if not ai_row.empty else None

    existing_override = (overrides or {}).get(task_id)
    if existing_override:
        current_cg = existing_override["cg_id"]
        st.caption(
            f"AI 建議：{ai_cg or '未指派'}　→　居督已覆寫為："
            f"**{current_cg or '未指派'}**（{existing_override['reason']}）"
        )
    else:
        current_cg = ai_cg
        st.caption(f"AI 建議：**{ai_cg or '未指派'}**（尚未覆寫）")

    if on_reassign is None:
        st.info("覆寫功能目前未啟用（呼叫端尚未提供 on_reassign callback）。")
        if st.button("✕ 關閉", key=f"close_task_override_noop_{task_id}"):
            st.session_state["calendar_active_dialog"] = None
            st.rerun()
        return

    avail_only_key = f"task_override_avail_only_{task_id}"
    show_available_only = st.toggle(
        "僅顯示有空檔且符合資格人選", value=False, key=avail_only_key,
        help="開啟後，只列出該時段無時間衝突、且符合性別/證照/環境等硬性資格條件的居服員；"
        "關閉時仍列出全部候選人，未達推薦條件者會標示具體原因。",
    )

    all_cg_ids = sorted(df_cg["居服員ID"].astype(str).unique().tolist()) if not df_cg.empty else []

    sel_key = f"task_override_sel_{task_id}"
    error_key = f"{sel_key}_error"

    ranked_by_id = {}
    available_ids, conflicted_ids = all_cg_ids, []
    if on_list_candidates is not None and all_cg_ids:
        ranked = on_list_candidates(task_id, all_cg_ids)
        ranked_by_id = {r["cg_id"]: r for r in ranked}
        available_ids = [r["cg_id"] for r in ranked if r["available"]]
        conflicted_ids = [r["cg_id"] for r in ranked if not r["available"]]

    if ranked_by_id:
        st.caption(f"🟢 {len(available_ids)} 位可派單　🔴 {len(conflicted_ids)} 位有衝突或不符資格")

    candidate_ids = available_ids if show_available_only else available_ids + conflicted_ids
    options = [OVERRIDE_UNASSIGN] + candidate_ids

    default_val = OVERRIDE_UNASSIGN if current_cg is None else str(current_cg)
    if st.session_state.get(sel_key) not in options:
        st.session_state[sel_key] = default_val if default_val in options else OVERRIDE_UNASSIGN

    chosen = st.selectbox(
        "指定居服員", options, key=sel_key,
        format_func=lambda v: _candidate_option_label(v, ranked_by_id, current_cg_id=current_cg),
    )

    if st.session_state.get(error_key):
        st.error(f"⚠️ 覆寫失敗：{st.session_state[error_key]}")

    chosen_cg = None if chosen == OVERRIDE_UNASSIGN else chosen
    if chosen_cg != current_cg:
        chosen_info = ranked_by_id.get(chosen_cg) if chosen_cg else None
        if chosen_info is not None and not chosen_info["available"]:
            st.warning(f"⚠️ **{chosen}**：{chosen_info['detail'] or '不符合建議派單條件'}。")
        reason = _reason_picker(sel_key)
        st.button(
            "✅ 確認覆寫", key=f"confirm_{sel_key}", width="stretch", disabled=not reason.strip(),
            on_click=_task_override_confirm_click,
            args=(task_id, current_cg, sel_key, error_key, success_key, on_reassign),
        )
    else:
        st.caption("選擇與目前生效指派不同的居服員／撤銷指派後，即可填寫原因並確認覆寫。")

    if existing_override and on_clear_override is not None:
        if st.button("↩️ 清除覆寫，還原 AI 建議", key=f"clear_{sel_key}", width="stretch"):
            on_clear_override(task_id)
            st.session_state["calendar_active_dialog"] = None
            st.rerun()

    if st.button("✕ 關閉", key=f"close_task_override_{task_id}"):
        st.session_state["calendar_active_dialog"] = None
        st.rerun()


def render_calendar_overview(
    df_result: pd.DataFrame,
    df_tasks: pd.DataFrame,
    df_cg: pd.DataFrame,
    overrides: dict = None,
    on_reassign=None,
    on_list_candidates=None,
) -> None:
    """渲染「🗓️ 月曆班表總覽」子區塊，作為結果區的首頁總覽。

    讀取既有 df_tasks／df_result／df_cg 做彙總，並套用 overrides（居督覆寫，與④區塊
    共用同一份 st.session_state["overrides"]）得到目前實際生效的班表，不重呼叫
    caregiver_engine.py 的排班演算法。僅在 df_tasks 具備「日期」欄位（月批次排班
    模式）時顯示完整功能；否則顯示提示並直接返回。

    on_reassign（快速改派 callback，見 _quick_reassign_dialog）與 on_list_candidates
    （依空檔＋資格排序候選人 callback）皆為選填：未提供時月曆總覽仍會正常顯示，僅
    快速改派／空檔排序功能不可用或退回未排序清單。候選名單一律取 df_cg 全體居服員
    （見 _quick_reassign_dialog），不依賴 Phase 1 的候選配對結果。點擊「未派單」
    徽章只設定 calendar_active_dialog（kind="task"）並導向該任務，實際 Modal 由
    呼叫端另外呼叫的 render_task_override_picker 集中派送（見該函式說明）。
    """
    st.subheader("🗓️ 月曆班表總覽")

    if "日期" not in df_tasks.columns:
        st.info("月曆總覽僅適用於含日期欄位的月批次排班，目前為單日排程模式，略過本區塊。")
        return

    overrides = overrides or {}
    df_result_effective = apply_overrides_to_result(df_result, overrides)

    st.caption(
        "點選日期可展開當週「居服員 x 日期」工時矩陣；點選「🚩 N 筆未派單」可直接"
        "查看該日未派單案件並快速指派居服員。"
    )

    valid_dates = [d for d in (_to_date(v) for v in df_tasks["日期"]) if d is not None]
    if not valid_dates:
        st.info("任務資料的「日期」欄位無法解析，暫無法顯示月曆總覽。")
        return

    months = sorted({f"{d.year}-{d.month:02d}" for d in valid_dates})
    if len(months) > 1:
        selected_month = st.selectbox("選擇年-月", months, index=len(months) - 1, key="calendar_selected_month")
    else:
        selected_month = months[0]
        st.caption(f"目前資料範圍：{selected_month}")

    year, month = (int(x) for x in selected_month.split("-"))

    daily_summary = build_daily_summary(df_result_effective, df_tasks, df_cg)
    weeks = _month_grid(year, month)
    selected_date = st.session_state.get("calendar_selected_date")

    header_cols = st.columns(7)
    for col, num in zip(header_cols, range(1, 8)):
        col.markdown(f"**{WEEKDAY_NUM_TO_NAME[num]}**")

    for week in weeks:
        cols = st.columns(7)
        for col, day_date in zip(cols, week):
            with col, st.container(border=True):
                if day_date is None:
                    st.write("")
                    continue

                info = daily_summary.get(day_date)
                if info is None or info["total_tasks"] == 0:
                    st.caption(str(day_date.day))
                    st.caption("－")
                    continue

                is_selected = day_date == selected_date
                if st.button(
                    str(day_date.day),
                    key=f"cal_day_{day_date.isoformat()}",
                    type="primary" if is_selected else "secondary",
                    width="stretch",
                ):
                    st.session_state["calendar_selected_date"] = day_date
                    # 切換日期一律視為「放棄任何待開啟的 Dialog」：即使先前有殘留
                    # 的 calendar_active_dialog（例如上一個 Dialog 是透過 st.dialog
                    # 原生關閉鈕而非本模組按鈕關閉，見模組頂端說明），點擊日期標題
                    # 也絕不該讓「任務詳情與快速改派」Modal 意外跳出。
                    st.session_state["calendar_active_dialog"] = None
                    st.session_state["calendar_scroll_to"] = "matrix"
                    st.rerun()

                st.caption(f"{info['assigned_count']}/{info['total_tasks']} 筆")
                if info["has_unassigned"]:
                    if st.button(
                        f"🚩 {info['unassigned_count']} 筆未派單",
                        key=f"cal_unassigned_badge_{day_date.isoformat()}",
                        width="stretch",
                        help="點擊查看本日未派單案件，選擇居服員快速指派。",
                    ):
                        st.session_state["calendar_selected_date"] = day_date
                        st.session_state["calendar_active_dialog"] = None
                        st.session_state["calendar_scroll_to"] = "unassigned"
                        st.rerun()
                if info["over_capacity_cg_count"] > 0:
                    st.caption(f":orange[{info['over_capacity_cg_count']} 位居服員超時]")
                if is_selected:
                    st.caption("**（已選取）**")

    if selected_date is not None:
        _render_week_matrix(
            df_result_effective, df_tasks, df_cg, selected_date, "日期",
            on_reassign, on_list_candidates,
        )


def render_task_override_picker(
    df_tasks: pd.DataFrame,
    df_result: pd.DataFrame,
    df_cg: pd.DataFrame,
    overrides: dict,
    on_reassign=None,
    on_clear_override=None,
    on_list_candidates=None,
) -> None:
    """④「居督人工覆寫」區塊的主要內容：取代原本逐筆列出全部任務的 expander
    清單，改為「搜尋／選擇單一任務 → 開啟覆寫 Modal（_task_override_dialog，與
    月曆視角未派單處置共用同一份實作）」＋「目前已覆寫任務」精簡清單（筆數受限
    於實際覆寫數，不受任務總數影響，故任務量再大也不會拖長頁面）。

    df_result 須為呼叫端的『AI 原始結果』（未套用 overrides），與 overrides 一起
    交給 _task_override_dialog 自行比對算出「AI 建議」vs「目前生效」；on_reassign／
    on_list_candidates 與月曆視角「一鍵調班」共用同一個 callback 與同一套
    calendar_active_dialog 狀態鍵（kind="task"），故不論是從這裡或從月曆「未派單」
    按鈕開啟，都是同一個 Modal、同一套稽核紀錄。
    """
    overrides = overrides or {}
    task_ids = df_tasks["任務ID"].tolist() if not df_tasks.empty else []
    cl_map = df_tasks.set_index("任務ID")["案家ID"] if "案家ID" in df_tasks.columns else {}
    assigned_map = dict(zip(df_result["任務ID"], df_result["派單居服員"])) if not df_result.empty else {}

    if not task_ids:
        st.info("目前無任務可供覆寫。")
    else:
        def _task_option_label(t_id) -> str:
            cl_id = cl_map.get(t_id, "")
            ai_cg = assigned_map.get(t_id) or "未指派"
            ov = overrides.get(t_id)
            if ov:
                cur = ov["cg_id"] if ov["cg_id"] is not None else "未指派"
                return f"{t_id}（案家{cl_id}）AI:{ai_cg} → 已覆寫:{cur}"
            return f"{t_id}（案家{cl_id}）AI 建議:{ai_cg}"

        picker_key = "task_override_picker_select"
        chosen_task = st.selectbox(
            "選擇任務進行覆寫", task_ids, key=picker_key, format_func=_task_option_label,
        )
        if st.button("🔍 開啟覆寫視窗", key="task_override_picker_open"):
            st.session_state["calendar_active_dialog"] = {"kind": "task", "task_id": chosen_task}
            st.rerun()

    if overrides:
        st.caption(f"目前已覆寫 {len(overrides)} 筆任務：")
        for t_id, ov in overrides.items():
            cur = ov["cg_id"] if ov["cg_id"] is not None else "未指派"
            c1, c2, c3 = st.columns([3, 2, 1])
            c1.write(f"任務 {t_id}（案家 {cl_map.get(t_id, '')}）→ **{cur}**")
            c2.caption(ov["reason"])
            if on_clear_override is not None and c3.button("↩️ 清除", key=f"override_list_clear_{t_id}"):
                on_clear_override(t_id)
                st.rerun()
    else:
        st.caption("目前尚無任何居督覆寫紀錄。")

    active_dialog = st.session_state.get("calendar_active_dialog")
    if active_dialog and active_dialog.get("kind") == "task":
        active_task_id = active_dialog.get("task_id")
        if active_task_id in set(task_ids):
            _task_override_dialog(
                active_task_id, df_tasks, df_cg, df_result, overrides,
                on_reassign, on_clear_override, on_list_candidates,
            )
        else:
            st.session_state["calendar_active_dialog"] = None
