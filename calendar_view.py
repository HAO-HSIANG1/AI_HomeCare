"""月曆式班表總覽 - Streamlit 前端呈現模組。

純粹讀取 app.py 既有的 st.session_state["last_result"]（df_tasks、phase2["df_result"]）
與 st.session_state["edit_cg"] 做彙總與呈現，不呼叫、不重複、也不修改
caregiver_engine.py 的任何排班演算法邏輯。僅在任務資料具備「日期」欄位
（月批次排班模式）時才顯示；單日排程模式下由呼叫端（app.py）略過本區塊。
"""

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


def _to_date(value):
    """將日期欄位值（可能是字串／Timestamp／date）正規化為 date；無法解析回傳 None。"""
    if pd.isna(value):
        return None
    try:
        return pd.to_datetime(value).date()
    except (ValueError, TypeError):
        return None


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


def _render_week_matrix(df_result: pd.DataFrame, df_tasks: pd.DataFrame, df_cg: pd.DataFrame, selected_date: date) -> None:
    week_start = selected_date - timedelta(days=selected_date.isoweekday() - 1)
    week_dates = [week_start + timedelta(days=i) for i in range(7)]

    st.subheader(f"📋 {selected_date.isoformat()} 所在週：居服員排班矩陣")

    matrix = build_caregiver_day_matrix(df_result, df_tasks, df_cg, week_dates)

    if matrix.empty:
        st.info("本週查無可顯示的居服員排班資料。")
    else:
        # 選取日期所在的那一欄，欄標題加上標記與其他 6 欄視覺區隔。
        col_labels = {
            d: f"{'▶ ' if d == selected_date else ''}{d.strftime('%m/%d')}({WEEKDAY_NUM_TO_NAME[d.isoweekday()][-1]}){' ◀' if d == selected_date else ''}"
            for d in week_dates
        }

        display_df = pd.DataFrame(index=matrix.index)
        status_df = pd.DataFrame(index=matrix.index)
        for d in week_dates:
            label = col_labels[d]
            display_df[label] = matrix[d].apply(
                lambda cell: "請假" if cell["status"] == "off" else f"{cell['hours']:.1f}h"
            )
            status_df[label] = matrix[d].apply(lambda cell: cell["status"])

        selected_label = col_labels[selected_date]

        def _row_styles(row: pd.Series):
            statuses = status_df.loc[row.name]
            styles = []
            for col_name in row.index:
                bg, fg = STATUS_COLORS.get(statuses[col_name], ("", ""))
                style = f"background-color:{bg};color:{fg};"
                if col_name == selected_label:
                    style += "font-weight:bold;"
                styles.append(style)
            return styles

        styled = display_df.style.apply(_row_styles, axis=1)
        st.table(styled)

    st.caption("圖例：")
    legend_html = "&nbsp;&nbsp;&nbsp;".join(
        f'<span style="background-color:{STATUS_COLORS[s][0]};color:{STATUS_COLORS[s][1]};'
        f'padding:2px 10px;border-radius:4px;border:1px solid rgba(0,0,0,0.08);font-size:0.85em;">'
        f'{STATUS_LABELS[s]}</span>'
        for s in ("normal", "warning", "danger", "off")
    )
    st.markdown(legend_html, unsafe_allow_html=True)

    if st.button("✕ 關閉細節", key="calendar_close_detail"):
        st.session_state["calendar_selected_date"] = None
        st.rerun()


def render_calendar_overview(df_result: pd.DataFrame, df_tasks: pd.DataFrame, df_cg: pd.DataFrame) -> None:
    """渲染「🗓️ 月曆班表總覽」子區塊，作為結果區的首頁總覽。

    純前端呈現：讀取既有 df_tasks／df_result／df_cg 做彙總，不修改、不重呼叫
    caregiver_engine.py 的任何排班演算法。僅在 df_tasks 具備「日期」欄位
    （月批次排班模式）時顯示完整功能；否則顯示提示並直接返回。
    """
    st.subheader("🗓️ 月曆班表總覽")

    if "日期" not in df_tasks.columns:
        st.info("月曆總覽僅適用於含日期欄位的月批次排班，目前為單日排程模式，略過本區塊。")
        return

    st.caption("點選日期可展開當週「居服員 x 日期」工時矩陣，快速掌握未派單與居服員超時狀況。")

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

    daily_summary = build_daily_summary(df_result, df_tasks, df_cg)
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
                    st.rerun()

                st.caption(f"{info['assigned_count']}/{info['total_tasks']} 筆")
                if info["has_unassigned"]:
                    st.caption(f":red[⚠ {info['unassigned_count']} 筆未派單]")
                if info["over_capacity_cg_count"] > 0:
                    st.caption(f":orange[{info['over_capacity_cg_count']} 位居服員超時]")
                if is_selected:
                    st.caption("**（已選取）**")

    if selected_date is not None:
        _render_week_matrix(df_result, df_tasks, df_cg, selected_date)
