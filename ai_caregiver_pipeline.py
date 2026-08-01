"""長照居家照顧派單系統 - 命令列進入點

實際運算邏輯位於 caregiver_engine.py，本檔僅負責依序呼叫三個階段並將結果印到
終端機，供不需要網頁介面時的快速執行（例如排程批次）。若要調整參數並以網頁
視覺化查看結果，請改用 `streamlit run app.py`。
"""

from ortools.linear_solver import pywraplp

from caregiver_engine import (
    DEFAULT_EXCEL_PATH,
    PipelineConfig,
    load_data,
    run_phase1_matching,
    run_phase2_optimization,
    run_phase3_did,
    validate_ba_codes,
)

EXCEL_PATH = DEFAULT_EXCEL_PATH


def main():
    config = PipelineConfig()

    print("=" * 60)
    print(" 啟動長照 AI 派單與效益分析系統")
    print("=" * 60)

    df_cg, df_cl, df_tasks, df_hist, tasks = load_data(EXCEL_PATH)

    # ------------------------------------------
    # Phase 0：BA 服務代碼併報法規防呆健檢（僅提示，不影響下方派單運算）
    # ------------------------------------------
    validated_tasks = validate_ba_codes(tasks)
    violation_count = int(validated_tasks["含違規代碼"].sum())
    print(f"\n[Phase 0] BA 服務代碼併報健檢：{len(validated_tasks)} 筆任務中，{violation_count} 筆疑似違規。")
    if violation_count > 0:
        print(validated_tasks.loc[validated_tasks["含違規代碼"], ["任務ID", "BA代碼檢核異常"]].to_string(index=False))

    # ------------------------------------------
    # Phase 1
    # ------------------------------------------
    print("\n[Phase 1] 正在執行硬性規則過濾與軟性適配度評分...")
    df_matches = run_phase1_matching(tasks, df_cg, config)
    print(f" -> 經硬性條件過濾後，共產生 {len(df_matches)} 組合格的候選配對。")

    # ------------------------------------------
    # Phase 2
    # ------------------------------------------
    print("\n[Phase 2] 正在檢查行程時間窗衝突並構建 OR-Tools 最佳化模型...")
    phase2 = run_phase2_optimization(df_matches, tasks, df_cg, config)
    print(f" -> 排除既定行程與轉場時間衝突後，剩餘 {len(phase2['df_valid'])} 組候選配對進入 MIP 求解。")

    if phase2["status"] == pywraplp.Solver.OPTIMAL:
        print("\n -> OR-Tools 最佳化求解成功！")
        df_result = phase2["df_result"]
        print(f"\n【派單結果總覽】 成功指派: {phase2['assigned_count']} / {len(df_tasks)} 筆任務")
        print(df_result.drop(columns=["地點緯度", "地點經度"]).to_string(index=False))
        total_revenue = df_result["預估長照申報點數(營收)"].sum()
        total_salary = df_result["預估居服員拆帳薪資"].sum()
        print(f"\n【財務試算】 預估長照申報總點數(營收): {total_revenue:,.0f} 點 | 預估居服員拆帳總薪資: {total_salary:,.0f} 元")
    else:
        print("無法找到最佳解，請檢查限制條件。")

    # ------------------------------------------
    # Phase 3
    # ------------------------------------------
    print("\n[Phase 3] 歷史數據因果效益回溯 (Treatment vs Control)...")
    did = run_phase3_did(df_hist)

    print("-" * 50)
    print(f"AI 派單組 (Treatment): 平均滿意度 {did['ai_sat']:.2f} 分 | 提早結案率 {did['ai_dropout']*100:.1f}%")
    print(f"人工派單組 (Control)  : 平均滿意度 {did['human_sat']:.2f} 分 | 提早結案率 {did['human_dropout']*100:.1f}%")
    print("-" * 50)
    print(
        f"【純粹增量效益 (Uplift)】 滿意度淨提升: +{did['uplift_sat']:.2f} 分 | 提早結案風險淨降低: {did['uplift_dropout']*100:.1f}%"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
