"""caregiver_engine.py 核心邏輯的單元測試（合成資料，不依賴 Excel 檔案）。

執行方式：
    python -m unittest tests.test_caregiver_engine -v
"""

import os
import sys
import unittest

import pandas as pd
from ortools.linear_solver import pywraplp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from caregiver_engine import (
    PipelineConfig,
    check_reassignment_conflict,
    rank_candidates_by_availability,
    run_monthly_batch_dispatch,
    run_phase1_matching,
    run_phase2_optimization,
)


def make_caregiver(
    cg_id,
    lat=25.05,
    lon=121.55,
    cert="單一級照服證照",
    used_hours=0.0,
    daily_cap=8.0,
    satisfaction=4.0,
    fatigue=0.0,
    lift_ok=1,
    gender="女",
    exclusion="無",
    busy1="無既定行程",
    busy2="無既定行程",
):
    return {
        "居服員ID": cg_id,
        "性別": gender,
        "服務起點_經度(家)": lon,
        "服務起點_緯度(家)": lat,
        "核心專長證照": cert,
        "每日工時上限(小時)": daily_cap,
        "當月累計服務時數(疲勞度)": fatigue,
        "歷史滿意度均值": satisfaction,
        "具備重度移位體力(0/1)": lift_ok,
        "今日既定行程1_時段": busy1,
        "今日既定行程1_地點經度": lon,
        "今日既定行程1_地點緯度": lat,
        "今日既定行程2_時段": busy2,
        "今日既定行程2_地點經度": lon,
        "今日既定行程2_地點緯度": lat,
        "特殊排斥條件": exclusion,
        "今日已佔用工時(小時)": used_hours,
    }


def make_task(
    t_id,
    c_id,
    start,
    end,
    duration_min,
    lat=25.05,
    lon=121.55,
    req_gender="無特殊要求",
    req_lift=0,
    req_type="一般家務與照顧",
    pref_cg="",
    priority="一般",
    client_env="無",
):
    return {
        "任務ID": t_id,
        "案家ID": c_id,
        "指定居服員性別": req_gender,
        "需重度移位協助(0/1)": req_lift,
        "案家環境特徵": client_env,
        "服務地點_緯度": lat,
        "服務地點_經度": lon,
        "服務歷時(分鐘)": duration_min,
        "歷史首選居服員ID": pref_cg,
        "特殊照護需求": req_type,
        "時間窗_開始": start,
        "時間窗_結束": end,
        "任務優先級": priority,
    }


class CapacityConstraintTests(unittest.TestCase):
    """Phase 2 MIP 必須阻擋「多筆任務加總後超派」，即便每筆任務個別未超時。"""

    def test_solver_never_exceeds_daily_hour_cap(self):
        config = PipelineConfig()
        df_cg = pd.DataFrame([make_caregiver("CG01", daily_cap=8.0, used_hours=0.0)])

        # 兩筆任務時間窗不重疊（含轉場緩衝），但總歷時 10 小時 > 每日工時上限 8 小時。
        tasks = pd.DataFrame(
            [
                make_task("TK01", "CL01", "08:00", "13:00", 300),  # 5 小時
                make_task("TK02", "CL02", "14:00", "19:00", 300),  # 5 小時
            ]
        )

        df_matches = run_phase1_matching(tasks, df_cg, config)
        # Phase 1 逐筆檢查未超時，兩筆候選配對都應存活。
        self.assertEqual(len(df_matches), 2)

        phase2 = run_phase2_optimization(df_matches, tasks, df_cg, config)
        self.assertEqual(phase2["status"], pywraplp.Solver.OPTIMAL)

        df_result = phase2["df_result"]
        assigned_minutes = 0
        if not df_result.empty:
            assigned_task_ids = df_result["任務ID"].tolist()
            assigned_minutes = tasks[tasks["任務ID"].isin(assigned_task_ids)]["服務歷時(分鐘)"].sum()

        # 已佔用工時(0) + 新派任務總歷時，不得超過每日工時上限(8小時=480分鐘)。
        self.assertLessEqual(assigned_minutes / 60.0, 8.0)
        # 兩筆任務加總必超派，因此不可能兩筆都指派給唯一的居服員。
        self.assertLessEqual(phase2["assigned_count"], 1)

    def test_capacity_constraint_accounts_for_already_used_hours(self):
        """Phase 2 求解器本身須獨立核算「已佔用工時」，不能只依賴 Phase 1 的逐筆過濾。

        直接手動建構 df_matches（略過 Phase 1），確保限制式 3 本身確實把
        今日已佔用工時 + 新派任務歷時 一併納入運算，而非只在 Phase 1 起作用。
        """
        config = PipelineConfig()
        # 今日已佔用 6 小時，上限 8 小時，僅剩 2 小時可派，新任務歷時 3 小時應被排除。
        df_cg = pd.DataFrame([make_caregiver("CG01", daily_cap=8.0, used_hours=6.0)])
        tasks = pd.DataFrame([make_task("TK01", "CL01", "08:00", "11:00", 180)])

        df_matches = pd.DataFrame(
            [
                {
                    "任務ID": "TK01",
                    "案家ID": "CL01",
                    "居服員ID": "CG01",
                    "適配度分數": 60.0,
                    "預估交通時間(分)": 0.0,
                    "任務開始時間": "08:00",
                    "任務結束時間": "11:00",
                    "優先級": "一般",
                    "地點緯度": 25.05,
                    "地點經度": 121.55,
                }
            ]
        )

        phase2 = run_phase2_optimization(df_matches, tasks, df_cg, config)
        self.assertEqual(phase2["assigned_count"], 0)


class SpecialCertHardFilterTests(unittest.TestCase):
    """法定 20 小時特照培訓認證：缺乏對應認證者，Phase 1 必須直接剔除該配對。"""

    def test_uncertified_caregiver_excluded_from_dementia_task(self):
        config = PipelineConfig()
        df_cg = pd.DataFrame(
            [
                make_caregiver("CG_no_cert", cert="單一級照服證照"),
                make_caregiver("CG_dementia", cert="失智症照顧專長"),
            ]
        )
        tasks = pd.DataFrame(
            [make_task("TK01", "CL01", "08:00", "10:00", 120, req_type="失智引導與精神陪伴")]
        )

        df_matches = run_phase1_matching(tasks, df_cg, config)
        matched_cg_ids = set(df_matches["居服員ID"])

        self.assertNotIn("CG_no_cert", matched_cg_ids)
        self.assertIn("CG_dementia", matched_cg_ids)

    def test_certified_pair_carries_cert_flag_columns(self):
        config = PipelineConfig()
        df_cg = pd.DataFrame([make_caregiver("CG_dementia", cert="失智症照顧專長")])
        tasks = pd.DataFrame(
            [make_task("TK01", "CL01", "08:00", "10:00", 120, req_type="失智引導與精神陪伴")]
        )

        df_matches = run_phase1_matching(tasks, df_cg, config)
        row = df_matches.iloc[0]
        self.assertEqual(row["具備失智症20小時認證(0/1)"], 1)
        self.assertEqual(row["具備精神疾病20小時認證(0/1)"], 0)


class ContinuityPriorityTests(unittest.TestCase):
    """照護連續性應為最高指導原則：不得為了微幅車程優化而更換熟悉且評價良好的居服員。"""

    def test_preferred_high_performing_caregiver_outscores_closer_alternative(self):
        config = PipelineConfig()
        df_cg = pd.DataFrame(
            [
                # 案家熟悉、表現優良的首選居服員，但距離較遠。
                make_caregiver("CG_preferred", lat=25.20, lon=121.70, satisfaction=4.6),
                # 明顯較近的替代居服員，滿意度普通。
                make_caregiver("CG_closer", lat=25.051, lon=121.551, satisfaction=4.0),
            ]
        )
        tasks = pd.DataFrame(
            [
                make_task(
                    "TK01", "CL01", "08:00", "10:00", 120,
                    lat=25.05, lon=121.55, pref_cg="CG_preferred",
                )
            ]
        )

        df_matches = run_phase1_matching(tasks, df_cg, config)
        score_preferred = df_matches.loc[df_matches["居服員ID"] == "CG_preferred", "適配度分數"].iloc[0]
        score_closer = df_matches.loc[df_matches["居服員ID"] == "CG_closer", "適配度分數"].iloc[0]

        self.assertGreater(score_preferred, score_closer)

        phase2 = run_phase2_optimization(df_matches, tasks, df_cg, config)
        df_result = phase2["df_result"]
        self.assertEqual(df_result.iloc[0]["派單居服員"], "CG_preferred")

    def test_non_performing_preferred_caregiver_gets_no_dynamic_bonus(self):
        config = PipelineConfig()
        df_cg = pd.DataFrame(
            [make_caregiver("CG_preferred", satisfaction=config.continuity_satisfaction_threshold - 0.5)]
        )
        tasks = pd.DataFrame([make_task("TK01", "CL01", "08:00", "10:00", 120, pref_cg="CG_preferred")])

        df_matches = run_phase1_matching(tasks, df_cg, config)
        score = df_matches.iloc[0]["適配度分數"]
        expected_without_dynamic_bonus = (
            config.base_score
            + config.preferred_caregiver_bonus
            + (df_cg.iloc[0]["歷史滿意度均值"] - config.satisfaction_baseline) * config.satisfaction_weight
        )
        self.assertAlmostEqual(score, round(expected_without_dynamic_bonus, 2), places=2)


class CaregiverChangeReasonTests(unittest.TestCase):
    """指派明細表「原首選替換原因」欄：新客戶或維持原首選居服員時應為空值，
    確實更換居服員時必須附上可解釋的具體原因，且須明確標註原首選居服員 ID
    以避免與本次獲派居服員混淆。
    """

    def test_new_client_without_preferred_caregiver_is_blank(self):
        config = PipelineConfig()
        df_cg = pd.DataFrame([make_caregiver("CG01")])
        tasks = pd.DataFrame([make_task("TK01", "CL01", "08:00", "10:00", 120, pref_cg="")])

        df_matches = run_phase1_matching(tasks, df_cg, config)
        phase2 = run_phase2_optimization(df_matches, tasks, df_cg, config)
        df_result = phase2["df_result"]

        self.assertEqual(df_result.iloc[0]["原首選替換原因"], "")

    def test_unchanged_preferred_caregiver_is_blank(self):
        config = PipelineConfig()
        df_cg = pd.DataFrame([make_caregiver("CG01")])
        tasks = pd.DataFrame([make_task("TK01", "CL01", "08:00", "10:00", 120, pref_cg="CG01")])

        df_matches = run_phase1_matching(tasks, df_cg, config)
        phase2 = run_phase2_optimization(df_matches, tasks, df_cg, config)
        df_result = phase2["df_result"]

        self.assertEqual(df_result.iloc[0]["派單居服員"], "CG01")
        self.assertEqual(df_result.iloc[0]["原首選替換原因"], "")

    def test_hard_constraint_failure_gives_specific_reason(self):
        config = PipelineConfig()
        df_cg = pd.DataFrame(
            [
                make_caregiver("CG_male_pref", gender="男"),
                make_caregiver("CG_female_alt", gender="女"),
            ]
        )
        tasks = pd.DataFrame(
            [
                make_task(
                    "TK01", "CL01", "08:00", "10:00", 120,
                    req_gender="限女性", pref_cg="CG_male_pref",
                )
            ]
        )

        df_matches = run_phase1_matching(tasks, df_cg, config)
        phase2 = run_phase2_optimization(df_matches, tasks, df_cg, config)
        df_result = phase2["df_result"]

        self.assertEqual(df_result.iloc[0]["派單居服員"], "CG_female_alt")
        change_reason = df_result.iloc[0]["原首選替換原因"]
        self.assertIn("原首選居服員[CG_male_pref]", change_reason)
        self.assertIn("性別不符", change_reason)

    def test_existing_schedule_conflict_gives_specific_reason(self):
        config = PipelineConfig()
        df_cg = pd.DataFrame(
            [
                make_caregiver("CG_busy_pref", busy1="08:00-10:00"),
                make_caregiver("CG_free_alt"),
            ]
        )
        tasks = pd.DataFrame(
            [make_task("TK01", "CL01", "08:30", "10:30", 120, pref_cg="CG_busy_pref")]
        )

        df_matches = run_phase1_matching(tasks, df_cg, config)
        phase2 = run_phase2_optimization(df_matches, tasks, df_cg, config)
        df_result = phase2["df_result"]

        self.assertEqual(df_result.iloc[0]["派單居服員"], "CG_free_alt")
        self.assertEqual(
            df_result.iloc[0]["原首選替換原因"],
            "原首選居服員[CG_busy_pref]今日既定行程與本任務時段衝突",
        )

    def test_double_booked_with_other_new_task_gives_specific_reason(self):
        config = PipelineConfig()
        # CG_pref 是 TK_A、TK_B 兩案家共同的歷史首選居服員，但兩任務時段重疊，
        # 無法同時服務兩者；TK_A 額外要求重度移位協助，只有 CG_pref 具備，
        # 因此系統必須把 CG_pref 留給 TK_A，TK_B 改派唯一的替代居服員 CG_alt。
        df_cg = pd.DataFrame(
            [
                make_caregiver("CG_pref", lift_ok=1),
                make_caregiver("CG_alt", lift_ok=0),
            ]
        )
        tasks = pd.DataFrame(
            [
                make_task(
                    "TK_A", "CL_A", "08:00", "10:00", 120,
                    req_lift=1, pref_cg="CG_pref",
                ),
                make_task(
                    "TK_B", "CL_B", "09:00", "11:00", 120,
                    req_lift=0, pref_cg="CG_pref",
                ),
            ]
        )

        df_matches = run_phase1_matching(tasks, df_cg, config)
        phase2 = run_phase2_optimization(df_matches, tasks, df_cg, config)
        df_result = phase2["df_result"].set_index("任務ID")

        self.assertEqual(df_result.loc["TK_A", "派單居服員"], "CG_pref")
        self.assertEqual(df_result.loc["TK_A", "原首選替換原因"], "")
        self.assertEqual(df_result.loc["TK_B", "派單居服員"], "CG_alt")
        self.assertEqual(
            df_result.loc["TK_B", "原首選替換原因"],
            "原首選居服員[CG_pref]該時段已媒合其他案家任務",
        )


class WeekdayHardConstraintTests(unittest.TestCase):
    """新版月批次排班資料表的「星期」／「可排班星期」防呆：欄位確實存在於資料表中時，
    若個別儲存格缺漏或格式無法解析，必須 fail-closed（保守判定當日不可派單），
    不可如舊版邏輯般被 `pd.notna()` 靜默跳過而 fail-open（誤判為全天候可排班）。
    """

    def test_missing_allowed_days_cell_is_rejected_not_silently_allowed(self):
        config = PipelineConfig()
        cg_with_days = make_caregiver("CG_scheduled")
        cg_with_days["可排班星期"] = "1,2,3,4,5"
        cg_missing_days = make_caregiver("CG_missing_days")
        cg_missing_days["可排班星期"] = None
        df_cg = pd.DataFrame([cg_with_days, cg_missing_days])

        task = make_task("TK01", "CL01", "08:00", "10:00", 120)
        task["星期"] = "星期一"
        tasks = pd.DataFrame([task])

        df_matches = run_phase1_matching(tasks, df_cg, config)

        matched_cg_ids = set(df_matches["居服員ID"])
        self.assertIn("CG_scheduled", matched_cg_ids)
        self.assertNotIn("CG_missing_days", matched_cg_ids)

    def test_unrecognized_task_weekday_is_rejected_not_silently_allowed(self):
        config = PipelineConfig()
        cg = make_caregiver("CG01")
        cg["可排班星期"] = "1,2,3,4,5"
        df_cg = pd.DataFrame([cg])

        task = make_task("TK01", "CL01", "08:00", "10:00", 120)
        task["星期"] = "週一"  # 非標準「星期一」格式，WEEKDAY_NAME_TO_NUM 無法辨識
        tasks = pd.DataFrame([task])

        df_matches = run_phase1_matching(tasks, df_cg, config)

        self.assertTrue(df_matches.empty)

    def test_legacy_table_without_weekday_columns_is_unaffected(self):
        """舊版資料表整體不含「星期」／「可排班星期」欄位時，此檢查應完全跳過，
        向下相容行為不變。"""
        config = PipelineConfig()
        df_cg = pd.DataFrame([make_caregiver("CG01")])
        tasks = pd.DataFrame([make_task("TK01", "CL01", "08:00", "10:00", 120)])

        df_matches = run_phase1_matching(tasks, df_cg, config)

        self.assertIn("CG01", set(df_matches["居服員ID"]))


class MandatoryBreakConstraintTests(unittest.TestCase):
    """規則1：居服員累計連續工作達 continuous_work_limit_mins(預設240分鐘)，
    下一段任務與前段之間須強制間隔 mandatory_break_mins(預設30分鐘)。
    """

    def test_chain_of_new_tasks_cannot_all_be_assigned_without_break(self):
        """四筆 90 分鐘任務、彼此間隔 20 分鐘(< 30 分鐘門檻)串成同一條連續鏈，
        累計於第 3 筆即達 240 分鐘門檻，第 4 筆不得與前 3 筆同時獲派。"""
        config = PipelineConfig()
        df_cg = pd.DataFrame([make_caregiver("CG01", daily_cap=12.0)])
        tasks = pd.DataFrame(
            [
                make_task("TK01", "CL01", "08:00", "09:30", 90),
                make_task("TK02", "CL02", "09:50", "11:20", 90),
                make_task("TK03", "CL03", "11:40", "13:10", 90),
                make_task("TK04", "CL04", "13:30", "15:00", 90),
            ]
        )

        df_matches = run_phase1_matching(tasks, df_cg, config)
        phase2 = run_phase2_optimization(df_matches, tasks, df_cg, config)

        self.assertLessEqual(phase2["assigned_count"], 3)

    def test_break_of_at_least_30min_resets_the_chain(self):
        """同樣四筆任務，只要在鏈中插入一段 >=30 分鐘的空檔，鏈即被截斷為兩段
        各自累計皆未達門檻，四筆任務應可全數獲派。"""
        config = PipelineConfig()
        df_cg = pd.DataFrame([make_caregiver("CG01", daily_cap=12.0)])
        tasks = pd.DataFrame(
            [
                make_task("TK01", "CL01", "08:00", "09:30", 90),
                make_task("TK02", "CL02", "09:50", "11:20", 90),
                make_task("TK03", "CL03", "11:55", "13:25", 90),  # 与 TK02 間隔 35 分鐘 (>=30)
                make_task("TK04", "CL04", "13:45", "15:15", 90),
            ]
        )

        df_matches = run_phase1_matching(tasks, df_cg, config)
        phase2 = run_phase2_optimization(df_matches, tasks, df_cg, config)

        self.assertEqual(phase2["assigned_count"], 4)

    def test_existing_appointments_alone_can_force_block_new_task(self):
        """既有既定行程本身（不可變動）累計已達 240 分鐘門檻時，緊接其後、間隔
        僅 20 分鐘的新候選任務必須被禁止指派，即使該任務本身完全符合其他條件。"""
        config = PipelineConfig()
        cg = make_caregiver("CG01", daily_cap=12.0, busy1="06:00-09:00", busy2="09:20-10:20")
        df_cg = pd.DataFrame([cg])
        tasks = pd.DataFrame([make_task("TK01", "CL01", "10:40", "11:40", 60)])

        df_matches = run_phase1_matching(tasks, df_cg, config)
        phase2 = run_phase2_optimization(df_matches, tasks, df_cg, config)

        self.assertEqual(phase2["assigned_count"], 0)


class PeriodicPriorityTests(unittest.TestCase):
    """規則2：週期性排班優先於臨時單次排班——月批次派單時，先鎖定週期性任務班表，
    臨時單次任務只能競爭剩餘的居服員產能與時段。
    """

    def _periodic_task(self, t_id, c_id, start, end, duration_min, date_str, is_periodic):
        task = make_task(t_id, c_id, start, end, duration_min)
        task["日期"] = date_str
        task["是否為週期性任務"] = is_periodic
        return task

    def test_periodic_task_wins_time_conflict_over_adhoc_task(self):
        config = PipelineConfig()
        df_cg = pd.DataFrame([make_caregiver("CG01", daily_cap=10.0)])
        tasks = pd.DataFrame(
            [
                self._periodic_task("TK_PERIODIC", "CL01", "08:00", "09:30", 90, "2026-08-03", True),
                self._periodic_task("TK_ADHOC", "CL02", "08:15", "09:45", 90, "2026-08-03", False),
            ]
        )

        batch = run_monthly_batch_dispatch(tasks, df_cg, config, date_column="日期")
        df_result_all = batch["df_result_all"]
        assigned_ids = set(df_result_all["任務ID"]) if not df_result_all.empty else set()

        self.assertIn("TK_PERIODIC", assigned_ids)
        self.assertNotIn("TK_ADHOC", assigned_ids)

    def test_adhoc_task_only_gets_leftover_capacity_after_periodic(self):
        config = PipelineConfig()
        df_cg = pd.DataFrame([make_caregiver("CG01", daily_cap=4.0)])  # 240 分鐘上限
        tasks = pd.DataFrame(
            [
                self._periodic_task("TK_P1", "CL01", "08:00", "09:10", 70, "2026-08-03", True),
                self._periodic_task("TK_P2", "CL02", "09:40", "10:50", 70, "2026-08-03", True),
                self._periodic_task("TK_P3", "CL03", "11:20", "12:30", 70, "2026-08-03", True),
                # 週期性任務共佔用 210 分鐘，僅剩 30 分鐘產能，臨時任務需 60 分鐘應被拒。
                self._periodic_task("TK_ADHOC", "CL04", "13:00", "14:00", 60, "2026-08-03", False),
            ]
        )

        batch = run_monthly_batch_dispatch(tasks, df_cg, config, date_column="日期")
        df_result_all = batch["df_result_all"]
        assigned_ids = set(df_result_all["任務ID"]) if not df_result_all.empty else set()

        self.assertTrue({"TK_P1", "TK_P2", "TK_P3"}.issubset(assigned_ids))
        self.assertNotIn("TK_ADHOC", assigned_ids)

    def test_legacy_table_without_periodic_column_is_unaffected(self):
        """舊版資料表不含「是否為週期性任務」欄位時，維持原本單一批次邏輯。"""
        config = PipelineConfig()
        df_cg = pd.DataFrame([make_caregiver("CG01", daily_cap=10.0)])
        task = make_task("TK01", "CL01", "08:00", "09:30", 90)
        task["日期"] = "2026-08-03"
        tasks = pd.DataFrame([task])

        batch = run_monthly_batch_dispatch(tasks, df_cg, config, date_column="日期")
        self.assertEqual(batch["total_assigned_count"], 1)


class MonthlyFatigueRollingTests(unittest.TestCase):
    """規則5：月排班動態時數滾動——每完成一天的排班，須將當天指派時數累加回
    居服員的累計疲勞度時數，隔天排班時納入權重計算(偏好累計時數較少者)。
    """

    def test_second_day_prefers_caregiver_with_less_accumulated_hours(self):
        config = PipelineConfig()
        df_cg = pd.DataFrame(
            [
                make_caregiver("CG_A", fatigue=0.0, daily_cap=10.0),
                make_caregiver("CG_B", fatigue=0.0, daily_cap=10.0),
            ]
        )

        day1_task = make_task("TK_DAY1", "CL01", "08:00", "12:00", 240, pref_cg="CG_A")
        day1_task["日期"] = "2026-08-03"
        day2_task = make_task("TK_DAY2", "CL02", "08:00", "10:00", 120, pref_cg="")
        day2_task["日期"] = "2026-08-04"
        tasks = pd.DataFrame([day1_task, day2_task])

        batch = run_monthly_batch_dispatch(tasks, df_cg, config, date_column="日期")
        df_result_all = batch["df_result_all"].set_index("任務ID")

        self.assertEqual(df_result_all.loc["TK_DAY1", "派單居服員"], "CG_A")
        self.assertEqual(df_result_all.loc["TK_DAY2", "派單居服員"], "CG_B")


class QuickReassignConflictTests(unittest.TestCase):
    """月曆視角「一鍵調班」的即時衝突檢查（check_reassignment_conflict）。"""

    def _tasks(self, rows):
        return pd.DataFrame(rows)

    def test_no_conflict_when_gap_covers_travel_and_buffer(self):
        config = PipelineConfig()
        df_tasks = self._tasks([
            {"任務ID": "T1", "案家ID": "C1", "日期": "2026-08-17", "時間窗_開始": "09:00", "時間窗_結束": "10:00", "服務歷時(分鐘)": 60},
            {"任務ID": "T2", "案家ID": "C2", "日期": "2026-08-17", "時間窗_開始": "10:30", "時間窗_結束": "11:00", "服務歷時(分鐘)": 30},
        ])
        df_result = pd.DataFrame([{"任務ID": "T1", "派單居服員": "CG1"}])
        df_cg = pd.DataFrame([{"居服員ID": "CG1", "每日工時上限(小時)": 8.0}])
        locations = {"T1": (25.05, 121.55), "T2": (25.05, 121.55)}

        err = check_reassignment_conflict("T2", "CG1", df_tasks, df_result, df_cg, config, task_locations=locations)
        self.assertIsNone(err)

    def test_conflict_when_transition_buffer_not_met(self):
        config = PipelineConfig()
        df_tasks = self._tasks([
            {"任務ID": "T1", "案家ID": "C1", "日期": "2026-08-17", "時間窗_開始": "09:00", "時間窗_結束": "10:00", "服務歷時(分鐘)": 60},
            {"任務ID": "T2", "案家ID": "C2", "日期": "2026-08-17", "時間窗_開始": "10:05", "時間窗_結束": "11:00", "服務歷時(分鐘)": 55},
        ])
        df_result = pd.DataFrame([{"任務ID": "T1", "派單居服員": "CG1"}])
        df_cg = pd.DataFrame([{"居服員ID": "CG1", "每日工時上限(小時)": 8.0}])
        locations = {"T1": (25.05, 121.55), "T2": (25.05, 121.55)}

        err = check_reassignment_conflict("T2", "CG1", df_tasks, df_result, df_cg, config, task_locations=locations)
        self.assertIsNotNone(err)
        self.assertIn("CG1", err)

    def test_conflict_when_daily_hour_cap_exceeded(self):
        config = PipelineConfig()
        df_tasks = self._tasks([
            {"任務ID": "T1", "案家ID": "C1", "日期": "2026-08-17", "時間窗_開始": "08:00", "時間窗_結束": "12:00", "服務歷時(分鐘)": 240},
            {"任務ID": "T2", "案家ID": "C2", "日期": "2026-08-17", "時間窗_開始": "13:00", "時間窗_結束": "17:00", "服務歷時(分鐘)": 240},
        ])
        df_result = pd.DataFrame([{"任務ID": "T1", "派單居服員": "CG1"}])
        df_cg = pd.DataFrame([{"居服員ID": "CG1", "每日工時上限(小時)": 6.0}])

        err = check_reassignment_conflict("T2", "CG1", df_tasks, df_result, df_cg, config)
        self.assertIsNotNone(err)
        self.assertIn("上限", err)

    def test_unknown_caregiver_is_rejected(self):
        config = PipelineConfig()
        df_tasks = self._tasks([
            {"任務ID": "T1", "案家ID": "C1", "日期": "2026-08-17", "時間窗_開始": "09:00", "時間窗_結束": "10:00", "服務歷時(分鐘)": 60},
        ])
        df_result = pd.DataFrame(columns=["任務ID", "派單居服員"])
        df_cg = pd.DataFrame([{"居服員ID": "CG1", "每日工時上限(小時)": 8.0}])

        err = check_reassignment_conflict("T1", "CG_NOT_EXIST", df_tasks, df_result, df_cg, config)
        self.assertIsNotNone(err)


class RankCandidatesByAvailabilityTests(unittest.TestCase):
    """月曆快速改派下拉選單的「空檔優先排序」（rank_candidates_by_availability）。

    與 check_reassignment_conflict 走同一套 _evaluate_reassignment 判定，這裡只驗證
    排序／標籤輸出，不重複驗證衝突判定本身的邊界條件（已由 QuickReassignConflictTests 涵蓋）。
    """

    def test_available_candidates_sorted_before_conflicted_ones(self):
        config = PipelineConfig()
        df_tasks = pd.DataFrame([
            {"任務ID": "T1", "案家ID": "C1", "日期": "2026-08-17", "時間窗_開始": "09:00", "時間窗_結束": "10:00", "服務歷時(分鐘)": 60},
            {"任務ID": "T2", "案家ID": "C2", "日期": "2026-08-17", "時間窗_開始": "09:15", "時間窗_結束": "10:15", "服務歷時(分鐘)": 60},
            {"任務ID": "T3", "案家ID": "C3", "日期": "2026-08-17", "時間窗_開始": "13:00", "時間窗_結束": "14:00", "服務歷時(分鐘)": 60},
        ])
        df_result = pd.DataFrame([
            {"任務ID": "T1", "派單居服員": "CG_BUSY"},
            {"任務ID": "T2", "派單居服員": "CG_FREE"},
        ])
        df_cg = pd.DataFrame([
            {"居服員ID": "CG_BUSY", "每日工時上限(小時)": 8.0},
            {"居服員ID": "CG_FREE", "每日工時上限(小時)": 8.0},
        ])

        # T3（13:00-14:00）欲從其他人改派：CG_BUSY 當天已有 T1(09:00-10:00)，與 T3 不重疊 -> 應為可派；
        # 額外驗證 CG_FREE（T2 09:15-10:15，同樣與 T3 不重疊）也應為可派，兩者皆列在前段。
        ranked = rank_candidates_by_availability(
            "T3", ["CG_BUSY", "CG_FREE"], df_tasks, df_result, df_cg, config,
        )
        self.assertTrue(all(r["available"] for r in ranked))

    def test_conflicted_candidate_ranked_after_available_one_with_detail(self):
        config = PipelineConfig()
        df_tasks = pd.DataFrame([
            {"任務ID": "T1", "案家ID": "C1", "日期": "2026-08-17", "時間窗_開始": "09:00", "時間窗_結束": "10:00", "服務歷時(分鐘)": 60},
            {"任務ID": "T2", "案家ID": "C2", "日期": "2026-08-17", "時間窗_開始": "09:10", "時間窗_結束": "10:00", "服務歷時(分鐘)": 50},
        ])
        df_result = pd.DataFrame([{"任務ID": "T1", "派單居服員": "CG_BUSY"}])
        df_cg = pd.DataFrame([
            {"居服員ID": "CG_BUSY", "每日工時上限(小時)": 8.0},
            {"居服員ID": "CG_FREE", "每日工時上限(小時)": 8.0},
        ])

        # 候選順序刻意把會衝突的 CG_BUSY 放在最前面：CG_BUSY 於 09:00-10:00 已有任務，
        # 與待改派的 T2（09:10-10:00）重疊 -> 應被排到 CG_FREE（無既有任務、必為可派）之後。
        ranked = rank_candidates_by_availability(
            "T2", ["CG_BUSY", "CG_FREE"], df_tasks, df_result, df_cg, config,
        )
        self.assertEqual([r["cg_id"] for r in ranked], ["CG_FREE", "CG_BUSY"])
        self.assertTrue(ranked[0]["available"])
        self.assertFalse(ranked[1]["available"])
        self.assertEqual(ranked[1]["reason"], "time_conflict")
        self.assertEqual(ranked[1]["conflict_task_id"], "T1")
        self.assertEqual(ranked[1]["conflict_start"], "09:00")
        self.assertEqual(ranked[1]["conflict_end"], "10:00")

    def test_check_reassignment_conflict_agrees_with_ranking(self):
        """單一事實來源：排序結果的 available 判定須與 check_reassignment_conflict 一致。"""
        config = PipelineConfig()
        df_tasks = pd.DataFrame([
            {"任務ID": "T1", "案家ID": "C1", "日期": "2026-08-17", "時間窗_開始": "09:00", "時間窗_結束": "10:00", "服務歷時(分鐘)": 60},
            {"任務ID": "T2", "案家ID": "C2", "日期": "2026-08-17", "時間窗_開始": "09:10", "時間窗_結束": "10:00", "服務歷時(分鐘)": 50},
        ])
        df_result = pd.DataFrame([{"任務ID": "T1", "派單居服員": "CG_BUSY"}])
        df_cg = pd.DataFrame([{"居服員ID": "CG_BUSY", "每日工時上限(小時)": 8.0}])

        ranked = rank_candidates_by_availability("T2", ["CG_BUSY"], df_tasks, df_result, df_cg, config)
        err = check_reassignment_conflict("T2", "CG_BUSY", df_tasks, df_result, df_cg, config)

        self.assertFalse(ranked[0]["available"])
        self.assertIsNotNone(err)
        self.assertEqual(ranked[0]["detail"], err)


class HardConstraintLabelingTests(unittest.TestCase):
    """回歸測試：月曆快速改派下拉選單過去只從 Phase 1 候選配對（df_matches）取人，

    導致性別不符／缺乏證照／環境排斥等「硬性條件不符」的居服員直接從選單消失，
    而非保留＋標記。修正後改由呼叫端傳入機構全體居服員 + df_cl，
    rank_candidates_by_availability 會用與 Phase 1 相同的 _check_hard_constraints
    標記（而非隱藏）這些人；check_reassignment_conflict（確認改派時的權威判定）
    則刻意不受影響，因為本系統無強制派單權限機制，居督仍可自行覆寫。
    """

    def _tasks(self, rows):
        return pd.DataFrame(rows)

    def _setup(self):
        config = PipelineConfig()
        df_tasks = self._tasks([
            {"任務ID": "T1", "案家ID": "C1", "日期": "2026-08-17", "時間窗_開始": "09:00", "時間窗_結束": "10:00", "服務歷時(分鐘)": 60},
        ])
        df_cl = pd.DataFrame([{
            "案家ID": "C1",
            "指定居服員性別": "限女性",
            "需重度移位協助(0/1)": 0,
            "案家環境特徵": "無",
            "特殊照護需求": "一般家務與照顧",
            "歷史首選居服員ID": "",
        }])
        df_cg = pd.DataFrame([
            make_caregiver("CG_MALE", gender="男"),
            make_caregiver("CG_FEMALE", gender="女"),
        ])
        df_result = pd.DataFrame(columns=["任務ID", "派單居服員"])
        return config, df_tasks, df_cl, df_cg, df_result

    def test_hard_constraint_failure_is_labeled_not_hidden_when_df_cl_provided(self):
        config, df_tasks, df_cl, df_cg, df_result = self._setup()

        ranked = rank_candidates_by_availability(
            "T1", ["CG_MALE", "CG_FEMALE"], df_tasks, df_result, df_cg, config, df_cl=df_cl,
        )

        self.assertEqual({r["cg_id"] for r in ranked}, {"CG_MALE", "CG_FEMALE"})
        by_id = {r["cg_id"]: r for r in ranked}
        self.assertTrue(by_id["CG_FEMALE"]["available"])
        self.assertFalse(by_id["CG_MALE"]["available"])
        self.assertEqual(by_id["CG_MALE"]["reason"], "hard_constraint")
        self.assertIn("性別不符", by_id["CG_MALE"]["detail"])
        # 可派者排在前面。
        self.assertEqual(ranked[0]["cg_id"], "CG_FEMALE")

    def test_hard_constraint_check_is_opt_in_via_df_cl(self):
        """未傳 df_cl 時完全跳過硬性條件檢查，維持修正前的行為（僅檢查時間衝突／工時）。"""
        config, df_tasks, df_cl, df_cg, df_result = self._setup()

        ranked = rank_candidates_by_availability(
            "T1", ["CG_MALE", "CG_FEMALE"], df_tasks, df_result, df_cg, config,
        )
        self.assertTrue(all(r["available"] for r in ranked))

    def test_check_reassignment_conflict_does_not_enforce_hard_constraints(self):
        """確認改派的權威判定（check_reassignment_conflict）刻意不檢查硬性資格條件，

        只標記在下拉選單供居督參考；本系統沒有強制派單權限機制，居督仍可能因臨時
        狀況刻意指派不符合建議條件的居服員。
        """
        config, df_tasks, df_cl, df_cg, df_result = self._setup()

        err = check_reassignment_conflict("T1", "CG_MALE", df_tasks, df_result, df_cg, config)
        self.assertIsNone(err)


if __name__ == "__main__":
    unittest.main()
