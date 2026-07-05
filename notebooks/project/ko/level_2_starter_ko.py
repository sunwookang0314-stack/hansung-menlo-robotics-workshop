from __future__ import annotations

"""Menlo AI 로봇 분류 챌린지용 Level 2 프로젝트 시작 파일입니다.

이 파일은 완성된 해답이 아니라 시작 파일입니다.

지원 코드 섹션은 반복해서 작성할 필요가 없는 작은 래퍼와 자료 구조를 제공합니다.
필요하면 읽고 수정할 수 있지만, 대부분의 팀은 지원 코드를 크게 바꾸지 않는 편이 좋습니다.
학생 TODO 섹션은 팀이 수정하고, 개선하고, test하고, presentation에서 설명해야 하는 부분입니다.

실행 설정:
- 기본 run(ctx)는 round1, round2, round3 또는 manual 시간을 묻습니다.
  라운드 제한 시간은 각각 5분, 10분, 15분이며, 모든 라운드는 최대 12개
  cube delivery에서 자동으로 멈춥니다.
- 일반 연습에서는 Enter를 눌러 round2를 사용하고 evaluation setup option은
  비워 두세요. 그러면 현재 scene과 robot pose를 그대로 사용합니다.
- 공통 평가 조건으로 연습할 때는 지정된 round와 1~50 사이 option 번호를
  입력하세요. Starter가 cube_color_order_key를 출력하고, viewer에서 해당
  key를 적용/reset한 뒤 결정된 시작 위치로 robot을 이동합니다.
- manual을 입력하면 원하는 제한 시간을 초 단위로 직접 입력할 수 있습니다.

Level 2 규칙: scene_state, 정확한 entity ID, coordinate go_to는 사용할 수 없습니다.
Camera observation, set_head, set_velocity, memory로 navigation을 구현하세요.
"""

import asyncio
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any

from menlo_runner.completion import CompletionConfig, CompletionTimeout, CompletionTracker
from menlo_runner.config import load_config
from menlo_runner.llm import ask_vlm, call_llm
from menlo_runner.perception import compress_jpeg, detect_color_blobs
from menlo_runner.programs.evaluation_setup import prepare_evaluation_round
from menlo_runner.scene import delivered_cube_ids, held_cube_info


# ---------------------------------------------------------------------------
# 지원 코드: 공통 과제 정의와 필수 LLM 결정 형식
# ---------------------------------------------------------------------------
# 과제 문장은 고정합니다. 목표는 cube 색상 순서와 시작 위치가 달라져도
# 소스 코드 변경 없이 처리하는 하나의 agent를 만드는 것입니다.
TASK = "Find and sort cubes from the source area into their matching destination pads."

# Notebook/Python starter에서 사용할 LLM 모델 선택입니다.
# 이 값을 바꾸거나 실행 전에 환경 변수/.env의 MENLO_LLM_MODEL을 설정하세요.
APPROVED_LLM_MODELS = (
    "minimaxai/minimax-m3",
    "qwen/qwen3.6-35b-a3b",
)
LLM_MODEL = os.environ.setdefault("MENLO_LLM_MODEL", "minimaxai/minimax-m3")
VLM_MODEL = os.environ.setdefault("MENLO_VLM_MODEL", "qwen/qwen3.6-35b-a3b")

# 고정 표지판 정보는 사용할 수 있습니다. 단, 이를 정확한 coordinate나 entity ID로
# 바꾸지 말고 관찰을 해석하는 데만 사용하세요.
DESTINATION_SIGN_RULES = {
    "red": "B",
    "green": "C",
    "blue": "D",
    "yellow": "E",
}
# ★수확용 역매핑(B→red …): VLM 응답에 '같이 보인' 다른 sign을 해당 색 기억에 등록할 때 사용.
LETTER_TO_COLOR = {v: k for k, v in DESTINATION_SIGN_RULES.items()}
SIGNAGE_NOTE = (
    "A는 conveyor/cube source area이며 destination이 아닙니다. "
    "Destination sign은 B red, C green, D blue, E yellow입니다."
)

# LLM은 아래 set에서 상위 단계 행동을 선택해야 합니다. 원시 속도 명령을
# 직접 출력하지 말고, 결정적 코드가 결정을 robot 행동으로 변환해야 합니다.
ALLOWED_NEXT_ACTIONS = {
    "search_cube",
    "navigate_to_cube",
    "pick_cube",
    "search_pad",
    "navigate_to_pad",
    "place_cube",
    "recover",
    "skip_target",
    "stop",
}


@dataclass
class AgentDecision:
    """LLM이 반환하고 코드가 검증한 상위 단계 결정입니다."""

    next_action: str
    target_color: str | None = None
    reason: str = ""
    recovery_strategy: str | None = None


@dataclass
class AgentMemory:
    """observe-decide-act cycle 사이에 agent가 유지하는 상태입니다.

    간단하게 시작한 뒤, 팀 전략에 필요한 field를 추가하세요. 예: target history,
    failed location, scan result, confidence score, held-object estimate 등.
    """

    delivered_count: int = 0
    held_color: str | None = None
    active_color: str | None = None
    stage: str = "need_cube"
    search_turns: int = 0
    failed_attempts: dict[str, int] = field(default_factory=dict)
    completed_colors: list[str] = field(default_factory=list)
    skipped_colors: list[str] = field(default_factory=list)
    # 런타임에 추정한 컨베이어 벨트/레일 색(제외가 아니라 획득 시 '후순위' 용도).
    belt_color: str | None = None
    # pick 연속 실패 횟수: 같은 큐브에 막히면 더 크게 relocate하기 위한 카운터.
    pick_fail_streak: int = 0
    # 마지막으로 실제 집은 색(get_held_cube_info ground truth).
    last_grabbed_color: str | None = None
    # 최근 pick 실패 메모(수명 ttl). 존재하면 recover가 더 크게 회전해 stuck 큐브를 벗어납니다.
    recent_pick_fail: dict[str, Any] | None = None
    # --- 경로 기억(성공 경험 greedy 재사용; 엄밀한 RL이 아닌 online heuristic 최적화) ---
    # pad_memory[color] = {"last_seen", "anchor", "successful_routes", "failed_routes",
    # "best_route"}. 전부 로봇 자신의 odometry pose(고유수용성) + 카메라/VLM 관찰에서 유도한
    # 학생 추정치이며 scene_state/entity ID가 아닙니다. 한 run 안에서만 축적됩니다(영속 캐시 없음).
    pad_memory: dict[str, dict[str, Any]] = field(default_factory=dict)
    # 현재 배송(pick 성공 → place 성공)의 step 단위 이동 기록. 성공 시 pad_memory로 승격.
    route_trace: list[dict[str, Any]] = field(default_factory=list)
    # 현재 배송의 누적 통계(t0, start_pose, vlm_calls, stalls, path_len_m).
    route_stats: dict[str, Any] = field(default_factory=dict)
    # stall(전진 병진 실패)이 났던 pose(x, y, 당시 yaw) 기록 — 같은 지점·같은 방향 재돌진 방지.
    stall_spots: list[dict[str, float]] = field(default_factory=list)
    # 우회가 실제로 병진을 확보한 지점·방향("성공한 우회") 기록 — 같은 지점 재방문 시 그 방향 우선.
    detour_wins: list[dict[str, float]] = field(default_factory=list)
    # ★이동 세그먼트 번호: 픽 성공/배송 성공마다 +1. 픽 단계에서 쌓인 stall 기억이
    #   배송 첫 전진을 선제 봉쇄하던 오염(라이브 확정)을 단계 태그로 차단.
    nav_segment: int = 0
    # ★큐브를 집은 자리(=소스 vicinity)의 자기 pose. 배송 후 다음 큐브 탐색 시 복귀 목표.
    #   매 pick 성공마다 갱신, 배송 사이에도 유지 → 멀티큐브 왕복의 나침반(랜덤 시작 안전).
    source_pose: dict[str, float] | None = None
    logs: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Observation:
    """LLM과 실행 코드에 전달할 간결한 관찰입니다."""

    robot_status: Any
    detections: list[Any]
    note: str = ""
    vlm_summary: str = ""


@dataclass(frozen=True)
class ScannedDetection:
    """해당 camera frame을 얻을 때 사용한 head pose가 함께 기록된 color detection입니다.

    이 구조는 특정 strategy에 묶이지 않도록 의도적으로 중립적입니다. 
    Level 1 팀은 coordinate estimate에 full bearing을 사용할 수 있고, 
    Level 2 팀은 closed-loop visual centering에 사용할 수 있습니다. 
    필요하면 confidence, target type, depth field를 추가하세요.
    """

    color: str
    angle_deg: float
    blob_area: int
    centroid: tuple[int, int]
    bbox: tuple[int, int, int, int]
    head_yaw: float
    head_pitch: float

    @property
    def full_bearing_deg(self) -> float:
        """대략적인 body-relative bearing입니다. Image angle에 head yaw를 더합니다."""
        return self.angle_deg + math.degrees(self.head_yaw)


def parse_agent_decision(text: str) -> AgentDecision | None:
    """필수 structured LLM JSON output을 parse하고 validate합니다."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    if not stripped.startswith("{"):
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            stripped = stripped[start : end + 1]
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        return None

    next_action = data.get("next_action")
    if next_action not in ALLOWED_NEXT_ACTIONS:
        return None

    target_color = data.get("target_color")
    if target_color is not None and not isinstance(target_color, str):
        return None

    return AgentDecision(
        next_action=next_action,
        target_color=target_color,
        reason=str(data.get("reason", "")),
        recovery_strategy=data.get("recovery_strategy"),
    )


def build_decision_context(
    task: str,
    observation: Observation,
    memory: AgentMemory,
    last_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Robot state를 LLM에 전달하기 좋은 간결한 text context로 변환합니다.

    VLM을 명시적으로 사용하는 경우가 아니라면 raw image는 이 text context에 넣지 마세요. LLM은 다음 high-level step을 고를 만큼의 정보만 받고, low-level control과 safety는 code가 처리해야 합니다.
    """
    visible = [
        {
            "color": detection.color,
            "angle_deg": detection.angle_deg,
            "full_bearing_deg": round(getattr(detection, "full_bearing_deg", detection.angle_deg), 1),
            "blob_area": detection.blob_area,
            "bbox": detection.bbox,
        }
        for detection in observation.detections
    ]
    return {
        "task": task,
        "visible_targets": visible,
        "held_color": memory.held_color,
        "active_color": memory.active_color,
        "stage": memory.stage,
        "delivered_count": memory.delivered_count,
        "completed_colors": memory.completed_colors,
        "skipped_colors": memory.skipped_colors,
        "failed_attempts": memory.failed_attempts,
        "last_result": last_result,
        "note": observation.note,
        "signage_note": SIGNAGE_NOTE,
        "vlm_summary": observation.vlm_summary,
    }


# ---------------------------------------------------------------------------
# 지원 코드: project 규칙에 맞는 SDK wrapper
# ---------------------------------------------------------------------------
# 이 래퍼들은 프로젝트 규칙에 맞는 input을 노출합니다. 아래 progress helper는
# completion과 robot이 cube를 들고 있는지 추적할 수 있도록 허용됩니다.
# Ground-truth coordinate, 정확한 target ID, global asset map은 추가하지 마세요.

async def get_robot_status(ctx: Any) -> Any:
    """Robot pose, motion status, neck state를 읽습니다."""
    return await ctx.state("robot_status")


async def get_camera_frame(
    ctx: Any,
    *,
    compressed: bool = False,
    max_width: int = 800,
    quality: int = 70,
) -> bytes:
    """POV camera frame을 가져오며, VLM용으로 resize/re-encode할 수 있습니다."""
    # ★get_vision(livekit RPC)은 자체 timeout이 없어 무응답이면 '무한 대기'다(SDK context.py 확인).
    #   10초 캡 + 1회 재시도 — 카메라 한 번 삐끗이 사이클/런을 통째로 세우는 것 방지.
    try:
        jpeg = await asyncio.wait_for(ctx.get_vision("pov"), timeout=10)
    except Exception as e:
        print(f"  camera 무응답({type(e).__name__}) → 1회 재시도")
        await asyncio.sleep(0.5)
        jpeg = await asyncio.wait_for(ctx.get_vision("pov"), timeout=10)
    if compressed:
        return compress_jpeg(jpeg, max_width=max_width, quality=quality)
    return jpeg


async def get_delivered_count(ctx: Any) -> int:
    """공통 workshop progress helper로 delivered cube 수를 셉니다."""
    return len(await delivered_cube_ids(ctx))


async def get_held_cube_info(ctx: Any) -> dict[str, str] | None:
    """Robot이 cube를 들고 있으면 현재 held cube id/color를 반환합니다."""
    held = await held_cube_info(ctx)
    return {"entity_id": held[0], "color": held[1]} if held else None


def build_signage_vlm_prompt(held_color: str | None = None) -> str:
    """고정 warehouse signage를 읽기 위한 strategy-neutral prompt를 만듭니다."""
    target = ""
    if held_color in DESTINATION_SIGN_RULES:
        target = f" Robot이 {held_color} cube를 들고 있으므로 target destination sign은 {DESTINATION_SIGN_RULES[held_color]}입니다."
    return (
        "이 robot camera frame에 보이는 warehouse sign을 읽으세요. "
        f"{SIGNAGE_NOTE} "
        "보이는 sign letter, color, 대략적인 left/center/right 위치, confidence를 JSON으로 반환하세요."
        + target
    )


async def ask_vlm_about_frame(
    ctx: Any,
    prompt: str,
    *,
    api_key: str,
    compressed: bool = True,
    max_width: int = 800,
    quality: int = 70,
) -> str:
    """Project에서 허용되는 VLM helper로 현재 POV frame에 대해 질문합니다."""
    jpeg = await get_camera_frame(
        ctx,
        compressed=compressed,
        max_width=max_width,
        quality=quality,
    )
    # ★to_thread: 동기 ask_vlm이 이벤트 루프를 얼리면 VLM 6~32초 동안 하트비트가 끊겨
    #   livekit RPC들이 연쇄로 죽는다(무응답 타임아웃의 유력 원인). 스레드로 빼서 루프를 살린다.
    # ★wait_for 45초: 무응답 콜이 시간을 무한정 태우지 못하게 캡. 호출부(_scan_pad_bearing)가
    #   예외를 잡고 '미검출'로 처리하므로 런은 계속 돈다. 실측 6~32s 분포 기준 여유 잡은 값.
    return await asyncio.wait_for(
        asyncio.to_thread(ask_vlm, jpeg, prompt, api_key=api_key),
        timeout=45,
    )


async def perceive(ctx: Any) -> list[Any]:
    """현재 camera frame에서 Workshop 2 color-blob detector를 실행합니다."""
    jpeg = await get_camera_frame(ctx)
    return detect_color_blobs(jpeg)


async def set_head(ctx: Any, *, yaw: float | None = None, pitch: float | None = None) -> Any:
    """Walking direction을 바꾸지 않고 camera 방향을 조정합니다."""
    args: dict[str, float] = {}
    if yaw is not None:
        args["yaw"] = yaw
    if pitch is not None:
        args["pitch"] = pitch
    try:
        return await ctx.invoke("set_head", args, timeout_s=12)
    except Exception as e:                       # ★RPC 무응답 1회 재시도(사이클 사망 방지)
        print(f"  set_head 무응답({type(e).__name__}) → 1회 재시도")
        await asyncio.sleep(0.8)
        return await ctx.invoke("set_head", args, timeout_s=12)


async def move_velocity(
    ctx: Any,
    *,
    vx: float = 0.0,
    vy: float = 0.0,
    wz: float = 0.0,
    duration_s: float = 1.0,
) -> Any:
    """짧은 body-frame velocity command를 보낸 뒤 멈춥니다.
    ★timeout 300→20초: 이동 명령은 길어야 수 초짜리다. 무응답 하나가 5분을 태우면
    900초 런에서 회복 불가 — 짧게 끊고 1회 재시도, 그래도 죽으면 상위로 올린다."""
    args = {"vx": vx, "vy": vy, "wz": wz, "duration_s": duration_s}
    tmo = max(duration_s + 8.0, 13.0)   # ★적응형+하한 13s: 짧은 명령이 일시 RTT 스파이크(livekit
                                        #   혼잡)에 2연속 타임아웃해 사이클을 뚫는 것 방지(검증 지적).
    try:
        return await ctx.invoke("set_velocity", args, timeout_s=tmo)
    except Exception as e:
        print(f"  move_velocity 무응답({type(e).__name__}) → 1회 재시도")
        await asyncio.sleep(0.8)
        return await ctx.invoke("set_velocity", args, timeout_s=tmo)


async def cancel_action(ctx: Any) -> Any:
    """현재 실행 중인 runtime action을 취소합니다."""
    return await ctx.invoke("cancel", {})


async def pick_nearest_cube(ctx: Any) -> Any:
    """Code가 robot을 시각적으로 충분히 위치시킨 뒤 nearest cube를 집습니다.
    ★timeout 900→120초: 무응답 픽 하나가 라운드의 대부분을 태우는 걸 방지.
    타임아웃이어도 실제로는 집었을 수 있으니 성공 판정은 호출측(held_cube_info)이 한다."""
    try:
        return await ctx.invoke(
            "pick_entity",
            {"target": {"kind": "entity", "entity_id": "cube"}},
            timeout_s=120,
        )
    except Exception as e:
        print(f"  pick 무응답({type(e).__name__}) → cancel 후 held 상태로 판정")
        try:
            await cancel_action(ctx)
        except Exception:
            pass
        return None


async def place_nearest_zone(ctx: Any) -> Any:
    """Matching pad에 도달한 뒤 nearest zone에 place합니다.
    ★timeout 900→120초 + cancel. 성공 판정은 delivered_cube_ids 증가로 한다."""
    try:
        return await ctx.invoke("place_entity", {}, timeout_s=120)
    except Exception as e:
        print(f"  place 무응답({type(e).__name__}) → cancel 후 delivered로 판정")
        try:
            await cancel_action(ctx)
        except Exception:
            pass
        return None


def result_summary(result: Any) -> dict[str, Any]:
    """SDK result를 log하기 쉬운 작은 dictionary로 변환합니다."""
    error = getattr(result, "error", None)
    status = getattr(result, "status", None)
    return {
        "status": str(status) if status is not None else None,
        "error": getattr(error, "message", None) if error else None,
    }


async def scan_head(
    ctx: Any,
    *,
    yaws: tuple[float, ...] = (-0.8, 0.0, 0.8),
    pitch: float = 0.15,
) -> list[Any]:
    """간단한 scan helper입니다. 더 나은 search 전략으로 교체할 수 있습니다."""
    all_detections: list[Any] = []
    for yaw in yaws:
        await set_head(ctx, yaw=yaw, pitch=pitch)
        await asyncio.sleep(0.4)
        for detection in await perceive(ctx):
            all_detections.append(
                ScannedDetection(
                    color=detection.color,
                    angle_deg=detection.angle_deg,
                    blob_area=detection.blob_area,
                    centroid=detection.centroid,
                    bbox=detection.bbox,
                    head_yaw=yaw,
                    head_pitch=pitch,
                )
            )
    return all_detections


# ---------------------------------------------------------------------------
# 학생 TODO: LLM decision 함수
# ---------------------------------------------------------------------------
async def decide_next_action(
    task: str,
    observation: Observation,
    memory: AgentMemory,
    last_result: dict[str, Any] | None = None,
) -> AgentDecision:
    """Text LLM을 사용해 다음 상위 단계 행동을 선택합니다.

    TODO:
    - decision_context로 명확한 prompt를 만드세요.
    - menlo_runner.llm.call_llm 또는 승인된 LLM helper를 호출하세요.
      helper가 synchronous/blocking이면 await asyncio.to_thread(...)로 감싸세요.
      그래야 strict round timer가 시간 초과 시 model 대기를 중단할 수 있습니다.
    - next_action, target_color, reason이 포함된 JSON을 요구하세요.
    - parse_agent_decision으로 validate하세요.
    - Validation이 실패하면 안전한 recovery decision을 반환하세요.

    아래 fallback은 의도적으로 약하게 만들어져 있습니다. 제출 전에는 교체하세요.
    """
    decision_context = build_decision_context(task, observation, memory, last_result)

    # Prompt 예시 형태:
    # system: 이 schema에 맞는 JSON만 반환하도록 요구합니다.
    # {"next_action": "search_cube", "target_color": "red", "reason": "..."}
    # user: json.dumps(decision_context)

    system_prompt = (
        "당신은 humanoid warehouse robot의 상위 단계 결정을 담당합니다. "
        "목표: source area(A)의 cube를 집어 색상과 일치하는 destination pad(B red, C green, D blue, E yellow)에 놓기. "
        f"다음 행동 중 정확히 하나만 고르세요: {', '.join(sorted(ALLOWED_NEXT_ACTIONS))}. "
        "raw 속도 명령은 출력하지 말고 상위 단계 결정만 내리세요. "
        "채점은 색 무관입니다(정확히 분류된 큐브당 동일 점수). 특정 색을 고집하지 마세요. "
        "규칙 — held_color가 null이면 '획득' 단계: clean_cubes에 큐브가 보이면 pick_cube를 고르세요 "
        "(pick_entity가 각도·거리와 무관하게 최근접 큐브를 스스로 접근·파지하므로 별도 정렬이 필요 없습니다). "
        "보이는 clean cube가 전혀 없을 때만 search_cube로 탐색하세요. 어떤 색이든 좋으니 target_color는 null로 두면 됩니다. "
        "held_color가 있으면 '배송' 단계: target_color를 반드시 held_color로 두고 그 색 pad로 "
        "navigate_to_pad 또는 place_cube 하세요. "
        "clean_cubes 힌트가 있으면 그 목록의 cube를 우선 노리고, belt_color는 후순위로 두세요. "
        "failed_attempts가 3 이상 쌓인 색은 skip_target 하세요. "
        "설명 없이 아래 schema의 JSON 하나만 반환하세요: "
        '{"next_action": "<action>", "target_color": "<color 또는 null>", "reason": "<짧은 이유>"}.'
    )
    user_prompt = json.dumps(decision_context, ensure_ascii=False)

    decision: AgentDecision | None = None
    try:
        config = load_config(require_tokamak=True)
        # ★to_thread: 동기 call_llm이 이벤트 루프를 얼리면 livekit 하트비트가 끊겨
        #   move_velocity 무응답의 원인이 된다(ask_vlm과 동일 원리). 45초 캡으로 무한 대기 차단.
        raw = await asyncio.wait_for(
            asyncio.to_thread(
                call_llm,
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                api_key=config.tokamak_api_key,
                timeout_s=40,   # ★스레드가 스스로 죽게(wait_for는 스레드를 못 죽여 워커 누수).
            ),
            timeout=45,
        )
        decision = parse_agent_decision(raw)
    except Exception:
        decision = None

    # LLM은 상위 시퀀서/ratifier로만 씁니다. 저수준 제어와 색 정합성은 코드가 강제합니다.
    if decision is not None:
        if memory.held_color:
            # 배송 단계: 실제 든 색으로 강제해 잘못된 pad 배송을 막습니다.
            decision.target_color = memory.held_color
        # 획득 단계에서는 target_color를 강제하지 않습니다(색맹 pick과 정합; 참고용).
        return decision

    # LLM 실패 시 rule-based로 degrade합니다(Tokamak 장애에도 동작을 이어갑니다).
    if memory.held_color:
        return AgentDecision(
            next_action="navigate_to_pad",
            target_color=memory.held_color,
            reason="대체 동작: LLM 실패, 든 색 pad로 이동.",
        )
    if decision_context["visible_targets"]:
        return AgentDecision(
            next_action="pick_cube",
            target_color=None,
            reason="대체 동작: LLM 실패, 보이는 cube를 pick(pick_entity가 최근접 큐브를 자체 접근·파지).",
        )
    return AgentDecision(next_action="search_cube", reason="대체 동작: LLM 실패, 보이는 target이 없어 탐색.")


# ---------------------------------------------------------------------------
# 학생 TODO: observation, execution, verification, memory
# ---------------------------------------------------------------------------
async def observe_world(ctx: Any, memory: AgentMemory) -> Observation:
    """LLM과 실행 코드를 위해 현재 관찰을 수집합니다.

    TODO:
    - 언제 set_head scan을 사용할지, 언제 single frame을 사용할지 결정하세요.
    - 필요하면 VLM output, confidence, target type, search note를 추가하세요.
      Signage에는 build_signage_vlm_prompt()와 ask_vlm_about_frame()을 사용하세요.
    - 제출 code에서는 scene_state와 정확한 entity ID를 사용하지 마세요.
    """
    robot_status = await get_robot_status(ctx)

    # cube를 찾는 단계에서는 head를 넓게 훑어 주변 target을 최대한 많이 봅니다.
    raw_detections = await scan_head(ctx)
    # scan이 끝나면 head를 정면으로 되돌려 이후 body-frame 판단이 흔들리지 않게 합니다.
    await set_head(ctx, yaw=0.0, pitch=0.15)
    # LLM에 넘기기 전에 perception 노이즈(컨베이어 레일/바닥 등 비현실적 blob)를 배제합니다.
    # 이렇게 하지 않으면 build_decision_context가 초대형 레일 blob을 "가장 큰 cube"로
    # LLM에 전달해 잘못된 target을 고르게 만듭니다.
    arrival_area = PAD_ARRIVAL_AREA if memory.held_color else CUBE_ARRIVAL_AREA
    detections = [d for d in raw_detections if _plausible_target(d, arrival_area)]
    visible_colors = sorted({d.color for d in detections})
    # 프레임을 뒤덮는 벨트/레일 색을 런타임 추정해 기억합니다(제외가 아니라 후순위 용도).
    belt = _detect_belt_color(raw_detections)
    if belt is not None:
        memory.belt_color = belt
    # 획득에 쓸 '깨끗한 큐브' 후보(색:크기)를 note에 실어 LLM이 벨트/노이즈에 안 휘둘리게 합니다.
    clean = sorted(
        (d for d in detections if _is_clean_cube(d, arrival_area)),
        key=lambda d: d.blob_area,
        reverse=True,
    )

    # cube를 들고 있으면 destination pad를 찾는 단계입니다. 같은 색 pad가 아직
    # 안 보일 때만 VLM으로 signage를 읽어 불필요한 호출과 시간 낭비를 줄입니다.
    # 추가 절감: 그 색의 경로 기억(best_route)이나 last_seen이 이미 있으면 결정용
    # 프리뷰 VLM도 생략합니다 — pad-nav가 필요한 시점에 스스로 look하므로 여기서
    # 또 읽는 것은 순수 중복입니다(사유는 note의 pad_cache로 남김).
    vlm_summary = ""
    pad_cache_note = ""
    if memory.held_color is not None and memory.held_color not in visible_colors:
        cached = memory.pad_memory.get(memory.held_color) or {}
        if cached.get("best_route") or cached.get("last_seen") or cached.get("anchor"):
            pad_cache_note = "pad_cache=hit(vlm_preview_skip)"
        else:
            try:
                config = load_config(require_tokamak=True)
                prompt = build_signage_vlm_prompt(memory.held_color)
                vlm_summary = await ask_vlm_about_frame(
                    ctx,
                    prompt,
                    api_key=config.tokamak_api_key,
                    max_width=SIGNAGE_VLM_MAX_WIDTH,
                    quality=SIGNAGE_VLM_QUALITY,
                )
            except Exception:
                vlm_summary = ""

    note_parts = [f"stage={memory.stage}", f"search_turns={memory.search_turns}"]
    if pad_cache_note:
        note_parts.append(pad_cache_note)
    if memory.held_color:
        target_pad = DESTINATION_SIGN_RULES.get(memory.held_color, "?")
        note_parts.append(f"held={memory.held_color}->pad {target_pad}")
    note_parts.append("visible=" + (",".join(visible_colors) if visible_colors else "none"))
    if memory.belt_color:
        note_parts.append(f"belt_color={memory.belt_color}(deprioritize)")
    if clean:
        note_parts.append("clean_cubes=" + ",".join(f"{d.color}:{d.blob_area}" for d in clean[:5]))

    return Observation(
        robot_status=robot_status,
        detections=detections,
        note="; ".join(note_parts),
        vlm_summary=vlm_summary,
    )


async def verify_outcome(ctx: Any, decision: AgentDecision, action_result: dict[str, Any]) -> dict[str, Any]:
    """마지막 action이 성공한 것처럼 보이는지 확인합니다.

    TODO:
    - 중요한 action 뒤에는 다시 observe하세요.
    - robot_status, camera evidence, SDK result status를 확인하세요.
    - 다음 LLM call이 recovery에 사용할 수 있는 정보를 반환하세요.
    """
    # 중요한 action 뒤에는 상태를 다시 읽어 실제로 성공했는지 근거를 모읍니다.
    held = await get_held_cube_info(ctx)
    # 느린 pick 직후 held가 아직 None으로 보일 수 있어 한 번 더 확인합니다(타이밍 헤지).
    if held is None and decision.next_action == "pick_cube":
        await asyncio.sleep(0.3)
        held = await get_held_cube_info(ctx)
    delivered_count = await get_delivered_count(ctx)
    robot_status = await get_robot_status(ctx)
    robot_motion = getattr(getattr(robot_status, "robot", None), "status", None)

    # 마지막 target 색이 아직 보이는지, 얼마나 크게 보이는지 다시 관찰합니다.
    target_still_visible: bool | None = None
    target_blob_area: int | None = None
    if decision.target_color is not None:
        try:
            matching = [d for d in await perceive(ctx) if d.color == decision.target_color]
            target_still_visible = len(matching) > 0
            target_blob_area = max((d.blob_area for d in matching), default=0)
        except Exception:
            target_still_visible = None

    # SDK result에 error가 실려 있으면 그대로 노출해 recovery 판단에 씁니다.
    result_error = None
    inner_result = action_result.get("result")
    if isinstance(inner_result, dict):
        result_error = inner_result.get("error")

    return {
        "decision": decision.__dict__,
        "action_result": action_result,
        "delivered_count": delivered_count,
        "held_cube": held,
        "held_color": held["color"] if held else None,
        "robot_motion": str(robot_motion) if robot_motion is not None else None,
        "target_still_visible": target_still_visible,
        "target_blob_area": target_blob_area,
        "result_error": result_error,
        # 경로 기억용 odometry pose(고유수용성). update_memory가 배송 시작/드롭 지점으로 씁니다.
        "pose": _pose_dict(robot_status),
    }


def update_memory(
    memory: AgentMemory,
    observation: Observation,
    decision: AgentDecision,
    verified: dict[str, Any],
) -> None:
    """각 cycle 뒤 지속 상태를 update합니다.

    TODO:
    - completed cube, held color, failed attempt, recovery history를 추적하세요.
    - interim/final presentation에서 보여줄 수 있는 간결한 log를 남기세요.
    """
    prev_delivered = memory.delivered_count
    new_delivered = int(verified.get("delivered_count", prev_delivered))
    new_held = verified.get("held_color")
    pose = verified.get("pose")

    action = decision.next_action
    color = decision.target_color
    action_result = verified.get("action_result") or {}

    # 경로 기록 수명주기: 획득→배송 전환(방금 집음) 시점에 새 배송 trace를 시작합니다.
    # 이 trace에 pad-nav의 모든 이동(기대/실제 이동량·stall·VLM look)이 쌓이고,
    # 배송이 성공하면 아래에서 pad_memory[색]으로 승격됩니다.
    # began_new_trace: 이 cycle에 새 배송 trace가 방금 시작됐는지(R2 커밋 가드의 입력).
    began_new_trace = memory.held_color is None and bool(new_held)
    if began_new_trace:
        memory.nav_segment += 1   # ★픽 성공 = 새 이동 세그먼트(픽 단계 stall 기억과 분리).
        if pose is not None:
            memory.source_pose = pose   # ★소스 위치 기억(멀티큐브 복귀 나침반).
        memory.route_trace = []
        memory.route_stats = {
            "target_color": new_held,
            "t0": time.monotonic(),
            "start_pose": pose,
            "vlm_calls": 0,
            "stalls": 0,
            "path_len_m": 0.0,
        }

    # cube를 찾거나 접근하는 동안의 목표 색을 active_color로 기억합니다.
    if action in {"search_cube", "navigate_to_cube"} and color:
        memory.active_color = color

    # 배달 성공: delivered_count가 늘면 방금 놓은 색을 완료 목록에 넣습니다.
    if new_delivered > prev_delivered:
        memory.nav_segment += 1   # ★배송 성공 = 다음 획득 세그먼트 시작.
        placed = memory.held_color or memory.active_color
        if placed and placed not in memory.completed_colors:
            memory.completed_colors.append(placed)
        if placed:
            memory.failed_attempts.pop(placed, None)
        # 성공 경로 승격(R2 커밋 가드): 이 cycle에 커밋해도 되는 경로일 때만 저장합니다.
        #  - t0가 없으면 추적된 배송이 아님(시작 전이거나 이미 커밋됨) → 커밋 금지.
        #  - 같은 cycle에 새 배송 trace가 시작됐으면(delivered가 1-cycle 늦게 잡힌 경우)
        #    route_stats는 '방금 집은 다음 큐브' 것이므로, 커밋하면 score≈0 쓰레기 경로가
        #    best_route(min-score라 이후에 못 이김)로 굳어짐 → 커밋 금지(학습 1건 생략 감수).
        # 커밋 색 키는 placed 추정이 아니라 route_stats["target_color"](pick 시점 ground truth).
        if _should_commit_route(memory.route_stats, began_new_trace):
            route_color = memory.route_stats.get("target_color") or placed
            entry = _pad_memory_entry(memory.pad_memory, route_color)
            stats = {k: v for k, v in memory.route_stats.items() if k != "t0"}
            stats["total_time_s"] = round(time.monotonic() - memory.route_stats["t0"], 1)
            waypoints = _route_waypoints(
                memory.route_stats.get("start_pose"), memory.route_trace, pose
            )
            route = _commit_successful_route(entry, waypoints, stats, drop_pose=pose)
            print(
                f"[route] {route_color} 배송 경로 저장: score={route['score']:.1f}"
                f" (time={stats['total_time_s']}s vlm={stats.get('vlm_calls', 0)}"
                f" stall={stats.get('stalls', 0)} path={stats.get('path_len_m', 0.0):.1f}m"
                f" wp={len(waypoints)})"
            )
            memory.route_trace = []
            memory.route_stats = {}
        memory.active_color = None
        memory.pick_fail_streak = 0
        memory.recent_pick_fail = None

    # pick 결과 판정: 이제 들고 있으면 성공, 아니면 실패 횟수를 늘립니다.
    if action == "pick_cube":
        if new_held:
            memory.failed_attempts.pop(color or new_held, None)
            memory.pick_fail_streak = 0
            memory.recent_pick_fail = None
            memory.last_grabbed_color = new_held
        else:
            key = color or memory.active_color
            if key:
                memory.failed_attempts[key] = memory.failed_attempts.get(key, 0) + 1
            # 같은 큐브에 막혀 색맹 pick이 헛집는 것을 막기 위해 relocate를 유도합니다.
            memory.pick_fail_streak += 1
            memory.recent_pick_fail = {"ttl": 2}

    # place 실패 판정: 배달 수 변화 없이 아직 들고 있으면 실패로 간주합니다.
    # ★즉시-place(search_pad/navigate_to_pad 분기가 도착 후 바로 place)도 포함한다 —
    #   place 결과(action_result["place"]["placed"]==False)를 안 세면 같은 색을 무한 재시도하고
    #   failed_attempts가 안 쌓여 skip_target(3회 규칙)이 영영 발동 안 한다(검증 지적).
    place_failed = (
        (action == "place_cube" and new_delivered == prev_delivered and new_held)
        or (isinstance(action_result.get("place"), dict)
            and action_result["place"].get("placed") is False and new_held)
    )
    if place_failed:
        memory.failed_attempts[new_held] = memory.failed_attempts.get(new_held, 0) + 1

    # navigate 실패 판정: 도착하지 못했으면 실패 횟수를 늘립니다.
    if action in {"navigate_to_cube", "navigate_to_pad"} and action_result.get("reached") is False and color:
        memory.failed_attempts[color] = memory.failed_attempts.get(color, 0) + 1

    # pad 접근 실패는 통계와 함께 진단용으로 남깁니다(다음 시도 전략·발표 근거).
    if action == "navigate_to_pad" and action_result.get("reached") is False and memory.held_color:
        _record_failed_route(
            _pad_memory_entry(memory.pad_memory, memory.held_color),
            {k: v for k, v in memory.route_stats.items() if k != "t0"},
            reason="navigate_to_pad_failed",
        )

    # search 진행도: 찾았으면 0으로, 못 찾았으면 누적해 무한 탐색을 감지합니다.
    if action in {"search_cube", "search_pad"}:
        memory.search_turns = 0 if action_result.get("found") else memory.search_turns + 1
    else:
        memory.search_turns = 0

    # skip 기록: 반복 실패로 건너뛴 색을 남기고 active에서 제외합니다.
    if action == "skip_target" and color and color not in memory.skipped_colors:
        memory.skipped_colors.append(color)
        memory.failed_attempts.pop(color, None)
        if memory.active_color == color:
            memory.active_color = None

    # 실패-회피 메모 수명 관리(존재 시 recover가 더 크게 회전; 시간이 지나면 소멸).
    if memory.recent_pick_fail is not None:
        memory.recent_pick_fail["ttl"] -= 1
        if memory.recent_pick_fail["ttl"] <= 0:
            memory.recent_pick_fail = None

    # 최종 상태 반영.
    memory.delivered_count = new_delivered
    memory.held_color = new_held
    memory.stage = "deliver_cube" if new_held else "need_cube"

    memory.logs.append({
        "observation": {
            "visible_count": len(observation.detections),
            "note": observation.note,
            "delivered_count": memory.delivered_count,
            "held_color": memory.held_color,
        },
        "llm_decision": decision.__dict__,
        "memory": {
            "active_color": memory.active_color,
            "completed_colors": list(memory.completed_colors),
            "skipped_colors": list(memory.skipped_colors),
            "failed_attempts": dict(memory.failed_attempts),
            "search_turns": memory.search_turns,
            "route_stats": {
                k: v for k, v in memory.route_stats.items() if k not in {"t0", "start_pose"}
            },
            "pad_routes": {
                c: len(e.get("successful_routes", [])) for c, e in memory.pad_memory.items()
            },
            "stall_spots": len(memory.stall_spots),
        },
        "verified": verified,
    })

# ---------------------------------------------------------------------------
# LEVEL 2 학생 TODO: vision-only action 구현
# ---------------------------------------------------------------------------
# Level 2는 go_to를 호출하면 안 됩니다. Camera observation, set_head,
# set_velocity, memory, recovery behavior로 navigate하세요.

# 아래 상수는 vision-only navigation의 튜닝 값입니다. 팀 전략에 맞게 조정하세요.
CENTER_TOLERANCE_DEG = 10.0   # 이 각도 안이면 target이 화면 중앙에 있다고 봅니다.
CUBE_ARRIVAL_AREA = 9000      # cube blob이 이만큼 크면 pick할 만큼 가깝다고 봅니다.
PAD_ARRIVAL_AREA = 20000      # pad blob이 이만큼 크면 place할 만큼 가깝다고 봅니다.
MIN_TARGET_AREA = 300         # 이보다 작은 blob은 noise로 무시합니다.
NAV_MAX_STEPS = 14            # navigation 한 번에서 최대 servo step 수.
SEARCH_MAX_ROTATIONS = 8      # search에서 body를 회전시키는 최대 횟수(대략 한 바퀴).

# --- perception noise 방어용 상수 ---
# 주의: 이 값들은 "이 씬의 좌표"가 아니라 "cube/pad라는 물체의 성질"에서 유도합니다.
# 그래서 시작 위치가 바뀌는 히든 평가에서도 성립하고, hardcoding에 해당하지 않습니다.
MAX_AREA_ARRIVAL_MULT = 4.0  # 단일 target은 도착 크기의 이 배수를 넘을 수 없음(컨베이어 레일/바닥 배제).
MIN_TARGET_ASPECT = 0.4      # cube/pad는 대략 정사각. 길고 얇은 blob(레일)은 배제.
MAX_TARGET_ASPECT = 2.5      # 정상 target이 걸러지면 이 범위를 넓히세요.
MAX_TARGET_WIDTH_FRAC = 0.6  # cube/pad는 프레임 폭의 이 비율을 넘지 않음(가로로 긴 컨베이어 레일 밴드 배제).

# --- "깨끗한 큐브" 판별 상수 ---
# cube면은 bbox를 꽉 채운 정사각으로 보입니다. 관측상 실제 큐브면 fill 0.92~0.93, aspect~1.0;
# 벨트/반사/구조물은 fill 0.45~0.67, aspect 1.3~1.5. 아래 값은 그 물리적 간극에서 유도합니다(씬 좌표 아님).
CLEAN_FILL_MIN = 0.80        # 이 이상 채워져야 단단한 큐브면으로 인정.
CLEAN_ASPECT_MIN = 0.7       # 큐브면은 거의 정사각. 가로로 긴 레일 밴드(≈1.3~1.5)를 배제.
CLEAN_ASPECT_MAX = 1.4
NAV_RESELECT_AREA_RATIO = 1.3  # 획득 nav에서 다른 큐브로 target을 바꾸려면 이 배수 이상 커야 함(진동 방지).
PICK_FAIL_RELOCATE = 2       # pick이 이만큼 연속 실패하면 크게 relocate해 다른 큐브가 최근접이 되게 함.
PICK_READY_AREA = 7000       # clean 큐브가 이만큼 크면(≈충분히 가까우면) 중앙정렬 없이 pick 발사.
                             # pick_entity는 각도 무관 최근접 큐브를 스스로 파지함(라이브 -26°/area~8000 즉시 성공).
                             # 학습 정책이 짧은 회전으로 큐브를 중앙 정렬하지 못하므로(ramp-up) 정렬을 요구하지 않음.

# --- 회전/pad-nav 상수 (학습 locomotion 정책 실측 기반) ---
# 이 로봇의 학습 보행 정책은 제자리 yaw 스핀이 포화되어 거의 안 돕니다(실측 ~3°/s, 큰 wz는 무효).
# 방향전환은 반드시 아크(vx>0 + wz)로 합니다(실측 vx0.3/wz1.2 ≈ 34°/s). 아래 값은 그 실측에서 유도합니다.
ARC_VX = 0.25                # 아크 조향 시 전진 속도(회전을 살리기 위한 최소 vx).
ARC_WZ = 1.2                 # 아크 조향 시 yaw rate(양수=좌회전).
SWEEP_VX = 0.2               # target을 잃었을 때 걸으며 시야를 훑는 아크의 전진 속도.
SWEEP_WZ = 1.0               # 스윕 아크의 yaw rate.
FORWARD_VX = 0.6             # pad 접근(stall 감지 있음) 직진 속도. 0.4→0.6: 워크숍01 검증 데모
                             # (vx0.8×3s≈2m) 범위 안. 기대 이동량 모델(_expected_advance_m)이 함께 스케일.
NAV_CUBE_VX = 0.4            # ★큐브 접근 전용 속도. pad 접근과 상수를 공유하면(FORWARD_VX 상향 시)
                             # 고정 0.8s 청크 이동량이 +50%가 되어 큐브 도착 판정(0.4 튜닝)을 지나쳐
                             # pick을 놓친다(검증 지적). 이 경로는 stall 감지도 없어 별도 속도로 고정.
VLM_MIN_CONFIDENCE = 0.5     # 이 미만 신뢰도의 sign은 무시하고 계속 탐색합니다.
# VLM 지연(6~32s/회)이 pad-nav 병목입니다. signage는 큰 글자(A~E)만 읽으면 되어 저해상도에
# 견고하므로, signage 판독 프레임을 기본(800px/q70)보다 줄여 업로드+vision 토큰을 낮춰 호출
# 시간을 단축합니다. 단 far C(작게 잡힘)를 너무 낮추면 놓칠 수 있어 중간값에서 시작 — 라이브로
# 지연 대 검출률을 측정해 조정하세요(더 내리면 빠르지만 C 놓칠 위험, compress_jpeg는 수정 가능).
SIGNAGE_VLM_MAX_WIDTH = 640  # 기본 800 → 640 (프레임 면적 ~36%↓).
SIGNAGE_VLM_QUALITY = 60     # 기본 70 → 60 (JPEG 품질↓로 payload↓).
PAD_SEARCH_TURN_DEG = 55.0   # head 스캔에서 sign을 못 찾으면 body를 이만큼 돌려 새 구역을 스캔.
PAD_TURN_TOL_DEG = 8.0       # 폐루프 회전이 목표 yaw 이 오차 안에 들면 종료.
HEAD_SCAN_YAWS_RAD = (0.0,)  # look은 1회 VLM(center)만. VLM이 매우 느려(~10-60s) 여러 각 스캔은 비쌈;
                             # 못 찾으면 전진으로 시야를 바꾸는 편이 회전보다 occlusion을 잘 품.
PAD_POS_OFFSET_DEG = 15.0    # VLM의 left/right를 대략 이 각도(도)로 환산해 body-bearing 계산.
PAD_FACE_TOL_DEG = 8.0       # 목표 방향이 이 안이면 이미 마주봤다고 보고 접근으로 넘어감.
                             # 반드시 PAD_POS_OFFSET_DEG보다 작아야 함: 같으면(둘 다 15) 'left'/'right'
                             # 검출의 face_turn=±15가 tol을 못 넘어(>15 거짓) 영영 회전하지 않음(라이브 확인).
PAD_OUTER_MAX = 18           # look→(전진/회전 or face+접근) 시도 최대 횟수. anchor goal-seek는
                             # VLM을 생략해 반복이 싸므로(~5s), 벽 통과 탐색 여유를 넉넉히 둡니다.
PAD_NAV_BUDGET_S = 150.0     # ★한 번의 visual_navigate_to_pad 벽시계 상한(초). 라이브 확정:
                             # 벽에 고착된 nav 하나가 900초를 통째로 태워 배송 0. 이 시간을 넘기면
                             # False 반환 → 상위가 recover(자세 변경)로 '다른 접근각'을 시도한다.
                             # 제자리 18회 재시도보다 새 pose 재시도가 벽 통과에 낫다.
# --- 색 우선 pad 항법 상수(색으로 접근, VLM은 글자 검증에만) ---
PAD_COLOR_SCAN_TURNS = 7      # 목표색이 안 보일 때 VLM 없이 회전 탐색할 횟수(≈한 바퀴).
PAD_VERIFY_MAX = 3           # 색 후보 도착했으나 VLM 글자 검증이 실패한 누적 상한(소스/오검출).
PAD_MAX_ARRIVAL_MULT = 2.5   # pad 후보 면적 상한(도착크기의 배수). 초과 blob은 들고 있는 큐브/
                             # 소스 표면 같은 거대 동색 물체 → pad 후보에서 제외(색 오인 차단).
PAD_PLACE_TRIES = 4          # C 확인 후 배송(delivered↑)될 때까지 place를 반복 시도할 최대 횟수
                             # (사용자 요구: 매번 place). C가 검증된 자리라 반복 place는 오배송-안전.
SOURCE_RETURN_MIN_M = 1.2    # 배송 후 다음 큐브를 못 찾으면 '큐브 집었던 자리'(소스)로 복귀 —
                             # 이 거리보다 멀 때만 복귀 이동(가까우면 제자리 스윕으로 국소 탐색).
                             # 자기 odometry pose 기준이라 랜덤 시작에도 성립(하드코딩 좌표 없음).
PAD_ADVANCE_DUR = 1.4        # 전진 한 청크의 길이(초). 시야 변경·pad 접근 공용.
PAD_FWD_BEFORE_TURN = 3      # 못 찾을 때 이 횟수-1 만큼 전진하고 매 이 횟수째에 회전.
TURN_MAX_ARCS = 5            # 폐루프 회전 한 번에서 최대 아크 명령 수(무한루프 방지).
# --- 회전 보장 상수(라이브 결함: 회전 미달·발진·막힘 무감지 = '명령해도 안 도는' 증상) ---
TURN_BLOCKED_MIN_DEG = 3.0   # 아크당 실측 회전이 이 미만이면 '회전 막힘'(벽 접촉 등)으로 판정.
TURN_RATE_INIT_DPS = 25.0    # 실효 회전률 EMA 초기값(도/초, 램프업 포함 실측 평균).
TURN_RATE_MIN_DPS = 8.0      # EMA 클램프 하한(장애물 스침 아크가 EMA를 붕괴시키지 못하게).
TURN_RATE_MAX_DPS = 45.0     # EMA 클램프 상한(odometry 노이즈 폭주 방지).
TURN_FINE_DEG = 18.0         # 이 미만 잔여각은 저속 아크로 미세 보정(최소 아크 ~25°의 발진 방지).
TURN_FINE_WZ = 0.35          # 미세 보정용 낮은 yaw rate(≈20°/s) — 과회전 구조적 차단.

# --- VLM 응답 정규화·비례 조향 상수 ---
# qwen 계열 VLM은 같은 프롬프트에도 응답 키 스키마가 흔들립니다(라이브 실측 5형:
# letter / sign_letter / label / text / text_content, label에는 글자 대신 'sign letter'
# 같은 서술어가 오기도 함). 아래 키 후보를 순서대로 훑되 "단일 알파벳" 값만 글자로 인정합니다.
VLM_LETTER_KEYS = ("letter", "sign_letter", "text_content", "text", "label")
# 라이브 실측: qwen이 글자를 위 키가 아니라 서술 문장에만 싣기도 함
# (label="sign", description="green square sign with white letter 'C'"). 이 키의 문장에서
# 'letter X' 뒤 글자나 단일 대문자 토큰을 별도로 추출합니다(_letter_from_phrase).
VLM_LETTER_DESC_KEYS = ("description", "desc", "caption")
VLM_DEFAULT_CONFIDENCE = 0.75  # 글자+bbox는 있는데 confidence 결측인 응답(라이브 실측)에 부여.
                               # 게이트(0.5)는 넘기되 명시적 high(0.9)보다는 낮게.
VLM_BBOX_SCALE = 1000.0        # qwen bbox_2d는 0~1000 정규화 좌표(640px 프레임 응답에서
                               # x=723 관측 → 픽셀 좌표 아님). 범위를 벗어나면 무시하고 fallback.
CAMERA_HFOV_DEG = 60.0         # perception.py의 angle 규약(±30° half-FOV)과 동일한 수평 화각.
                               # bbox 중심 x → 방위각 비례 환산에 사용(±15° 양자화 제거).

# --- 전진 stall(막힘) 감지·우회 상수 (라이브 실측 기반) ---
# fwd 0.4/1.4s의 정상 병진 ≈0.37m(실측 → 실효 속도 FORWARD_EFF_SPEED_MPS). 학습 정책은
# 구조물(source ledge 등)에 막혀도 명령을 수용해 병진 0·회전만 남고, navpad는 이를 몰라
# 제자리 배회했습니다(라이브 확정: x≈1.1 고착). 전진 전후 odometry 거리(상대량)를
# 기대 이동량(속도×시간 운동 모델)과 비교한 '이동 효율'로 stall을 감지해 우회합니다.
FORWARD_EFF_SPEED_MPS = 0.27  # (구) vx=0.4 전용 실효 속도 — _expected_advance_m가 대체(참조용으로 유지).
FWD_EFF_RATIO = 0.83          # 정상상태 병진 효율(워크숍01: 0.8×3.0s≈2.0m에서 유도).
FWD_RAMP_S = 0.35             # 램프업 병진 손실 근사(실측 0.4×1.4s=0.37m와 정합: 0.83×0.4×1.05≈0.35m).
PAD_ADVANCE_FAR_DUR = 2.8     # anchor 원거리 장청크 — 램프업 1회로 2청크 거리(움찔 1회 제거).
PAD_FAR_DIST_M = 2.5          # anchor가 이보다 멀면 장청크 허용(근거리는 1.4s 유지 = 정지 정밀도).
STALL_EFF_RATIO = 0.3         # 실제/기대 이동 효율이 이 미만이면 stall(옛 0.11m/0.37m 게이트와 동치).
STALL_ABS_FLOOR_M = 0.04      # odometry 노이즈 하한(이 미만 병진은 어떤 기대치든 stall).
PAD_STALL_BACKUP_S = 0.8      # stall 시 후진 시간(우회 공간 확보).
PAD_STALL_DETOUR_DEG = 50.0   # stall 시 우회 회전각(옆걸음 실패 시의 '폴백' 전용으로 강등).
# --- 진짜 옆걸음(vy 게걸음): 막힘 탈출 1차 수단 ---
# 기존 우회는 몸을 돌려서(±50/80°) 패드를 시야에서 잃고 느렸다(사용자 지적: "옆걸음이 약해").
# vy(측면속도, 문서상 +=좌, 클립 |vy|<=1.5)로 '몸 방향을 유지한 채' 옆으로 미끄러지면 패드가
# 계속 정면에 남고, 회전 우회보다 강하고 빠르다. 안 먹히면(정책 미지원/옆도 막힘) 회전으로 폴백.
STRAFE_VY = 0.8               # 옆걸음 측면속도(+=좌). 클립 1.5의 절반 남짓 — 강하되 낙상 마진 확보.
STRAFE_DUR = 1.3             # 옆걸음 한 스텝 시간(초).
STRAFE_MIN_M = 0.15          # 이 미만 이동이면 옆걸음이 안 먹힌 것 -> 회전 우회 폴백.
STRAFE_VX_ASSIST = 0.0       # 옆걸음 중 전진 성분(0=순수 측면. 앞이 막혔으니 전진은 재충돌 위험).
STRAFE_MAX_STEPS = 5         # ★한 번의 막힘 탈출에서 '같은 방향으로' 옆걸음할 최대 횟수.
                             # 사용자 지적: 몇 번만 더 가면 장애물 끝인데 전진 막혔다고 방향을
                             # 뒤집어 제자리걸음이 됐다. 옆걸음이 먹히는 한 방향을 유지하고
                             # 매 스텝 전진이 뚫리는지 확인 → 뚫리면 종료, 방향 전환은 옆도 막힐 때만.
# 측면 우회(lateral bypass): 짧은 detour로도 못 뚫는 선형 구조물(벨트 등)에 반복해 막히면,
# 표지 재조준을 잠시 멈추고 목표 쪽으로 ~90° 꺾어 여러 청크를 '따라 이동'해 구조물의 끝/틈을
# 지나갑니다(표준 bug-following, 카메라·odometry만). 라이브 확정: pad가 벨트 너머라 직진
# 접근만으론 x≈1.1에서 영구 고착 -> R6 비수렴 신호와 함께 escalate.
PAD_BYPASS_STALL_TRIGGER = 2  # 연속 hard-stall(직진+detour 모두 실패)이 이만큼이면 측면 우회 발동.
PAD_BYPASS_TURN_DEG = 80.0    # 측면 우회 시 목표 쪽으로 꺾는 각(구조물과 대략 평행하게 이동).
PAD_BYPASS_MAX_CHUNKS = 6     # 한 번의 측면 우회에서 따라 이동할 최대 옆걸음 스텝(escalate 상한). 4→6.
STALL_SPOT_RADIUS_M = 0.45    # 기억된 stall 지점 반경 — 이 안에서 같은 방향 전진이면 선제 우회.
STALL_HEADING_TOL_DEG = 40.0  # stall '같은 방향' 판정 폭. 60→40: 목표가 45° 옆인 전진까지 동일
                              # 방향으로 묶여 선제 봉쇄되던 것 방지(라이브: anchor +45°에서 -50° 우회).
# --- 최후 탈출(왔던 길 후진) 상수: 전진·우회·측면 우회 전부 막혔을 때의 보장 탈출 ---
RETREAT_MAX_M = 1.0           # 후진 총 거리 상한(후방 카메라 없음 → 짧게).
RETREAT_CHUNK_S = 1.2         # 후진 청크 길이(청크마다 odometry로 병진 검증).
RETREAT_BACK_TOL_DEG = 45.0   # 탈출 목표가 '정후방'으로 인정되는 각 허용치.

# --- 경로 기억(route memory)·last_seen·VLM 절감 상수 ---
# 성공한 배송 경로를 점수화해 저장하고 다음 같은 색 배송에서 greedy 재사용하는 online
# heuristic 최적화입니다(엄밀한 RL 학습이 아니라 성공/실패 경험 축적형 탐욕 선택).
# 모든 좌표는 로봇 자신의 odometry(고유수용성) 기준 — scene_state가 아니며, 한 run
# 안에서만 유효합니다(프로세스 간 영속 캐시 없음 = 특정 setup hardcoding 아님).
ADVANCE_MIN_S = 0.7            # 램프업 미달로 거의 안 걷는 초단시간 전진 명령 방지 하한.
WAYPOINT_TOL_M = 0.4           # waypoint 도달 판정 반경(유클리드 거리).
ROUTE_REPLAY_CHUNKS_PER_WP = 4 # waypoint당 최대 전진 청크 수(초과 시 replay 중단).
ROUTE_MIN_WAYPOINT_GAP_M = 0.5 # waypoint 압축 최소 간격(기록 폭증·미세 진동 방지).
LAST_SEEN_MAX_REUSE = 2        # last_seen 연속 재조준 상한(초과 시 VLM 재확인 강제).
LAST_SEEN_MAX_DRIFT_M = 2.5    # 목격 pose에서 이만큼 멀어지면 ray(방향) 가정을 불신.
ROUTE_SCORE_VLM_W = 10.0       # 경로 점수: score = time + vlm*10 + stall*5 + path*1.5.
ROUTE_SCORE_STALL_W = 5.0      #   (낮을수록 좋음. VLM 호출이 지배 비용이라 가장 무겁게.)
ROUTE_SCORE_PATH_W = 1.5
FAILED_ROUTES_KEEP = 3         # 실패 경로 기록 보관 상한(진단용).
DETOUR_WIN_KEEP = 20           # 성공 우회 방향 기록 보관 상한.

# --- R6: 접근 수렴 판정(변화율 proxy) 상수 — 현재 '관측 모드'(로그·기록만, 행동 불변) ---
# 단안이라 pad 거리 d를 직접 못 재므로, 원근 투영의 면적 ∝ 1/d² 관계로 색블롭 면적의
# 변화율을 거리 변화율의 단조 대용으로 씁니다. 라이브(R4)에서 임계를 보정한 뒤에만
# 행동(전략 변경) 트리거로 승격합니다.
APPROACH_MIN_SAMPLES = 4         # 수렴 판정 최소 표본 수(전/후반 중앙값 비교가 성립하는 최소).
APPROACH_AREA_GROWTH_MIN = 1.15  # 후반 면적 중앙값이 전반의 이 배수 이상이면 '접근 중'.
                                 # 면적 ∝ 1/d²: 5m 거리에서 한 청크(≈0.37m) 전진 시 (5/4.63)² ≈ 1.17
                                 # — 먼 거리에서도 청크당 증가분을 감지하는 하한에서 유도.

# --- pad anchor(목격 기반 대략적 위치 기억) 상수 ---
# last_seen ray(방향만)의 한계: 목격 pose에서 벗어나거나 몸이 많이 회전하면 재조준각이
# ±160° 같은 발산 회전을 낳습니다(라이브 확정: x=2.63까지 진출 후 스핀→전도). bbox 면적으로
# 거리까지 추정해 sign을 world '점'으로 기억하면 재조준각을 매 반복 '현재 pose'에서 새로
# 계산하므로(폐루프) 회전·이동으로 낡지 않습니다 — 사람이 목적지를 한 번 보면 대략적 위치를
# 기억해 두고, 그 뒤로는 안 보여도 장애물을 우회하며 그 방향으로 가는 방식(고전 Bug 알고리즘의
# goal + 국소 회피)과 같습니다. 점 좌표는 카메라 관찰(방위+크기)을 자기 odometry 프레임에
# 투영한 학생 추정치일 뿐 scene_state가 아닙니다(Level 2 합법).
PAD_SIGN_DIST_K = 0.34          # d ≈ K/√area_frac(면적 ∝ 1/d²의 역산). 라이브 실측 far-C bbox
                                # (area_frac≈0.0093, 당시 pad까지 ~3.5m 추정)에서 유도한 잠정값.
                                # 과소추정(짧게 멈춤→가까운 재목격으로 자가보정)이 과대추정(구조물
                                # 돌진)보다 안전해 낮은 쪽을 택함 — 라이브 로그의 d 추정치로 보정하세요.
PAD_ANCHOR_MIN_D = 0.8          # 거리 추정 하한(과대 bbox가 0m 근처 추정을 내는 것 방지).
PAD_ANCHOR_MAX_D = 6.0          # 거리 추정 상한(원거리 미소 bbox의 노이즈 폭주 방지).
PAD_ANCHOR_MIN_CONF = 0.6       # anchor 융합 신뢰도 게이트 — nav 게이트(0.5)보다 엄격하게 잡아
                                # 낮은 확신의 오검출(A를 C로 오독 등)이 점 추정을 오염시키지 않게 함.
PAD_ANCHOR_OUTLIER_M = 2.0      # n≥2로 자리잡은 anchor 평균에서 이 이상 벗어난 새 목격은 기각
                                # (한 번의 오독이 평균을 끌고 가는 것 방지; Mahalanobis-lite).
PAD_ANCHOR_NEAR_M = 1.2         # anchor 근접 반경 — 이 안이면 goal-seek 대신 VLM 확인으로 전환
                                # (점 추정 오차 안이므로 더 걸어봤자 목표를 지나칠 뿐).
PAD_ANCHOR_MAX_REUSE = 5        # anchor 연속 goal-seek 상한 — 초과 시 VLM 재확인(오염 방지).
PAD_ANCHOR_NEAR_MISS_LIMIT = 3  # anchor 근접 VLM 연속 미검출 → anchor 폐기(오염 자가치유).
PAD_PLACE_GATE_M = 2.0          # ★anchor가 이보다 멀면 색blob 도착판정·confirm VLM을 기각.
                                # 면적∝1/d²: 진짜 도착이면 anchor_dist ≤ 실거리 1.5m+오차.
                                # 4m급 모순 blob = 초록 컨베이어/동색 오브젝트(라이브 pad7/8 확정).
PAD_GATE_VETO_MAX = 4           # 연속 veto 상한 — anchor 오염 코너케이스에 confirm 재개 기회.
PAD_ANCHOR_STALE_S = 90.0       # anchor 마지막 갱신 후 이 시간이 지나면 '낡음'(예외 허용 조건).
CONFIRM_COOLDOWN_CHUNKS = 2     # confirm 실패 후 이만큼 전진 청크 소화 전 재confirm 금지(VLM 스팸 차단).
BELT_CONFUSABLE_FILL_MIN = 0.70 # held색==벨트색일 때 도착 blob 최소 fill(실측: 벨트 0.45~0.67, 큐브면 0.92+).
SOURCE_EXCLUSION_RADIUS_M = 1.5 # 이번 배송 pick 지점 반경 — 이 안의 색blob 도착판정은 무효(source 오인).
PAD_WIDE_SCAN_YAWS_RAD = (-0.8, 0.0, 0.8)  # 첫 탐색 전용 head 팬(커버 ~150°, locomotion 0).
PAD_ANCHOR_MAX_REAIM_DEG = 90.0  # 한 반복의 anchor 재조준각 상한 — 점 기반이라 발산하진 않지만
                                 # 큰 후방 회전을 두 반복에 나눠(회전→전진→재평가) 전도 위험을 줄임.
PAD_ANCHOR_W_CAP = 6.0          # 융합 가중치 누적 상한 — 옛 목격 더미가 새(더 가까운) 목격을
                                # 압도하지 못하게 해 잘못 초기화된 anchor도 재목격으로 씻겨나감.
# --- 국소 탐색(anchor 근접 폐기 직후 '그 자리' 훑기)·복귀 게이트·중간 재확인 상수 ---
# 라이브 회귀: 화면 끝 단일 목격으로 잘못 박힌 anchor로 0.74m까지 접근 → 폐기 → 그 순간
# '목격 지점 복귀'가 출발점으로 173°·2.9m 되돌아가 7.2m 진전을 통째로 버림(배송 0).
LS_RETURN_MAX_TURN_DEG = 90.0   # 복귀가 이보다 크게 돌아야 하면 '진전 되돌리기' → 복귀 금지, 국소 탐색.
LS_RETURN_MAX_BACK_M = 2.0      # 복귀 후진 거리 상한 → 초과면 복귀 대신 국소 탐색.
PAD_LOCAL_SEARCH_MAX = 2        # nav당 국소 탐색 발동 상한(무한 재탐색 방지).
PAD_LOCAL_SEARCH_VLM_MAX = 4    # 국소 탐색 1회의 VLM 콜 상한(광각 3 + 여유 1). VLM은 지배 비용.
PAD_LOCAL_SPIRAL_STEPS = 3      # 재획득 실패 시 훑을 나선 스텝 수(옆걸음+전진 1쌍씩, 반경 1~2m).
ANCHOR_RECONFIRM_DIST_M = 2.0   # 단일 목격(n<2) anchor로 접근하다 이 거리에 처음 닿으면 VLM 1회
                                # 재확인해 anchor 교정(가까울수록 거리추정 정확) — 0.74m까지 틀린 채
                                # 가서야 폐기하던 것 방지.


def _frame_width_from(detection: Any) -> float | None:
    """detection의 centroid.x와 angle_deg로 카메라 프레임 폭(px)을 역산합니다.

    perception은 angle_deg = (cx - W/2)/(W/2)*HFOV_HALF_DEG(30) 로 각도를 계산하므로,
    이를 뒤집으면 W = 60*cx/(angle_deg + 30) 입니다. 해상도를 하드코딩하지 않고
    관찰값에서 유도하므로 시작 포즈가 바뀌는 히든 평가에서도 성립합니다.
    """
    cx = detection.centroid[0]
    denom = detection.angle_deg + 30.0  # == HFOV_HALF_DEG
    if denom <= 0 or cx <= 0:
        return None
    return 60.0 * cx / denom


def _plausible_target(detection: Any, arrival_area: int) -> bool:
    """color blob이 '진짜 cube/pad'로 보이는지 검사합니다(카메라 기반, scene_state 미사용).

    네 가지 물리 prior로 환경 노이즈를 배제합니다:
    - 면적 하한: MIN_TARGET_AREA 미만은 noise.
    - 면적 상한: 단일 target은 도착 크기의 몇 배를 넘길 수 없음 → 화면을 뒤덮는
      컨베이어 파란 레일/바닥 같은 초대형 blob을 배제.
    - aspect(가로/세로): cube/pad는 대략 정사각. 길고 얇은 레일은 배제.
    - 폭 비율: cube/pad는 프레임 폭의 일부만 차지. 프레임을 가로지르는 레일 밴드를 배제
      (정사각으로 잡혀 aspect·면적을 통과하는 레일 조각까지 걸러냄).
    """
    if detection.blob_area < MIN_TARGET_AREA:
        return False
    if detection.blob_area > arrival_area * MAX_AREA_ARRIVAL_MULT:
        return False
    _, _, width, height = detection.bbox
    if height <= 0:
        return False
    aspect = width / height
    if not (MIN_TARGET_ASPECT <= aspect <= MAX_TARGET_ASPECT):
        return False
    frame_width = _frame_width_from(detection)
    if frame_width is not None and width / frame_width > MAX_TARGET_WIDTH_FRAC:
        return False
    return True


def _fill_ratio(detection: Any) -> float:
    """blob_area / bbox 넓이. 꽉 찬 정사각 큐브면은 1에 가깝고, 벨트/반사/구조물은 낮습니다."""
    _, _, width, height = detection.bbox
    if width <= 0 or height <= 0:
        return 0.0
    return detection.blob_area / (width * height)


def _is_clean_cube(detection: Any, arrival_area: int) -> bool:
    """detection이 '고립된 단단한 큐브면'으로 보이는지 검사합니다(색 무관).

    _plausible_target(노이즈/레일 배제)에 더해 두 물리 prior를 요구합니다:
    - fill(면적/bbox) >= CLEAN_FILL_MIN: 큐브면은 bbox를 꽉 채움. 벨트/반사는 성깁니다.
    - aspect가 거의 정사각(CLEAN_ASPECT_MIN..MAX): 큐브면은 ~1.0, 가로로 긴 레일은 1.3~1.5.
    """
    if not _plausible_target(detection, arrival_area):
        return False
    _, _, width, height = detection.bbox
    if height <= 0:
        return False
    aspect = width / height
    if not (CLEAN_ASPECT_MIN <= aspect <= CLEAN_ASPECT_MAX):
        return False
    return _fill_ratio(detection) >= CLEAN_FILL_MIN


def _detect_belt_color(detections: list[Any]) -> str | None:
    """프레임을 뒤덮는 초대형 수평 구조물(컨베이어 벨트/레일)의 색을 런타임에 추정합니다.

    벨트는 큐브 도착 크기의 몇 배를 넘는 가로로 긴 blob으로 나타납니다. 이 색은 '제외'가
    아니라 '후순위'로만 쓰므로(획득 시 다른 색을 먼저 치움) 오탐이 나도 치명적이지 않습니다.
    특정 씬 좌표가 아니라 '거대·수평' 물리 성질에서 유도하므로 히든 평가에서도 성립합니다.
    """
    oversized = [
        d for d in detections
        if d.blob_area > CUBE_ARRIVAL_AREA * MAX_AREA_ARRIVAL_MULT
        and d.bbox[2] >= d.bbox[3]  # 가로 >= 세로 (수평 밴드)
    ]
    if not oversized:
        return None
    return max(oversized, key=lambda d: d.blob_area).color


def _select_acquire_target(
    candidates: list[Any],
    belt_color: str | None,
    locked_color: str | None,
) -> Any | None:
    """획득 모드에서 다가갈 큐브를 고릅니다(벨트색 후순위 + 진동 방지 hysteresis)."""
    if not candidates:
        return None
    # 벨트색이 아닌 큐브를 우선, 그다음 큰(가까운) 순.
    best = max(candidates, key=lambda d: (d.color != belt_color, d.blob_area))
    # 이미 어떤 색을 추적 중이면, 새 후보가 충분히(NAV_RESELECT_AREA_RATIO배) 더 커야만
    # target을 바꿔 등거리 큐브 사이에서 프레임마다 흔들리는 것을 막습니다.
    if locked_color is not None and best.color != locked_color:
        locked_now = [d for d in candidates if d.color == locked_color]
        if locked_now:
            locked_best = max(locked_now, key=lambda d: d.blob_area)
            if best.blob_area < locked_best.blob_area * NAV_RESELECT_AREA_RATIO:
                return locked_best
    return best


async def _clean_cube_ready(ctx: Any) -> tuple[bool, str | None]:
    """pick 준비 여부(ready)와 color-blind pick이 실제로 집을 큐브 색을 함께 반환합니다.

    color-blind pick_entity는 부채꼴(콘)과 무관하게 3D 최근접 큐브 엔티티를 잡습니다. 그래서
    '집을 색'은 콘을 무시하고 전체에서 가장 크게(=가장 가깝게) 보이는 '깨끗한' 큐브의 색으로
    리포트해 실제 pick 결과와 일치시킵니다(옛 콘-내 최댓값 리포트는 pick이 콘 밖 큐브를 잡을 때
    라벨이 어긋났음 — 배송엔 무해했지만 진단이 오해를 부름). ready 게이트는 '충분히 가까운
    (area >= PICK_READY_AREA) 깨끗한 큐브'가 보이는지만 봅니다(콘/중앙 정렬 요구 없음 — 아래 주석).
    실제 잡은 색은 pick 뒤 get_held_cube_info로 최종 확정하므로, 이 색은 어디까지나 예측/진단용입니다.
    """
    clean = [d for d in await perceive(ctx) if _is_clean_cube(d, CUBE_ARRIVAL_AREA)]
    if not clean:
        return (False, None)
    grab = max(clean, key=lambda d: d.blob_area)  # 콘 무관 최근접 ≈ pick_entity가 잡을 큐브.
    # 중앙 정렬을 요구하지 않습니다: pick_entity는 각도 무관하게 최근접 큐브를 스스로 파지하고
    # (라이브에서 -26°/area~8000 즉시 성공), 이 학습 정책은 짧은 회전으로 큐브를 중앙에 정렬하지
    # 못합니다(ramp-up으로 제자리 맴돌기만 함). 그래서 '충분히 가까운(area) clean 큐브'면 준비 완료.
    ready = grab.blob_area >= PICK_READY_AREA
    return (ready, grab.color)


async def _pad_color_blob(ctx: Any, target_color: str | None) -> Any | None:
    """목표 pad 색의 'pad다운' 최대 blob을 반환합니다(색 우선 항법의 눈).

    들고 있는 큐브·소스 표면도 같은 색이라 거대 blob으로 잡히므로(라이브: green 68879px),
    도착크기의 PAD_MAX_ARRIVAL_MULT배를 넘는 초대형은 pad 후보에서 제외합니다 — 그게 없으면
    '들고 있는 큐브 = pad'로 착각해 제자리 도착 판정이 납니다. 글자 오검출은 VLM 검증이 마저 거릅니다.
    """
    if target_color is None:
        return None
    cap = int(PAD_ARRIVAL_AREA * PAD_MAX_ARRIVAL_MULT)
    cands = [
        d for d in await perceive(ctx)
        if d.color == target_color and _plausible_target(d, PAD_ARRIVAL_AREA)
        and d.blob_area <= cap
    ]
    return max(cands, key=lambda d: d.blob_area) if cands else None


async def _best_color_blob(ctx: Any, target_color: str | None, arrival_area: int) -> Any | None:
    """target_color의 plausible blob 중 최대(≈최근접)를 반환합니다(없으면 None).

    _target_in_range의 인식 경로를 분리한 것 — R6 수렴 관측이 도착 게이트와 동일한
    인식으로 면적 표본을 얻게 하기 위함입니다(별도 인식이면 표본과 게이트가 어긋남).
    """
    if target_color is None:
        return None
    matching = [
        d for d in await perceive(ctx)
        if d.color == target_color and _plausible_target(d, arrival_area)
    ]
    return max(matching, key=lambda d: d.blob_area) if matching else None


async def _target_in_range(ctx: Any, target_color: str | None, arrival_area: int) -> bool:
    """target_color blob이 arrival_area 이상으로 크고 화면 중앙 근처에 보이면 True입니다."""
    best = await _best_color_blob(ctx, target_color, arrival_area)
    return (
        best is not None
        and best.blob_area >= arrival_area
        and abs(best.angle_deg) <= CENTER_TOLERANCE_DEG * 1.5
    )


async def visual_search(
    ctx: Any, target_color: str | None = None, *, memory: AgentMemory | None = None
) -> bool:
    """Camera movement와 robot motion으로 cube 또는 pad를 search합니다.

    TODO:
    - set_head 또는 body rotation을 사용하는 scan pattern을 설계하세요.
    - 필요하면 cube와 pad를 어떻게 구분할지 결정하세요.
    - Visual centering에 도움이 되면 detection.full_bearing_deg를 사용하세요.
    - 유용한 target을 찾았는지 반환하세요.
    """
    # cube를 찾는지(pick) pad를 찾는지(place)에 따라 크기 상한이 달라집니다.
    held = await get_held_cube_info(ctx)
    arrival_area = PAD_ARRIVAL_AREA if held else CUBE_ARRIVAL_AREA

    # body를 조금씩 회전시키며 매 방향마다 head를 훑어 넓게 search합니다.
    for _ in range(SEARCH_MAX_ROTATIONS):
        detections = await scan_head(ctx)
        # 이후 body-frame 조향을 위해 head를 정면으로 되돌립니다.
        await set_head(ctx, yaw=0.0, pitch=0.15)

        if target_color is None:
            # 획득: 색 고정 없이 '깨끗한' 큐브만 후보. 벨트색은 제외가 아니라 후순위(다른 색 먼저).
            belt = _detect_belt_color(detections)
            pool = [d for d in detections if _is_clean_cube(d, arrival_area)]
        else:
            # pad 탐색 등 색이 지정된 경우: 해당 색의 plausible blob만.
            belt = None
            pool = [d for d in detections if d.color == target_color and _plausible_target(d, arrival_area)]
        if pool:
            best = max(pool, key=lambda d: (d.color != belt, d.blob_area))
            # full_bearing_deg(head yaw 포함)로 target 쪽으로 body를 대략 정렬합니다.
            # 제자리 회전은 학습 정책상 안 먹히므로 아크(vx>0+wz)로 살짝 걸으며 정렬합니다.
            bearing = best.full_bearing_deg
            if abs(bearing) > CENTER_TOLERANCE_DEG:
                wz = -ARC_WZ if bearing > 0 else ARC_WZ
                await move_velocity(ctx, vx=ARC_VX, wz=wz, duration_s=min(abs(bearing) / 40.0, 1.2))
            return True

        # 못 찾음: 획득 모드에서 '큐브 집었던 자리'(소스)를 알고 멀면 그쪽으로 복귀합니다 —
        # 큐브는 소스에 있으므로, 배송 후 멀리서 헛도는 대신 소스로 돌아가 찾습니다(멀티큐브).
        # 자기 pose 기준이라 랜덤 시작에도 성립. 소스 모름/근처면 기존 아크-스윕으로 시야만 이동.
        if target_color is None and memory is not None and memory.source_pose is not None:
            pose_now = await _get_pose(ctx)
            dist, turn = _face_turn_to(pose_now, memory.source_pose)
            if dist > SOURCE_RETURN_MIN_M:
                if abs(turn) > PAD_FACE_TOL_DEG:
                    await _turn_by_deg(ctx, turn)
                await _advance_or_detour(ctx, 1.0, memory=memory, goal_turn_deg=turn)
                continue
        # 이 방향에서 target을 못 찾았으면 아크-스윕(걸으며 회전)으로 시야를 옮긴 뒤 다시 훑습니다.
        await move_velocity(ctx, vx=SWEEP_VX, wz=SWEEP_WZ, duration_s=0.8)
    return False


async def visual_navigate_to_target(ctx: Any, target_color: str | None, *, verbose: bool = False) -> bool:
    """카메라 피드백만 사용해 cube 또는 pad 앞까지 폐루프 이동합니다.

    이 함수는 Level 2 규칙에 맞게 scene_state, 좌표, go_to 없이 동작합니다.
    매 step마다 현재 POV frame을 다시 인식하고, target의 화면상 각도와 blob 크기만으로
    `회전하며 전진할지`, `직진할지`, `도착으로 볼지`, `실패로 빠질지`를 결정합니다.

    동작 모드:
    - `target_color is None`: cube 획득 모드입니다. 색을 미리 고정하지 않고
      `_is_clean_cube()`를 통과한 깨끗한 큐브 후보 중 하나를 따라갑니다. 컨베이어 벨트/레일
      색은 `_detect_belt_color()`로 후순위 처리하고, `_select_acquire_target()`의
      hysteresis로 프레임마다 다른 큐브로 흔들리는 것을 막습니다.
    - `target_color is not None`: 색 지정 모드입니다. 주로 delivery pad 접근에 쓰며,
      해당 색의 `_plausible_target()` 후보 중 가장 큰 blob을 따라갑니다.

    도착 판정:
    - robot이 cube를 들고 있지 않으면 `CUBE_ARRIVAL_AREA`, 들고 있으면
      `PAD_ARRIVAL_AREA`를 기준으로 삼습니다.
    - blob 면적이 기준 이상이고, 동시에 `CENTER_TOLERANCE_DEG` 안에 들어와야
      도착으로 인정합니다. 면적만 보면 옆으로 지나가는 레일 조각도 도착으로 오판할 수
      있어서 중앙 조건을 함께 둡니다.

    반환값:
    - `True`: target이 충분히 크고 중앙에 보여서 다음 pick/place를 시도해도 되는 상태입니다.
    - `False`: target을 3회 연속 잃었거나 `NAV_MAX_STEPS` 안에 도착하지 못한 상태입니다.

    부작용:
    - `set_head()`로 시선을 정면에 맞춥니다.
    - `move_velocity()`로 짧은 전진/회전 명령을 반복합니다.
    """
    # cube를 향하는지(pick) pad를 향하는지(place)에 따라 도착 판정 크기가 다릅니다.
    held = await get_held_cube_info(ctx)
    arrival_area = PAD_ARRIVAL_AREA if held else CUBE_ARRIVAL_AREA

    # body-frame servoing을 위해 head를 정면으로 맞춥니다(이미지 각도≈몸통 방위).
    await set_head(ctx, yaw=0.0, pitch=0.15)

    lost_streak = 0
    locked_color: str | None = None
    for _step in range(1, NAV_MAX_STEPS + 1):
        raw = await perceive(ctx)
        if target_color is None:
            # 획득 모드: 색 고정 없이 '깨끗한' 큐브로. 벨트색은 후순위, hysteresis로 진동 방지.
            belt = _detect_belt_color(raw)
            matching = [d for d in raw if _is_clean_cube(d, arrival_area)]
            best = _select_acquire_target(matching, belt, locked_color)
            if best is not None:
                locked_color = best.color
        else:
            # 색 지정(주로 pad): 해당 색의 plausible blob 중 가장 큰 것.
            matching = [
                d for d in raw
                if d.color == target_color and _plausible_target(d, arrival_area)
            ]
            best = max(matching, key=lambda d: d.blob_area) if matching else None

        if best is None:
            # target loss: 아크-스윕으로 걸으며 재획득을 시도하고, 계속 못 찾으면 실패로 종료합니다.
            lost_streak += 1
            if verbose:
                print(f"  [nav {_step}] lost (streak={lost_streak}) -> sweep")
            if lost_streak >= 3:
                if verbose:
                    print("  [nav] FAIL: target 3연속 손실")
                return False
            await move_velocity(ctx, vx=SWEEP_VX, wz=SWEEP_WZ, duration_s=0.8)
            continue
        lost_streak = 0

        area = best.blob_area
        angle = best.angle_deg

        if area >= arrival_area and abs(angle) <= CENTER_TOLERANCE_DEG:
            # 도착: 충분히 가깝고 target이 화면 중앙에 있습니다. 중앙 조건이 없으면
            # 옆으로 지나가는 레일 밴드가 area만으로 조기 도착을 유발할 수 있습니다.
            if verbose:
                print(f"  [nav {_step}] ARRIVE {best.color} area={area} angle={angle:.1f}")
            return True

        if abs(angle) > CENTER_TOLERANCE_DEG:
            # 아직 중앙이 아니면 아크(vx>0+wz)로 target 쪽으로 선회합니다(제자리 회전 불가).
            wz = -ARC_WZ if angle > 0 else ARC_WZ
            if verbose:
                print(f"  [nav {_step}] arc  {best.color} area={area} angle={angle:.1f} wz={wz:+.1f}")
            await move_velocity(ctx, vx=ARC_VX, wz=wz, duration_s=0.5)
        else:
            # 중앙에 있으면 똑바로 전진합니다.
            if verbose:
                print(f"  [nav {_step}] fwd  {best.color} area={area} angle={angle:.1f}")
            await move_velocity(ctx, vx=NAV_CUBE_VX, duration_s=0.8)   # ★큐브 도착 판정(0.4)과 정합

    if verbose:
        print(f"  [nav] FAIL: {NAV_MAX_STEPS} step 내 도착 실패")
    return False


def _parse_signs(text: str) -> list[dict[str, Any]]:
    """VLM signage 응답에서 sign 목록을 견고하게 parse합니다.

    응답은 보통 [{letter, color, position, confidence}, ...] JSON이지만 코드펜스나
    설명이 섞일 수 있어, 첫 JSON 배열/객체만 추출해 해석합니다. 실패하면 빈 목록.
    """
    stripped = text.strip()
    if "```" in stripped:
        for part in stripped.split("```"):
            p = part.strip()
            if p.lower().startswith("json"):
                p = p[4:].strip()
            if p.startswith("[") or p.startswith("{"):
                stripped = p
                break
    starts = [i for i in (stripped.find("["), stripped.find("{")) if i >= 0]
    if not starts:
        return []
    start = min(starts)
    end = max(stripped.rfind("]"), stripped.rfind("}"))
    if end <= start:
        return []
    try:
        data = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        data = data["signs"] if isinstance(data.get("signs"), list) else [data]
    if not isinstance(data, list):
        return []
    # VLM(qwen)은 글자 키를 'letter'가 아니라 'sign_letter'/'label'/'text'/'text_content'로,
    # 위치를 'position'이 아니라 'approximate_position'으로 반환하기도 합니다(라이브 실측).
    # 다양한 키 별칭을 표준 키('letter','position','confidence')로 정규화해 하위 소비자가
    # 일관되게 읽게 합니다. confidence 없이 bbox+글자만 주는 형식도 실측됐으므로 결측 시
    # 기본 신뢰도를 부여해 검출이 게이트에서 통째로 기각되지 않게 합니다.
    normalized: list[dict[str, Any]] = []
    for s in data:
        if not isinstance(s, dict):
            continue
        letter = _extract_sign_letter(s)
        if not letter:
            continue
        out = dict(s)
        out["letter"] = letter
        out["position"] = s.get("position") or s.get("approximate_position") or ""
        if s.get("confidence") is None:
            # 결측·null confidence는 방위 근거(bbox나 position)가 있는 검출만 기본 신뢰도로
            # 구제합니다. 근거가 하나도 없는 글자-단독 검출(환각 가능)은 기본값을 주면
            # _sign_offset_deg가 0°(정면)로 fallback해 엉뚱한 직진을 유발하므로 게이트가
            # 거르도록 둡니다. ('not in s' 가드는 명시적 null을 통과시켜 유효 검출을 버렸음.)
            bbox = s.get("bbox_2d") or s.get("bbox")
            has_bearing = (
                isinstance(bbox, (list, tuple)) and len(bbox) == 4
            ) or bool(out["position"])
            if has_bearing:
                out["confidence"] = VLM_DEFAULT_CONFIDENCE
        normalized.append(out)
    return normalized


def _letter_from_phrase(text: str) -> str | None:
    """서술 문장에서 표지 글자 하나를 추출합니다("...white letter 'C'" → 'C').

    라이브 실측: qwen이 글자를 별도 키가 아니라 description 문장에만 싣고 인용부호로 감싸
    ('C') 단순 토큰 분리로는 못 잡는 경우가 있습니다. 인용부호·구두점을 공백으로 바꿔
    토큰화한 뒤 (1) 'letter' 바로 뒤 단일 알파벳, (2) 단일 '대문자' 토큰 순으로 봅니다.
    대문자를 요구해 관사 'a'/'an' 같은 소문자 단일 글자를 글자로 오인하지 않습니다.
    """
    for ch in "'\"-.,:;()[]":
        text = text.replace(ch, " ")
    tokens = text.split()
    for i, tok in enumerate(tokens[:-1]):
        if tok.lower() == "letter":
            nxt = tokens[i + 1]
            if len(nxt) == 1 and nxt.isalpha():
                return nxt.upper()
    for tok in tokens:
        if len(tok) == 1 and tok.isalpha() and tok.isupper():
            return tok
    return None


def _extract_sign_letter(s: dict[str, Any]) -> str | None:
    """sign dict에서 표지 글자(단일 알파벳)를 견고하게 추출합니다.

    qwen은 글자를 VLM_LETTER_KEYS 중 아무 키에나 싣고, label에는 글자 대신 'sign letter' 같은
    서술어가 올 수도 있습니다(라이브 실측). 단일 알파벳 값을 우선 인정하고, 'sign C' 같은 혼합
    표기에서는 단일 글자 토큰을 찾으며, 서술어만 있으면 글자로 오인하지 않고 버립니다.
    글자 키가 하나도 안 잡히면 description/desc/caption 서술 문장에서 마지막으로 시도합니다.
    """
    values = [str(s[k]).strip() for k in VLM_LETTER_KEYS if s.get(k)]
    for v in values:
        if len(v) == 1 and v.isalpha():
            return v
    for v in values:
        tokens = [t for t in v.replace("-", " ").split() if len(t) == 1 and t.isalpha()]
        if tokens:
            return tokens[0]
    for k in VLM_LETTER_DESC_KEYS:
        if s.get(k):
            letter = _letter_from_phrase(str(s[k]))
            if letter:
                return letter
    return None


def _as_confidence(v: Any) -> float:
    """VLM confidence를 float으로 안전 변환합니다.

    qwen은 confidence를 0.95 같은 수치뿐 아니라 'high'/'medium'/'low' 문자열로도 반환합니다
    (라이브 실측: `float('high')`가 navpad 전체를 ValueError로 크래시시킴). 문자열 등급을
    대표 수치로 매핑하고, 파싱 불가한 값은 0.0으로 떨궈 안전하게 무시합니다.
    """
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().lower()
    named = {"high": 0.9, "medium": 0.6, "med": 0.6, "low": 0.3}
    if s in named:
        return named[s]
    try:
        return float(s)
    except (TypeError, ValueError):
        return 0.0


def _find_target_sign(signs: list[dict[str, Any]], letter: str) -> dict[str, Any] | None:
    """parse된 sign 목록에서 목표 글자와 일치하는 sign을 confidence 최고로 고릅니다."""
    cands = [s for s in signs if str(s.get("letter", "")).strip().upper() == letter.upper()]
    if not cands:
        return None
    return max(cands, key=lambda s: _as_confidence(s.get("confidence", 0)))


def _sign_offset_deg(target: dict[str, Any]) -> float:
    """target sign의 화면상 수평 위치를 카메라 기준 방위 오프셋(도, +=우측)으로 환산합니다.

    bbox_2d(qwen 0~1000 정규화)가 있으면 중심 x의 화면 비율로 비례 환산합니다 — left/right
    ±PAD_POS_OFFSET_DEG 양자화는 far-left 표지(실측 방위 ~-27°)를 한 번에 15°만 보정해
    영영 정면을 못 맞춥니다(라이브 확정). bbox가 없거나 정규화 범위를 벗어나면(픽셀 좌표 등)
    기존 left/center/right 양자화로 fallback합니다.
    """
    bbox = target.get("bbox_2d") or target.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        try:
            x1, _, x2, _ = (float(v) for v in bbox)
        except (TypeError, ValueError):
            x1, x2 = -1.0, -1.0
        if 0.0 <= x1 < x2 <= VLM_BBOX_SCALE:
            frac = ((x1 + x2) / 2.0) / VLM_BBOX_SCALE
            return (frac - 0.5) * CAMERA_HFOV_DEG
    position = str(target.get("position", "center")).strip().lower()
    if "left" in position:
        return -PAD_POS_OFFSET_DEG
    if "right" in position:
        return PAD_POS_OFFSET_DEG
    return 0.0


# ---------------------------------------------------------------------------
# 학생 TODO: 경로 기억(route memory) 순수 헬퍼 — unit test 대상
# ---------------------------------------------------------------------------
# 아래 헬퍼들은 로봇 자신의 odometry pose(고유수용성)와 카메라/VLM 관찰에서 유도한
# "학생 추정치"만 다룹니다. scene_state, entity ID, coordinate go_to는 일절 쓰지 않고
# 이동은 언제나 set_velocity 폐루프(_turn_by_deg/_advance_or_detour)로만 수행하므로
# Level 2에서 합법입니다(발표에서 명시할 것). ctx가 없는 순수 함수라 unit test로 잠급니다.


def _pose_dict(robot_status: Any) -> dict[str, float]:
    """robot_status에서 {x, y, yaw_deg}만 뽑은 간결한 pose dict를 만듭니다(결측은 0.0)."""
    pose = getattr(getattr(robot_status, "robot", None), "pose", None)
    pos = getattr(pose, "position", None) or (0.0, 0.0, 0.0)
    return {
        "x": float(pos[0]),
        "y": float(pos[1]),
        "yaw_deg": float(getattr(pose, "yaw_deg", 0.0) or 0.0),
    }


def _face_turn_to(pose: dict[str, float], target: dict[str, float]) -> tuple[float, float]:
    """현재 pose에서 target 지점까지의 (유클리드 거리 m, 필요 회전각 도)를 계산합니다.

    벡터 to_target=(dx,dy)의 세계 방위각 atan2(dy,dx)와 현재 yaw의 차를 (-180,180]로
    정규화한 값이 face_turn입니다(양수=좌회전, _turn_by_deg 규약과 동일).
    """
    dx = float(target["x"]) - float(pose["x"])
    dy = float(target["y"]) - float(pose["y"])
    dist = math.hypot(dx, dy)
    bearing_deg = math.degrees(math.atan2(dy, dx))
    return dist, _angle_diff_deg(bearing_deg, float(pose.get("yaw_deg", 0.0)))


def _movement_efficiency(expected_m: float, actual_m: float) -> float:
    """실제 이동량/기대 이동량(속도×시간 운동 모델). 기대가 0 이하면 0.0(판단 불가)."""
    if expected_m <= 0.0:
        return 0.0
    return actual_m / expected_m


def _is_stalled(expected_m: float, actual_m: float) -> bool:
    """이동 효율 기반 stall 판정: 실제 병진이 기대의 STALL_EFF_RATIO 미만이면 막힌 것.

    기대 이동량에 비례하므로 거리 기반으로 짧아진 전진 청크에도 같은 기준이 성립하고,
    STALL_ABS_FLOOR_M 하한이 odometry 노이즈 오탐을 막습니다.
    """
    return actual_m < max(STALL_ABS_FLOOR_M, expected_m * STALL_EFF_RATIO)


def _expected_advance_m(vx: float, duration_s: float) -> float:
    """운동 모델 기대 병진: 효율×vx×(duration-램프업). stall 게이트가 vx와 함께 스케일된다.
    (기존 FORWARD_EFF_SPEED_MPS=0.27은 vx=0.4 전용 매직넘버라 속도를 올리면 감지가 다 틀어짐.)"""
    return FWD_EFF_RATIO * vx * max(0.0, duration_s - FWD_RAMP_S)


def _advance_duration_s(dist_m: float) -> float:
    """남은 거리 기반 전진 시간(속도×시간 모델의 역산: t = d / v_실효).

    [ADVANCE_MIN_S, PAD_ADVANCE_DUR]로 클램프합니다 — 너무 짧으면 램프업 미달로 안 걷고,
    너무 길면 waypoint를 지나쳐 폐루프 재보정 기회를 잃습니다.
    """
    if dist_m <= 0.0:
        return ADVANCE_MIN_S
    return min(
        max(dist_m / (FWD_EFF_RATIO * FORWARD_VX) + FWD_RAMP_S, ADVANCE_MIN_S),
        PAD_ADVANCE_FAR_DUR,
    )


def _route_score(stats: dict[str, Any]) -> float:
    """낮을수록 좋은 경로 점수 = 시간 + VLM×10 + stall×5 + 경로길이×1.5 (결측 키는 0)."""
    return (
        float(stats.get("total_time_s", 0.0))
        + float(stats.get("vlm_calls", 0)) * ROUTE_SCORE_VLM_W
        + float(stats.get("stalls", 0)) * ROUTE_SCORE_STALL_W
        + float(stats.get("path_len_m", 0.0)) * ROUTE_SCORE_PATH_W
    )


def _select_best_route(routes: list[dict[str, Any]]) -> dict[str, Any] | None:
    """score 최소(=최선) 경로를 고릅니다. waypoints가 없는 항목은 재사용 불가라 제외."""
    valid = [r for r in routes if r.get("waypoints")]
    if not valid:
        return None
    return min(valid, key=lambda r: float(r.get("score", math.inf)))


def _nearest_waypoint_index(waypoints: list[dict[str, float]], x: float, y: float) -> int:
    """현재 위치에서 유클리드 최근접 waypoint 인덱스(동률이면 경로 후반 쪽)를 반환합니다.

    후반 쪽 동률 우선은 이미 지나온 앞부분으로 되돌아가는 낭비를 막기 위함입니다.
    """
    best_i, best_d = 0, math.inf
    for i, wp in enumerate(waypoints):
        d = math.hypot(float(wp["x"]) - x, float(wp["y"]) - y)
        if d <= best_d:
            best_i, best_d = i, d
    return best_i


def _compress_waypoints(
    points: list[dict[str, float] | None],
    min_gap_m: float = ROUTE_MIN_WAYPOINT_GAP_M,
) -> list[dict[str, float]]:
    """연속 waypoint 간격을 min_gap 이상으로 솎아냅니다(첫 점과 마지막 점은 반드시 유지).

    마지막 점은 드롭(place) 지점이라 간격 미달이어도 버리지 않고 직전 점을 대체합니다.
    """
    pts = [{"x": float(p["x"]), "y": float(p["y"])} for p in points if p is not None]
    if not pts:
        return []
    out = [pts[0]]
    for p in pts[1:]:
        if math.hypot(p["x"] - out[-1]["x"], p["y"] - out[-1]["y"]) >= min_gap_m:
            out.append(p)
    last = pts[-1]
    if (out[-1]["x"], out[-1]["y"]) != (last["x"], last["y"]):
        if len(out) > 1 and math.hypot(last["x"] - out[-1]["x"], last["y"] - out[-1]["y"]) < min_gap_m:
            out[-1] = last
        else:
            out.append(last)
    return out


def _route_waypoints(
    start_pose: dict[str, float] | None,
    trace: list[dict[str, Any]],
    drop_pose: dict[str, float] | None,
) -> list[dict[str, float]]:
    """배송 trace에서 실제 병진에 성공한 step의 pose만 뽑아 waypoint 목록으로 압축합니다."""
    points: list[dict[str, float] | None] = [start_pose]
    for step in trace:
        if step.get("stall"):
            continue
        if step.get("action") in {"advance", "replay_advance", "detour_advance"}:
            points.append(step.get("pose"))
    points.append(drop_pose)
    return _compress_waypoints(points)


def _make_last_seen(
    pose: dict[str, float],
    face_turn_deg: float,
    *,
    confidence: float = 0.0,
    position: str = "",
    source: str = "vlm",
) -> dict[str, Any]:
    """sign 목격 기록: 목격 pose와 sign의 world 방위(yaw+face_turn)를 저장합니다.

    단안 VLM이라 거리를 모르므로 sign을 '위치'가 아니라 '목격 pose에서의 방향(ray)'으로
    기억합니다. 재조준은 그 world 방위를 현재 yaw와 비교해 계산하고, 목격 pose에서
    멀어질수록 ray 가정의 기하 오차가 커지므로 LAST_SEEN_MAX_DRIFT_M 밖에서는 불신합니다.
    """
    return {
        "pose": dict(pose),
        "world_heading_deg": _angle_diff_deg(float(pose.get("yaw_deg", 0.0)) + face_turn_deg, 0.0),
        "confidence": confidence,
        "position": position,
        "source": source,
    }


def _last_seen_face_turn(last_seen: dict[str, Any] | None, pose: dict[str, float]) -> float | None:
    """last_seen 기반 재조준 각(도, 양수=좌회전). 신뢰반경 밖이면 None(=VLM 필요)."""
    if not last_seen:
        return None
    seen_pose = last_seen.get("pose") or {}
    drift = math.hypot(
        float(pose["x"]) - float(seen_pose.get("x", 0.0)),
        float(pose["y"]) - float(seen_pose.get("y", 0.0)),
    )
    if drift > LAST_SEEN_MAX_DRIFT_M:
        return None
    return _angle_diff_deg(float(last_seen.get("world_heading_deg", 0.0)), float(pose.get("yaw_deg", 0.0)))


def _estimate_sign_distance(area_frac: float | None) -> float | None:
    """bbox 면적 비율 → 대략적 거리(m): d = K/√area_frac (면적 ∝ 1/d²의 역산).

    단안 카메라의 조야한 sensor model이라 ±수십% 오차를 전제합니다 — anchor는 '대략 그
    부근'만 맞으면 충분하고(최종 도착 판정은 어차피 VLM+색블롭 게이트가 책임), 가까운
    목격일수록 큰 융합 가중치가 걸려 접근할수록 점점 정밀해집니다. 결측/0이면 None.
    """
    if not area_frac or area_frac <= 0.0:
        return None
    return min(max(PAD_SIGN_DIST_K / math.sqrt(float(area_frac)), PAD_ANCHOR_MIN_D), PAD_ANCHOR_MAX_D)


def _project_point(pose: dict[str, float], face_turn_deg: float, dist_m: float) -> dict[str, float]:
    """현재 pose에서 face_turn 방향(양수=좌)으로 dist만큼 떨어진 지점의 world 좌표.

    world 방위 = yaw + face_turn (_make_last_seen의 world_heading 규약과 동일),
    좌표계는 로봇 자신의 odometry 프레임입니다.
    """
    heading_rad = math.radians(float(pose.get("yaw_deg", 0.0)) + float(face_turn_deg))
    return {
        "x": float(pose["x"]) + float(dist_m) * math.cos(heading_rad),
        "y": float(pose["y"]) + float(dist_m) * math.sin(heading_rad),
    }


def _anchor_weight(confidence: float, dist_m: float) -> float:
    """anchor 융합 가중치 = confidence / max(거리, 1m).

    가까운 목격일수록 bbox가 커서 거리 추정의 상대 오차가 작으므로 더 크게 신뢰합니다.
    confidence 하한 0.1은 conf 결측(0.0) 목격도 미미하게나마 반영되게 합니다.
    """
    return max(float(confidence), 0.1) / max(float(dist_m), 1.0)


def _fuse_anchor(
    anchor: dict[str, Any] | None,
    point: dict[str, float],
    weight: float,
) -> dict[str, Any]:
    """목격 지점 추정치를 가중 평균으로 누적해 anchor를 갱신합니다(없으면 초기화).

    w_sum을 PAD_ANCHOR_W_CAP으로 클램프해 옛 목격 더미가 새 목격을 압도하지 못하게
    합니다 — 초기 오추정 anchor도 재목격 몇 번이면 실제 위치 쪽으로 씻겨 갑니다.
    이상치 기각: 목격 2회 이상으로 자리잡은 평균에서 PAD_ANCHOR_OUTLIER_M 넘게 벗어난
    새 점은 오독(다른 표지를 목표 글자로 착각)일 공산이 커 기각합니다 — pad는 움직이지
    않으므로 정상 목격은 항상 같은 부근에 모입니다.
    """
    w_new = max(float(weight), 0.0)
    has_anchor = anchor is not None and float(anchor.get("w_sum", 0.0)) > 0.0
    if has_anchor and w_new <= 0.0:
        return anchor  # 0-가중 갱신은 정보가 없으므로 기존 anchor 유지.
    if not has_anchor:
        return {"x": float(point["x"]), "y": float(point["y"]), "w_sum": w_new, "n": 1,
                "t": time.monotonic()}
    if int(anchor.get("n", 0)) >= 2:
        off = math.hypot(float(point["x"]) - float(anchor["x"]), float(point["y"]) - float(anchor["y"]))
        if off > PAD_ANCHOR_OUTLIER_M:
            # ★자가치유: 모순 목격이 '연속 3회'면 자리잡은 평균 쪽이 오염이라 보고 새 목격으로
            #   재초기화한다. 기존엔 오염 anchor가 교정 목격을 영원히 기각(잠금)했다.
            rej = int(anchor.get("rej", 0)) + 1
            if rej >= 3:
                return {"x": float(point["x"]), "y": float(point["y"]),
                        "w_sum": w_new, "n": 1, "t": time.monotonic()}
            out = dict(anchor)
            out["rej"] = rej
            return out  # 크게 벗어난 목격은 일단 기각(연속 모순 카운트만 증가).
    w_old = min(float(anchor["w_sum"]), PAD_ANCHOR_W_CAP)
    w_total = w_old + w_new
    return {
        "x": (float(anchor["x"]) * w_old + float(point["x"]) * w_new) / w_total,
        "y": (float(anchor["y"]) * w_old + float(point["y"]) * w_new) / w_total,
        "w_sum": min(w_total, PAD_ANCHOR_W_CAP),
        "n": int(anchor.get("n", 0)) + 1,
        "t": time.monotonic(),   # 갱신 시각 — anchor '낡음' 판정(게이트 예외 조건)용.
        "rej": 0,                # 정상 융합 = 모순 목격 연속 카운트 리셋.
    }


def _side_name(side: float) -> str:
    """우회 방향 부호(+1=좌회전)를 실패 카운터 키로 변환합니다."""
    return "left" if side > 0 else "right"


def _choose_detour_side(
    preferred: float,
    fails: dict[str, int],
    goal_turn_deg: float | None = None,
) -> float:
    """우회 방향 선택: 목표(anchor) 방위가 뚜렷하면 그쪽이 최우선, 실패 이력은 격차 2회부터만 뒤집습니다.

    detour_fails는 nav 전역 카운터라 다른 위치의 실패 1회가 전 맵의 우회를 반대로 밀었습니다
    (라이브 확정: anchor +45°인데 -50° 선제 우회 → 목표 반대방향 이탈). 목표 쪽으로 꺾어야
    우회 후 재조준각이 작아지고 표지가 시야에 남습니다.
    """
    goal_known = goal_turn_deg is not None and abs(goal_turn_deg) > 15.0
    if goal_known:
        preferred = 1.0 if goal_turn_deg > 0 else -1.0
    other = -preferred
    margin = 2 if goal_known else 1   # 목표가 뚜렷하면 실패 이력에 더 완고하게.
    if fails.get(_side_name(preferred), 0) - fails.get(_side_name(other), 0) >= margin:
        return other
    return preferred


def _near_known_stall(
    spots: list[dict[str, float]],
    x: float,
    y: float,
    yaw_deg: float,
    *,
    radius_m: float = STALL_SPOT_RADIUS_M,
    heading_tol_deg: float = STALL_HEADING_TOL_DEG,
) -> bool:
    """기억된 stall 지점(위치+당시 진행 방향) 근처에서 같은 방향으로 또 전진하려는지 검사합니다.

    방향까지 비교하는 이유: 장애물은 stall 당시 진행 방향 앞에 있으므로, 같은 지점이라도
    다른 방향의 전진은 막지 않아야 우회 경로 자체가 봉쇄되지 않습니다.
    """
    for s in spots:
        if math.hypot(x - float(s["x"]), y - float(s["y"])) > radius_m:
            continue
        if abs(_angle_diff_deg(yaw_deg, float(s.get("yaw_deg", yaw_deg)))) <= heading_tol_deg:
            return True
    return False


def _preferred_side_from_history(
    wins: list[dict[str, float]],
    x: float,
    y: float,
    yaw_deg: float,
    *,
    radius_m: float = STALL_SPOT_RADIUS_M,
    heading_tol_deg: float = STALL_HEADING_TOL_DEG,
) -> float | None:
    """과거에 우회가 성공했던 지점·방향 근처면 그때의 우회 side(+1/-1)를 반환합니다.

    stall 지점 기억(_near_known_stall)의 쌍둥이: 실패는 피하고 성공은 재사용합니다.
    같은 구조물을 라운드 안에서 반복 통과할 때 좌/우 탐색 없이 한 번에 뚫립니다.
    최근 기록 우선(뒤에서부터 검색) — 같은 지점의 오래된 기록보다 새 경험을 신뢰합니다.
    """
    for win in reversed(wins):
        if math.hypot(x - float(win["x"]), y - float(win["y"])) > radius_m:
            continue
        if abs(_angle_diff_deg(yaw_deg, float(win.get("yaw_deg", yaw_deg)))) <= heading_tol_deg:
            return float(win["side"])
    return None


def _sign_bbox_area_frac(target: dict[str, Any]) -> float | None:
    """VLM sign 검출의 bbox 면적을 프레임 대비 비율(0~1)로 환산합니다(면적 ∝ 1/d² 대용).

    qwen bbox_2d는 0~1000 정규화 좌표(_sign_offset_deg와 동일 규약). 범위를 벗어나거나
    형식이 깨지면 None — 소비자(수렴 판정 2차 신호)는 결측을 무시합니다.
    """
    bbox = target.get("bbox_2d") or target.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        try:
            x1, y1, x2, y2 = (float(v) for v in bbox)
        except (TypeError, ValueError):
            return None
        if 0.0 <= x1 < x2 <= VLM_BBOX_SCALE and 0.0 <= y1 < y2 <= VLM_BBOX_SCALE:
            return ((x2 - x1) / VLM_BBOX_SCALE) * ((y2 - y1) / VLM_BBOX_SCALE)
    return None


def _approach_converging(
    history: list[dict[str, Any]],
    *,
    min_samples: int = APPROACH_MIN_SAMPLES,
    growth_min: float = APPROACH_AREA_GROWTH_MIN,
) -> bool | None:
    """접근 수렴 판정(사용자 스펙 distance_error_rate의 단안 proxy) — 현재 관측 모드 전용.

    history는 접근 반복마다의 {"area": 색블롭 면적, "face_turn": 조준각(도)} 표본입니다.
    1차 신호: 면적 ∝ 1/d²이므로 후반 표본의 면적 중앙값이 전반 대비 growth_min배 이상이면
    거리가 줄고 있는 것(수렴). 중앙값 비교라 green flicker 단발 노이즈에 강합니다.
    2차 신호: 면적이 안 늘어도 |face_turn|이 첫 표본보다 줄어 정면 허용각 안이면 조준 수렴.
    반환: None=표본 부족(판정 불가), True=수렴 중, False=비수렴(전진해도 진전 없음).
    """
    samples = [h for h in history if h.get("area") is not None]
    if len(samples) < min_samples:
        return None
    areas = [float(h["area"]) for h in samples]
    half = len(areas) // 2
    early, late = sorted(areas[:half]), sorted(areas[half:])
    early_med = early[len(early) // 2]
    late_med = late[len(late) // 2]
    if late_med > 0 and (early_med == 0 or late_med >= early_med * growth_min):
        return True
    turns = [abs(float(h["face_turn"])) for h in history if h.get("face_turn") is not None]
    if len(turns) >= 2 and turns[-1] <= PAD_FACE_TOL_DEG and turns[-1] < turns[0]:
        return True
    return False


def _should_commit_route(route_stats: dict[str, Any], began_new_trace: bool) -> bool:
    """R2 단순 가드: 이번 cycle의 경로 커밋 허용 여부.

    - t0가 없으면 추적된 배송이 아님(시작 전이거나 이미 커밋됨) → 금지.
    - 같은 cycle에 새 배송 trace가 방금 시작됐으면(began_new_trace) delivered 증가는
      '지연 도착한 직전 배송' 신호이고 현재 stats는 새 배송 것 → 커밋하면 score≈0
      쓰레기 경로가 best_route(min-score)로 영구 고정되므로 금지(학습 1건 생략을 감수).
    """
    if began_new_trace:
        return False
    return route_stats.get("t0") is not None


def _pad_memory_entry(pad_memory: dict[str, dict[str, Any]], color: str) -> dict[str, Any]:
    """색상별 pad 기억 슬롯을 얻거나 만듭니다."""
    return pad_memory.setdefault(
        color,
        {
            "last_seen": None,
            "anchor": None,  # sign의 world '점' 추정치 {x, y, w_sum, n} — ray와 달리 회전에 불변.
            "successful_routes": [],
            "failed_routes": [],
            "best_route": None,
        },
    )


def _commit_successful_route(
    entry: dict[str, Any],
    waypoints: list[dict[str, float]],
    stats: dict[str, Any],
    drop_pose: dict[str, float] | None,
) -> dict[str, Any]:
    """배송 성공 경로를 저장하고 best_route(score 최소)를 갱신합니다."""
    route = {
        "waypoints": waypoints,
        "score": _route_score(stats),
        "stats": dict(stats),
        "drop_pose": drop_pose,
    }
    entry.setdefault("successful_routes", []).append(route)
    entry["best_route"] = _select_best_route(entry["successful_routes"])
    return route


def _record_failed_route(entry: dict[str, Any], stats: dict[str, Any], reason: str) -> None:
    """실패한 pad 접근의 통계를 진단용으로 남깁니다(최근 FAILED_ROUTES_KEEP개만 보관)."""
    failed = entry.setdefault("failed_routes", [])
    failed.append({"stats": dict(stats), "reason": reason})
    del failed[:-FAILED_ROUTES_KEEP]


def _bump_stat(memory: AgentMemory | None, key: str, amount: float = 1) -> None:
    """현재 배송 route_stats의 카운터를 증가시킵니다(memory 없으면 no-op)."""
    if memory is None:
        return
    memory.route_stats[key] = memory.route_stats.get(key, 0) + amount


def _trace_step(memory: AgentMemory | None, **fields: Any) -> None:
    """현재 배송 route_trace에 step 하나를 기록합니다(memory 없으면 no-op).

    한 배송 trace는 보통 수십 step이지만, 비정상 장기 배회에 대비해 상한을 둡니다
    (초과 시 앞부분을 버리므로 그 배송의 waypoint 초반부가 소실될 수 있음 — 허용).
    """
    if memory is None:
        return
    fields.setdefault("step", len(memory.route_trace))
    memory.route_trace.append(fields)
    if len(memory.route_trace) > 1000:
        del memory.route_trace[:200]


def _record_stall(memory: AgentMemory | None, pose: dict[str, float]) -> None:
    """stall 발생 지점(위치+진행 방향)을 기록합니다 — 이후 같은 방향 재돌진을 선제 우회."""
    if memory is None:
        return
    _bump_stat(memory, "stalls", 1)
    memory.stall_spots.append(
        {"x": float(pose["x"]), "y": float(pose["y"]),
         "yaw_deg": float(pose["yaw_deg"]), "seg": memory.nav_segment}
    )
    del memory.stall_spots[:-40]


async def _get_pose(ctx: Any) -> dict[str, float]:
    """odometry pose {x, y, yaw_deg}를 한 번의 상태 읽기로 얻습니다(고유수용성; scene 아님)."""
    return _pose_dict(await get_robot_status(ctx))


async def _get_yaw_deg(ctx: Any) -> float:
    """로봇 자신의 body yaw(도)를 읽습니다.

    이는 고유수용성(gyro/IMU 상당)이며 scene_state가 아닙니다. 폐루프 회전에서 '상대 변화량'만
    쓰므로 Level 2에서 합법입니다(큐브/pad 위치 같은 scene 정보는 전혀 사용하지 않음).
    """
    status = await get_robot_status(ctx)
    pose = getattr(getattr(status, "robot", None), "pose", None)
    return float(getattr(pose, "yaw_deg", 0.0) or 0.0)


def _angle_diff_deg(a: float, b: float) -> float:
    """a-b를 (-180, 180] 범위로 정규화한 차이(도)."""
    d = (a - b) % 360.0
    return d - 360.0 if d > 180.0 else d


_turn_rate_dps = TURN_RATE_INIT_DPS   # 실효 회전률 EMA(도/초) — run 전체 공유(같은 정책이므로).


async def _turn_by_deg(ctx: Any, delta_deg: float) -> bool:
    """아크(vx>0 + wz)로 body를 상대 delta_deg만큼 회전합니다(양수=좌회전). 반환: 목표각 도달 여부.

    이 학습 정책은 제자리 회전이 불가하고 짧은 명령은 ramp-up으로 거의 안 돌기 때문에, 개루프
    회전은 부정확합니다. 폐루프에 세 가지를 보강했습니다(라이브 결함 교정):
    ① 아크 전후 yaw '실측'으로 실효 회전률 EMA를 갱신 — 하드코딩 25°/s 가정이 틀려도
       (wz 클립 등) 다음 아크부터 자동 적응해 과회전/발진을 줄인다.
    ② 잔여각이 작으면(<TURN_FINE_DEG) wz 자체를 낮춘다 — 기존엔 최소 0.9s 아크가 ~25°를
       돌아 8° 허용오차에 수렴 불가(좌우 발진=움찔거림, 아크마다 vx 동반이라 배회까지).
    ③ 아크당 실측 회전 <TURN_BLOCKED_MIN_DEG면 '회전 막힘'(벽 접촉): 1차 후진 후 재시도,
       2차 후진 아크(vx<0)로 벽에서 떨어지며 회전 — 그래도 0이면 False(호출부가 탈출 처리).
       기존엔 막혀도 vx>0 아크 5회로 장애물을 밀기만 하고 '조용히' 리턴했다(명령해도 안 돎).
    """
    global _turn_rate_dps
    if abs(delta_deg) < 1.0:
        return True
    target = (await _get_yaw_deg(ctx)) + delta_deg
    blocked = 0
    force_full = False   # ★저속 아크가 데드밴드로 안 돌면 다음 아크를 정속으로 승급(막힘 오판 방지).
    for _ in range(TURN_MAX_ARCS):
        yaw_now = await _get_yaw_deg(ctx)
        remaining = _angle_diff_deg(target, yaw_now)
        if abs(remaining) <= PAD_TURN_TOL_DEG:
            return True
        fine_arc = abs(remaining) < TURN_FINE_DEG and not force_full
        if not fine_arc:
            wz = ARC_WZ if remaining > 0 else -ARC_WZ
            dur = min(max(abs(remaining) / _turn_rate_dps, 0.7), 1.4)
        else:
            wz = TURN_FINE_WZ if remaining > 0 else -TURN_FINE_WZ   # 저속 미세 보정(발진 차단)
            dur = min(max(abs(remaining) / max(_turn_rate_dps * 0.55, 10.0), 0.7), 1.2)
        await move_velocity(ctx, vx=ARC_VX, wz=wz, duration_s=dur)
        turned = abs(_angle_diff_deg(await _get_yaw_deg(ctx), yaw_now))
        if turned >= TURN_BLOCKED_MIN_DEG:
            blocked = 0
            force_full = False
            if not fine_arc:   # 정속 아크만 EMA에 반영(저속 아크는 률이 다름)
                _turn_rate_dps = min(
                    max(0.7 * _turn_rate_dps + 0.3 * (turned / dur), TURN_RATE_MIN_DPS),
                    TURN_RATE_MAX_DPS,
                )
            continue
        if fine_arc:
            # ★저속(wz 0.35) 조합은 실측된 적 없어 정책 데드밴드로 안 돌 수 있다 —
            #   '막힘'으로 오판해 후진 댄스를 추지 말고 다음 아크를 정속으로 재시도.
            force_full = True
            continue
        blocked += 1
        if blocked == 1:
            await move_velocity(ctx, vx=-0.2, duration_s=0.8)             # 1차: 접촉 떼기
        else:
            await move_velocity(ctx, vx=-ARC_VX, wz=wz, duration_s=1.2)   # 2차: 후진 아크
            if abs(_angle_diff_deg(await _get_yaw_deg(ctx), yaw_now)) < TURN_BLOCKED_MIN_DEG:
                return False   # 후진 아크로도 회전 0 = 완전 봉쇄 → 호출부가 탈출 처리.
    # 아크 상한 소진: '부분 회전'은 실패가 아니다 — 상위 루프가 매 반복 재조준한다.
    # False는 물리적 회전 봉쇄(벽 접촉) 전용: 호출부가 stall 기록+후퇴를 실행하기 때문에,
    # 단순 미수렴에 False를 주면 가짜 stall 기억+불필요 1m 후퇴가 생긴다(검증 지적).
    return True


async def _pose_str(ctx: Any) -> str:
    """디버그용 위치/방향 문자열(x, y, yaw). 상대 이동 추적에만 씁니다(scene 정보 아님)."""
    status = await get_robot_status(ctx)
    pose = getattr(getattr(status, "robot", None), "pose", None)
    pos = getattr(pose, "position", None) or (0.0, 0.0, 0.0)
    yaw = float(getattr(pose, "yaw_deg", 0.0) or 0.0)
    return f"pos=({pos[0]:+.2f},{pos[1]:+.2f}) yaw={yaw:+.0f}°"


async def _strafe(ctx: Any, side: float, *, duration_s: float = STRAFE_DUR) -> float:
    """몸 방향을 유지한 채 vy로 옆걸음(게걸음)하고, 실제 이동 거리(m)를 반환합니다.

    side=+1이면 좌측(vy 양수), -1이면 우측. 회전이 없으므로 패드가 시야에서 안 벗어나고,
    앞이 막힌 상황에서 옆으로 빠져나가는 가장 직접적인 방법입니다. 반환 거리가 작으면
    (STRAFE_MIN_M 미만) 정책이 그 방향으로 게걸음을 못 하는 것 — 호출부가 폴백을 씁니다.
    """
    p0 = await _get_pose(ctx)
    await move_velocity(ctx, vy=side * STRAFE_VY, vx=STRAFE_VX_ASSIST, duration_s=duration_s)
    p1 = await _get_pose(ctx)
    return math.hypot(p1["x"] - p0["x"], p1["y"] - p0["y"])


async def _advance_or_detour(
    ctx: Any,
    side: float,
    *,
    duration_s: float = PAD_ADVANCE_DUR,
    memory: AgentMemory | None = None,
    action: str = "advance",
    goal_turn_deg: float | None = None,
    verbose: bool = False,
) -> bool:
    """전진 한 청크를 실행하되, 병진이 죽으면(구조물 stall) 후진+아크 우회로 전환합니다.

    학습 정책은 장애물에 막혀도 전진 명령을 수용해 병진 0·미세 회전만 남고, 호출부는 이를 몰라
    같은 자리를 배회합니다(라이브 확정: navpad가 x≈1.1 source ledge에 고착). 전진 전후
    odometry 거리(실제 이동량)를 속도×시간 기대 이동량과 비교(_is_stalled)해 stall을 감지하고,
    감지 시 후진→side로 꺾음→우회 전진→역회전(재조준)으로 장애물을 옆으로 비껴갑니다
    (bug-avoidance: turn-advance-turn-back). 추가로 두 가지를 기록/활용합니다:
    - 모든 청크의 기대/실제 이동량·효율을 route_trace에 기록(사후 분석·경로 승격용).
    - stall 지점(위치+당시 방향)을 기억하고, 같은 지점·같은 방향 전진이면 직진을 생략하고
      선제 우회합니다(_near_known_stall) — 같은 구조물에 반복 돌진하는 낭비 제거.

    역회전이 필수인 이유: 우회각(PAD_STALL_DETOUR_DEG=50°)이 카메라 half-FOV(30°)를 넘어,
    꺾은 채로 두면 직전까지 마주보던 표지가 반드시 프레임 밖으로 나가고, not-found 복구의
    고정 +55° 회전이 같은 방향으로 오차를 누적시켜 재획득이 사실상 불가합니다. 원래 방위로
    되돌려 표지를 시야에 복귀시키고 다음 look의 bbox 비례 조향이 미세 보정하게 합니다.

    반환: 병진을 확보했으면(직진 성공 또는 우회 전진 성공) True, 우회 전진마저 stall이면
    False — 호출부는 False일 때만 side를 토글합니다(성공한 우회의 방향을 뒤집으면 진동함).
    """
    pose0 = await _get_pose(ctx)
    expected = _expected_advance_m(FORWARD_VX, duration_s)
    goal_side = (
        (1.0 if goal_turn_deg > 0 else -1.0)
        if goal_turn_deg is not None and abs(goal_turn_deg) > 15.0
        else None
    )
    # 같은 지점·같은 방향에서 과거에 성공한 우회 방향이 있으면 호출부의 side보다 우선합니다
    # (실패 이력 회피와 대칭인 '성공 이력 재사용' — 좌/우 재탐색 없이 한 번에 통과).
    if memory is not None:
        win_side = _preferred_side_from_history(
            memory.detour_wins, pose0["x"], pose0["y"], pose0["yaw_deg"]
        )
        # ★목표 방위가 뚜렷할 때 그 '반대쪽' 성공 이력은 무시 — 픽 단계 옆걸음 성공 기록이
        #   배송 우회 방향을 목표 반대로 오염시키던 것 차단(라이브 확정).
        if win_side is not None and win_side != side and (
            goal_side is None or win_side == goal_side
        ):
            if verbose:
                print(f"           과거 성공 우회 방향({_side_name(win_side)}) 우선 적용")
            side = win_side
    known_spots: list[dict[str, float]] = []
    if memory is not None:
        def _stall_cluster(s: dict[str, float]) -> int:
            # 반경 내 기록 수(자기 포함). 2 이상 = 반복 확인된 지속 장애물(예: 컨베이어 턱).
            return sum(
                1 for t in memory.stall_spots
                if math.hypot(t["x"] - s["x"], t["y"] - s["y"]) < STALL_SPOT_RADIUS_M
            )
        # ★단계 오염 차단(라이브 확정): 다른 세그먼트(픽 단계 등)의 1회짜리 stall 기록은
        #   선제 봉쇄에 못 쓴다 — 직진을 먼저 시도한다. 같은 세그먼트이거나 2회+ 뭉친 지점만 유효.
        known_spots = [
            s for s in memory.stall_spots
            if s.get("seg", -1) == memory.nav_segment or _stall_cluster(s) >= 2
        ]
    if memory is not None and _near_known_stall(
        known_spots, pose0["x"], pose0["y"], pose0["yaw_deg"]
    ):
        if verbose:
            print(
                f"           기억된 stall 지점 근접(±{STALL_SPOT_RADIUS_M}m·동일 방향)"
                f" -> 직진 생략, {side * PAD_STALL_DETOUR_DEG:+.0f}° 선제 우회"
            )
        _trace_step(
            memory, action=action, pose=pose0, expected_m=round(expected, 3),
            actual_m=0.0, efficiency=0.0, stall=True, note="preempt_known_stall",
        )
    else:
        await move_velocity(ctx, vx=FORWARD_VX, duration_s=duration_s)
        pose1 = await _get_pose(ctx)
        moved = math.hypot(pose1["x"] - pose0["x"], pose1["y"] - pose0["y"])
        efficiency = _movement_efficiency(expected, moved)
        stalled = _is_stalled(expected, moved)
        _trace_step(
            memory, action=action, pose=pose1, expected_m=round(expected, 3),
            actual_m=round(moved, 3), efficiency=round(efficiency, 2), stall=stalled,
        )
        if not stalled:
            _bump_stat(memory, "path_len_m", moved)
            return True
        _record_stall(memory, pose0)
        if verbose:
            print(
                f"           전진 stall(moved={moved:.2f}m, 기대 {expected:.2f}m,"
                f" 효율 {efficiency:.0%}) -> 후진 {PAD_STALL_BACKUP_S}s"
                f" + {side * PAD_STALL_DETOUR_DEG:+.0f}° 우회 전진 후 재조준"
            )
    # ★막힘 탈출 = '회전'이 아니라 '옆걸음(vy)'을 먼저 쓴다(사용자 요구: 옆걸음 강화).
    #   몸 방향을 유지해 패드가 시야에 남고, 회전 우회보다 강하고 빠르다.
    # ★막힘 탈출 = 옆걸음(vy)을 '같은 방향으로 끈질기게'. 옆으로 비킬 때마다 전진이 뚫리는지
    #   확인하고, 뚫리면 즉시 종료. 전진이 아직 막혀도 '옆걸음이 계속 먹히는 한' 방향을 안 바꾼다
    #   — 방향 전환은 옆걸음 자체가 벽에 막혔을 때만. (기존엔 옆걸음 1회 후 전진이 막히면 바로
    #   방향을 뒤집어, 장애물 끝을 몇 스텝 앞두고 제자리걸음이 됐다 — 사용자 확정.)
    await move_velocity(ctx, vx=-0.2, duration_s=PAD_STALL_BACKUP_S)   # 접촉 해제(후진)
    d0 = pose0
    d1 = pose0
    detour_moved = 0.0
    detour_stalled = True
    strafe_steps = 0
    for _ in range(STRAFE_MAX_STEPS):
        strafed = await _strafe(ctx, side)
        if strafed < STRAFE_MIN_M:
            break   # 그 방향 옆도 막힘 → 루프 종료(아래에서 폴백 or 방향전환 위임)
        strafe_steps += 1
        d0 = await _get_pose(ctx)
        await move_velocity(ctx, vx=FORWARD_VX, duration_s=duration_s)
        d1 = await _get_pose(ctx)
        detour_moved = math.hypot(d1["x"] - d0["x"], d1["y"] - d0["y"])
        detour_stalled = _is_stalled(expected, detour_moved)
        if not detour_stalled:
            if verbose:
                print(f"           옆걸음 {strafe_steps}회({_side_name(side)}) 후 전진 뚫림 {detour_moved:.2f}m")
            break
        if verbose:
            print(f"           옆걸음 {strafe_steps}회({_side_name(side)}), 전진 여전히 막힘 -> 같은 방향 계속")
    if strafe_steps == 0:
        # 옆걸음이 처음부터 안 먹힘(정책 미지원/그 방향 즉시 벽) → 회전 우회 폴백.
        if verbose:
            print(f"           옆걸음 안 먹힘 -> {side * PAD_STALL_DETOUR_DEG:+.0f}° 회전 우회 폴백")
        await _turn_by_deg(ctx, side * PAD_STALL_DETOUR_DEG)
        d0 = await _get_pose(ctx)
        await move_velocity(ctx, vx=FORWARD_VX, duration_s=duration_s)
        d1 = await _get_pose(ctx)
        await _turn_by_deg(ctx, -side * PAD_STALL_DETOUR_DEG)
        detour_moved = math.hypot(d1["x"] - d0["x"], d1["y"] - d0["y"])
        detour_stalled = _is_stalled(expected, detour_moved)
    _trace_step(
        memory, action="detour_advance", pose=d1, expected_m=round(expected, 3),
        actual_m=round(detour_moved, 3),
        efficiency=round(_movement_efficiency(expected, detour_moved), 2),
        stall=detour_stalled, note=_side_name(side),
    )
    if detour_stalled:
        _record_stall(memory, d0)
    else:
        _bump_stat(memory, "path_len_m", detour_moved)
        if memory is not None:
            # 성공한 우회의 (진입 지점·진입 방향·side)를 기억 — 재방문 시 이 방향을 우선.
            memory.detour_wins.append(
                {"x": pose0["x"], "y": pose0["y"], "yaw_deg": pose0["yaw_deg"], "side": side}
            )
            del memory.detour_wins[:-DETOUR_WIN_KEEP]
    if verbose:
        print(
            f"           우회 전진 moved={detour_moved:.2f}m ->"
            f" {'OK' if not detour_stalled else '우회도 stall'}"
        )
    return not detour_stalled


def _bypass_chunks(
    streak: int,
    *,
    trigger: int = PAD_BYPASS_STALL_TRIGGER,
    cap: int = PAD_BYPASS_MAX_CHUNKS,
) -> int:
    """연속 hard-stall streak에서 측면 우회로 '따라 이동'할 청크 수를 정합니다.

    trigger 미만이면 0(아직 측면 우회 안 함, 짧은 detour로 계속 시도). trigger 이상이면
    2청크에서 시작해 streak가 커질수록 1씩 늘려 cap까지 escalate합니다 — 같은 구조물에 오래
    막힐수록 더 멀리 따라 이동해 끝/틈을 지나갈 확률을 높입니다.
    """
    if streak < trigger:
        return 0
    return min(streak - trigger + 2, cap)


async def _lateral_bypass(
    ctx: Any,
    side: float,
    chunks: int,
    *,
    memory: AgentMemory | None = None,
    probe: Any = None,
    verbose: bool = False,
) -> bool:
    """선형 구조물(벨트 등)에 반복해 막히면 몸 방향을 유지한 채 vy 옆걸음으로 여러 스텝
    비켜 이동해 구조물의 끝/틈을 지납니다.

    기존엔 목표 쪽으로 ~80° '꺾어' 걸었으나(패드를 시야에서 잃고 복귀 회전까지 필요), 옆걸음은
    방향을 유지해 패드가 계속 정면에 남고 복귀 회전이 불필요합니다. side는 목표(표지)가 있던
    방향(+1=좌). 옆도 막히면(옆걸음 실패) 조기 종료·False로 호출부가 반대쪽을 시도합니다.
    """
    if verbose:
        print(f"           반복 stall -> 옆걸음 우회({_side_name(side)}) 최대 {chunks}스텝")
    entry_pose = await _get_pose(ctx)  # 진입 지점·방위 — 성공 시 detour_wins 기록용.
    moved_any = False
    for _ in range(chunks):
        moved = await _strafe(ctx, side, duration_s=PAD_ADVANCE_DUR)
        p1 = await _get_pose(ctx)
        _trace_step(
            memory, action="bypass_strafe", pose=p1,
            actual_m=round(moved, 3), stall=(moved < STRAFE_MIN_M),
        )
        if moved < STRAFE_MIN_M:
            break  # 옆도 막힘 -> 중단(호출부가 반대쪽 시도).
        _bump_stat(memory, "path_len_m", moved)
        moved_any = True
        # ★한 걸음마다 place 검토: 옆으로 비켜 이동하다 pad를 스쳐 지나가는 것 방지.
        #   (회전이 없으므로 감지 즉시 중단해도 패드가 이미 정면 — 복귀 회전 불필요.)
        if probe is not None and await probe():
            if verbose:
                print("           옆걸음 우회 중 place 후보 감지 -> 조기 중단")
            break
    # (회전을 안 했으므로 복귀 회전 없음 — 방향은 처음부터 유지됨.)
    if moved_any and memory is not None:
        # 성공한 측면 우회의 (진입 지점·진입 방위·side)도 detour_wins에 기록합니다 — 지난
        # 런에서 검증된 북쪽 통과(+80° 2회 진전)가 기록되지 않아, 엉뚱한 방향의 옆걸음 detour
        # 성공이 방향 선택을 오염시켜 포켓에 갇혔습니다(라이브 확정). 최근 기록 우선 검색이라
        # 진짜 통과 경험이 낡은/우연한 기록을 자연히 이깁니다.
        memory.detour_wins.append(
            {
                "x": entry_pose["x"],
                "y": entry_pose["y"],
                "yaw_deg": entry_pose["yaw_deg"],
                "side": side,
            }
        )
        del memory.detour_wins[:-DETOUR_WIN_KEEP]
    if verbose:
        print(f"           측면 우회 {'진전' if moved_any else '실패(측면도 막힘)'}")
    return moved_any


async def _retreat_along_trace(
    ctx: Any, memory: AgentMemory | None, *, verbose: bool = False
) -> bool:
    """전진·우회·측면 우회가 전부 막혔을 때의 '보장 탈출': 왔던 길로 후진.

    route_trace에서 실제로 이동에 성공했던(actual_m≥0.15) 최근 pose를 골라 되돌아갑니다 —
    직전에 지나온 길은 유일하게 통행이 검증된 방향입니다. 후방 카메라가 없으므로 후진은
    저속·거리 제한·청크별 odometry 검증으로만 하고, 목표가 후방이 아니면(그쪽으로 회전이
    가능한 상황) 기존 waypoint 폐루프(_replay_route)를 재사용합니다.
    """
    pose = await _get_pose(ctx)
    goal: dict[str, Any] | None = None
    if memory is not None:
        for step in reversed(memory.route_trace[-30:]):
            p = step.get("pose")
            if p is None or float(step.get("actual_m", 0.0) or 0.0) < 0.15:
                continue   # 성공한 이동의 종점만 후보(막혔던 지점으로는 안 돌아감).
            d = math.hypot(pose["x"] - float(p["x"]), pose["y"] - float(p["y"]))
            if 0.5 <= d <= 1.5:
                goal = p
                break
    if goal is None:
        # trace 없음/후보 없음 → 최소한의 순수 후진으로 공간 확보(이동량 실측 검증).
        await move_velocity(ctx, vx=-0.2, duration_s=RETREAT_CHUNK_S * 2)
        p1 = await _get_pose(ctx)
        return math.hypot(p1["x"] - pose["x"], p1["y"] - pose["y"]) >= STALL_ABS_FLOOR_M
    _, back_turn = _face_turn_to(pose, goal)
    if abs(abs(back_turn) - 180.0) <= RETREAT_BACK_TOL_DEG:
        # 목표가 거의 정후방 → 회전 없이 후진(회전 자체가 막힌 상황에서도 성립).
        if verbose:
            print(f"           최후 탈출: 왔던 길로 후진(목표 후방 {back_turn:+.0f}°)")
        moved_total = 0.0
        while moved_total < RETREAT_MAX_M:
            p0 = await _get_pose(ctx)
            await move_velocity(ctx, vx=-0.2, duration_s=RETREAT_CHUNK_S)
            p1 = await _get_pose(ctx)
            step_m = math.hypot(p1["x"] - p0["x"], p1["y"] - p0["y"])
            _trace_step(memory, action="retreat", pose=p1, actual_m=round(step_m, 3))
            if step_m < STALL_ABS_FLOOR_M:
                return False   # 후방도 막힘 → 상위 recover로.
            moved_total += step_m
            if math.hypot(p1["x"] - float(goal["x"]), p1["y"] - float(goal["y"]))                     <= WAYPOINT_TOL_M:
                break
        return True
    # 후방이 아니면 그쪽으로 회전이 가능한 상황 → 기존 waypoint 폐루프 재사용.
    return await _replay_route(
        ctx, [{"x": float(goal["x"]), "y": float(goal["y"])}], memory, verbose=verbose
    )


def _other_pad_here(
    memory: AgentMemory | None,
    held_color: str,
    pose: dict[str, float],
    *,
    radius_m: float = PAD_ANCHOR_NEAR_M,
) -> str | None:
    """지금 위치 근처(radius)에 '다른' destination pad의 수확 anchor가 있으면 그 글자를 반환.

    target 표지를 못 찾았는데 다른 pad가 바로 여기 있다면, target anchor가 오독으로 그 pad
    위치에 박힌 것입니다(라이브: green→C인데 화면 끝 오독으로 C anchor가 실제 blue pad D
    위치에 박혀 로봇이 D로 향함, 접근하니 수확이 D를 잡음). 강한 오독 신호입니다.
    """
    if memory is None:
        return None
    for color, e in memory.pad_memory.items():
        if color == held_color or not isinstance(e, dict):
            continue
        a = e.get("anchor")
        if not a:
            continue
        if math.hypot(float(a["x"]) - pose["x"], float(a["y"]) - pose["y"]) <= radius_m:
            return DESTINATION_SIGN_RULES.get(color, color)
    return None


async def _harvest_other_signs(
    ctx: Any,
    signs: list[dict[str, Any]],
    target_letter: str,
    head_yaw_rad: float,
    memory: AgentMemory | None,
    *,
    verbose: bool = False,
) -> None:
    """★공짜 앵커 수확: 이미 지불한 VLM 응답에 '같이 보인' 다른 pad sign(B~E)도
    해당 색의 last_seen/anchor로 기억합니다. 기존엔 목표 글자만 쓰고 나머지를 버려서
    배송 색이 바뀔 때마다 맨땅 VLM 탐색부터 다시 했습니다 — 추가 VLM 비용 0으로 없앱니다.
    게이트는 anchor 융합과 동일하게 엄격(PAD_ANCHOR_MIN_CONF): 수확은 기회 관측이라
    오독→오배송 경로의 오염 리스크를 목표 관측보다 보수적으로 봅니다. 목표 글자 자체는
    _look_for_sign이 더 정밀하게 갱신하므로 여기서 건드리지 않습니다(같은 목격 이중 가중 방지)."""
    if memory is None or not signs:
        return
    pose: dict[str, float] | None = None
    for s in signs:
        letter = _extract_sign_letter(s)
        if letter is None or letter == target_letter:
            continue
        color = LETTER_TO_COLOR.get(letter)
        if color is None:                        # 'A'(source)는 destination pad가 아님
            continue
        conf = _as_confidence(s.get("confidence", 0))
        if conf < PAD_ANCHOR_MIN_CONF:
            continue
        if pose is None:
            pose = await _get_pose(ctx)          # 수확할 게 있을 때만 1회 조회
        face_turn = -(_sign_offset_deg(s) + math.degrees(head_yaw_rad))
        entry = _pad_memory_entry(memory.pad_memory, color)
        entry["last_seen"] = _make_last_seen(
            pose, face_turn, confidence=conf, position=str(s.get("position", ""))
        )
        est_d = _estimate_sign_distance(_sign_bbox_area_frac(s))
        if est_d is not None:
            point = _project_point(pose, face_turn, est_d)
            entry["anchor"] = _fuse_anchor(
                entry.get("anchor"), point, _anchor_weight(conf, est_d)
            )
        _trace_step(
            memory, action="harvest", source="vlm", found=True,
            note=f"sign {letter}({color}) conf={conf:.2f}", pose=pose,
        )
        if verbose:
            print(f"    수확: sign {letter}({color}) 기억 (conf={conf:.2f})")


async def _ask_vlm_jpeg(jpeg: bytes, prompt: str, api_key: str) -> str:
    """ask_vlm_about_frame의 bytes 입력판 — 프레임을 먼저 다 찍어두고 '병렬'로 묻기 위함."""
    return await asyncio.wait_for(
        asyncio.to_thread(ask_vlm, jpeg, prompt, api_key=api_key),
        timeout=45,
    )


async def _scan_pad_bearing_wide(
    ctx: Any,
    letter: str,
    held_color: str,
    api_key: str,
    *,
    memory: AgentMemory | None = None,
    verbose: bool = False,
) -> dict[str, Any] | None:
    """첫 탐색 전용 광각 look: head 팬 3방향 프레임을 '선캡처'한 뒤 VLM 3콜을 병렬 발사.

    벽시계 ≈ max(1콜) ≈ 10~33s — 순차 3콜(라이브 실측 99s)의 약 1/3, 커버리지 ~150°.
    head 팬은 locomotion이 없어 세 프레임 모두 같은 body pose — 기존 bearing 환산
    (_sign_offset_deg + head_yaw)과 표지판 수확(harvest) 규약이 그대로 성립합니다.
    """
    frames: list[tuple[float, bytes]] = []
    for hy in PAD_WIDE_SCAN_YAWS_RAD:
        await set_head(ctx, yaw=hy, pitch=0.15)
        await asyncio.sleep(0.35)              # head 정착 대기(scan_head와 동일 패턴).
        try:
            jpeg = await get_camera_frame(
                ctx, compressed=True,
                max_width=SIGNAGE_VLM_MAX_WIDTH, quality=SIGNAGE_VLM_QUALITY,
            )
        except Exception:
            continue                           # 프레임 1장 실패가 스캔 전체를 죽이지 않게.
        frames.append((hy, jpeg))
    await set_head(ctx, yaw=0.0, pitch=0.15)
    if not frames:
        return None
    prompt = build_signage_vlm_prompt(held_color)
    _bump_stat(memory, "vlm_calls", len(frames))
    raws = await asyncio.gather(
        *[_ask_vlm_jpeg(j, prompt, api_key) for _, j in frames],
        return_exceptions=True,                # 한 콜 실패가 전체 스캔을 죽이지 않게.
    )
    best: dict[str, Any] | None = None
    for (hy, _), raw in zip(frames, raws):
        if isinstance(raw, BaseException):
            continue
        signs = _parse_signs(raw)
        await _harvest_other_signs(ctx, signs, letter, hy, memory, verbose=verbose)
        target = _find_target_sign(signs, letter)
        conf = _as_confidence(target.get("confidence", 0)) if target else 0.0
        if verbose:
            pos_txt = target.get("position") if target else "?"
            print(f"    wide head={hy:+.1f}rad -> '{letter}':{pos_txt} conf={conf:.2f}")
        if target and conf >= VLM_MIN_CONFIDENCE and (best is None or conf > best["confidence"]):
            body_bearing = _sign_offset_deg(target) + math.degrees(hy)
            best = {
                "face_turn": -body_bearing,
                "confidence": conf,
                "position": str(target.get("position", "")),
                "bbox_area_frac": _sign_bbox_area_frac(target),
            }
    return best


async def _scan_pad_bearing(
    ctx: Any,
    letter: str,
    held_color: str,
    api_key: str,
    *,
    memory: AgentMemory | None = None,
    verbose: bool = False,
) -> dict[str, Any] | None:
    """head를 여러 각도로 팬하며 VLM으로 목표 sign을 찾습니다. 못 찾으면 None.

    반환: {"face_turn": 도(양수=좌회전), "confidence": float, "position": str} —
    face_turn 외 필드는 last_seen 기록용입니다. VLM 실호출마다 route_stats.vlm_calls를
    셉니다(경로 점수의 지배 비용).

    head 팬(set_head)은 locomotion이 없어 로봇이 움직이지 않으므로 스캔 중 위치를 잃지 않습니다
    (아크로 돌며 스캔하면 전진 병진으로 배회함). VLM이 느리므로(7~20s) center부터 보고 처음
    확신 검출에서 조기 종료합니다(1~3회). body-bearing = image_offset + head_yaw(코드의
    full_bearing 규약: +=우측), face_turn = -body_bearing(우측이면 우회전=음수 delta).
    """
    for hy in HEAD_SCAN_YAWS_RAD:
        await set_head(ctx, yaw=hy, pitch=0.15)
        _bump_stat(memory, "vlm_calls", 1)
        try:
            raw = await ask_vlm_about_frame(
                ctx,
                build_signage_vlm_prompt(held_color),
                api_key=api_key,
                max_width=SIGNAGE_VLM_MAX_WIDTH,
                quality=SIGNAGE_VLM_QUALITY,
            )
        except Exception:
            raw = ""
        signs = _parse_signs(raw)
        await _harvest_other_signs(ctx, signs, letter, hy, memory, verbose=verbose)
        target = _find_target_sign(signs, letter)
        conf = _as_confidence(target.get("confidence", 0)) if target else 0.0
        if verbose:
            pos_txt = target.get("position") if target else "?"
            print(f"    scan head={hy:+.1f}rad -> '{letter}':{pos_txt} conf={conf:.2f}")
        if target and conf >= VLM_MIN_CONFIDENCE:
            # bbox 비례 환산(가능하면) 또는 left/right 양자화(fallback)로 오프셋을 얻습니다.
            body_bearing = _sign_offset_deg(target) + math.degrees(hy)
            await set_head(ctx, yaw=0.0, pitch=0.15)
            return {
                "face_turn": -body_bearing,
                "confidence": conf,
                "position": str(target.get("position", "")),
                # bbox 면적 비율(∝ 1/d²): R6 수렴 판정의 2차(희소) 신호. 결측이면 None.
                "bbox_area_frac": _sign_bbox_area_frac(target),
            }
    await set_head(ctx, yaw=0.0, pitch=0.15)
    return None


async def _look_for_sign(
    ctx: Any,
    letter: str,
    held_color: str,
    api_key: str,
    memory: AgentMemory | None,
    entry: dict[str, Any] | None,
    *,
    reason: str,
    verbose: bool = False,
) -> float | None:
    """VLM look 1회(호출 사유 로그 필수). 성공 시 last_seen을 갱신하고 face_turn을 반환합니다.

    VLM은 pad-nav의 지배 비용(6~32s/회)이라, 호출부는 cached route/last_seen이 모두 소진된
    경우에만 이 함수에 도달해야 합니다. 왜 호출했는지가 로그에 남아야 이후 절감 튜닝이 가능합니다.
    """
    if verbose:
        print(f"    VLM 호출 이유: {reason}")
    first_search = "첫 탐색" in reason and (
        entry is None
        or (entry.get("anchor") is None and entry.get("last_seen") is None)
    )   # ★confirm 실패로 기억이 소거된 'place 전 확인'까지 광각 3콜이 나가던 것 차단.
    if first_search:
        # ★첫 탐색(기억 없음): head 팬 3방향 프레임 선캡처 후 VLM 3콜 병렬 발사 —
        #   순차 3콜(라이브 실측 99s)의 벽시계를 max(1콜)≈10~33s로, 커버리지 ~150°.
        sighting = await _scan_pad_bearing_wide(
            ctx, letter, held_color, api_key, memory=memory, verbose=verbose
        )
    else:
        sighting = await _scan_pad_bearing(
            ctx, letter, held_color, api_key, memory=memory, verbose=verbose
        )
    pose = await _get_pose(ctx)
    if sighting is None:
        _trace_step(memory, action="look", source="vlm", found=False, note=reason, pose=pose)
        return None
    est_d: float | None = None
    if entry is not None:
        # head 팬만 했으므로 scan 동안 body pose는 그대로 — 지금 pose가 목격 pose입니다.
        entry["last_seen"] = _make_last_seen(
            pose,
            sighting["face_turn"],
            confidence=sighting.get("confidence", 0.0),
            position=sighting.get("position", ""),
        )
        # bbox 면적이 있으면 거리까지 추정해 sign의 world '점'(anchor)을 융합 갱신합니다.
        # 융합 게이트는 nav 게이트보다 엄격(0.6): 낮은 확신의 오독이 점 추정을 오염시키면
        # ray보다 수명이 길어(회전 불변) 피해가 크기 때문입니다.
        conf = float(sighting.get("confidence", 0.0) or 0.0)
        est_d = _estimate_sign_distance(sighting.get("bbox_area_frac"))
        if est_d is not None and conf >= PAD_ANCHOR_MIN_CONF:
            point = _project_point(pose, sighting["face_turn"], est_d)
            entry["anchor"] = _fuse_anchor(
                entry.get("anchor"), point, _anchor_weight(conf, est_d)
            )
            if verbose:
                a = entry["anchor"]
                print(
                    f"    anchor 갱신: d≈{est_d:.1f}m -> ({a['x']:+.2f},{a['y']:+.2f})"
                    f" (목격 {a['n']}회, w={a['w_sum']:.2f})"
                )
    _trace_step(
        memory, action="look", source="vlm", found=True,
        face_turn_deg=round(sighting["face_turn"], 1),
        confidence=sighting.get("confidence"),
        bbox_area_frac=sighting.get("bbox_area_frac"),
        est_dist_m=None if est_d is None else round(est_d, 2),
        note=reason, pose=pose,
    )
    return sighting["face_turn"]


async def _register_local_sighting(
    ctx: Any,
    entry: dict[str, Any] | None,
    sighting: dict[str, Any],
    *,
    verbose: bool = False,
) -> bool:
    """국소 탐색이 얻은 sighting을 last_seen(+가능하면 anchor)으로 기록합니다.

    _look_for_sign의 기록 규약과 동일. head 팬만 했으므로 지금 body pose가 곧 목격 pose입니다.
    anchor는 근접 폐기 직후라 None → _fuse_anchor가 새 목격으로 신선하게 재초기화합니다.
    """
    if entry is None:
        return False
    pose = await _get_pose(ctx)
    entry["last_seen"] = _make_last_seen(
        pose, sighting["face_turn"],
        confidence=sighting.get("confidence", 0.0),
        position=sighting.get("position", ""),
    )
    conf = float(sighting.get("confidence", 0.0) or 0.0)
    est_d = _estimate_sign_distance(sighting.get("bbox_area_frac"))
    if est_d is not None and conf >= PAD_ANCHOR_MIN_CONF:
        point = _project_point(pose, sighting["face_turn"], est_d)
        entry["anchor"] = _fuse_anchor(entry.get("anchor"), point, _anchor_weight(conf, est_d))
        if verbose:
            a = entry["anchor"]
            print(f"    [local] anchor 재설정 d≈{est_d:.1f}m -> ({a['x']:+.2f},{a['y']:+.2f})")
    return True


async def _local_pad_search(
    ctx: Any,
    letter: str,
    held_color: str,
    api_key: str,
    memory: AgentMemory | None,
    entry: dict[str, Any] | None,
    *,
    nav_start: float,
    verbose: bool = False,
) -> bool:
    """anchor를 근접에서 폐기한 '그 자리'를 중심으로 표지를 국소 재탐색합니다(되돌아가지 않음).

    폐기 지점은 '패드가 있을 거라 믿고 온' 곳이고 anchor 오차(1~2m)만큼 어긋났을 뿐 실제
    pad는 근처일 공산이 큽니다. 출발점으로 173° 되돌아가 진전을 버리는 대신 지금 위치 반경
    1~2m를 훑습니다. VLM은 지배 비용(30~40s/콜)이라 콜 수를 제한하고, 나선 스텝 사이엔 값싼
    place probe(VLM 0)로 pad 옆 스침을 먼저 봅니다. 재획득 시 last_seen(+anchor) 갱신 후 True.
    """
    vlm_left = PAD_LOCAL_SEARCH_VLM_MAX
    # 1) 제자리 광각 head 스캔(좌/중/우 3프레임 병렬 VLM). locomotion 0이라 위치를 안 잃습니다.
    sighting = await _scan_pad_bearing_wide(
        ctx, letter, held_color, api_key, memory=memory, verbose=verbose
    )
    vlm_left -= len(PAD_WIDE_SCAN_YAWS_RAD)
    if sighting is not None:
        if verbose:
            print("    [local] 광각 스캔에서 표지 재획득 -> anchor 재설정")
        return await _register_local_sighting(ctx, entry, sighting, verbose=verbose)
    # 2) 못 찾으면 작은 나선: 옆걸음 1스텝 + 전진 1스텝을 좌/우 번갈아 반경 1~2m를 훑습니다.
    for side in (1.0, -1.0, 1.0, -1.0)[:PAD_LOCAL_SPIRAL_STEPS]:
        if time.monotonic() - nav_start > PAD_NAV_BUDGET_S:
            if verbose:
                print("    [local] 예산 초과 -> 국소 탐색 중단(상위 recover)")
            break
        await _strafe(ctx, side)                       # 몸 방향 유지 옆걸음(패드가 시야에 남음).
        await _advance_or_detour(ctx, side, memory=memory, verbose=verbose)
        probe_hit = await _place_probe(ctx, held_color, entry, memory)  # 값싼 후보 검사(VLM 0)
        if vlm_left > 0:
            sighting = await _scan_pad_bearing(
                ctx, letter, held_color, api_key, memory=memory, verbose=verbose
            )
            vlm_left -= 1
            if sighting is not None:
                if verbose:
                    print("    [local] 나선 스텝에서 표지 재획득 -> anchor 재설정")
                return await _register_local_sighting(ctx, entry, sighting, verbose=verbose)
        elif probe_hit and verbose:
            print("    [local] place 후보 감지했으나 VLM 예산 소진 -> 상위 confirm 티어로")
    if verbose:
        print("    [local] 국소 탐색 실패(표지 미발견)")
    return False


async def _replay_route(
    ctx: Any,
    waypoints: list[dict[str, float]],
    memory: AgentMemory | None,
    *,
    probe: Any = None,
    verbose: bool = False,
) -> bool:
    """기억된 성공 경로의 waypoint들을 odometry 폐루프로 따라갑니다(카메라·VLM 불필요).

    현재 위치에서 최근접 waypoint로 합류해 경로 후반만 재사용합니다. waypoint마다 거리/방위
    벡터(_face_turn_to)를 다시 계산해 회전 후 거리 기반 시간(_advance_duration_s)만큼
    전진하는 것을 반복하고, stall 우회까지 실패하거나 waypoint를 청크 예산 안에 못 따라가면
    False를 반환해 상위(VLM 탐색)로 넘깁니다. odometry 오차는 waypoint마다 재계산하는
    폐루프가 흡수하고, 최종 도착 판정은 어차피 상위의 VLM+색블롭 게이트가 책임집니다.
    """
    pose = await _get_pose(ctx)
    start_i = _nearest_waypoint_index(waypoints, pose["x"], pose["y"])
    if verbose:
        print(f"  [replay] waypoint {start_i + 1}/{len(waypoints)}부터 합류  {await _pose_str(ctx)}")
    for wi, wp in enumerate(waypoints[start_i:], start=start_i):
        for _chunk in range(ROUTE_REPLAY_CHUNKS_PER_WP):
            pose = await _get_pose(ctx)
            dist, face_turn = _face_turn_to(pose, wp)
            if dist <= WAYPOINT_TOL_M:
                break
            if verbose:
                print(f"  [replay wp{wi}] dist={dist:.2f}m turn={face_turn:+.0f}°")
            if abs(face_turn) > PAD_FACE_TOL_DEG:
                await _turn_by_deg(ctx, face_turn)
            side = 1.0 if face_turn > 0 else -1.0
            if not await _advance_or_detour(
                ctx, side, duration_s=_advance_duration_s(dist),
                memory=memory, action="replay_advance", verbose=verbose,
            ):
                if verbose:
                    print("  [replay] 우회까지 stall -> replay 중단, VLM 탐색으로 전환")
                return False
            # ★한 걸음마다 place 검토: 경유 중 pad 옆 스침 포착(VLM 0회).
            if probe is not None and await probe():
                if verbose:
                    print("  [replay] 경유 중 place 후보 감지 -> 조기 종료, 종점 confirm으로")
                return True
        else:
            pose = await _get_pose(ctx)
            dist, _ = _face_turn_to(pose, wp)
            if dist > WAYPOINT_TOL_M * 2:
                if verbose:
                    print(f"  [replay] wp{wi} 미도달(dist={dist:.2f}m) -> replay 중단")
                return False
            # 살짝 어긋난 정도면 다음 waypoint로 계속(폐루프가 흡수).
    return True


def _blob_arrival_suspect(
    blob: Any,
    held_color: str | None,
    memory: AgentMemory | None,
    pose: dict[str, float],
) -> str | None:
    """pad 도착 blob이 '컨베이어/source 오인'으로 의심되면 사유 문자열, 아니면 None.

    초록 배송처럼 held_color가 런타임 추정 벨트색과 같을 때만 강화 검사합니다
    (다른 색 배송의 정상 도착은 건드리지 않음 — 오탐 부작용 최소화).
    """
    if memory is None or held_color is None or held_color != memory.belt_color:
        return None
    # (a) source 근접: 이번 배송 pick 지점 반경 안이면 색blob 도착 자체가 구조적 오인.
    start = (memory.route_stats or {}).get("start_pose")
    if start:
        d_src = math.hypot(
            float(pose["x"]) - float(start["x"]), float(pose["y"]) - float(start["y"])
        )
        if d_src <= SOURCE_EXCLUSION_RADIUS_M:
            return f"pick 지점 {d_src:.2f}m 이내(source 오인 권역)"
    # (b) fill 게이트: 벨트/반사는 성기고(실측 0.45~0.67) pad 정면은 bbox를 거의 채움(0.92+).
    if _fill_ratio(blob) < BELT_CONFUSABLE_FILL_MIN:
        return f"fill {_fill_ratio(blob):.2f} < {BELT_CONFUSABLE_FILL_MIN}(벨트 성김 의심)"
    return None


async def _place_probe(
    ctx: Any,
    held_color: str,
    entry: dict[str, Any] | None,
    memory: AgentMemory | None = None,
    *,
    gate_m: float = PAD_PLACE_GATE_M,
) -> bool:
    """'값싼 도착(후보) 검사'(VLM 0회, ~0.5s) — 한 걸음마다 place 검토 + 도착 재게이트 공용.

    참 조건 = 색블롭 근접 AND 벨트/소스 오인 아님 AND anchor 근접(gate_m 이내; 없으면 생략).
    ★적대적 검증에서 probe/replay 경로가 벨트 veto와 anchor 게이트를 '우회'해 오배송으로
    이어질 수 있음이 확인돼, 메인 루프 도착 게이트와 같은 기준으로 통일했다.
    참이어도 '후보'일 뿐, place 확정은 VLM confirm이 한다(오배송 방지 불변식 유지)."""
    try:
        blob = await _best_color_blob(ctx, held_color, PAD_ARRIVAL_AREA)
        if (
            blob is None
            or blob.blob_area < PAD_ARRIVAL_AREA
            or abs(blob.angle_deg) > CENTER_TOLERANCE_DEG * 1.5
        ):
            return False
        pose = await _get_pose(ctx)
        if _blob_arrival_suspect(blob, held_color, memory, pose) is not None:
            return False
        anchor = entry.get("anchor") if entry is not None else None
        if anchor is not None:
            dist, _ = _face_turn_to(pose, anchor)
            if dist > gate_m:
                return False
        return True
    except Exception:
        return False   # ★보조 검사는 절대 사이클을 죽이지 않는다(카메라 딸꾹질 = 후보 아님).


async def _place_after_arrival(ctx: Any, memory: AgentMemory) -> dict[str, Any]:
    """pad 도착 확정(VLM confirm+블롭 게이트 통과) 직후 '같은 cycle'에서 즉시 place하고
    delivered 증가로 검증합니다.

    도착 후 다음 LLM 사이클(관찰+왕복 수십 초)을 기다리는 동안 방향이 틀어지면
    place_nearest_zone('가장 가까운 존')이 엉뚱한 존을 잡을 수 있습니다. LLM의 배송 의도
    (search_pad/navigate_to_pad)를 결정적 코드가 행동으로 변환하는 범위 — 기존 place_cube
    분기가 이미 nav+place를 한 action으로 합성하는 것과 같은 granularity(Level 2 적법)입니다.
    """
    before = await get_delivered_count(ctx)
    held0 = memory.held_color
    result = None
    # ★배송(delivered↑)될 때까지 place를 반복 시도한다(사용자 요구: 매번 place). 이 지점은
    #   색+VLM으로 목표 pad C가 확인된 자리이므로 반복 place는 오배송-안전(색 무관 nearest-zone
    #   place가 위험한 건 '틀린 pad'에서인데, 여기선 C가 검증됨). 매 시도 사이 살짝 전진해
    #   존 위에 확실히 서고, 목표색 pad를 벗어나면 즉시 멈춘다(상위 nav가 재접근).
    for i in range(PAD_PLACE_TRIES):
        result = await place_nearest_zone(ctx)
        await asyncio.sleep(0.5)                    # 느린 카운트 반영 헤지.
        after = await get_delivered_count(ctx)
        if after > before:
            print(f"  place 성공(delivered {before}->{after}, {i+1}회째)")
            break
        if await get_held_cube_info(ctx) is None:
            break   # 손 비었는데 카운트 안 오름 = 이미 처리/이상 → 중단.
        # 아직 배송 안 됨: 목표색 pad가 여전히 근접이면 살짝 전진 후 재시도, 벗어났으면 중단.
        if held0 is not None and await _target_in_range(ctx, held0, PAD_ARRIVAL_AREA):
            if i < PAD_PLACE_TRIES - 1:
                print(f"  place 미배송 -> 살짝 전진 후 재시도({i+2}/{PAD_PLACE_TRIES})")
                await move_velocity(ctx, vx=0.25, duration_s=0.6)
        else:
            print("  place 미배송 + 목표색 pad 벗어남 -> 중단(상위 nav 재접근)")
            break
    after = await get_delivered_count(ctx)
    return {
        "placed": after > before,
        "result": result_summary(result) if result is not None else None,
        "delivered_before": before,
        "delivered_after": after,
    }


async def visual_navigate_to_pad(
    ctx: Any,
    held_color: str,
    *,
    memory: AgentMemory | None = None,
    verbose: bool = False,
) -> bool:
    """색 우선 + VLM 글자 검증(단순 항법). 목표 pad는 held_color와 같은 색이므로, 그 색 blob을
    색 서보잉으로 접근하고(=VLM 0회, 빠르고 오독에 강함), 충분히 가까워지면 VLM으로 글자를 딱
    1회 '검증'해 맞을 때만 도착으로 본다.

    핵심: 오독(D를 C로)은 '글자'에서 나지 '색'에서 안 난다. 초록 배송에 초록 색을 따라가면
    파란 pad로 새는 것이 구조적으로 불가능하다. VLM은 두 곳에만 — ①색이 안 보일 만큼 멀 때
    방향 1회 ②색 후보 도착 시 글자 검증 1회. 나머지 항법·접근·장애물 회피는 전부 색+odometry.

    Level 2 준수: scene 좌표/go_to 미사용, robot_status(자기 pose/yaw)·카메라·색blob·VLM·메모리만.
    """
    letter = DESTINATION_SIGN_RULES.get(held_color)
    if letter is None:
        return False
    try:
        config = load_config(require_tokamak=True)
    except Exception:
        return False
    api_key = config.tokamak_api_key
    entry = _pad_memory_entry(memory.pad_memory, held_color) if memory is not None else None

    await set_head(ctx, yaw=0.0, pitch=0.15)   # 색blob 각도를 몸 기준으로 읽기 위해 고개 정면.
    nav_start = time.monotonic()
    no_color_turns = 0     # 목표색이 안 보여 회전한 연속 횟수(한 바퀴 넘으면 VLM 방향 1회).
    verify_fails = 0       # 색 후보 도착했으나 VLM 글자 검증 실패 누적(소스 A/오검출).
    hard_stall = 0         # 국소 옆걸음으로도 못 뚫은 연속 횟수(측면 바이패스 escalate 트리거).
    detour_side = 1.0

    for attempt in range(PAD_OUTER_MAX):
        if time.monotonic() - nav_start > PAD_NAV_BUDGET_S:
            if verbose:
                print(f"  [pad] 예산 {PAD_NAV_BUDGET_S:.0f}s 초과 -> 포기(recover 후 재시도)")
            return False

        # 1) 색으로 목표 pad blob 찾기(VLM 0). 없으면 회전 탐색, 한 바퀴+ 못 보면 VLM 방향 1회.
        blob = await _pad_color_blob(ctx, held_color)
        if blob is None:
            no_color_turns += 1
            if no_color_turns <= PAD_COLOR_SCAN_TURNS:
                if verbose:
                    print(f"  [pad {attempt}] 목표색({held_color}) 안 보임 -> {PAD_SEARCH_TURN_DEG:.0f}° 색 탐색 회전")
                await _turn_by_deg(ctx, PAD_SEARCH_TURN_DEG)
            else:
                # 한 바퀴 돌아도 색이 안 잡힘 = pad가 멀다. VLM 광각으로 '방향만' 잡고 그쪽 전진.
                if verbose:
                    print(f"  [pad {attempt}] 한 바퀴 색 못 봄 -> VLM 방향 1회 후 전진")
                sighting = await _scan_pad_bearing_wide(
                    ctx, letter, held_color, api_key, memory=memory, verbose=verbose
                )
                no_color_turns = 0
                if sighting is not None and abs(sighting["face_turn"]) > PAD_FACE_TOL_DEG:
                    await _turn_by_deg(ctx, sighting["face_turn"])
                await _advance_or_detour(ctx, detour_side, memory=memory, verbose=verbose)
            continue
        no_color_turns = 0
        angle = blob.angle_deg
        area = blob.blob_area

        # 2) 색 후보에 충분히 가까움(크고 중앙) -> VLM으로 글자 '검증'(딱 1회).
        if area >= PAD_ARRIVAL_AREA and abs(angle) <= CENTER_TOLERANCE_DEG * 1.5:
            if verbose:
                print(f"  [pad {attempt}] 색 후보 근접(area={area}) -> VLM 글자 '{letter}' 검증  {await _pose_str(ctx)}")
            # ★검증은 좌/중/우 '광각'으로: 패드 C 표지판이 정면이 아니라 옆에 치우쳐 있어도 잡는다
            #   (라이브: 첫 스캔에서 C가 head +0.8 한 쪽에서만 보였음 → 정면 1장 검증은 진짜 C를
            #   놓쳐 오기각). VLM 3콜은 병렬이라 벽시계는 1콜과 같음.
            sighting = await _scan_pad_bearing_wide(
                ctx, letter, held_color, api_key, memory=memory, verbose=verbose,
            )
            if sighting is not None:
                # 글자 확인됨(정면이든 옆이든) = 진짜 목표 pad. 그쪽으로 정렬 후 색 근접이면 도착.
                face = sighting["face_turn"]
                if abs(face) > PAD_FACE_TOL_DEG:
                    await _turn_by_deg(ctx, face)
                if await _target_in_range(ctx, held_color, PAD_ARRIVAL_AREA):
                    if verbose:
                        print(f"  [pad {attempt}] 색+글자 검증 통과 -> 도착  {await _pose_str(ctx)}")
                    _trace_step(memory, action="arrive", source="color+vlm", pose=await _get_pose(ctx))
                    return True
                continue   # C 쪽으로 틀었으나 아직 근접 아님 -> 재접근.
            # 색은 맞는데 글자가 아님(소스 A/다른 초록 물체/오검출) -> 이 후보 버리고 딴 데.
            verify_fails += 1
            if verbose:
                print(f"  [pad {attempt}] 글자 검증 실패({verify_fails}/{PAD_VERIFY_MAX}) -> 이 색 후보 기각")
            if verify_fails >= PAD_VERIFY_MAX:
                if verbose:
                    print("  [pad] 색 후보 글자 검증 반복 실패 -> nav 포기(상위 recover)")
                return False
            # 이 초록 후보를 벗어나 다른 초록 후보를 찾도록 크게 회전 + 한 발.
            await _turn_by_deg(ctx, PAD_SEARCH_TURN_DEG)
            await _advance_or_detour(ctx, detour_side, memory=memory, verbose=verbose)
            continue

        # 3) 색 후보가 보이지만 아직 멀거나 안 중앙 -> 색 서보잉으로 접근.
        if abs(angle) > CENTER_TOLERANCE_DEG:
            if verbose:
                print(f"  [pad {attempt}] 색 조준 angle={angle:+.0f}° area={area}")
            await _turn_by_deg(ctx, -angle)   # angle+ = 오른쪽 -> 오른쪽으로(음수 delta).
            hard_stall = 0
        else:
            if verbose:
                print(f"  [pad {attempt}] 색 접근 전진 area={area}")
            detour_side = 1.0 if angle <= 0 else -1.0   # 막히면 blob 쪽으로 옆걸음.
            if await _advance_or_detour(
                ctx, detour_side, memory=memory, goal_turn_deg=-angle, verbose=verbose
            ):
                hard_stall = 0
            else:
                # ★국소 옆걸음으로도 못 뚫음 = 장애물에 wedge. 같은 자리서 또 시도하면 제자리
                #   thrash(라이브 확정)이므로 '크게 우회'로 escalate: 목표 쪽으로 측면 바이패스
                #   (여러 스텝 옆걸음), 그쪽도 막히면 반대쪽, 그래도 막히면 왔던 길로 후진해
                #   접근각 자체를 바꾼다. 색 우선 nav에 이 escalation을 빠뜨린 게 wedge의 원인.
                hard_stall += 1
                chunks = min(PAD_BYPASS_MAX_CHUNKS, 2 + hard_stall)
                bypass_side = 1.0 if angle <= 0 else -1.0   # 목표(색 blob) 쪽으로 우회.
                if verbose:
                    print(f"  [pad {attempt}] 국소 옆걸음 실패 {hard_stall}회 -> 측면 바이패스 {chunks}스텝")
                if not await _lateral_bypass(ctx, bypass_side, chunks, memory=memory, verbose=verbose):
                    if not await _lateral_bypass(ctx, -bypass_side, chunks, memory=memory, verbose=verbose):
                        # 양쪽 다 막힘 → 왔던 길로 후진해 다른 접근각 확보(마지막 수단).
                        if verbose:
                            print(f"  [pad {attempt}] 양쪽 바이패스 실패 -> 후진(접근각 변경)")
                        await _retreat_along_trace(ctx, memory, verbose=verbose)
                        hard_stall = 0
        continue

    if verbose:
        print(f"  [pad] FAIL: {PAD_OUTER_MAX} attempt 내 도착 실패")
    return False


async def recover_motion(ctx: Any, memory: AgentMemory, reason: str | None = None) -> dict[str, Any]:
    """Target loss, blocked motion, failed manipulation에서 recover합니다.

    TODO:
    - Step back, rotate, rescan, detour 선택, LLM skip 요청 등을 구현하세요.
    - 같은 failed action을 무한 반복하지 않도록 memory를 사용하세요.
    """
    # 반복 실패를 memory로 파악해 회전량을 키우고, 너무 잦으면 skip을 권고합니다.
    color = memory.active_color or memory.held_color
    fails = memory.failed_attempts.get(color, 0) if color else 0

    # 1) 뒤로 물러나 manipulation/navigation을 위한 공간을 확보합니다.
    await move_velocity(ctx, vx=-0.15, duration_s=0.8)
    # 2) 실패가 쌓일수록 더 오래 아크로 선회해 다른 접근 경로(detour)를 찾습니다(제자리 회전 불가).
    #    pick이 반복 실패하면(같은 큐브에 막힘) 선회량을 더 키워 다른 큐브가 최근접이 되게 합니다.
    extra = 0.3 * memory.pick_fail_streak if memory.recent_pick_fail else 0.0
    await move_velocity(ctx, vx=SWEEP_VX, wz=SWEEP_WZ, duration_s=min(0.8 + 0.4 * fails + extra, 2.5))
    # 3) 회전 뒤 주변을 다시 훑어 target 재획득 기회를 만들고 head를 정면으로 둡니다.
    await scan_head(ctx)
    await set_head(ctx, yaw=0.0, pitch=0.15)

    # 4) 같은 target에서 반복 실패하면(LLM이 skip_target을 고르도록) skip을 권고합니다.
    suggest_skip = color is not None and fails >= 3
    return {
        "action": "recover",
        "reason": reason,
        "color": color,
        "fails": fails,
        "suggest_skip": suggest_skip,
        "status": "stepped_back_and_rotated",
    }


async def execute_decision(
    ctx: Any,
    decision: AgentDecision,
    observation: Observation,
    memory: AgentMemory,
) -> dict[str, Any]:
    """검증된 LLM 결정 하나를 Level 2 robot 행동으로 변환합니다.

    TODO:
    - go_to 없이 search/navigation을 구현하세요.
    - Intended cube 가까이에 visual positioning한 뒤에만 pick하세요.
    - Matching pad 가까이에 visual positioning한 뒤에만 place하세요.
    - Target이 사라지거나 movement가 실패하면 recovery를 사용하세요.
    """
    action = decision.next_action
    # 아직 큐브를 안 들었으면 '획득' 단계(색 고정 없이 최근접 깨끗한 큐브), 들었으면 '배송'
    # 단계(실제 든 색 held_color로 pad를 찾음). color-blind pick과 싸우지 않는 것이 핵심입니다.
    acquiring = memory.held_color is None

    if action in {"search_cube", "search_pad"}:
        if action == "search_pad" and memory.held_color:
            # pad는 색블롭이 아니라 VLM signage로 찾습니다(같은 색 source/큐브와 구분).
            found = await visual_navigate_to_pad(ctx, memory.held_color, memory=memory, verbose=True)
            if found:
                # ★도착 확정(VLM confirm+블롭 게이트 통과) 즉시 place — 다음 LLM 왕복(수십 초)
                #   동안 방향이 틀어지면 '가장 가까운 존'이 엉뚱한 존이 될 수 있다.
                place_info = await _place_after_arrival(ctx, memory)
                return {"action": action, "found": True, "place": place_info}
            return {"action": action, "found": False}
        search_target = None if (action == "search_cube" and acquiring) else (memory.held_color or decision.target_color)
        found = await visual_search(ctx, search_target, memory=memory)
        return {"action": action, "found": found}

    if action in {"navigate_to_cube", "navigate_to_pad"}:
        if action == "navigate_to_pad" and memory.held_color:
            # 배송: pad는 VLM signage로 방향을 잡아 아크로 접근합니다(색블롭 불가).
            reached = await visual_navigate_to_pad(ctx, memory.held_color, memory=memory, verbose=True)
            if reached:
                place_info = await _place_after_arrival(ctx, memory)   # ★도착 즉시 place(위와 동일 근거).
                return {"action": action, "reached": True, "place": place_info}
        elif action == "navigate_to_cube" and acquiring:
            # 획득: clean 큐브가 이미 pick 가능할 만큼 가까우면 정렬 nav를 생략합니다. 짧은 회전은
            # 학습 정책 ramp-up으로 거의 안 돌아 근접 큐브 주위를 맴돌 뿐이고, pick_entity는 각도
            # 무관 최근접 큐브를 스스로 파지하므로 정렬이 불필요합니다(다음 cycle에서 pick_cube).
            ready, _seen = await _clean_cube_ready(ctx)
            reached = ready or await visual_navigate_to_target(ctx, None)
        else:
            nav_target = memory.held_color or decision.target_color
            reached = await visual_navigate_to_target(ctx, nav_target)
        if not reached:
            # 도착 실패 -> recovery로 자세를 바꾼 뒤 다음 cycle에서 재시도합니다.
            recovery = await recover_motion(ctx, memory, reason=f"{action}_failed")
            return {"action": action, "reached": False, "recovery": recovery}
        return {"action": action, "reached": True}

    if action == "pick_cube":
        # color-blind pick_entity는 정면 최근접 큐브를 색 무관하게 잡습니다. 그래서 특정 색을
        # 맞추려 싸우지 않고, 정면에 '깨끗한' 큐브가 집을 준비가 됐는지만 확인합니다. 실제 잡은
        # 색은 이후 get_held_cube_info로 확정해 그 색 pad로 배송합니다(채점은 색 무관 30pt/개).
        ready, seen = await _clean_cube_ready(ctx)
        if not ready:
            # 준비 안 됨: 반복 실패가 쌓였으면 크게 relocate, 아니면 한 번 더 접근 후 재확인.
            if memory.pick_fail_streak >= PICK_FAIL_RELOCATE:
                recovery = await recover_motion(ctx, memory, reason="pick_relocate")
                return {"action": "pick_cube", "result": None, "positioned": False, "recovery": recovery}
            await visual_navigate_to_target(ctx, None)
            ready, seen = await _clean_cube_ready(ctx)
            if not ready:
                recovery = await recover_motion(ctx, memory, reason="pick_no_clean_cube")
                return {"action": "pick_cube", "result": None, "positioned": False, "recovery": recovery}
        result = await pick_nearest_cube(ctx)
        return {"action": "pick_cube", "result": result_summary(result), "positioned": True, "seen_color": seen}

    if action == "place_cube":
        # ★빈손 가드: 든 큐브가 없으면 place는 성립하지 않는다(배송 직후 관성 place_cube가
        #   방금 놓은 pad blob으로 빈손 place_entity를 쏴 최대 120s를 태우던 것 차단 — 검증 지적).
        if memory.held_color is None:
            return {"action": "place_cube", "status": "no_held_cube", "positioned": False}
        # 실제 들고 있는 색(ground truth)을 우선해 그 색 pad로 이동/place합니다.
        pad_color = memory.held_color or decision.target_color
        if memory.held_color:
            # 배송: pad는 VLM signage로 접근해야 합니다. 색블롭 도착판정은 같은 색 source(A)에서도
            # 참이 되어 엉뚱한 zone에 place할 수 있으므로, VLM pad-nav로 목표 pad에 확실히 붙습니다.
            if not await visual_navigate_to_pad(ctx, memory.held_color, memory=memory, verbose=True):
                recovery = await recover_motion(ctx, memory, reason="place_positioning_failed")
                return {"action": "place_cube", "result": None, "positioned": False, "recovery": recovery}
        elif not await _target_in_range(ctx, pad_color, PAD_ARRIVAL_AREA):
            reached = await visual_navigate_to_target(ctx, pad_color)
            if not reached:
                recovery = await recover_motion(ctx, memory, reason="place_positioning_failed")
                return {"action": "place_cube", "result": None, "positioned": False, "recovery": recovery}
        place_info = await _place_after_arrival(ctx, memory)
        return {"action": "place_cube", "result": place_info["result"], "positioned": True,
                "place": place_info}

    if action == "recover":
        return await recover_motion(ctx, memory, decision.recovery_strategy)

    if action == "skip_target":
        # 실제 스킵 기록은 update_memory가 처리합니다.
        return {"action": "skip_target", "target_color": decision.target_color, "status": "skipped"}

    return {"action": action, "status": "no_op"}


async def run_agent(
    ctx: Any,
    *,
    max_cycles: int = 10_000,
    completion: CompletionConfig | None = None,
) -> AgentMemory:
    """얇은 observe-LLM-act loop입니다. 이 loop만이 아니라 TODO 함수들을 수정하세요."""
    memory = AgentMemory()
    last_result: dict[str, Any] | None = None
    tracker = CompletionTracker(completion) if completion is not None else None

    async def run_step(awaitable: Any, label: str) -> Any:
        if tracker is None:
            return await awaitable
        return await tracker.wait_for_remaining(awaitable, label)

    consecutive_errors = 0   # ★연속 사이클 예외 카운터 — RPC 연쇄 사망 감지용.
    for cycle in range(1, max_cycles + 1):
        print(f"\n[Level 2] Cycle {cycle}")
        try:
            if tracker is not None:
                first_cycle = tracker.started_at is None
                tracker.start_first_cycle()
                if first_cycle:
                    tracker.print_start()
                reason = await tracker.stop_reason_from_scene(ctx)
                if reason is not None:
                    tracker.mark_ended(reason)
                    print(f"Completion target reached before cycle action: {reason}.")
                    break

            observation = await run_step(observe_world(ctx, memory), "observe_world")
            decision = await run_step(
                decide_next_action(TASK, observation, memory, last_result),
                "LLM decision",
            )
            print("LLM decision:", decision)

            if decision.next_action == "stop":
                break

            action_result = await run_step(
                execute_decision(ctx, decision, observation, memory),
                "execute action",
            )
            verified = await run_step(
                verify_outcome(ctx, decision, action_result),
                "verify outcome",
            )
            update_memory(memory, observation, decision, verified)
            last_result = verified
            consecutive_errors = 0   # 사이클 정상 완료 → 실패 카운터 리셋.
            if tracker is not None:
                reason = await tracker.stop_reason_from_scene(ctx)
                if reason is not None:
                    tracker.mark_ended(reason)
                    print(f"Completion target reached after cycle action: {reason}.")
                    break
        except CompletionTimeout as exc:
            if tracker is not None:
                tracker.mark_ended(str(exc))
            print(f"Completion timer expired: {exc}.")
            break
        except Exception as exc:
            # ★기존엔 CompletionTimeout만 잡아서 RPC 무응답(재시도 2차 실패) 예외 하나가
            #   런 '전체'를 죽였다(검증 지적 — probe가 카메라 호출을 늘려 노출면 확대).
            #   사이클 단위로 흡수하고, 연속 과다면 연결 사망으로 보고 종료한다.
            consecutive_errors += 1
            print(
                f"  ⚠ Cycle {cycle} 실패({consecutive_errors}/6):"
                f" {type(exc).__name__}: {str(exc)[:100]}"
            )
            if consecutive_errors >= 6:
                print("  연속 실패 과다 → 연결이 죽은 듯. 런 종료.")
                break
            await asyncio.sleep(1.0)

    if tracker is not None:
        await tracker.print_summary_from_scene(ctx)
    return memory


async def run(ctx: Any) -> None:
    print(TASK)
    print("Level 2 autonomous-vision project starter 실행")
    completion = await prepare_evaluation_round(ctx, level=2)
    memory = await run_agent(
        ctx,
        max_cycles=10_000,
        completion=completion,
    )
    print("\n실행 완료.")
    print(f"Delivered count: {memory.delivered_count}")
    if memory.pad_memory:
        # 발표·디버그용 경로 기억 요약: 색상별 성공 경로 수와 best score(낮을수록 좋음).
        print("Route memory:")
        for pad_color, entry in memory.pad_memory.items():
            best = entry.get("best_route")
            n_ok = len(entry.get("successful_routes", []))
            n_fail = len(entry.get("failed_routes", []))
            if best:
                print(
                    f"  {pad_color}: 성공 {n_ok}회/실패 {n_fail}회,"
                    f" best score={best['score']:.1f}"
                    f" (vlm={best['stats'].get('vlm_calls', 0)},"
                    f" stall={best['stats'].get('stalls', 0)},"
                    f" wp={len(best['waypoints'])})"
                )
            else:
                print(f"  {pad_color}: 성공 0회/실패 {n_fail}회 (best_route 없음)")
    print("Logs:")
    for item in memory.logs:
        print(item)


