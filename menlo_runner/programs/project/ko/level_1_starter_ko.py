from __future__ import annotations

"""Menlo AI 로봇 분류 챌린지용 Level 1 프로젝트 시작 파일입니다.

이 파일은 완성된 해답이 아니라 시작 파일입니다.

지원 코드 섹션은 반복해서 작성할 필요가 없는 작은 래퍼와 자료 구조를 제공합니다.
필요하면 읽고 수정할 수 있지만, 대부분의 팀은 지원 코드를 크게 바꾸지 않는 편이 좋습니다.
학생 TODO 섹션은 팀이 수정하고, 개선하고, test하고, presentation에서 설명해야 하는 부분입니다.

Level 1 규칙: scene_state와 정확한 entity ID는 사용할 수 없습니다. Coordinate go_to는
학생 시스템이 관찰로 추정하거나 기록한 좌표에만 사용할 수 있습니다.
"""

import asyncio
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any

from menlo_runner.completion import CompletionConfig, CompletionTracker
from menlo_runner.llm import ask_vlm
from menlo_runner.perception import detect_color_blobs
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
    "minimaxai/minimax-m2.7",
    "qwen/qwen3.6-35b-a3b",
)
LLM_MODEL = os.environ.setdefault("MENLO_LLM_MODEL", "minimaxai/minimax-m3")

# 고정 표지판 정보는 사용할 수 있습니다. 단, 이를 정확한 coordinate나 entity ID로
# 바꾸지 말고 관찰을 해석하는 데만 사용하세요.
DESTINATION_SIGN_RULES = {
    "red": "B",
    "green": "C",
    "blue": "D",
    "yellow": "E",
}
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

    이 구조는 특정 strategy에 묶이지 않도록 의도적으로 중립적입니다. Level 1 팀은 coordinate estimate에 full bearing을 사용할 수 있고, Level 2 팀은 closed-loop visual centering에 사용할 수 있습니다. 필요하면 confidence, target type, depth field를 추가하세요.
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


async def get_camera_frame(ctx: Any) -> bytes:
    """POV camera frame을 가져옵니다."""
    return await ctx.get_vision("pov")


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


async def ask_vlm_about_frame(ctx: Any, prompt: str, *, api_key: str) -> str:
    """Project에서 허용되는 VLM helper로 현재 POV frame에 대해 질문합니다."""
    jpeg = await get_camera_frame(ctx)
    return ask_vlm(jpeg, prompt, api_key=api_key)


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
    return await ctx.invoke("set_head", args, timeout_s=10)


async def move_velocity(
    ctx: Any,
    *,
    vx: float = 0.0,
    vy: float = 0.0,
    wz: float = 0.0,
    duration_s: float = 1.0,
) -> Any:
    """짧은 body-frame velocity command를 보낸 뒤 멈춥니다."""
    return await ctx.invoke(
        "set_velocity",
        {"vx": vx, "vy": vy, "wz": wz, "duration_s": duration_s},
        timeout_s=30,
    )


async def cancel_action(ctx: Any) -> Any:
    """현재 실행 중인 runtime action을 취소합니다."""
    return await ctx.invoke("cancel", {})


async def pick_nearest_cube(ctx: Any) -> Any:
    """Code가 robot을 시각적으로 충분히 위치시킨 뒤 nearest cube를 집습니다."""
    return await ctx.invoke(
        "pick_entity",
        {"target": {"kind": "entity", "entity_id": "cube"}},
        timeout_s=300,
    )


async def place_nearest_zone(ctx: Any) -> Any:
    """Matching pad에 도달한 뒤 nearest zone에 place합니다."""
    return await ctx.invoke("place_entity", {}, timeout_s=300)


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

    visible = decision_context["visible_targets"]
    if not visible:
        return AgentDecision(next_action="search_cube", reason="대체 동작: 보이는 target이 없습니다.")

    largest = max(visible, key=lambda item: item["blob_area"])
    return AgentDecision(
        next_action="navigate_to_cube",
        target_color=largest["color"],
        reason="대체 동작: 가장 큰 visible color blob을 선택합니다.",
    )


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
    detections = await scan_head(ctx)
    return Observation(robot_status=robot_status, detections=detections)


async def verify_outcome(ctx: Any, decision: AgentDecision, action_result: dict[str, Any]) -> dict[str, Any]:
    """마지막 action이 성공한 것처럼 보이는지 확인합니다.

    TODO:
    - 중요한 action 뒤에는 다시 observe하세요.
    - robot_status, camera evidence, SDK result status를 확인하세요.
    - 다음 LLM call이 recovery에 사용할 수 있는 정보를 반환하세요.
    """
    held = await get_held_cube_info(ctx)
    return {
        "decision": decision.__dict__,
        "action_result": action_result,
        "delivered_count": await get_delivered_count(ctx),
        "held_cube": held,
        "held_color": held["color"] if held else None,
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
    if "delivered_count" in verified:
        memory.delivered_count = int(verified["delivered_count"])
    memory.held_color = verified.get("held_color")

    memory.logs.append({
        "observation": {
            "visible_count": len(observation.detections),
            "note": observation.note,
            "delivered_count": memory.delivered_count,
            "held_color": memory.held_color,
        },
        "llm_decision": decision.__dict__,
        "verified": verified,
    })

# ---------------------------------------------------------------------------
# LEVEL 1 학생 TODO: coordinate-guided action 구현
# ---------------------------------------------------------------------------
# Level 1은 go_to를 사용할 수 있지만 observation으로 추정한 coordinate에만 사용할 수 있습니다.
# Entity ID, scene_state, ground-truth object coordinate를 사용하지 마세요.


def estimate_target_xy_from_observation(observation: Observation, target_color: str | None) -> tuple[float, float] | None:
    """Camera observation으로 target world coordinate를 추정합니다.

    TODO:
    - 원하는 cube 또는 pad에 해당하는 detection을 선택하세요.
    - Head yaw가 포함되도록 가능하면 detection.full_bearing_deg를 사용하세요.
    - Depth, calibration, blob size, camera geometry 등을 사용해 distance를 추정하세요.
    - Robot pose, bearing, distance를 결합해 world x/y로 변환하세요.
    - Confidence가 너무 낮으면 None을 반환하세요.
    """
    return None


async def go_to_xy(ctx: Any, x: float, y: float) -> Any:
    """Coordinate-based go_to입니다. 학생 시스템이 추정한 x/y에만 사용하세요."""
    return await ctx.invoke(
        "go_to",
        {
            "target": {
                "kind": "pose",
                "pose": {"frame_id": "world", "position": [x, y, 0]},
            }
        },
        timeout_s=300,
    )


async def execute_decision(
    ctx: Any,
    decision: AgentDecision,
    observation: Observation,
    memory: AgentMemory,
) -> dict[str, Any]:
    """검증된 LLM 결정 하나를 Level 1 robot 행동으로 변환합니다.

    TODO:
    - Search action에서는 안전하게 scan하거나 reposition하세요.
    - Navigation action에서는 vision으로 x/y를 추정하고 go_to_xy를 호출하세요.
    - Pick/place action에서는 robot이 intended target 가까이에 있는지 verify하세요.
    - recover/skip/stop은 팀 policy에 맞게 구현하세요.
    """
    if decision.next_action in {"search_cube", "search_pad"}:
        await scan_head(ctx)
        return {"action": decision.next_action, "status": "scanned"}

    if decision.next_action in {"navigate_to_cube", "navigate_to_pad"}:
        target_xy = estimate_target_xy_from_observation(observation, decision.target_color)
        if target_xy is None:
            return {"action": decision.next_action, "status": "failed", "reason": "coordinate estimate 없음"}
        result = await go_to_xy(ctx, *target_xy)
        return {"action": decision.next_action, "target_xy": target_xy, "result": result_summary(result)}

    if decision.next_action == "pick_cube":
        result = await pick_nearest_cube(ctx)
        return {"action": "pick_cube", "result": result_summary(result)}

    if decision.next_action == "place_cube":
        result = await place_nearest_zone(ctx)
        return {"action": "place_cube", "result": result_summary(result)}

    if decision.next_action == "recover":
        await move_velocity(ctx, vx=-0.15, duration_s=0.8)
        return {"action": "recover", "status": "stepped_back"}

    return {"action": decision.next_action, "status": "no_op"}


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

    for cycle in range(1, max_cycles + 1):
        print(f"\n[Level 1] Cycle {cycle}")
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

        observation = await observe_world(ctx, memory)
        decision = await decide_next_action(TASK, observation, memory, last_result)
        print("LLM decision:", decision)

        if decision.next_action == "stop":
            break

        action_result = await execute_decision(ctx, decision, observation, memory)
        verified = await verify_outcome(ctx, decision, action_result)
        update_memory(memory, observation, decision, verified)
        last_result = verified
        if tracker is not None:
            reason = await tracker.stop_reason_from_scene(ctx)
            if reason is not None:
                tracker.mark_ended(reason)
                print(f"Completion target reached after cycle action: {reason}.")
                break

    if tracker is not None:
        await tracker.print_summary_from_scene(ctx)
    return memory


async def run(ctx: Any) -> None:
    print(TASK)
    print("Level 1 adaptive-navigation project starter 실행")
    memory = await run_agent(
        ctx,
        max_cycles=10_000,
        completion=CompletionConfig(level=1, max_elapsed_s=600),
    )
    print("\n실행 완료.")
    print(f"Delivered count: {memory.delivered_count}")
    print("Logs:")
    for item in memory.logs:
        print(item)



