# Menlo Robot Workshop Programs

[English](README.md) | [한국어](README.ko.md)

이 저장소는 네 개의 workshop notebook과 같은 흐름으로 구성되어 있습니다.

- Workshop 1: SDK 기초, viewer, robot state, scene state, 기본 action
- Workshop 2: camera frame 기반 perception, HSV color blob, angle estimation, depth hook
- Workshop 3: global state와 vision-only control을 활용한 custom navigation
- Workshop 4: tool-calling LLM agent

학생용 notebook은 언어별로 분리되어 있습니다.

- `notebooks/student/en/`: 영어 학생용 notebook
- `notebooks/student/ko/`: 한국어 학생용 notebook

## 작업 방식 선택

### Option 1: Notebook / Google Colab

workshop을 notebook에서 진행하려면 이 방식을 사용하세요. 저장소를 clone하거나 Python을 로컬에 설치하거나 IDE scaffold를 설정할 필요가 없습니다.

1. 영어 또는 한국어 workshop notebook을 엽니다.
2. Google Colab에 업로드하거나 원하는 notebook 환경에서 엽니다.
3. notebook의 setup cell을 따라 package와 API key를 설정합니다. Colab에서는 key를 notebook cell에 직접 쓰지 말고 Secrets manager에 저장하세요.
4. 안내가 나오면 출력된 viewer URL을 Google Chrome에서 엽니다.

### Option 2: Local IDE Scaffold

VS Code, PyCharm 등 로컬 IDE에서 작업하려면 이 저장소를 clone하고 `menlo_runner/` 아래의 reusable module을 사용하세요.

## Local IDE Setup

이 폴더에서 package를 설치합니다.

```powershell
py -m pip install -e .
```

`.env.example`을 `.env`로 복사하거나 다음 내용으로 `.env`를 만듭니다.

```text
MENLO_API_KEY=...
TOKAMAK_API_KEY=...
```

`TOKAMAK_API_KEY`는 LLM/VLM agent 예제에서만 필요합니다.

## Workshop Demo 실행

`menlo-run` script가 PATH에 없다면 다음 long form을 사용하세요.

```powershell
py -m menlo_runner.cli basics-demo
py -m menlo_runner.cli perception-demo
py -m menlo_runner.cli navigation-demo
py -m menlo_runner.cli agent-demo
py -m menlo_runner.cli student-program
py -m menlo_runner.cli level-0-starter
py -m menlo_runner.cli level-1-starter
py -m menlo_runner.cli level-2-starter
py -m menlo_runner.cli level-0-starter-ko
py -m menlo_runner.cli level-1-starter-ko
py -m menlo_runner.cli level-2-starter-ko
```

demo는 simulated robot을 만들고 viewer URL을 출력합니다. Chrome에서 viewer를 연 뒤 선택한 program을 실행하고 마지막에 robot을 정리합니다.

## Interactive Session

하나의 robot과 viewer를 유지한 채 여러 workshop demo를 실행하려면 다음 명령을 사용하세요.

```powershell
py -m menlo_runner.cli session
```

사용 가능한 command:

```text
programs                 내장 program 목록 표시
run <program>            내장 program 실행
custom <module>          async def run(ctx)를 제공하는 custom module 실행
scene                    robot, pad, cube 상태 요약 출력
position                 robot 위치와 상태 출력
screenshot [path]        robot POV image 저장
skills                   viewer skill 목록 표시
viewer                   viewer URL 다시 출력
reset                    viewer UI의 reset button 사용
quit                     연결 해제, robot 삭제, 종료
```

## Module Map

- `menlo_runner.basics`: Workshop 1의 기본 SDK action wrapper
- `menlo_runner.perception`: Workshop 2 color blob detection, `perceive`, annotation, depth hook
- `menlo_runner.navigation`: Workshop 3 `turn_to_face`, `my_go_to_global`, `my_go_to_visual`
- `menlo_runner.agents`: Workshop 4 tool registry, executor, ReAct-style `WorkshopAgent`
- `menlo_runner.scene`: scene-state helper와 cube/pad utility
- `menlo_runner.programs`: student notebook에서 배운 개념을 실행 가능한 예제로 정리한 module

Exercise solution은 의도적으로 포함하지 않았습니다. Student notebook의 exercise cell은 직접 완성하세요. IDE에서 작업할 때는 같은 기능을 `student_program.py`에 작성할 수 있습니다.

최종 프로젝트는 `notebooks/project/`의 level별 starter notebook을 사용하세요.

영어 Python starter:

- `menlo_runner/programs/project/en/level_0_starter.py`
- `menlo_runner/programs/project/en/level_1_starter.py`
- `menlo_runner/programs/project/en/level_2_starter.py`

한국어 Python starter:

- `menlo_runner/programs/project/ko/level_0_starter_ko.py`
- `menlo_runner/programs/project/ko/level_1_starter_ko.py`
- `menlo_runner/programs/project/ko/level_2_starter_ko.py`
