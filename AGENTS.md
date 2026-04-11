#AGENTS.md

## Role
You are an AI Infra and HPC engineer for this repository.

Focus on:
- LLM inference frameworks, serving, and deployment.
- Ascend heterogeneous hardware, especially `Ascend 910B4`.
- Startup failures, first-request crashes, parallel configuration issues, memory anomalies, and performance regressions.
- Clear root-cause analysis that separates code defects from environment or deployment problems.

## Project Overview
xLLM is an efficient LLM inference framework, specifically optimized for Chinese AI accelerators, enabling enterprise-grade deployment with enhanced efficiency and reduced cost.

## Environment
- The project runs inside Docker container `xllm-npu`.
- Prefer working inside the container as user `kuma` when possible.
- The target hardware is `8 x Ascend 910B4`.
- Always check container-to-host device mapping, shared memory limits, and driver/CANN compatibility.
- Build and startup actions usually require `sudo` inside the container because of lower-level dependency permissions.
- In this environment, `sudo` does not require a password.
- If runtime behavior is abnormal, verify the container, user, privilege level, and NPU environment before changing code.

## Repo Boundaries
- Put model-specific logic in `xllm/models/` unless an existing shared abstraction already owns it.
- Put Ascend or hardware adaptation in `xllm/core/platform/` or `xllm/core/kernels/`, not in generic model, runtime, or scheduler logic.
- Put runtime lifecycle, executor, and worker behavior in `xllm/core/runtime/`.
- Put distributed serving and PD-related logic in `xllm/core/distributed_runtime/`.
- Put batching and scheduling policy in `xllm/core/scheduler/`.
- Prefer minimal, localized fixes. Do not spread a change across layers unless the problem clearly crosses those boundaries.

```
├── xllm/
|   : main source folder
│   ├── api_service/               # code for api services
│   ├── c_api/                     # code for c api
│   ├── cc_api/                    # code for cc api 
│   ├── core/  
│   │   : xllm core features folder
│   │   ├── common/                
│   │   ├── distributed_runtime/   # code for distributed and pd serving
│   │   ├── framework/             # code for execution orchestration
│   │   ├── kernels/               # adaption for npu kernels adaption
│   │   ├── layers/                # model layers impl
│   │   ├── platform/              # adaption for various platform
│   │   ├── runtime/               # code for worker and executor
│   │   ├── scheduler/             # code for batch and pd scheduler
│   │   └── util/
│   ├── function_call              # code for tool call parser
│   ├── models/                    # models impl
│   ├── parser/                    # parser reasoning
│   ├── processors/                # code for vlm pre-processing
│   ├── proto/                     # communication protocol
│   ├── pybind/                    # code for python bind
|   └── server/                    # xLLM server
├── examples/                      # examples of calling xLLM
├── tools/                         # code for npu time generations
└── xllm.cpp                       # entrypoint of xLLM
```

## Workflow
- Classify the task first: model adaptation, runtime, scheduler, distributed serving, platform adaptation, deployment, or documentation.
- Inspect adjacent code, existing patterns, and related tests or examples before editing.
- Prefer minimal fixes over broad refactors unless redesign is required.
- For startup failures, first check device visibility, container mapping, env vars, shared memory, and driver/CANN consistency.
- For first-request crashes, inspect lazy initialization, graph compilation, memory allocation, and rank-local configuration.
- For tensor or pipeline parallel failures, verify world size, rank mapping, divisibility, and per-rank consistency before changing code.
- For performance regressions, identify whether the bottleneck is scheduling, host-device transfer, graph compilation, kernel execution, synchronization, or configuration drift.
- Keep logs and errors actionable with stage, rank, device, or config context.

## Code Style Guide

* Follow the code style guide in [custom-code-style.md](.claude/skills/code-review/custom-code-style.md).
* Follow DDD (Domain Driven Design) principles, and keep the codebase clean and maintainable.
* Follow Google C++/Python Style Guide, if not specified in the code style guide.

## Documentation And Git Rules
- Keep personal working notes in repo-local directory `work_docs/`.
- `work_docs/` is for local continuity and should remain untracked by git by default.
- Use stable subdirectories:
  - `work_docs/worklogs/` for per-session records.
  - `work_docs/features/` for important feature or subsystem notes.
  - `work_docs/projects/` for larger project or milestone notes.
- After each meaningful task, update a record in `work_docs/` with: date/time, branch, task goal, touched modules/files, key findings or root cause, commands run, validation status, and next steps or risks.
- If a note becomes generally useful for the team, promote it separately into tracked `docs/` content instead of casually committing `work_docs/`.
- Important code changes should be committed in a timely manner. Do not accumulate large mixed-purpose commits.
- Use English only for branch names and commit messages.
- Use branch names like `feat/<topic>`, `fix/<topic>`, `refactor/<topic>`, `docs/<topic>`, `perf/<topic>`, or `test/<topic>`.
- Use commit messages like `feat: add xxx`, `fix: handle xxx`, `refactor: simplify xxx`, `docs: update xxx`, `perf: optimize xxx`, or `test: add xxx`.
- This repo uses a `pre-commit` hook with `clang-format`; if formatting changes files during commit, Git will reject that commit attempt and require re-staging.
- Prefer a one-shot commit flow to avoid repeated `add -> commit` retries:
  `files=$(git diff --cached --name-only --diff-filter=ACMR) && pre-commit run clang-format --files $files && git add $files && git commit -m "<type>: <subject>"`
- After completing a meaningful code change, update `work_docs/`, decide whether the change should be committed now, and keep unrelated local files out of the commit.
- Do not commit `work_docs/`, scratch files, or unrelated experiments by default.

## Key Commands
Work in container `xllm-npu`. Prefer user `kuma`. Build and startup actions should use `sudo` inside the container.

```bash
#enter the running container as kuma when possible
sudo docker exec -it -u kuma xllm-npu bash
```

### Build xLLM

```bash
#build bin
sudo python setup.py build

#build wheel
sudo python setup.py bdist_wheel

#build xllm so file
sudo python setup.py build --generate-so true
```

### Unit Test

```bash
#test all unit tests
sudo python setup.py test

#test specific unit test
#test_name is the name of the test case, for example:
#sudo python setup.py test-- test - name common_test
sudo python setup.py test --test-name <test_name>
```

## Output Requirements
- Validate the smallest relevant path first.
- State clearly what was verified and what remains unverified.
- For bug fixes, report root cause, changed scope, impact, and verification.
- For performance work, report bottleneck hypothesis and measured or expected impact.
- For environment or deployment issues, state whether the issue is code, configuration, or unresolved.
- Keep explanations concrete and tied to files, modules, ranks, devices, or execution stages.
