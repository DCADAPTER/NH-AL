# 30개 AL 신뢰성 해석 문헌의 4모듈 재정의
## Reference + 추가 NH 최적화 프레임워크

작성일: 2026-10-15  
기반 자료: 2026-10-14 작성한 30편 문헌 조사. 문헌 번호 P01–P30은 그대로 유지한다.  
이번 산출물: **논문별 모듈 매핑 + 공통 인터페이스/실험 규약 + 실행 가능한 연결 예제**. 전체 30편의 재현 구현을 완료했다는 뜻은 아니다.

> **이번 정의가 이전 조사와 달라지는 핵심:** NH는 기존 후보 풀 검색을 제거하는 모듈이 아니라, 기존 Reference 절차가 선택한 점에서 시작하여 같은 acquisition을 추가 최적화하는 플러그인이다. 따라서 방법명은 우선 **Reference + NH refinement**로 표현한다. 후보 풀을 없애는 방법은 별도의 연구 variant다.

## 1. 전체 구조

사용자가 제안한 네 모듈을 최상위 단위로 둔다.

| 최상위 모듈 | 소유하는 책임 | 다음 단계 |
|---|---|---|
| 1. g(x) 계산 | 문제 정의, 실제 해석, 입력/출력/실행상태/시간/호출수 저장 | 메타모델 학습 |
| 2. 메타모델 학습 | 모델 선택, 전처리, 고정된 학습정책, 학습자료로 fit, 예측·미분 제공 | 모델 평가 |
| 3. 적응 샘플링 | acquisition, Reference selector, gradient adapter, 선택적 NH, 최종점 선택·시간 기록 | g(x) 계산 |
| 4. 모델 평가 | Pf 추정, 학습/분류 지표, 종료 판단 | 종료 또는 적응 샘플링 |

NH는 **3번 모듈이 호출하는 독립 구현체**다. 메타모델을 직접 학습하거나 실제 g를 호출하지 않는다.

~~~mermaid
flowchart TD
    G["1. g(x) 계산·저장"] --> M["2. 메타모델 학습"]
    M --> E["4. 모델 평가"]
    E -->|계속| A["3. 적응 샘플링"]
    E -->|종료| R["결과 저장"]
    A -->|NH 사용| N["NH 추가 최적화"]
    N --> A
    A -->|최종 선택점| G
~~~

초기 DOE에 대한 g 호출로 시작한다. 위 A–N 왕복은 한 iteration당 **한 번의 선택적 호출**이며, NH 내부 trajectory가 끝나면 최종 선택점을 g 모듈로 넘긴다.

### Reference 경로

$$
\mathcal D_n \to \widehat g_n \to \mathrm{Evaluate}_n
\to \mathrm{ReferenceSelect}(a_n,\mathcal C_n)
\to x_{\mathrm{ref}}\to g(x_{\mathrm{ref}}).
$$

### NH 경로

$$
\mathcal D_n \to \widehat g_n \to \mathrm{Evaluate}_n
\to \mathrm{ReferenceSelect}(a_n,\mathcal C_n)
\to x_{\mathrm{ref}}
\to \mathrm{NHRefine}(x_{\mathrm{ref}},a_n,\nabla V_n)
\to x_{\mathrm{selected}}\to g(x_{\mathrm{selected}}).
$$

**공통 단계까지 동일하게 실행한 뒤 NH를 추가한다.** NH 경로에서도 x_ref와 a_n(x_ref)를 반드시 보관한다. x_ref를 별도로 g에 넣지는 않는다. 두 점을 모두 실제 평가하면 평가 예산과 batch 크기가 달라지기 때문이다.

## 2. 네 모듈의 입출력 계약

### 2.1 g(x) 계산 모듈

**입력:** 물리 좌표 x, 필요한 경우 component_id/time_id/replicate_id, run_id, iteration.  
**출력:** 실제 응답과 계산 레코드. 데이터 저장을 완료한 뒤 학습 모듈로 전달한다.

필수 저장 항목:

| 필드 | 정의 |
|---|---|
| x_physical / x_model | 실제 해석 입력과 전처리 좌표. 두 좌표계를 혼동하지 않음 |
| y_raw | scalar, component vector, trajectory 등 원 응답 |
| failure_label | 문제 정의에 따른 실패 여부. raw response와 함께 보관 |
| problem_id / definition_version | 분포·단위·threshold·system logic 포함 |
| wall_seconds / worker_id | 실제 계산시간과 실행 worker |
| call_id / n_calls_attempted / n_calls_success | 실패한 solver 호출도 비용에 포함 |
| status / error | success, solver_failed, invalid_input 등 |
| component_id / time_id / seed | 시스템·시간·stochastic simulator 추적정보 |
| is_cache_hit | 실제 재실행인지 구분 |

**중요:** 계산 실패를 g>0인 안전점으로 바꾸지 않는다. 분류 SVM을 사용해도 가능한 경우 원 g를 보관하고 label 변환은 고정된 규칙으로 수행한다.

문제 adapter는 최소한 domain, input distribution, failure event, g.evaluate를 제공한다. 시스템은 성분별 관측 마스크가 필요하며, 성분 하나를 계산하고 모든 성분을 계산한 것처럼 기록하지 않는다. stochastic simulator의 반복평가는 같은 x여도 다른 관측이므로 결정론적 cache와 구분한다.

### 2.2 메타모델 학습 모듈

**입력:** 성공한 g 평가자료, model/training config.  
**출력:** model snapshot, training log, prediction interface, capability 선언.

“타입별 학습법 고정”은 **학습 정책 고정**을 의미한다. Reference와 NH 양쪽에서 동일한 커널/기저/architecture, hyperparameter 최적화법, 재시작 수, early-stopping 규칙을 사용한다. 매 iteration fit된 hyperparameter 값까지 강제로 같게 하는 것은 아니다. NH로 학습자료가 달라지면 적합 결과도 달라질 수 있다.

Capability 예:

- mean, epistemic_variance, observation_noise_variance
- decision_function, ensemble_responses, conditional_failure_probability
- input_gradient, posterior_draws, component_predictions

SVM에 없는 GP variance를 임의로 만들어 반환하지 않는다. PCE bootstrap에는 ensemble을, SPCE에는 조건부 실패확률을, conformal 모델에는 구간을 제공하도록 별도 adapter를 둔다.

전처리는 데이터 누출 없이 fit한다. 한 iteration의 Reference→NH 동안 전처리와 모델을 **동결**한다. 입력이 x=T(z)로 변환되면 $\nabla_z V=J_T(z)^\top\nabla_x V$를 적용한다. 표준화된 모델에서 threshold 0의 위치도 일관되게 변환해야 한다.

### 2.3 적응 샘플링 모듈

다음 네 부분을 내부에 둔다.

1. CandidateProvider: 원 후보 분포·영역·batch context를 생성한다.
2. Acquisition: score(x), direction(min/max), required_capabilities를 제공한다.
3. ReferenceSelector: 논문 고유 정책에 따라 점 또는 batch/성분 tuple을 선택한다.
4. OptionalRefiner: NH가 추가 후보를 생성하고 원 score로 최종 선정한다.

Reference가 확률적 weighted sampling이면 그 정책을 그대로 유지한다. argmax로 바꾼 뒤 같은 Reference라고 부르지 않는다.

**Gradient는 g(x)의 gradient가 아니라 acquisition/energy의 입력 gradient다.** gradient 계산에 실제 고비용 g를 쓰는 구현은 별도 method/예산으로 구별한다.

NH optimizer에 전달하는 최소 계약:

~~~python
refine(
    x_reference,
    acquisition,          # 원 score와 방향, energy/gradient adapter
    frozen_context,      # model, scaler, posterior draws, stage/cluster
    constraints,
    rng,
    budget
) -> CandidateTrace
~~~

NH는 seed에서 추가 후보를 반환한다. 최종 실제 g 평가점은 적응 샘플링 모듈의 acceptance 규칙이 결정한다.

### 2.4 평가 모듈

평가 모듈 안에 **Pf estimator**와 **stopping policy**를 서브모듈로 둔다. 이는 최상위 네 모듈을 늘리는 것이 아니다.

- deterministic scalar: $P_f=P(g(X)\leq0)$.
- system: 정의된 논리조합으로 실패 event 계산.
- time-variant: $P_f(T)=P(\min_{t\in[0,T]}g(X,t)\leq0)$.
- stochastic response: $P_f=E_X[P(Y\leq0\mid X)]$.
- noisy latent model: 잠재 g의 실패와 노이즈 포함 관측 실패를 별도 target으로 정의.

MCS/IS/SS/line sampling은 estimator adapter가 소유한다. sampling 모듈과 공유해야 하는 proposal/level state는 명시적으로 전달한다. 모듈 분리가 동일한 i.i.d. MCS를 모든 문헌에 강제한다는 뜻은 아니다.

**종료 결과:** continue / converged / budget_exhausted / failed. 예산 소진을 정확도 수렴으로 표기하지 않는다. 검증을 위해 true g를 추가 호출하면 별도 검증예산과 실제 호출수를 모두 기록한다.

## 3. NH 전후 비교 규약

### 3.1 동일 상태의 점수 비교

iteration n의 model snapshot을 고정하고

$$
a_{\mathrm{ref}}=a_n(x_{\mathrm{ref}}),\qquad
a_{\mathrm{NH}}=a_n(x_{\mathrm{NH}})
$$

를 저장한다. 최소화 문제이면 gain은 $a_{\mathrm{ref}}-a_{\mathrm{selected}}$, 최대화이면 반대 차이다.

기본 acceptance:

$$
x_{\mathrm{selected}}
=\operatorname{best}_{a_n}\bigl(\{x_{\mathrm{ref}}\}\cup
\{\text{유효한 NH 후보}\}\bigr).
$$

- 원 Reference 점을 항상 포함한다.
- domain, stage/level, component, batch 중복 제약을 재검사한다.
- 원 score가 같으면 Reference를 유지한다. 다른 tie-break를 쓰면 명시한다.
- 개선점이 없거나 NH가 수치적으로 실패하면 Reference로 되돌아간다.
- smooth proxy로 이동해도 acceptance는 원 acquisition으로 한다.

이 규약은 **해당 iteration의 acquisition 값이 나빠지지 않는다는 것만** 보장한다. 최종 Pf 오차나 실제 g 호출수가 반드시 좋아진다는 보장은 아니다.

### 3.2 U용 gradient 예시

$v=\sigma^2>0$, $V=U^2=\mu^2/v$라면

$$
\nabla V = \frac{2\mu}{v}\nabla\mu
-\frac{\mu^2}{v^2}\nabla v.
$$

예측분산 floor는 정의와 derivative를 일치시킨다. energy transformation, variance floor, step size는 config에 기록한다. 별도의 밀도 가중치나 기존 NHSVM repulsion을 추가하면 원 acquisition 위의 추가 최적화와 구분하여 **energy-proxy variant**로 표시한다.

### 3.3 Sampling 시간 정의

$$
t_{\mathrm{sampling,NH}}=
t_{\mathrm{candidate}}+
t_{\mathrm{reference}}+
t_{\mathrm{NH}}+
t_{\mathrm{rescore/accept}}.
$$

Reference의 시간은 앞의 두 항이다. 따라서 **현재 설계에서는 NH를 추가한 한 iteration이 Reference보다 저렴하다고 전제하면 안 된다.** 연구 가설은 추가 탐색비용을 지불하여 실제 g 호출수, 전체 iteration, 최종 Pf 오차 또는 전체 wall time에서 이득을 얻는가이다.

필수 로그:

~~~text
run_id, iteration, model_snapshot_id, acquisition_id, direction
x_reference, score_reference, x_nh_best, score_nh_best
x_selected, score_selected, acquisition_gain, selection_reason
candidate_seconds, reference_seconds, nh_seconds, rescore_seconds
n_score_points, n_energy_points, n_gradient_points, n_g_calls
seed, component_id, time_id, cluster_id, stage_id
gradient_kind, energy_definition, proxy_used, failure_status
~~~

### 3.4 두 종류의 비교실험

**Frozen-state 비교:** 같은 D_n·모델·후보 풀에서 Reference와 NH 전후 score/시간을 비교한다. 그 iteration에서의 직접 개선효과를 측정한다.

**End-to-end 비교:** 같은 초기 DOE와 예산, 난수 stream 설계로 Reference run과 NH run을 각각 실행한다. 첫 선택 후 학습자료가 달라지므로 이후의 서로 다른 모델에서 계산한 score를 같은 목적함수의 직접 개선량처럼 비교하지 않는다. NH run 내부에 저장한 shadow Reference score가 직접 비교 기준이다.

Batch, portfolio, system, time-dependent 방법은 이산 선택을 고정한 x-refinement를 기본 variant로 둔다. component/time/cluster까지 함께 바꾸는 방법은 별도의 실험으로 이름을 붙인다.

## 4. g 모듈 adapter 종류

| adapter | 실제 계산과 저장 | 실패 event |
|---|---|---|
| scalar_deterministic | scalar g와 원 입력 | $g \leq 0$ |
| scalar_classification | 가능한 경우 raw g + sign label | 지정된 failure label |
| component_system | g_k 또는 vector, 관측 마스크와 성분 비용 | series/parallel/arbitrary logic |
| noisy_latent | noisy y와 noise metadata; latent truth와 구별 | 설정된 잠재 g의 실패 |
| stochastic_response | 동일 x에서 달라질 수 있는 Y, replicate/seed | 조건부 확률을 적분 |
| time_trajectory | trajectory 또는 (x,t) 응답과 평가 범위 | 시간구간 중 실패 |

각 논문이 사용한 benchmark 하나를 논문 자체와 동일시하지 않는다. profile은 모델·학습/선택/평가 정책을 정의하고, problem은 별도 지정한다. 서로 호환되지 않는 problem/profile 조합은 실행 전에 차단한다.

## 5. 30편의 통일 모듈 매핑

아래 표에서 **원문 기반 설명**은 전날 조사에서 확인한 범위다. **NH 확장**과 모듈 경계는 이번 연구를 위한 제안이다. 수치 설정을 확인하지 못한 항목은 빈 설정/TODO로 남겨야 하며 임의로 원 논문 설정인 것처럼 채우지 않는다. 논문별 실행 profile이 되려면 마지막 “미확인” 항목을 먼저 해소해야 한다.

### P01. AK-MCS (2011)

[원문/출처](https://doi.org/10.1016/j.strusafe.2011.01.002)

| 구분 | 이 프레임워크에서의 정의 |
|---|---|
| 1. g(x) | `scalar_deterministic` adapter. §4의 실제 평가·저장 계약 적용. |
| 2. 메타모델 | 고정 학습정책의 Kriging/GP; predict_mean, predict_epistemic_var, 입력 미분 제공. |
| 3-A. Reference 샘플링 | MCS 후보 전체의 U 최소 또는 EFF 최대. 선택점과 원점수 저장. |
| 3-B. Acquisition | $U(x)=\|\mu(x)\|/\sigma(x)$ (min); EFF (max)를 별도 variant로 등록. |
| 3-C. Gradient | $U^2$의 analytic/autodiff gradient. EFF는 폭 $\epsilon(x)$의 미분까지 포함. |
| 3-D. NH 추가 최적화 **[제안]** | Reference 최선점으로 시작해 같은 U/EFF를 개선; U가 같을 때 기본적으로 Reference 유지. |
| 4. 평가·종료 | MCS Pf와 적분 COV; U 조건과 표본 정밀도 조건을 구분. 임계값은 variant 설정. |
| 구현 전 확인사항 | 원문별 kernel, DOE, 재학습·종료 수치; 동봉 GP-U는 공통 예제이며 원논문 완전 복제 아님. |

### P02. AK-IS (2013)

[원문/출처](https://doi.org/10.1016/j.ress.2012.10.008)

| 구분 | 이 프레임워크에서의 정의 |
|---|---|
| 1. g(x) | `scalar_deterministic` adapter. §4의 실제 평가·저장 계약 적용. |
| 2. 메타모델 | Kriging; mean/variance; IS 밀도는 메타모델과 별도 state. |
| 3-A. Reference 샘플링 | IS 관련 후보/단계별 selection을 보존; proposal id와 density 저장. |
| 3-B. Acquisition | AK learning 기준; 단계별 원식 확인 후 고정. |
| 3-C. Gradient | 점별 GP 기준 gradient + 필요한 log-density gradient. |
| 3-D. NH 추가 최적화 **[제안]** | IS proposal을 고정한 채 Reference 점을 추가 최적화. NH 점을 무가중 IS 표본으로 사용하지 않음. |
| 4. 평가·종료 | 가중 IS Pf, weight variance/COV, surrogate 오차를 평가. |
| 구현 전 확인사항 | 단계별 정확한 acquisition·support·종료식. |

### P03. AK-SYS (2014)

[원문/출처](https://doi.org/10.1016/j.ress.2013.10.010)

| 구분 | 이 프레임워크에서의 정의 |
|---|---|
| 1. g(x) | `component_system` adapter. §4의 실제 평가·저장 계약 적용. |
| 2. 메타모델 | 성분별 Kriging; 각 g_k의 별도 데이터 마스크와 모델. |
| 3-A. Reference 샘플링 | 시스템 논리로 (x, component_id)를 선택. |
| 3-B. Acquisition | 시스템·성분 선택 함수; 단순 min U_k로 대체 금지. |
| 3-C. Gradient | 성분별 연속 gradient; 시스템 min/max는 piecewise 또는 명시적 proxy. |
| 3-D. NH 추가 최적화 **[제안]** | component_id를 고정하고 x만 refine; 최종 시스템 admissibility 재검사. |
| 4. 평가·종료 | 시스템 indicator로 Pf 평가; 시스템 학습오차와 성분 호출비용 기록. |
| 구현 전 확인사항 | 시스템별 원 학습식·성분 선택식·종료식. |

### P04. AK-SS (2016)

[원문/출처](https://doi.org/10.1016/j.strusafe.2015.12.003)

| 구분 | 이 프레임워크에서의 정의 |
|---|---|
| 1. g(x) | `scalar_deterministic` adapter. §4의 실제 평가·저장 계약 적용. |
| 2. 메타모델 | Kriging + 현재 SS level b_j를 별도 state로 전달. |
| 3-A. Reference 샘플링 | 현재 SS 수준의 조건부 후보 중 선택; chain/level id 유지. |
| 3-B. Acquisition | 수준별 경계 불확실성; 공통 표기는 $U_j(x)=|\mu(x)-b_j|/\sigma(x)$. |
| 3-C. Gradient | 현재 b_j를 고정한 점별 gradient. |
| 3-D. NH 추가 최적화 **[제안]** | level을 고정하고 허용 조건부 영역 안에서 추가 탐색. integration chain에는 NH 궤적을 섞지 않음. |
| 4. 평가·종료 | SS 조건부 확률 곱, 상관을 반영한 불확실성; 수준 종료와 전체 종료 분리. |
| 구현 전 확인사항 | 원 단계별 scoring·level transition·종료 수치. |

### P05. H learning (2015)

[원문/출처](https://doi.org/10.1016/j.camwa.2015.07.004)

| 구분 | 이 프레임워크에서의 정의 |
|---|---|
| 1. g(x) | `scalar_deterministic` adapter. §4의 실제 평가·저장 계약 적용. |
| 2. 메타모델 | Kriging; Gaussian response density 제공. |
| 3-A. Reference 샘플링 | H 기준 후보 선택; 원문 line-sampling 결합을 보존. |
| 3-B. Acquisition | 경계 응답구간의 -p(y|x) log p(y|x) 적분, max. |
| 3-C. Gradient | 적분 경계 epsilon(x) 및 mean/variance의 미분. Bernoulli entropy로 대체 금지. |
| 3-D. NH 추가 최적화 **[제안]** | Reference 점에서 H 추가 최대화; H가 음수 가능하므로 log(H) energy 금지. |
| 4. 평가·종료 | 원 적분법의 Pf·오차를 평가; i.i.d. MCS COV로 통일하지 않음. |
| 구현 전 확인사항 | 구간 폭, 정규화, 후보 생성, stop 원식. |

### P06. LIF (2017)

[원문/출처](https://doi.org/10.1016/j.ress.2016.09.003)

| 구분 | 이 프레임워크에서의 정의 |
|---|---|
| 1. g(x) | `scalar_deterministic` adapter. §4의 실제 평가·저장 계약 적용. |
| 2. 메타모델 | Kriging; mean/variance 및 필요한 국소 적분정보. |
| 3-A. Reference 샘플링 | 원 LIF 후보 절차·영역 선별 유지. |
| 3-B. Acquisition | 국소 오분류 개선·입력 확률질량을 이용한 LIF; 원식 등록 필요. |
| 3-C. Gradient | 내부 적분/국소영역 의존성 포함. 단순 $f_X(x)\Phi(-U(x))$로 대체 금지. |
| 3-D. NH 추가 최적화 **[제안]** | Reference 이후 동일 LIF 추가 최적화; inner quadrature/난수는 iteration 내 고정. |
| 4. 평가·종료 | 원 Pf 추정 및 stop; 내부 표본 비용 별도 집계. |
| 구현 전 확인사항 | 정확한 LIF 식·방향·반경·적분·종료식. |

### P07. REIF / REIF2 (2019)

[원문/출처](https://doi.org/10.1016/j.ress.2019.01.014)

| 구분 | 이 프레임워크에서의 정의 |
|---|---|
| 1. g(x) | `scalar_deterministic` adapter. §4의 실제 평가·저장 계약 적용. |
| 2. 메타모델 | Kriging; folded-normal 계산에 필요한 mean/variance. |
| 3-A. Reference 샘플링 | 대표·저불일치·절단 후보의 REIF 또는 REIF2 argmax. |
| 3-B. Acquisition | REIF와 density 보정 REIF2를 서로 다른 variant로 정의. |
| 3-C. Gradient | folded-normal 식과 REIF2 density 미분; 절단영역 제약 유지. |
| 3-D. NH 추가 최적화 **[제안]** | Reference와 같은 variant 및 유효영역에서 refine. |
| 4. 평가·종료 | 원 reliability estimator 및 truncation/학습 종료를 각각 기록. |
| 구현 전 확인사항 | folded-normal 평균만으로 전체 acquisition 복원 금지; 계수·보정·stop 확인. |

### P08. FNEIF (2020)

[원문/출처](https://www.sciencedirect.com/science/article/pii/S0951832019304788)

| 구분 | 이 프레임워크에서의 정의 |
|---|---|
| 1. g(x) | `scalar_deterministic` adapter. §4의 실제 평가·저장 계약 적용. |
| 2. 메타모델 | Kriging; mean/variance. |
| 3-A. Reference 샘플링 | FNEIF 최대 후보 선택. |
| 3-B. Acquisition | Folded-normal expected improvement, max. |
| 3-C. Gradient | 저자 supplementary 원식을 이식한 후 analytic/autodiff와 수치미분 대조. |
| 3-D. NH 추가 최적화 **[제안]** | 동일 FNEIF를 추가 최대화; REIF 구현을 재사용한 것으로 간주하지 않음. |
| 4. 평가·종료 | 원 stop와 Pf 정밀도 조건; supplementary 설정 확인. |
| 구현 전 확인사항 | FNEIF threshold·전체 식·종료식; supplementary 확보 우선. |

### P09. REAK (2019)

[원문/출처](https://doi.org/10.1016/j.ress.2018.10.004)

| 구분 | 이 프레임워크에서의 정의 |
|---|---|
| 1. g(x) | `scalar_deterministic` adapter. §4의 실제 평가·저장 계약 적용. |
| 2. 메타모델 | Kriging; 잘못 분류될 가능성과 오류 추정에 필요한 출력. |
| 3-A. Reference 샘플링 | 유효영역에서 원 enrichment policy 수행. |
| 3-B. Acquisition | 점별 criterion과 effective-region 필터를 별도 등록. |
| 3-C. Gradient | 점별 score 미분; 현재 effective region은 고정. |
| 3-D. NH 추가 최적화 **[제안]** | 유효영역 안에서 추가 탐색; 원 error audit용 표본은 유지. |
| 4. 평가·종료 | 오분류에 의한 Pf error estimate를 중심으로 stop. MC 분산과 별도 기록. |
| 구현 전 확인사항 | 정확한 점수·effective region update·error bound 구현. |

### P10. AK-MCSi (2018)

[원문/출처](https://doi.org/10.1016/j.strusafe.2018.01.002)

| 구분 | 이 프레임워크에서의 정의 |
|---|---|
| 1. g(x) | `scalar_deterministic` adapter. §4의 실제 평가·저장 계약 적용. |
| 2. 메타모델 | Kriging; batch 데이터 추가 후 동일 학습정책 적용. |
| 3-A. Reference 샘플링 | 후보 관리·군집·배치 대표점 선택 유지. |
| 3-B. Acquisition | 점별 AK score + batch 선정 규칙을 분리. |
| 3-C. Gradient | 점별 score gradient; 군집/대표점 선택 자체는 별도 이산 정책. |
| 3-D. NH 추가 최적화 **[제안]** | Reference 배치 각 seed를 refine하고 중복·군집 제약 재검사. |
| 4. 평가·종료 | Pf/COV·학습 stop 및 배치 비용; 실제 호출수와 wall time 구분. |
| 구현 전 확인사항 | 군집 가중치·정확한 variant·stop. |

### P11. RLCB (2020)

[원문/출처](https://doi.org/10.1007/s00158-020-02622-3)

| 구분 | 이 프레임워크에서의 정의 |
|---|---|
| 1. g(x) | `scalar_deterministic` adapter. §4의 실제 평가·저장 계약 적용. |
| 2. 메타모델 | Kriging; mean/variance 및 density weight. |
| 3-A. Reference 샘플링 | 원 RLCB 선택 및 후보생성 유지. |
| 3-B. Acquisition | Reliability lower confidence bounding; 원식·방향 등록. |
| 3-C. Gradient | mean/variance뿐 아니라 PDF 기반 가중치 미분 포함. |
| 3-D. NH 추가 최적화 **[제안]** | iteration의 weight/계수는 고정하고 원 RLCB만 추가 최적화. |
| 4. 평가·종료 | 논문의 상대 Pf error 기반 stop와 estimator 사용. |
| 구현 전 확인사항 | 일반 BO의 $\mu-\kappa\sigma$로 대체 금지; 정확한 수식·threshold. |

### P12. Portfolio allocation (2023)

[원문/출처](https://www.sciencedirect.com/science/article/pii/S0045782523001901)

| 구분 | 이 프레임워크에서의 정의 |
|---|---|
| 1. g(x) | `scalar_deterministic` adapter. §4의 실제 평가·저장 계약 적용. |
| 2. 메타모델 | Kriging; portfolio의 공통 예측정보. |
| 3-A. Reference 샘플링 | portfolio가 함수 id를 결정한 뒤 해당 Reference selector 실행. |
| 3-B. Acquisition | 여러 learning function + 선택/배분 정책. |
| 3-C. Gradient | 선택된 함수의 gradient; 함수 id와 보상 state 고정. |
| 3-D. NH 추가 최적화 **[제안]** | 선택된 acquisition_id를 고정하고 refine. NH 후 정책 업데이트는 실제 관측을 얻은 뒤 수행. |
| 4. 평가·종료 | 원 Pf/stop 및 portfolio 보상 기록. |
| 구현 전 확인사항 | portfolio 구성·reward·할당식·stop. |

### P13. Noisy-model U_N (2024)

[원문/출처](https://arxiv.org/html/2401.10796v1)

| 구분 | 이 프레임워크에서의 정의 |
|---|---|
| 1. g(x) | `noisy_latent` adapter. §4의 실제 평가·저장 계약 적용. |
| 2. 메타모델 | GP; epistemic variance와 observation noise variance를 분리. |
| 3-A. Reference 샘플링 | 원 후보/SS 및 batch 절차에서 U_N 최대 선택. |
| 3-B. Acquisition | $U_N(x)=\Phi\!\left(-\frac{|\mu(x)|}{\sigma(x)}\right)-\Phi\!\left(-\frac{|\mu(x)|\sqrt{\sigma^2(x)+\sigma_{\mathrm{noise}}^2(x)}}{\sigma(x)\sigma_{\mathrm{noise}}(x)}\right)$, max. |
| 3-C. Gradient | noise model과 평균·분산을 고정한 analytic/autodiff; 절댓값 주의. |
| 3-D. NH 추가 최적화 **[제안]** | 원 batch 구조를 유지한 추가 최적화; 반복관측 허용 여부는 실험정책으로 고정. |
| 4. 평가·종료 | 잠재 deterministic g의 Pf와 추정오차. 물리적 stochastic-output Pf와 구별. |
| 구현 전 확인사항 | 후보/cluster·원 종료값; noise가 0인 극한 처리. |

### P14. Arbitrary systems (2024)

[원문/출처](https://doi.org/10.1016/j.ress.2024.110150)

| 구분 | 이 프레임워크에서의 정의 |
|---|---|
| 1. g(x) | `component_system` adapter. §4의 실제 평가·저장 계약 적용. |
| 2. 메타모델 | 성분 Kriging/PCK; 성분별 학습데이터와 불확실성. |
| 3-A. Reference 샘플링 | SS·DBSCAN·성분 sensitivity를 이용하는 원 선택. |
| 3-B. Acquisition | 시스템 score + component selection + cluster policy. |
| 3-C. Gradient | component/cluster를 고정한 x-gradient; 논리경계 처리. |
| 3-D. NH 추가 최적화 **[제안]** | Reference tuple의 이산 항목은 고정, x refine 후 동일 군집·성분 유효성 검사. |
| 4. 평가·종료 | arbitrary system indicator·SS Pf·시스템 오차 판단. |
| 구현 전 확인사항 | 시스템 score·민감도 사용방식·stop 원식. |

### P15. AK-MCS-C2 (2026 preprint)

[원문/출처](https://arxiv.org/html/2606.20191v1)

| 구분 | 이 프레임워크에서의 정의 |
|---|---|
| 1. g(x) | `scalar_deterministic` adapter. §4의 실제 평가·저장 계약 적용. |
| 2. 메타모델 | Kriging + cross-conformal 구간 [m,M]; calibration state 고정. |
| 3-A. Reference 샘플링 | 0을 포함하는 구간의 후보 중 (M-m)/2 최대. |
| 3-B. Acquisition | $a(x)=(M(x)-m(x))/2$, max; admissible iff $m(x)\leq 0\leq M(x)$. |
| 3-C. Gradient | quantile/구간 gate가 비매끄러움; exact piecewise 또는 명시적 proxy. |
| 3-D. NH 추가 최적화 **[제안]** | ambiguous set 제약을 유지해 refine; gate 위반점 배제. |
| 4. 평가·종료 | $P_f^{\mathrm{lower}}=\operatorname{mean}(\mathbf{1}[M\leq 0])$, $P_f^{\mathrm{upper}}=\operatorname{mean}(\mathbf{1}[m\leq 0])$; gap과 적분오차. 보장조건 별도 확인. |
| 구현 전 확인사항 | v1 상하한 부호 문제, 적응 calibration 유효성; 정확한 미분 방식. |

### P16. ASVM-MCS (2017)

[원문/출처](https://doi.org/10.1016/j.strusafe.2017.04.006)

| 구분 | 이 프레임워크에서의 정의 |
|---|---|
| 1. g(x) | `scalar_classification` adapter. §4의 실제 평가·저장 계약 적용. |
| 2. 메타모델 | SVM classifier; raw g 저장 후 label 생성. kernel/C/전처리/CV 정책 고정. |
| 3-A. Reference 샘플링 | 원 경계 근접성·다양성 score로 MCS 후보 선택. |
| 3-B. Acquisition | SVM decision margin과 기존 표본 거리 결합; 원식 필요. |
| 3-C. Gradient | RBF decision gradient + nearest-neighbor 거리의 piecewise gradient. |
| 3-D. NH 추가 최적화 **[제안]** | Reference 점을 seed로 원 score 추가 최적화; 기존 NH의 별도 repulsion 추가는 별도 variant. |
| 4. 평가·종료 | SVM 실패분류 기반 MCS Pf; 원 learning stop 별도 확인. |
| 구현 전 확인사항 | 정확한 거리 정규화·결합식·원 stop. abs(s)만 쓰는 baseline과 구별. |

### P17. SS-SVM (2011)

[원문/출처](https://doi.org/10.1016/j.strusafe.2011.06.001)

| 구분 | 이 프레임워크에서의 정의 |
|---|---|
| 1. g(x) | `scalar_classification` adapter. §4의 실제 평가·저장 계약 적용. |
| 2. 메타모델 | SS 수준별 label을 쓰는 SVM; margin 제공, GP variance는 없음. |
| 3-A. Reference 샘플링 | SS level별 조건부 후보 및 원 adaptive classifier 선택. |
| 3-B. Acquisition | 원 SVM enrichment criterion; 수준별 정의 필요. |
| 3-C. Gradient | 고정 b_j와 classifier의 입력 gradient; 원식 미확인 시 실행 보류. |
| 3-D. NH 추가 최적화 **[제안]** | 조건부 영역과 level 유지 후 refine; HMC-SS sampler와 NH optimizer 분리. |
| 4. 평가·종료 | SS 조건부 확률과 상관된 표본오차; 수준/전체 종료. |
| 구현 전 확인사항 | 수준별 selection·threshold·stop. |

### P18. Sequential adaptive SVR (2020)

[원문/출처](https://doi.org/10.1016/j.ress.2020.106948)

| 구분 | 이 프레임워크에서의 정의 |
|---|---|
| 1. g(x) | `scalar_deterministic` adapter. §4의 실제 평가·저장 계약 적용. |
| 2. 메타모델 | SVR regression; kernel/regularization/epsilon 및 고정 tuning protocol. |
| 3-A. Reference 샘플링 | MCS 후보를 단계적으로 선별한 뒤 원 최종선택. |
| 3-B. Acquisition | stage filter + 최종 score; 표준 GP-U로 치환 금지. |
| 3-C. Gradient | 최종 scalar score의 gradient; stage 조건은 고정/유효성 검사. |
| 3-D. NH 추가 최적화 **[제안]** | Reference가 결정한 stage/영역 안에서 x 추가 최적화. |
| 4. 평가·종료 | SVR 예측 sign으로 Pf; 원 학습 종료식 확인. |
| 구현 전 확인사항 | 각 stage·최종 score·stop 수식. |

### P19. Bayesian SVR: Cheng–Lu (2021)

[원문/출처](https://doi.org/10.1016/j.ress.2020.107286)

| 구분 | 이 프레임워크에서의 정의 |
|---|---|
| 1. g(x) | `scalar_deterministic` adapter. §4의 실제 평가·저장 계약 적용. |
| 2. 메타모델 | Bayesian LS-SVR/epsilon-SVR의 원 variant; 예측 mean/variance. |
| 3-A. Reference 샘플링 | 원 candidate rule 및 Bayesian learning score. |
| 3-B. Acquisition | Bayesian SVR의 원 acquisition; variance 제공만으로 U로 단정 금지. |
| 3-C. Gradient | 원 Bayesian predictive moments와 score의 미분. |
| 3-D. NH 추가 최적화 **[제안]** | Reference와 같은 posterior/variant에서 refine. |
| 4. 평가·종료 | 원 Pf 계산법·분류오차/종료 기준. |
| 구현 전 확인사항 | learning 식·candidate law·정확한 training/stop. |

### P20. ABSVR: Wang et al. (2021)

[원문/출처](https://doi.org/10.1016/j.cma.2021.114172)

| 구분 | 이 프레임워크에서의 정의 |
|---|---|
| 1. g(x) | `scalar_deterministic` adapter. §4의 실제 평가·저장 계약 적용. |
| 2. 메타모델 | 논문 고유 Bayesian SVR; 19번과 별도 model/training profile. |
| 3-A. Reference 샘플링 | 원 ABSVR learning 및 후보선정 구현. |
| 3-B. Acquisition | 원 learning function; PDF 포함 여부와 계수 확정 필요. |
| 3-C. Gradient | posterior·score 미분; 확인 전 구현 차단. |
| 3-D. NH 추가 최적화 **[제안]** | Reference 출력 이후 추가 optimizer만 연결. |
| 4. 평가·종료 | 원 reliability estimator 및 수렴판정. |
| 구현 전 확인사항 | accepted manuscript의 score·학습정책·stop 확보. |

### P21. Bootstrap PCE (2018)

[원문/출처](https://doi.org/10.1016/j.strusafe.2018.06.003)

| 구분 | 이 프레임워크에서의 정의 |
|---|---|
| 1. g(x) | `scalar_deterministic` adapter. §4의 실제 평가·저장 계약 적용. |
| 2. 메타모델 | sparse PCE bootstrap ensemble; 기저·차수·희소선택·bootstrap 정책 고정. |
| 3-A. Reference 샘플링 | FBR 최소 및 원 batch 군집 규칙. |
| 3-B. Acquisition | $\mathrm{FBR}=|B_{\mathrm{safe}}-B_{\mathrm{fail}}|/B=|1-2p_{\mathrm{vote}}|$, min. |
| 3-C. Gradient | hard vote는 미분 거의 0; smooth proxy gradient 사용 여부 명시. |
| 3-D. NH 추가 최적화 **[제안]** | Reference batch seed에서 proxy로 이동하되 원 FBR로 accept; 동률은 Reference 유지. |
| 4. 평가·종료 | ensemble 실패분류/Pf와 원 bootstrap stopping criterion. |
| 구현 전 확인사항 | 원 stop·군집 설정; proxy-NH는 원 score의 정확한 gradient 방법 아님. |

### P22. EIFPE-PCE (2020)

[원문/출처](https://doi.org/10.1002/nme.6351)

| 구분 | 이 프레임워크에서의 정의 |
|---|---|
| 1. g(x) | `scalar_deterministic` adapter. §4의 실제 평가·저장 계약 적용. |
| 2. 메타모델 | PCE basis covariance 기반 Gaussian prior/posterior; $\mathrm{mean}=\mathrm{PCE}$ 및 variance. |
| 3-A. Reference 샘플링 | EIFPE가 큰 후보 선택; 원 후보분포 확인. |
| 3-B. Acquisition | Expected indicator function prediction error, max; variance-bias 구성. |
| 3-C. Gradient | variance/bias terms 모두 미분; 동적 weight는 iteration 내 고정. |
| 3-D. NH 추가 최적화 **[제안]** | 동일 EIFPE 위에서 추가 최대화; expected improvement로 오기하지 않음. |
| 4. 평가·종료 | 원 Pf 및 accuracy/종료 지표. |
| 구현 전 확인사항 | 전체 EIFPE·bias estimate·weight·stop. |

### P23. Active sparse PCE systems (2020)

[원문/출처](https://doi.org/10.1016/j.ress.2020.107025)

| 구분 | 이 프레임워크에서의 정의 |
|---|---|
| 1. g(x) | `component_system` adapter. §4의 실제 평가·저장 계약 적용. |
| 2. 메타모델 | BCS sparse PCE; 성분/system posterior와 기저선정 정책. |
| 3-A. Reference 샘플링 | 시스템 복합 learning으로 점/성분 선택. |
| 3-B. Acquisition | 논문의 composite learning function. |
| 3-C. Gradient | PCE polynomial gradient + system piecewise 처리. |
| 3-D. NH 추가 최적화 **[제안]** | 성분 선택 보존 후 위치 추가 최적화. |
| 4. 평가·종료 | 시스템 Pf와 원 수렴기준; 성분 호출 마스크 기록. |
| 구현 전 확인사항 | 시스템 score·normalization·후보·stop. |

### P24. PAS-PCE (2022)

[원문/출처](https://www.sciencedirect.com/science/article/abs/pii/S0951832022003726)

| 구분 | 이 프레임워크에서의 정의 |
|---|---|
| 1. g(x) | `scalar_deterministic` adapter. §4의 실제 평가·저장 계약 적용. |
| 2. 메타모델 | prediction-oriented sparse PCE; 원 기저선택/회귀 알고리즘 고정. |
| 3-A. Reference 샘플링 | prediction-oriented optimality criterion으로 후보 선택. |
| 3-B. Acquisition | 예측용 optimality 및 국소 불확실성 기준; 전체 식 확인. |
| 3-C. Gradient | 현재 design matrix/기저 고정 후 후보 위치에 대한 미분. |
| 3-D. NH 추가 최적화 **[제안]** | Reference point를 refine; virtual design update는 실제 g 호출 없이 수행. |
| 4. 평가·종료 | PCE sign 기반 Pf와 원 adaptive design stop. |
| 구현 전 확인사항 | 정확한 optimality·선택방향·matrix update·stop. |

### P25. Enhanced bPCE (2024)

[원문/출처](https://doi.org/10.1016/j.strusafe.2024.102467)

| 구분 | 이 프레임워크에서의 정의 |
|---|---|
| 1. g(x) | `scalar_deterministic` adapter. §4의 실제 평가·저장 계약 적용. |
| 2. 메타모델 | bootstrap sparse PCE; ensemble 및 기저 선택 고정. |
| 3-A. Reference 샘플링 | 오분류 가능성과 joint PDF를 반영한 원 선택. |
| 3-B. Acquisition | misclassification + PDF의 논문 고유 결합. |
| 3-C. Gradient | hard vote + density; proxy를 사용하면 별도 variant. |
| 3-D. NH 추가 최적화 **[제안]** | 동일 density 및 원 score로 accept; 임의 단순 곱으로 변경 금지. |
| 4. 평가·종료 | 원 bootstrap Pf/stop; MC 정밀도 분리. |
| 구현 전 확인사항 | 정확한 결합·정규화·stop. |

### P26. AL-SPCE (2025/2026)

[원문/출처](https://arxiv.org/html/2507.04553v1)

| 구분 | 이 프레임워크에서의 정의 |
|---|---|
| 1. g(x) | `stochastic_response` adapter. §4의 실제 평가·저장 계약 적용. |
| 2. 메타모델 | stochastic PCE; 조건부 failure s(x), 계수 불확실성 sample 제공. |
| 3-A. Reference 샘플링 | 재표집 MCS 후보 + weighted K-medoids 및 cluster별 최댓점. |
| 3-B. Acquisition | $a(x)=f_X(x)\operatorname{Var}_{\mathrm{ep}}[\hat{s}(x)]$, max. |
| 3-C. Gradient | density와 계수표본 기반 variance의 gradient; 계수 draws 고정. |
| 3-D. NH 추가 최적화 **[제안]** | 각 cluster seed를 refine; cluster/중복 constraints 유지. 원 s 분산 사용. |
| 4. 평가·종료 | $P_f=\mathbb{E}_X[\hat{s}(X)]$를 고정 적분풀에서 계산; epistemic 오류와 적분 SE 분리. |
| 구현 전 확인사항 | 원 stop·batch 수치; s(1-s)를 epistemic variance로 쓰지 않음. |

### P27. PC-Kriging (2016/2017)

[원문/출처](https://doi.org/10.1061/AJRUA6.0000870)

| 구분 | 이 프레임워크에서의 정의 |
|---|---|
| 1. g(x) | `scalar_deterministic` adapter. §4의 실제 평가·저장 계약 적용. |
| 2. 메타모델 | PCE trend + GP residual; 기저선정·kernel fitting protocol 고정. |
| 3-A. Reference 샘플링 | AK-MCS형 learning을 PCK 예측으로 수행. |
| 3-B. Acquisition | U/EFF 등 선택한 variant의 원식. |
| 3-C. Gradient | PCE trend와 GP residual의 미분 모두 포함. |
| 3-D. NH 추가 최적화 **[제안]** | Reference 후 동일 PCK posterior의 score 추가 최적화. |
| 4. 평가·종료 | PCK 기반 MCS Pf·learning stop·COV. |
| 구현 전 확인사항 | PCK variant 및 원 학습/종료 설정. |

### P28. MC-dropout AL (2025)

[원문/출처](https://doi.org/10.1016/j.ress.2025.111169)

| 구분 | 이 프레임워크에서의 정의 |
|---|---|
| 1. g(x) | `scalar_deterministic` adapter. §4의 실제 평가·저장 계약 적용. |
| 2. 메타모델 | dropout NN; architecture/optimizer/epoch·early-stop·마스크 수 정책 고정. |
| 3-A. Reference 샘플링 | 논문 고유 불확실성 learning; input candidate law 확인 필요. |
| 3-B. Acquisition | 원 acquisition 미확인; U라고 추정하지 않음. |
| 3-C. Gradient | iteration 동안 dropout masks 고정 후 input autodiff. |
| 3-D. NH 추가 최적화 **[제안]** | Reference와 NH가 동일 masks·모델 snapshot 사용; 진단 resampling은 별도 기록. |
| 4. 평가·종료 | 논문의 두 convergence criteria 및 Pf estimator 확인. |
| 구현 전 확인사항 | input pool·acquisition·두 stop 식·training settings. |

### P29. DBAL time-variant (2026)

[원문/출처](https://doi.org/10.1007/s00158-026-04351-5)

| 구분 | 이 프레임워크에서의 정의 |
|---|---|
| 1. g(x) | `time_trajectory` adapter. §4의 실제 평가·저장 계약 적용. |
| 2. 메타모델 | BDNN; g(x,t) 또는 trajectory output 모델, posterior draws 고정. |
| 3-A. Reference 샘플링 | 예측 극값점·예측 실패점에서 batch 대표점 선택. |
| 3-B. Acquisition | 대표점 score + trajectory/time selection. |
| 3-C. Gradient | 시간 index를 고정한 x-gradient 또는 연속 시간 joint variant 구분. |
| 3-D. NH 추가 최적화 **[제안]** | 기본형은 time_id 고정 후 x refine; joint time optimization은 별도 변경. |
| 4. 평가·종료 | $P_f(T)=P\!\left(\min_t g(X,t)\leq 0\right)$; confidence stop와 trajectory MCS. |
| 구현 전 확인사항 | 대표점 score·시간 discretization·confidence stop·model training. |

### P30. Weighted-sampling DNN (2020)

[원문/출처](https://doi.org/10.1016/j.ymssp.2020.106684)

| 구분 | 이 프레임워크에서의 정의 |
|---|---|
| 1. g(x) | `scalar_deterministic` adapter. §4의 실제 평가·저장 계약 적용. |
| 2. 메타모델 | deterministic DNN; BNN variance 없음; 학습정책 고정. |
| 3-A. Reference 샘플링 | 원 weight에 따른 확률적 표본선정; argmax로 바꾸지 않음. |
| 3-B. Acquisition | sampling weight와 분포; 최적화 score가 존재하는지 확인. |
| 3-C. Gradient | 원 weight 미분이 가능하면 이동용으로 사용 가능하나 정책 변화 발생. |
| 3-D. NH 추가 최적화 **[제안]** | 기본 공정 비교군은 NH 비활성. weight-optimization hybrid는 별도 제안 variant로만 허용. |
| 4. 평가·종료 | DNN Pf 및 원 수렴기준. |
| 구현 전 확인사항 | weight·stop 원식; NH 후 표본분포가 원 Reference와 달라짐. |

## 6. Current implementation and execution

The runtime uses the metamodel plugin package and one complete method YAML per
reference. Model implementations and tests have separate responsibilities.

| Path | Responsibility |
|---|---|
| run.py | CLI, method selection and paper lookup |
| pipeline.py | Component construction, preflight validation, AL loop and logs |
| metamodels/ | Prediction contract, registry and GP/SVM/PCE/BNN plugins |
| g.py | Problem definition and recorded true-model evaluations |
| adaptive.py | Acquisition, Reference selector and original-score acceptance |
| nh.py | Optional NH prototype |
| evaluation.py | Surrogate Pf integration and stopping |
| configs/*.yaml | Standalone method specifications; default: Kriging_GP.yaml |
| papers.yaml | P01-P30 literature definitions and unresolved settings |
| tests/ | Numerical model tests, acquisition tests and pipeline integration |
| README.md | Current installation and usage |

P01-P30 remain specification_only. The executable method configurations use
profile=metamodel_demo. These plugins do not constitute full reproductions of
all the cited papers. The NH implementation is a prototype, to be replaced by a
validated NH adapter for research use.

### 6.1 Execution

Python 3.10+ and the dependencies in requirements.txt are required.

~~~powershell
python -m pip install -r requirements.txt
python run.py --check-config
python run.py --list-metamodels
python run.py --list-papers
python run.py --paper P16
python run.py --config configs/Kriging_GP.yaml --mode paired --output runs/trial_01
python -m unittest discover -s tests -v
~~~

pipeline.prepare constructs named Pipeline components and validates capability
requirements without fitting or calling true g. pipeline.execute preflights
both Reference and NH before starting a paired comparison. Results are written
to separate reference/nh directories; existing output paths are never overwritten.

The example problems use independent standard-normal inputs. search_bound
limits acquisition search, without truncating the Pf integration distribution.
SVM needs both failure and safe observations, so its YAML uses beta=0; the other
examples use beta=2.5. Align problems, designs, seeds, budgets and evaluation
policies before comparing methods.

### 6.2 Method configuration

~~~yaml
profile: metamodel_demo
mode: paired
problem:
  factory: g:Problem
  type: linear_normal
  dimension: 2
metamodel:
  plugin: gp
  optimize: true
acquisition:
  factory: adaptive:UAcquisition
sampling:
  factory: adaptive:PoolSelector
nh:
  factory: nh:NestedHamiltonianPrototype
evaluation:
  factory: evaluation:MCEvaluation
~~~

This excerpt illustrates component selection. Full executable settings are in
configs/Kriging_GP.yaml, configs/SVM.yaml, configs/PCE.yaml and configs/BNN.yaml.
Each reference should have its own complete YAML without implicit inheritance.
External models can use metamodel.factory=module:class instead of a registry name.

Prediction returns explicit fields, with unsupported values set to None.
GP/BNN provide mean and epistemic variance, SVM provides decision scores,
and PCE provides a polynomial mean and optional bootstrap ensemble.
PCE bootstrap variance is not declared as a GP posterior variance.
See metamodels/README.md for model settings and distribution constraints.

### 6.3 Records and limitations

- g_calls.jsonl: actual inputs, outputs, timings and call statuses.
- selection.jsonl: Reference and selected points, original acquisition gain,
  candidate/Reference/NH/rescore costs.
- evaluation.jsonl: Pf, conditional MC error, stopping status, true call count,
  model capabilities, fitted snapshot and training logs.
- fixed_samples.json: shared integration and audit pools.
- training.csv, effective_config.json and summary.json.

Sampling cost includes candidate generation, Reference selection, NH and
acceptance rescoring. Fitting, true evaluation and integration costs are separate.
GP kernel optimization warnings and PCE rank deficiency are recorded.

U-audit stopping is an example heuristic, not a global convergence certificate.
The budget-only evaluation policy makes no convergence claim. Scalar single-point
execution is supported; system/batch/noisy/time-dependent adapters require the
additional contracts described above.


## 7. 구현 우선순위

검증 기록: analytic energy gradient와 중앙차분 비교, min/max acceptance 및 제약 검사, paired run의 실제 호출수·공통 초기 snapshot·shadow Reference 기록, P01–P30 사양의 중복/누락 검사를 통과했다. 이는 연결 예제의 구현 검증이며 NH의 정확도·효율 우위를 입증한 benchmark 결과가 아니다.

| 단계 | 대상 | 이유 |
|---|---|---|
| 1 | P01 GP-U → GP-EFF | 동일 acquisition의 추가 최적화와 시간/호출수 trade-off 확인 |
| 2 | P16 ASVM-MCS | 기존 NHSVM을 원 SVM acquisition refinement로 정렬 |
| 3 | P27 PCK | model adapter 교체 효과와 search 효과 분리 |
| 4 | P21 bPCE | hard vote의 gradient 문제와 proxy의 효과 검증 |
| 5 | P13 noisy GP, P26 SPCE | 서로 다른 uncertainty/target을 보존하면서 확장 |
| 6 | system/batch/time methods | component/time/cluster 고정 refinement 후 joint variant |
| 별도 | P30 weighted DNN | 원 확률적 policy와 argmax refinement는 동등 비교가 아님 |

BLL은 기존 30편에 직접 대응 논문이 확인되지 않았으므로 새 모델 adapter로 추가할 수 있다. 그것을 기존 논문 하나의 재현 profile로 표시하지 않는다.

## 8. 이번 연구에서 검증할 가설

> 원 AL 방법의 Reference 선택을 그대로 수행한 뒤, 그 선택점에서 NH로 동일 acquisition을 추가 최적화하면, 추가적인 surrogate/gradient 계산비용을 지불하는 대신 실제 모델 호출수 또는 최종 신뢰성 추정 정확도에서 이득을 얻을 수 있는가?

최소 보고 지표는 **실제 g 호출수, 전체 wall time, sampling 시간의 세부 항목, 동일 상태 acquisition gain, 최종 Pf 오차, seed별 변동**이다. 동일 실제 평가 예산 실험과 동일 wall-time 예산 실험을 나눠 보고한다.

기존 프레임워크와 제안 프레임워크의 차이는 config의 NH 활성화 및 NH 계산비용이어야 한다. kernel·acquisition·DOE·종료법까지 함께 달라지면 NH의 기여를 분리할 수 없다.
