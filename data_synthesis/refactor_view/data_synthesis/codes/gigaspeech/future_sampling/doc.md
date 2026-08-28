每个 chunk
chunk -> 累积 observed source

if empty chunk (非最后) -> READ
if last chunk -> Instruct LLM(1次, translate_final) -> WRITE/READ
if observed_words < min_observed_words -> READ

否则进入 Case 4:
observed source -> Base LLM(1次) -> M个 future source

(observed + each future) x M -> Instruct LLM(翻译 M次, 并发请求非batch) -> M个候选中文

M个候选中文 + (observed+future) x M -> Align model(awesome-align, M次, CPU) -> 对齐截断 safe_prefix

safe_prefix -> 单调性检查 + delta提取 + 最小长度过滤 -> valid candidates

valid candidates -> Instruct LLM(1次, judge打分) -> scores

scores -> 阈值/共识率 -> direction check -> word_head_guard -> WRITE or READ

全部 chunk 结束 -> last chunk 已经通过 translate_final 做补尾




每个进入 Case 4 的 chunk 调用次数（默认 M=10）
Base LLM: 1次
Instruct LLM: M次翻译 + 1次打分 = 11次
Align model: M次 = 10次 