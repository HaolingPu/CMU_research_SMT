# Where future prediction actually works — verified cases from the 100-case bundle

Source: `trajectory-viewer-v2-20260902/data/review.json` (run `ambiguity-q38-gemma-q36-fsetv2-prefixnorm-strict-40k-r1-20260831`, first 100 rows of AUD0000000003). Every step below was read in full: the observed prefix, all retained futures (Gemma-4-E2B + Qwen3.8-27B, plausible + contrastive), the READ/WRITE decision, the later disambiguating source, and the committed Chinese.

"Works" means: the Chinese realization of the *already-observed* words depended on source that had not yet arrived, the futures spanned the competing readings, the decoder READ instead of committing, and the eventual commit matched the resolution.

Viewer links use `http://127.0.0.1:8766/#case=<utt_id>` locally.

## A. Lexical / structural ambiguity — futures contained both readings

| # | Case | Prefix at decision | What the futures disagreed on | Decision | Resolution → commit |
|---|---|---|---|---|---|
| A1 | 48 `AUD0000000003_108` step 10 | `The door of the pit was hardly closed` | Gemma: "hardly" = barely shut (*it was left ajar, not fastened securely, leaking air*). Qwen: "hardly … when" temporal (*before the blast of wind, when the first cold draft, until a shriek*) | READ, READ | `when the bear rushed` → **坑口刚关上，熊就** (刚…就). A present-only reading would give 几乎没关上 (barely closed), the wrong sense. |
| A2 | 34 `AUD0000000003_1059` step 9 | `The king sat deep` | Physical (*in the throne, in the corner, in the river, in a pit*) vs mental (*in thought, in disbelief, in his own mind*) | READ | `in thought,` → **国王陷入沉思** (陷入 not 坐在). No Chinese verb is committable before the PP. |
| A3 | 8 `AUD0000000003_1010` step 2 | `It began to get` | Weather "it" (*colder, dark outside, cold so we lit the fire*) vs situational "it" (*harder to understand, crowded, dangerous, personal*) | READ | `colder, and Niels` → **天气开始变冷** (subject 天气 inserted; 它 avoided). |
| A4 | 87 `AUD0000000003_1159` step 13 | `…they thought that by the next morning she would be` | Gemma: recovery (*fully recovered, safe from the illness, back to normal*). Qwen plausible: death (*dead, gone, cold to the touch*). Qwen contrastive: miracle (*healed, saved*) | READ | `lost to them.` → **到第二天早上，她就会永远离开他们**. Polarity split came from the two samplers disagreeing. |
| A5 | 3 `AUD0000000003_100` step 8 | `…and asked if it had been` | Cleft focus unknown (*he who had done this deed, you who caused distress, it the king's fault, it really him or his brother*) | READ | `he who had come to their` → **是否是他**, then `rescue.` → 前来救援. |
| A6 | 37 `AUD0000000003_1066` steps 2–3 | `One of these boxes` | Container reading (*contains the documents, holds the records*) vs property reading (*is heavier, should be opened first, is sealed shut*) | READ, READ | `contains the proofs of your birth.` → **其中一个盒子里装有你的出生**. The locative 里 on 盒子 exists only under the "contains" reading. |

## B. Chinese word order — modifier or adjunct must precede the head, so the head cannot be committed yet

| # | Case | Prefix at decision | Futures | Decision | Resolution → commit |
|---|---|---|---|---|---|
| B1 | 50 `AUD0000000003_1082` steps 13–15 | `They had everything they` | Relative-clause content varies (*needed to survive, could possibly ask for, couldn't afford to keep, didn't actually possess*) | READ ×3 | `could possibly desire for their comfort,` → **拥有了一切可能让它们感到舒适的东西** (clause before 东西). |
| B2 | 60 `AUD0000000003_1106` steps 4–7 | `…took care not to look over the fence into the field` | Adverbial continuation (*because the field was dark, hoping she wouldn't see, so she kept her eyes on the gate*) vs relative clause on *field* (*where the horses grazed, where the fence had collapsed*) | READ ×4 | `where the donkey was feeding.` → **刻意避免越过栅栏看向驴子正在吃草的田地**. Committing 田地 early would have been irreversible. |
| B3 | 9 `AUD0000000003_1011` steps 14–18 | `…lifted whole bucketfuls of broth and great joints of meat out of an enormous` | Noun (*cauldron, pot, tub, kettle, meal, feast*) and whether a relative clause follows | READ ×5 | `pot which was set on the ground between them.` → **从放在他们之间地上的一个大锅里舀起满满一桶**. Source-location phrase and relative clause both precede the verb 舀起. Same utterance as failure case 9 (它们 at step 3): the method failed early and worked late. |
| B4 | 14 `AUD0000000003_1019` steps 23–25 | `and want to get the upper hand of him` | Means phrase present (*by offering aid, by showing strength, by undermining his position*) vs absent (*before the battle, in this conflict*) | READ ×3 | `by carrying off the princess,` → **企图通过劫持公主来压倒他**. The 通过… phrase must come before 压倒他. |
| B5 | 22 `AUD0000000003_1033` steps 17–19 | `…saw the three headless giants lying in a heap in the` | Location (*courtyard, dirt, ruins, pit of the dungeon*) vs figurative (*dream, vision, memories*) | READ ×3 | `courtyard,` → **庭院里堆着三具无头巨人的尸体** (location-first existential). |
| B6 | 76 `AUD0000000003_1130` step 18 | `…returned home from hunting and saw a baby` | *lying in the nursery, crying in the cradle, who looked familiar, girl wrapped in a blanket on the steps* | READ | `lying in the cradle.` → **看到摇篮里躺着一个婴儿时**. |
| B7 | 59 `AUD0000000003_1105` steps 19–20 | `Crowning her glossy` | Head noun (*hair, gown, smile, surface, leather, metal*) | READ ×2 | `black hair.` → **为她乌黑发亮的秀发增添了光彩**. |
| B8 | 78 `AUD0000000003_1137` step 12 | `After you` | *arrive, finish your work, have eaten, leave the room, pay the bill* | READ | `have eaten` → **吃完饭后** (postposition 后 needs the clause). |
| B9 | 92 `AUD0000000003_1174` step 18 | `…nothing should happen while he` | *was resting, was sleeping, was away, was alone, was questioning the guard* | READ | `was away,` → **在他外出期间** (在…期间 frame). |
| B10 | 20 `AUD0000000003_103` step 12 | `The shepherd came and stood before` | Qwen: object noun (*the throne, the king, the altar, the mirror*). Gemma: new clause (*he asked, he bowed*) | READ | `the throne,` → **来到王座前** (postposition 前). |
| B11 | 15 `AUD0000000003_1021` step 28 | `So high that even the` | *mountain, king's guards, giants' own shadows, walls, oldest giants* | READ | `giants could not touch the top of it.` → **高得连巨人都无法触及墙顶** (连…都 needs the NP). |
| B12 | 16 `AUD0000000003_1025` step 22 | `Whoever drinks the wine I hold can wield` | *influence, the authority, the sword hidden beneath the dais, the wine back into the horn* | READ | `the sword that hangs above` → **可执 / 悬于 / 上方的利剑** committed in three safe pieces. |
| B13 | 12 `AUD0000000003_1016` steps 15–17 | `and then neither of you can` | *agree, blame the third party, say anything, escape* | READ ×3 | `blame the other` → **这样你们谁也不能责怪谁**. |

## C. Safe partial commits — the decoder wrote only the stable frame

| # | Case | Prefix | Committed | Postponed until |
|---|---|---|---|---|
| C1 | 1 `AUD0000000003_0` steps 3–7 | `And these introductions are inevitably both` | **而这些介绍** (topic only) | `monotonous and unavailing.` → 不可避免地既单调又无效 |
| C2 | 35 `AUD0000000003_106` steps 2–3 | `No, I won't` | **不，** only | `princess for my wife` → 我不会说的，直到我得到公主作为我的妻子 |

## Aggregate observations (all 100 cases, 2,382 sampled steps)

- READ rate is 53% overall, and ~56% whether the prefix ends in a content word, a determiner, or a preposition; it drops to ~40% only at commas and sentence ends. So the decision is not a surface-syntax heuristic; it is driven by whether the futures' translation distributions agree.
- The futures rarely *predict* the true continuation: the real next content word appears among the futures in 23% of steps, and the real first two words in 8%. The method does not need the futures to be right, only to span the alternatives. Every success above has the true continuation absent or nearly absent from the future set.
- Counter-example that defines the boundary: case 9 step 3 (`They were so huge that the` → 它们). The futures were diverse (people, animals, objects) but the probe mapped all of them to the same token 它们, so strict consensus passed. Success requires that semantic divergence among futures becomes divergence in the probe's next-token distribution; when the probe has a fixed bias, more futures cannot help.
- Caveat carried over from the audit: these 100 cases are contiguous from one audiobook recording and are illustrative, not a benchmark.
