# RealNVP optimizacija: tehnična referenca

1. 10. 2026 · Gaber Zupan Škrlj

Podroben del predaje: rezultati po problemih, protokol, pasti in poti po repozitoriju. **Začni z uvodom v [`PROJECT_HANDOFF.md`](PROJECT_HANDOFF.md).**

## Povzetek

Projekt (september – 1. oktober 2026) je preverjal, ali se RealNVP normalizing flow da uporabiti kot generativni black-box optimizer za diskretne probleme. Ideja: namesto iskanja posamezne rešitve se naučimo distribucijo nad rešitvami, iz nje vzorčimo kandidate, jih ocenimo z objective funkcijo in distribucijo premaknemo proti boljšim. Raziskovalno vprašanje se je med delom premaknilo od "ali RealNVP zna optimizirati?" k "zakaj dobro rešitev najde, a je ne obdrži?".

Potek dela:

1. **Binarni benchmark.** Pet IOH/PBO problemov (n = 100, 100 seedov) in Jump_k proti (1+1)-EA.
2. **Permutacijski benchmark.** Random-key reprezentacija (argsort) na TSP-20, QAP Nug20 in PFSP Ta001 z enakim zamrznjenim protokolom; TSP-50 kot sanity check.
3. **Poskusi popravka retentiona na TSP-20.** Cosine LR + elite, Gaussian exploration, annealed KL (s čistim testom ene intervencije) in Boltzmann varianta KL.
4. **Prenos KL na QAP in PFSP**, da preverimo, ali je efekt splošen.
5. **Budget-matched (1+1)-EA** za vse tri permutacijske probleme in anytime krivulje (best-so-far proti številu evalvacij).
6. **Kontrola brez flowa.** Diagonalni Gaussian namesto RealNVP na TSP-20, vse drugo enako.
7. **2D basin študija.** Toy problem z znanim Boltzmann targetom, da ločimo kapaciteto modela od dinamike treninga: temperature, MLE fit, warm start in annealing.

## Glavni rezultati

RealNVP + REINFORCE na nobenem testiranem problemu ni premagal preprostega (1+1)-EA; EA doseže enak ali boljši best-ever na vseh paired seedih (strogo boljši: TSP-20 2/10, QAP Nug20 10/10, PFSP Ta001 10/10) in je izrazito bolj sample-efficient.

Glavne ugotovitve:

- **Discovery ≠ retention.** Model optimum pogosto najde med treningom, končna distribucija pa ga skoraj nikoli ne generira (TSP-20: P(optimum) = 0 na 10/10 seedih pri RealNVP + KL).
- **Kolaps ni specifičen za flow.** Diagonalni Gaussian (40 parametrov) na TSP-20 doseže primerljiv discovery kot RealNVP (KL: 7/10 vs 8/10 hitov); coupling layerji pri trenutnem protokolu niso pokazali merljive prednosti. Retention in diversity se razlikujeta: Gaussian + KL obdrži optimum na 3/10 seedih, RealNVP na 0/10.
- **V 2D toy problemu (T = 0.05) failure iz scratcha izvira iz dinamike treninga.** Model zna alocirati maso med oba bazena in dvobazenska rešitev ima nižji loss; REINFORCE jo drži, če začne iz nje, iz scratcha pa je ne doseže (0/10 seedov). Oblika gostote znotraj bazenov ostaja omejena s kapaciteto.
- **Annealed KL** poveča permutacijsko diversity na vseh treh problemih, discovery pa izboljša le na TSP-20; na PFSP se izboljšanje ni repliciralo.
- Prenos 2D diagnoze na permutacijske probleme je hipoteza, ne rezultat.

## Glavna tabela

Povprečen best-ever objective čez 10 seedov (42–51) in število seedov, ki dosežejo optimum oz. best-known. Nižje je bolje. Kjer obstajata dve seriji runov, sta navedeni obe: *saved* = kanoničen shranjen run (README v `results/permutation_optimization/`), *trace* = rerun z anytime trace (`anytime/`).

| Problem (optimum) | RealNVP | RealNVP + KL | Gaussian | Gaussian + KL | (1+1)-EA |
| --- | --- | --- | --- | --- | --- |
| TSP-20 (3.513669) | 3.5865, 3/10 (saved); 3.5786, 4/10 (trace) | 3.5154, 8/10 (clean KL; saved in trace enako) | 3.5198, 3/10 | 3.5163, 7/10 | 3.513669, 10/10 |
| QAP Nug20 (2570) | 2805.8 (9.18 %), 0/10 (saved) | 2794.8 (8.75 %) saved; 2797.2 (8.84 %) trace; 0/10 | ni testirano | ni testirano | 2627.6 (2.24 %), 0/10 |
| PFSP Ta001 (1278) | 1295.8 (1.39 %), 0/10 (saved) | 1294.9 (1.32 %), 0/10 (trace, čist baseline + KL) | ni testirano | ni testirano | 1278.0 (0.00 %), 10/10 |

- Budget: QAP/PFSP 3,317,760 evalvacij za vse metode; TSP RealNVP/Gaussian 2,216,960 (2000 epoch), EA 3,317,760, vendar EA optimum najde že po 720–69,742 evalvacijah (mediana ~10k).
- Saved vs trace: rerun ni bit-exact (število niti spremeni float rezultate; isti basini, drugačne per-seed številke). Gaussian je primerjan paired s trace runi in anytime grafi so iz trace runov; README tabele v `results/` uporabljajo saved rune. EA rerun je byte-identičen saved runu.
- Legacy PFSP KL (1293.1) je drug skript z drugim knjigovodstvom in ni v tej tabeli (glej PFSP razdelek).
- Gaussian = diagonalni Gaussian random keys, LR tunan (baseline 1e-2, KL 3e-2); RealNVP ima fiksen LR 1e-4.

## Metoda in protokol

RealNVP generira zvezen vektor, nediferenciabilen decoder ga pretvori v diskretno rešitev, gradient pa gre prek REINFORCE na log qθ(y), ne skozi diskretizacijo.

1. z ~ N(0, I) → RealNVP → y (zvezno).
2. Decoder: binarno threshold, permutacije argsort(y) (random keys).
3. Objective na diskretni rešitvi; pri minimizaciji reward = −cost.
4. log qθ(y) prek `model.inverse(y.detach())`: `gaussian_log_prob(z) + inverse_log_det`.
5. Advantage = reward − leave-one-out baseline, nato standardizacija (enako (r − mean)/std).

$$
\mathcal{L} = -\overline{A \cdot \log q_\theta(y)} + T \cdot \mathrm{KL}, \qquad \mathrm{KL} = \overline{\log \mathcal{N}(z) - \log|\det J_f| - \log \mathcal{N}(y)}
$$

Annealed KL: T geometrično 0.1 → 0.001. Arhitektura: affine coupling (s s tanh, t linearen), izmenični flip, Gaussova baza. Straight-through estimator ni uporabljen.

**Zamrznjen permutacijski protokol** (TSP, QAP, PFSP): DIMENSION 20, 4 layerji / 64 hidden, batch 1024, 3000 epoch (TSP 2000), Adam LR 1e-4, validacija 4096 vzorcev vsakih 50 epoch, test 16384 vzorcev, grad norm 5.0, seedi 42..51, checkpoint = najnižji validation mean.

**Budget:** šteje se vsaka objective evalvacija, tudi validacijske (lahko posodobijo best-ever); testne ne. Wall-clock ni mera, ker so runi tekli paralelno.

**Metrike:**

- *Discovery:* best-ever, gap, evalvacija prvega hita, število hitov.
- *Retention:* končni mean in best na checkpointu (16384 vzorcev), retention_loss = final_best − best_ever.
- *Diversity:* unique permutacij / 16384, delež vzorcev na optimumu (P(optimum)).

## Binarni del (IOH/PBO, zaključeno)

RealNVP + threshold decoder ni boljši od (1+1)-EA na nobenem od petih problemov, vsak problem pa pokaže drug failure mode. Nastavitev: n = 100, 100 seedov (42–141), budget 4,096,000 evalvacij, višje je bolje.

| Problem | RealNVP mean best ± std | EA mean best ± std | Hiti RealNVP | Hiti EA | Failure mode |
| --- | --- | --- | --- | --- | --- |
| OneMax (100) | 100.0 ± 0.0 | 100.0 ± 0.0 | 100/100 | 100/100 | učinkovitost: 134,574 vs 1,027 evalvacij do optimuma |
| LeadingOnes (100) | 77.3 ± 12.4 | 100.0 ± 0.0 | 1/100 | 100/100 | velika varianca med seedi, najslabši 35 |
| ConcatenatedTrap (20) | 16.002 ± 0.020 | 16.35 ± 0.26 | 0/100 | 0/100 | skoraj determinističen kolaps v deceptive lokalni optimum |
| NKLandscapes | −0.30309 ± 0.00360 | −0.29265 ± 0.00093 | – | – | povprečen EA run boljši od najboljšega RealNVP runa |
| IsingTorus (200) | 172.16 ± 7.77 | 195.0 ± 8.70 | 1/100 | 75/100 | RealNVP hit pri 406,528, EA povprečno ~2,660 evalvacij |

Vir: `results/benchmark_100seeds/README.md`. Starejši posamezni runi (npr. LeadingOnes 97/100) so iz `results/benchmark/` in niso del te primerjave.

**Jump_k** (n = 100, seed 42, en run na k): optimum najde do k = 6 (~240k evalvacij), od k = 7 naprej kolabira na n − k. Hit se zgodi le v kratkem oknu (~20 epoch), ko je distribucija še široka: model past "prehiti", preskočiti je ne zna. Rezultat je en seed na k, zato je le indic.

## TSP-20

Na TSP-20 RealNVP optimum najde, ga pa zanesljivo ne obdrži: pri clean baseline in clean RealNVP + KL končna distribucija optimuma ne generira na nobenem od 10 seedov; nekateri starejši, confounded runi (cosine + elite 3/10, original KL 1/10) ga občasno obdržijo, vendar retention ni zanesljiv. Instanca: Euclidean, instance seed 12345, optimum 3.513669; pogost lokalni minimum je 3.522437.

| Varianta | Hiti | Seedi s P(optimum) > 0 | Unique tour / 16384 | Opomba |
| --- | --- | --- | --- | --- |
| RealNVP baseline (trace rerun) | 4/10 | 0/10 | 2–8 | shranjen run 3/10 |
| Cosine LR + elite | 8/10 | 3/10 | – | confounded (8 layerjev, batch 4096, 3000 epoch, brez standardizacije), retention nestabilen |
| Gaussian exploration ε = 0 / 0.10 / 0.25 | 4/10, 5/10, 5/10 | – | – | več diversity, discovery ne izboljša |
| Original KL | 10/10 | 1/10 | – | confounded, isto kot cosine |
| Baseline + samo KL (trace rerun) | 8/10 | 0/10 | 7–20 | vsi kolabirajo na 3.522437; shranjen run 8/10 |
| Boltzmann (KL brez standardizacije) | 9/10 | 0/10 | 53–106 | test mean slabši na 10/10 |
| Diagonal Gaussian | 3/10 | 1/10 | 23–748 | σ divergira (537–14040) |
| Diagonal Gaussian + KL | 7/10 | 3/10 | 1–26 | na 3 seedih generator = optimum (99.9 %) |

**Čist KL test.** Baseline + samo KL (T 0.1 → 0.001) da 8/10 vs 3/10 na shranjenih runih, boljši test mean na 9/10. KL torej izboljša discovery na TSP, retention pa ne.

**Boltzmann run in efektivna temperatura.** Če advantage standardiziramo in nato dodamo T · KL, je efektivna temperatura glede na dolžino T · std(length), ki se krči, ko se distribucija koncentrira. Brez standardizacije (pravi Boltzmann target ∝ exp(−length/T)) je hitov 9/10, napoved P(optimum) > 0 pa je ovržena: 0/10, test best povsod 3.522437. Flow zaostaja za schedulom (val mean pri epochu 1000: 3.71 vs 3.54).

**Diagonal Gaussian random keys** (y = μ + σ · z, 40 parametrov, vse drugo enako, preverjeno z diffom). Gaussian + KL najde optimum prej (122k–160k vs 403k–799k evalvacij) in ga na 3 seedih obdrži, RealNVP na 0. Discovery je primerljiv; coupling layerji pri trenutnem protokolu niso pokazali merljive prednosti. **Caveat:** LR je bil za Gaussian tunan (sweep na seedih 0–2), za RealNVP ne.

## QAP Nug20 in PFSP Ta001

Na QAP in PFSP RealNVP nikoli ne doseže optimuma; KL konsistentno poveča permutacijsko diversity, kvalitete iskanja pa zanesljivo ne izboljša.

### QAP Nug20 (optimum 2570)

| | Baseline | Annealed KL | KL (trace rerun) |
| --- | --- | --- | --- |
| Mean best cost (gap) | 2805.8 (9.18 %) | 2794.8 (8.75 %) | 2797.2 (8.84 %) |
| Hiti optimuma | 0/10 | 0/10 | 0/10 |
| Mean final generator cost | 2871.8 | 2838.1 | 2837.5 |
| Mean retention loss | 66.0 | 43.2 | 40.2 |
| Mean unique permutacij / 16384 | 4.2 | 11.2 | 11.8 |

- KL ima več unique permutacij na 10/10 paired seedih, boljši best-ever pa le na 6/10 (razlika 11 pri std 34–40, v šumu).
- Baseline generator ima final best = final mean: vseh 16384 vzorcev ima isti cost.
- RealNVP najde best-ever pozno: mediana 750k evalvacij (KL 1.70M).

### PFSP Taillard Ta001 (20 jobov × 5 strojev, best-known 1278)

Referenci: NEH 1286, identiteta 1448.

| | Baseline | Original KL (legacy) | Čist baseline + KL |
| --- | --- | --- | --- |
| Mean best makespan (gap) | 1295.8 (1.39 %) | 1293.1 (1.18 %) | 1294.9 (1.32 %) |
| Best / worst run | 1290 / 1297 | 1288 / 1297 | 1287 / 1297 |
| Hiti 1278 | 0/10 | 0/10 | 0/10 |
| Boljši od baselinea (paired) | – | 8/10 | 5/10 (2 izenačena, 3 slabši) |
| Mean final generator makespan | 1297.003 | 1297.006 | 1297.007 |
| Mean retention loss | 1.2 | 3.9 | 2.1 |
| Mean unique permutacij / 16384 | 1550.0 | 13754.4 | 13980.9 |

- **"KL izboljša PFSP discovery" se ne replicira** (std ~3); replicira se le permutacijska diversity.
- Vsi generatorji končajo na platoju 1297: KL ohrani permutacijsko diversity, ne pa objective diversity. Makespan ima velike platoje, zato veliko različnih permutacij dobi isto objective vrednost; vpliv tega na gradientni signal ni bil neposredno izmerjen.
- Original KL (`legacy/.../pfsp/realnvp_pfsp_kl.py`) ima drugačno knjigovodstvo (validacija ne posodablja best-ever, 61 validacij, druga CSV shema). Kanonična je `realnvp_pfsp_baseline_kl.py`.

## Budget-matched (1+1)-EA in anytime krivulje

EA doseže enak ali boljši best-ever kot RealNVP + KL na vseh paired seedih in je izrazito bolj sample-efficient. Na TSP doseže optimum na 10/10 seedih, RealNVP + KL na 8/10 (na teh 8 seedih sta izenačena), mediani časa do optimuma pa sta ~11k proti ~550k evalvacij (~50×). Na QAP in PFSP je EA strogo boljši na 10/10 seedih. Na QAP je EA mediana pri 1k evalvacijah (2715) boljša od RealNVP mediane na koncu budgeta (2802).

**EA:** trenutna permutacija → 1 + Poisson(1) naključnih potez (TSP inversion, QAP swap, PFSP insertion) → polna evalvacija → accept if not worse (`<=`, zaradi PFSP platojev). Budget 3,317,760 evalvacij, seedi 42..51. Hiter evaluator je ob zagonu preverjen proti `objective.py`.

- TSP-20: 10/10 hitov po 720–69,742 evalvacijah (mediana ~10k).
- QAP: 2596–2664, optimuma 2570 ne najde. Po 10k evalvacijah je vsak EA seed že boljši od RealNVP best-ever po celotnem budgetu.
- PFSP: 1278 na 10/10 po 10,665–104,965 evalvacijah (mediana ~40k). Pri 1k evalvacijah je EA na platoju 1297, kjer končajo vsi RealNVP generatorji.

![TSP-20: konvergenca EA vs RealNVP + KL](results/permutation_optimization/anytime/tsp20_clean_convergence.png)

![QAP Nug20: konvergenca EA vs RealNVP + KL](results/permutation_optimization/anytime/qap_nug20_clean_convergence.png)

![PFSP Ta001: konvergenca EA vs RealNVP + KL](results/permutation_optimization/anytime/pfsp_ta001_clean_convergence.png)

Grafi: `plotting/clean_anytime_plots.py`, podatki v `results/permutation_optimization/anytime/`. Črta = mediana čez 10 seedov, pas = 25.–75. percentil. Na linearni y osi se razlika 3.522437 vs 3.513669 pri TSP ne vidi.

## 2D basin študija: zakaj flow izgubi bazen

V 2D toy problemu RealNVP zna alocirati maso med več bazenov; kolaps iz scratcha pri nizki temperaturi izvira iz dinamike treninga: ko se redek bazen izprazni, REINFORCE nima vzorcev in zato nima signala za vrnitev. Problem: dva bazena, globalni levo, desni za Δf višji; brez T standardiziran REINFORCE, s T cost f + T · log q (reverse KL do Boltzmanna ∝ exp(−f/T)). LR 1e-4, 2000 epoch, seedi 42–51, brez plateau restore, poročanje na 100k svežih vzorcih, bazen "zaseden" pri ≥ 1 % mase. Skripte in ukazi: `experiments/continuous_optimization/README.md`.

| Korak | Nastavitev | Rezultat |
| --- | --- | --- |
| 1 | Δf = 0.2, brez T | 10/10 samo globalni bazen; desni pade pod 1 % pri epochu 100–140 |
| 2 | Δf = 0, brez T | 3/10 oba, 3 levo, 4 desno; odločeno do epocha ~200, prazen bazen se ne vrne |
| 3 | Δf = 0.2, T ∈ {1, 0.5, 0.3, 0.2, 0.1, 0.05} | T ≥ 0.2: 10/10 oba, p_right/target 0.99–1.00; T = 0.1: 10/10 oba, a ratio 0.885 in ni konvergirano; T = 0.05: 0/10 |
| 4 | 4 minimumi, f = 0/0.1/0.2/0.3 | brez T: 10/10 samo globalni; s T: vsi 4 bazeni 10/10 pri vseh T, TV do targeta 0.003 (T = 1) … 0.020 (T = 0.1) |
| 5 | T = 0.05: scratch / MLE fit / warm start | scratch 0/10 oba; MLE 10/10; warm 10/10 oba, KL nižji od scratch na 10/10 |
| 6 | Anneal T 0.2 → 0.05 | bazen ostane, a pod 1 % (0/10); KL nižji od scratch na 10/10, višji od warm na 10/10 |

![Masa v desnem bazenu proti Boltzmann targetu pri Δf = 0.2](experiments/continuous_optimization/basin2d_delta0.2_temperature_p_right.png)

**Korak 5 (ključna kontrola).** Pri T = 0.05 je target p_right = 0.0179, opustitev desnega bazena stane 0.018 nats reverse KL. MLE fit na točnih Boltzmann vzorcih pokaže, da arhitektura oba bazena predstavi (forward KL 0.028, konvergiran). REINFORCE, ki začne iz MLE fita (*warm*), drži oba bazena na 10/10 seedih: reverse KL 0.0074–0.0099, p_right 0.0148–0.0165. Iz scratcha je reverse KL 0.029–0.037 in p_right ≤ 0.0003. Kolabirana rešitev je torej po lossu **slabša**, ne optimum; desni bazen se izprazni do epocha ~250–500 in se ne vrne.

![Scratch vs MLE warm start pri Δf = 0.2](experiments/continuous_optimization/basin2d_mle_delta0.2_p_right_kl.png)

**Korak 6 (anneal, 1. 10.).** T geometrično 0.2 → 0.05 v 1000 epochih, nato 1000 pri 0.05, paired s scratch. p_right 0.0005–0.0087 (scratch ≤ 0.0003, warm ~0.016), reverse KL 0.021–0.030. Annealing delno pomaga, warm start rešitve pa ne doseže.

**Omejitve.** Fiksen LR 1e-4 in 2000 epoch; pri T = 0.2/0.1 scratch še ni konvergiran (KL 2–4× višji od warm). Mean f je nad targetom na vseh seedih (+0.003–0.006), oblika znotraj bazena torej ni točna. Spread znotraj bazena (~0.139) je omejen s kapaciteto: tanh na s omeji log det, zato ima gostota strop.

## Glavne ugotovitve in obseg veljavnosti

Vsaka ugotovitev ima naveden dokaz in mejo, do katere velja.

| Ugotovitev | Dokaz | Velja za | Ne velja / ni testirano |
| --- | --- | --- | --- |
| RealNVP ni boljši od (1+1)-EA | 5 binarnih problemov (100 seedov); TSP/QAP/PFSP: EA enak ali boljši na 10/10 paired seedih (strogo boljši 2/10, 10/10, 10/10) | vse testirane probleme pri n = 100 oz. 20 | močnejši baselini (tabu, iterated greedy) niso potrebni za to trditev |
| Discovery ≠ retention | TSP: P(optimum) = 0 na 10/10 pri RealNVP + KL; QAP retention loss 40–66 | TSP-20, QAP, PFSP | Gaussian + KL optimum obdrži na 3/10; confounded RealNVP runi občasno (cosine 3/10, original KL 1/10) |
| Kolaps ni specifičen za flow | Gaussian vs RealNVP na TSP-20: 3/10 vs 4/10, 7/10 vs 8/10 | TSP-20 | QAP/PFSP; LR za RealNVP ni tunan |
| KL poveča permutacijsko diversity | unique permutacij višji na vseh treh problemih | TSP, QAP, PFSP | objective diversity (PFSP plato 1297) |
| KL izboljša discovery | TSP-20 čist test 8/10 vs 3/10 | TSP-20 | PFSP se ne replicira (5/10 paired), QAP v šumu |
| Kolaps iz scratcha izvira iz dinamike treninga, ne iz alokacijske kapacitete ali optimuma lossa | 2D: warm start drži oba bazena z nižjim KL, scratch ne | 2D toy, T = 0.05, LR 1e-4 | prenos na permutacije je hipoteza |
| Kapaciteta zadostuje za razporeditev mase med bazene | MLE fit, korak 3–4 | 2D toy | oblika znotraj bazena je omejena (tanh cap na log det) |

Popravki pogostih napačnih formulacij:

- Večja ekspresivnost ni bila testirana; testirana je bila *manjša* (Gaussian) in ni škodila.
- "Šibek gradient signal" kot vzrok ni izmerjen. Izmerjeno je: brez vzorcev v bazenu ni signala za vrnitev.
- Redundanca random-key reprezentacije je verjetna, kot vzrok pa ni testirana.

## Pasti za naslednika

Naslednje stvari niso razvidne iz kode na prvi pogled in so že povzročile napačne zaključke ali neprimerljive tabele.

- **Napačen estimator v starih toy skriptah.** `objective_2d_baseline.py`, `objective_2d_experimental.py`, `objective_2d_test.py` in `objective_10d_baseline.py` uporabljajo `loss = -(weights * log_det).mean()`, kar ni score-function estimator. Pravilen je v `objective_2d_gaussian.py` in `basin_2d_*.py`.
- **Implicitna temperatura.** Standardiziran advantage + T · KL ni fiksen Boltzmann target: efektivna temperatura je T · std(objective) in se krči s koncentracijo.
- **Confounded TSP runi.** Cosine + elite in original KL se od baselinea razlikujeta v petih stvareh hkrati (8 layerjev, batch 4096, 3000 epoch, cosine LR, brez standardizacije). Za KL efekt uporabi `realnvp_tsp20_baseline_kl.py`.
- **Knjigovodstvo PFSP KL.** Legacy `realnvp_pfsp_kl.py` ne šteje validacije v best-ever in ima drugo CSV shemo; ni primerljiv z baselineom.
- **Rerun ni bit-exact.** Število niti spremeni float rezultate (isti basini, drugačne per-seed številke, npr. QAP do 72). Grafe in številke ob njih vedno jemlji iz istega runa. Pri paralelnih seedih nastavi `torch.set_num_threads(...)`.
- **Plateau restore** (checkpoint po najnižjem mean) lahko favorizira bolj koncentrirano distribucijo. V 2D diagnostiki je bil zato izklopljen kot možen confounder; kot samostojen vzrok ni abliran.
- **Retention loss** mora biti povsod definiran enako (final_best − best_ever, best-ever vključuje validacijo), sicer tabele niso primerljive.
- **Okolje.** Python je `~/Real-nvp-learning/.venv/bin/python` (sistemski nima torcha). CUDA na big.ijs.si (Tesla K80, driver 470) ne dela s trenutnim PyTorch buildom; vse teče na CPU. Dolge rune poženi z `nohup python -u ... > log 2>&1 &` (brez `-u` je stdout bufferiran).
- **Wall-clock ni mera.** Runi so tekli paralelno; primerjaj število evalvacij.

## Kje je kaj

Vse je v repozitoriju [gaberzupanskrlj/Real-nvp-learning](https://github.com/gaberzupanskrlj/Real-nvp-learning), veja `main`. Skripte se poganjajo iz svoje mape.

| Kaj | Pot |
| --- | --- |
| Binarni benchmark (skripte) | `experiments/discrete_optimization/benchmarks/binary/` |
| Binarni rezultati, 100 seedov | `results/benchmark_100seeds/` |
| TSP / QAP / PFSP skripte (RealNVP, Gaussian, EA, skupni `objective.py`) | `experiments/discrete_optimization/benchmarks/permutation/{tsp,qap,pfsp}/` |
| Anytime CSV-ji in grafi (trace runi) | `results/permutation_optimization/anytime/` |
| Starejši permutacijski rezultati | `results/permutation_optimization/{tsp20,tsp50,qap,pfsp}/` |
| 2D basin študija (skripte, CSV, README) | `experiments/continuous_optimization/` |
| Plot skripte | `plotting/` (`clean_anytime_plots.py`, `basin_temperature_plots.py`, `basin_mle_plots.py`) |
| Legacy in raziskovalne skripte | `experiments/discrete_optimization/legacy/` |
| Povzetek dne 30. 9. za mentorja | `experiments/README_2026-09-30.md` |

Povezani dokumenti:

- [RealNVP Permutation Benchmark: Results Broadsheet](https://claude.ai/code/artifact/87917475-f0eb-468d-ace0-7ec2fd65941c): podroben permutacijski del.
- [2D Basin Study: How RealNVP Converges](https://claude.ai/code/artifact/8743c156-50bb-4e9a-b1f5-674a95b9f091): koraki 1–4 2D študije.

## Odprta vprašanja in naslednji koraki

Največ informacije bi dal test, ali 2D diagnoza (izgubljen bazen brez vzorcev nima signala) velja tudi na permutacijah. Predlogi po prioriteti:

1. **RealNVP LR sweep na TSP-20** (teče 1. 10.): `tsp/realnvp_tsp20_lr_trace.py`, LR 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, seedi 0–2, isto pravilo izbire kot za Gaussian, nato main run na seedih 42–51. Odloči, ali trditev "coupling layerji ne prinesejo prednosti" velja brez caveata.
2. **Warm start ali annealing na TSP-20.** Analog koraka 5–6: začni iz distribucije, ki že ima maso na optimumu, in preveri, ali REINFORCE + KL maso obdrži.
3. **Diagnostika na permutacijah:** delež batcha z različnimi objective vrednostmi, gradient SNR, effective sample size, entropija, čas med discovery in izgubo elite rešitve. To neposredno testira hipotezo o signalu.
4. **CMA-ES nad random keys.** Loči vpliv reprezentacije od načina učenja distribucije.
5. **Gaussian na QAP in PFSP**, da se trditev "kolaps ni specifičen za flow" razširi čez TSP-20.
6. **Jump_k z 10 seedi na k** (zdaj en seed na k).
7. Plot `basin_multi_plots.py` (2D gostota, 4 minimumi) čaka na rerun koraka 4 z grid snapshoti.

Preprosto dodajanje novih arhitektur ali heurističnih popravkov ni smiselno, dokler ni jasno, ali je ozko grlo signal treninga ali reprezentacija.
