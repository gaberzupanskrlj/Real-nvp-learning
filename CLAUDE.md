# RealNVP za diskretno in permutacijsko optimizacijo

Raziskovalni projekt (IJS). Preverjamo, ali se da RealNVP normalizing flow uporabiti kot generativni black-box optimizer za diskretne probleme. Iščemo dejanske failure mode-e in preverjamo hipoteze, ne navijamo za rezultate.

## Pravila sodelovanja

- **Eksperimentov ne poganjaj.** Poganja jih uporabnik. Ti pišeš ali pregleduješ kodo in interpretiraš output, ki ga prilepi.
- Odgovarjaj v slovenščini, kratko. Brez dolge teorije, razen če jo uporabnik želi.
- Potek dela: koda → uporabnik požene → prilepi output → skupaj interpretiramo → naslednji eksperiment.
- Ko nekaj ne dela, najprej diagnosticiraj konkretni output ali error.
- **Ena intervencija naenkrat.** Ne spreminjaj več stvari hkrati, primerjave morajo biti paired in controlled.
- Metodološko napako povej direktno.
- Ne ustvarjaj novih map ali datotek brez potrebe. Legacy kode ne mešaj nazaj v aktivne eksperimente.

## Stil kode

- Preprosta raziskovalna koda: ena berljiva, reproducibilna skripta je boljša od frameworka.
- **Brez dekorativnih blokov komentarjev** (`# ====...`). Uporabljaj kratke komentarje: `# Settings`, `# RealNVP model`, `# Validation`, `# Final evaluation`.
- Brez abstrakcij in refactorjev, ki spremenijo metodo.
- Ohrani: RealNVP matematiko, invertibility, change-of-variables log-prob, seede in eksplicitne hiperparametre.

## Metoda

```
z ~ N(0, I) → RealNVP → y (zvezno) → nediferenciabilen decoder → x (diskretno) → objective
```

- Gradient **ne** gre skozi diskretizacijo. Uporabljamo REINFORCE (score function) na `log q_θ(y)`.
- `log q_θ(y)` se izračuna prek `model.inverse(y.detach())`: `gaussian_log_prob(z) + inverse_log_det`.
- Advantage: `reward − leave-one-out baseline`, nato standardizacija. Po standardizaciji je to enako `(r − mean)/std`, LOO sam ne prispeva ničesar dodatnega.
- `loss = -(advantage.detach() * log_prob).mean()`. Pri minimizaciji je `reward = -cost`.
- Ne uporabljaj straight-through estimatorja.
- Arhitektura: affine coupling (`s` s `tanh`, `t` linearen), izmenični `flip`, Gaussova baza.

## Permutacijski protokol (zamrznjen)

Pri vseh treh problemih je isti: `R^20 → argsort(y) → permutacija`. Spreminja se samo objective landscape.

| Nastavitev | Vrednost |
|---|---|
| DIMENSION | 20 |
| NUM_LAYERS / HIDDEN_DIM | 4 / 64 |
| BATCH_SIZE | 1024 |
| EPOCHS | 3000 |
| LR | 1e-4 (Adam) |
| VALIDATION_SIZE / VALIDATE_EVERY | 4096 / 50 |
| TEST_SIZE | 16384 |
| MAX_GRAD_NORM | 5.0 |
| Seedi | 42..51 |
| Checkpoint | najnižji validation mean objective |

Validacijske evalvacije se štejejo v budget in lahko posodobijo best-ever. Testne se ne.

### Metrike: discovery, retention, diversity

- **Discovery:** `best_*` (best-ever), `best_gap_percent`, `best_found_eval`, `target_hit`, `target_eval`.
- **Retention:** `final_mean_*` in `final_best_*` na checkpointu (16384 test vzorcev), `retention_loss = final_best − best_ever`.
- **Diversity:** `final_unique_permutations` / 16384, `final_optimum_fraction`.

Retention loss mora biti za vse probleme izračunan enako, sicer tabela ni primerljiva.

## Rezultati do zdaj

### Binarni del: IOH/PBO, n = 100, 100 seedov, budget 4.096M evalvacij (zaključeno)

RealNVP proti (1+1)-EA:
- **OneMax:** oba rešita problem, RealNVP v povprečju potrebuje ~134.6k evalvacij, EA ~1k.
- **LeadingOnes:** RealNVP 77.3 ± 12.4 (1/100 hitov), EA 100/100.
- **ConcatenatedTrap:** oba 0/100. RealNVP skoraj deterministično kolabira v deceptive lokalni optimum.
- **NKLandscapes:** EA je boljši in stabilnejši.
- **IsingTorus:** RealNVP 1/100 hitov, EA 75/100.

Zaključek: trenutni RealNVP + REINFORCE + threshold decoder ni boljši od (1+1)-EA, izpostavi pa različne failure mode-e.

**Jump_k, n = 100, seed 42 (en run na k):**
- Optimum najde do k = 6 (okoli 240k evalvacij), od k = 7 naprej kolabira na n−k.
- Hit se zgodi le v kratkem oknu (~20 epoch), ko je distribucija še široka. Model "prehiti" past, preskočiti je ne zna.
- Manjka več seedov.

### TSP-20 (Euclidean, instance seed 12345, referenčni optimum 3.513669)

- RealNVP pogosto najde optimum, nato distribucija kolabira v drug basin (retention problem).
- **Cosine LR + elite checkpoint:** discovery 8/10, retention nestabilen.
- **Annealed KL:** discovery 10/10, manj kolapsa. Optimum ostane dominanten mode le v ~1/10 runov, pogost near-optimal basin je 3.522437.

### QAP Nug20 (optimum 2570)

| | Baseline | Annealed KL |
|---|---|---|
| Mean best cost | 2805.8 | 2794.8 |
| Mean gap | 9.18 % | 8.75 % |
| Hiti optimuma | 0/10 | 0/10 |
| Mean final generator cost | 2871.8 | 2838.1 |
| Mean retention loss | 66.0 | 43.2 |
| Mean unique permutacij | 4.2 | 11.2 |

KL ima več unique permutacij na vseh 10/10 paired seedih. Anti-collapse efekt je konsistenten, izboljšanje kvalitete iskanja pa manj.

## Trenutno: PFSP Taillard Ta001

- 20 jobov × 5 strojev, best-known makespan 1278. Optimizira se samo permutacija jobov, cilj je makespan.
- `objective.py` je preverjen: identity 1448, NEH 1286 (gap 0.63 %), fast in slow evaluator se ujemata.
- `realnvp_pfsp_baseline.py` je metodološko identičen QAP baselineu (preverjeno 25. 9.). Razlika je samo objective in dodaten stolpec `retention_loss`. **10-seed run teče.**
- `realnvp_pfsp_kl.py` piše uporabnik. Edina intervencija glede na baseline:
  - `y, forward_log_det = model(z)` brez `no_grad` in brez detach.
  - `kl_loss = mean(gaussian_log_prob(z) − forward_log_det − gaussian_log_prob(y))`.
  - `loss = reinforce_loss + T * kl_loss`, pri čemer je T geometrično anneal-an 0.1 → 0.001 z `progress = (epoch−1)/(EPOCHS−1)`. Preveri, da je enako kot pri QAP KL.
  - Checkpoint kriterij ostane enak baselineu.
  - `RESULTS_CSV` mora biti drugačen od baselinea (`pfsp20_realnvp_kl_10seeds.csv`).
- Raziskovalno vprašanje: ali se anti-collapse efekt KL prenese na flow-shop scheduling? Uspeh je že več unique permutacij, manjši retention loss in podobna ali boljša discovery, tudi brez doseženega 1278.
- Pozor: makespan ima velike platoje (veliko permutacij ima enak Cmax), zato je REINFORCE signal šibek.

Po PFSP baseline + KL: primerjalna tabela TSP/QAP/PFSP (discovery, retention, diversity, gapi, KL efekt), nato urediti README in results.

## Benchmark principi

- Glavna mera je število objective evalvacij. Wall-clock ni zanesljiv, kadar več procesov teče paralelno.
- Pravi timing: workers = 1, isti hardware in ista meritev za vse metode.
- Klasičen PFSP baseline: trenutna permutacija → random swap ali insertion → celoten makespan → accept if better. 1 kandidat = 1 evalvacija. NEH je samo referenca, ne budget-matched optimizer.
- Pri TSP inversion baselineu delta evaluation ni isto kot black-box evalvacija, zato bodi previden pri primerjavi eval-countov.

## Struktura repozitorija

- Primarna veja: `main`.
- Permutacijski eksperimenti: `experiments/discrete_optimization/benchmarks/permutation/{tsp,qap,pfsp}/`.
- Rezultati: `results/permutation_optimization/{tsp20,tsp50,qap}/`. PFSP mapo ustvari, ko bodo rezultati.
- Stari in raziskovalni eksperimenti so v legacy/archive.

## Hardware

- Lokalno: Windows, VS Code.
- Remote: big.ijs.si, 2× Xeon E5-2680 v3 (48 niti), ~1 TiB RAM.
- 2× Tesla K80 z driverjem 470 / CUDA 11.4 ne deluje s trenutnim PyTorch buildom (cu130). **Ne predpostavljaj, da CUDA dela.** Večina runov teče na CPU.
- Pri paralelnih seedih na CPU nastavi `torch.set_num_threads(...)`, da se procesi ne dušijo.
