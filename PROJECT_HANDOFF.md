# RealNVP za diskretno optimizacijo: predaja projekta

Gaber Zupan Škrlj, Institut Jožef Stefan, 1. 10. 2026

## Uvod

Ta dokument je namenjen tistemu, ki projekt prevzema. Septembra 2026 sem na IJS raziskoval, ali lahko normalizing flow model, konkretno RealNVP, služi kot optimizacijski algoritem za diskretne probleme. Tu je opisano, s kakšno idejo sem začel, kako se je delo razvijalo, do česa sem prišel in kje bi po mojem mnenju nadaljeval.

Dokument je namenoma kratek. Vse številke, nastavitve eksperimentov in tehnične podrobnosti so v [`PROJECT_HANDOFF_TECHNICAL.md`](PROJECT_HANDOFF_TECHNICAL.md), 2D študija pa ima svoj opis v [`experiments/continuous_optimization/README.md`](experiments/continuous_optimization/README.md). Predlagam, da najprej prebereš ta dokument, tehničnega pa uporabljaš kot priročnik, ko začneš delati s kodo.

## Ideja

Klasični optimizacijski algoritmi, na primer evolucijski, vzdržujejo eno ali nekaj trenutnih rešitev in jih postopoma izboljšujejo. Ideja projekta je bila drugačna: namesto posameznih rešitev se naučimo celotno verjetnostno porazdelitev nad rešitvami. Iz nje vzorčimo kandidate, jih ocenimo in porazdelitev premaknemo proti boljšim. Če bi to delovalo, bi model lahko hkrati ohranjal več dobrih območij prostora rešitev, kar je pri težkih problemih z mnogo lokalnimi optimumi zanimiva lastnost.

RealNVP generira zvezne vektorje. Diskretno rešitev dobimo tako, da vektor zaokrožimo (binarni problemi) ali uredimo po velikosti (permutacije, t. i. random keys). Ker ta korak ni odvedljiv, se model uči z metodo REINFORCE, ki potrebuje samo vrednost kriterijske funkcije, ne pa njenega gradienta.

## Kako je delo potekalo

Najprej sem metodo preizkusil na petih standardnih binarnih problemih in jo primerjal s preprostim (1+1) evolucijskim algoritmom. Metoda je delovala, a je bila povsod slabša ali precej počasnejša. Pokazala pa je zanimiv vzorec: model je dobre rešitve pogosto našel, nato pa se je njegova porazdelitev zožila na eno samo, ne nujno najboljšo.

Glavni del dela je bil na permutacijskih problemih:TSP-20, QAP in PFSP. Pri TSP se je isti vzorec pokazal zelo jasno. Model je optimalno pot med učenjem pogosto našel, ob koncu učenja pa je skoraj nikoli ni več generiral. Temu sem rekel razlika med *odkrivanjem* in *ohranjanjem* rešitve in to je postalo osrednje vprašanje projekta.

Sledilo je več poskusov, da bi model optimum ohranil: drugačen potek učne stopnje, dodatno raziskovanje, KL člen s temperaturo, ki porazdelitev drži širšo. KL člen je izboljšal odkrivanje na TSP in povečal raznolikost na vseh treh problemih, ohranjanja optimuma pa ni rešil nobeden od poskusov.

Za pošteno primerjavo sem nato na vseh treh permutacijskih problemih pognal (1+1) evolucijski algoritem z enakim številom evalvacij. Rezultat je bil jasen in je glavni zaključek primerjalnega dela.

Na koncu sem želel razumeti, zakaj model izgubi dobre rešitve. Naredil sem dve kontroli. Prva zamenja RealNVP z najpreprostejšim možnim modelom, diagonalno Gaussovo porazdelitvijo, da vidimo, ali je težava v samem flowu. Druga je 2D problem z dvema minimumoma, kjer je ciljna porazdelitev točno znana, zato lahko ločimo, ali model nima dovolj kapacitete ali pa je težava v samem učenju.

## Kaj sem ugotovil

Na nobenem testiranem problemu RealNVP ni bil boljši od preprostega evolucijskega algoritma. Na permutacijskih problemih je algoritem dosegel enako dobro ali boljšo rešitev na vseh desetih semenih, pri TSP pa je optimum našel približno 50-krat hitreje (mediana čez semena).

| Problem | RealNVP + KL | (1+1)-EA |
| --- | --- | --- |
| TSP-20 | optimum na 8/10 semenih | optimum na 10/10 semenih |
| QAP Nug20 | povprečno približno 9 % nad optimumom | povprečno približno 2 % nad optimumom |
| PFSP Ta001 | optimum na 0/10 semenih | optimum na 10/10 semenih |

Drugi rezultat je razlika med odkrivanjem in ohranjanjem. Model optimum najde, končna porazdelitev pa ga praviloma ne vsebuje. Noben od preizkušenih popravkov tega ni zanesljivo rešil; nekateri starejši runi so optimum občasno obdržali.

Tretji rezultat je, da težava ni specifična za RealNVP. Gaussov model s samo 40 parametri je na TSP-20 dosegel približno enako kot RealNVP, v nekaterih pogledih celo bolje. Dodatna izraznost flowa torej ni pomagala. Učna stopnja je bila za oba modela izbrana z enakim postopkom; tudi RealNVP pri najboljši učni stopnji optimuma ne obdrži.

Četrti in po mojem najzanimivejši rezultat izhaja iz 2D študije. Model zna predstaviti porazdelitev z več minimumi in ko začne iz pravilne porazdelitve, jo med učenjem tudi obdrži. Ko pa se uči od začetka, redek minimum izgubi, čeprav bi bila rešitev z obema minimumoma boljša tudi po kriteriju, ki ga optimizira. Razlog je preprost: ko v nekem območju ni več vzorcev, učenje o njem ne dobi nobene informacije in se tja ne more vrniti. Težava je torej v dinamiki učenja, ne v kapaciteti modela. Ali isti mehanizem povzroča izgubo optimuma tudi pri TSP, še ni preverjeno.

## Kje bi nadaljeval

Najprej bi preveril, ali 2D ugotovitev velja tudi za permutacije: učenje pri TSP začeti iz porazdelitve, ki optimum že vsebuje, in pogledati, ali ga model obdrži. Če ga, je težava tudi tam v dinamiki učenja. Smiselne so še primerjava s CMA-ES na isti predstavitvi, Gaussov model na QAP in PFSP ter več ponovitev eksperimenta Jump_k. Podroben seznam je na koncu tehničnega dokumenta.

Ne bi pa nadaljeval z dodajanjem novih arhitektur ali hevrističnih popravkov, dokler ni jasno, ali je ozko grlo v učenju ali v predstavitvi problema.

## Kako začeti

1. Preberi ta dokument, nato razdelek "Pasti za naslednika" v [`PROJECT_HANDOFF_TECHNICAL.md`](PROJECT_HANDOFF_TECHNICAL.md). Tam so stvari, ki niso razvidne iz kode in so že povzročile napačne zaključke.
2. Koda permutacijskih eksperimentov je v `experiments/discrete_optimization/benchmarks/permutation/`, rezultati v `results/permutation_optimization/`, 2D študija v `experiments/continuous_optimization/`.
3. Skripte poganjaj iz njihove mape z `~/Real-nvp-learning/.venv/bin/python`. Na strežniku big.ijs.si CUDA ne deluje, vse teče na procesorju.

