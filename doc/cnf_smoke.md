# Pilot CNF su CPU — branch `cnf`

Esecuzione nell'ambiente `.sunenv`, float64/complex128, con
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`. Gli artifact sono in
`results/cnf_smoke/`; i comandi di utilizzo sono in `doc/cnf.md`.

## Training REX senza integrazione SNF

Cinque training su reticolo 2², beta in [0.5, 1], due repliche,
due catene per replica, due sweep di warmup e tre epoche. Tutti hanno
applicato 3/3 aggiornamenti con gradienti finiti e salvato un checkpoint.

| Campo | PG media, ultima epoca | PINN media fra repliche, ultima epoca |
| --- | ---: | ---: |
| CloverChair | -0.00903068 | 0.0368880 |
| Smearing | -0.00533206 | 0.00636559 |
| Hyper-smearing | -0.00681513 | 0.00990168 |
| Residual | -0.00210021 | 0.00711184 |
| Hyper-residual | -0.00234298 | 0.0104477 |

Questi numeri dimostrano l'esecuzione del percorso, non una classifica:
i campioni sono pochi, le inizializzazioni consumano RNG diversi e le catene
sono correlate. La PG non è una loss positiva e non deve essere interpretata
come la varianza residua: quella è la PINN.

Ulteriori pilot:

- **CloverChair 4³:** tre repliche, due catene, dieci sweep di warmup e
  dieci epoche. 10/10 aggiornamenti applicati. PINN media dell'ultimo batch
  0.00763348. In 3D sono attivi anche i percorsi chair.
- **CloverChair con difetto 2⁴:** beta fisso 6, BC in [0,1], difetto di lato 1,
  due repliche e due epoche. 2/2 aggiornamenti applicati; PINN finita.
- **Smearing funzionale, hyper-smearing per-link e residual con due
  convoluzioni:** un'epoca ciascuno su 4²; backward PG e aggiornamento
  optimizer riusciti. Il test funzionale verifica anche che congelare il
  residuo non modifichi i parametri allenabili tra repliche successive.
- **Prior Haar (beta=0):** un'epoca residual con due repliche in [0,1],
  completata con gradienti e diagnostiche finiti.
- **Resume residual:** tre epoche + ripresa della quarta hanno prodotto
  pesi, catene REX e stato RNG identici bit per bit a quattro epoche continue.

## Ricostruzione SNF

I cinque checkpoint 2D sono stati valutati con due blocchi SNF, due passi CF4
per blocco e quattro traiettorie complessive. Tutti hanno prodotto lavoro,
delta F ed ESS finiti.

Per CloverChair 4³: lavoro medio 91.7405, ESS frazionaria 0.964071
(solo quattro campioni, non una stima affidabile di performance).
Il controllo separato sull'intervallo intero ha dato:

- differenza massima logdet fra 2 e 4 passi CF4: `1.27e-6`;
- errore massimo sui link dopo andata/ritorno a 4 passi: `1.30e-9`;
- residuo massimo fra logdet diretto e inverso: `1.82e-8`.

Il caso con difetto 4D è più impegnativo: la differenza logdet scende da
`2.05e-2` (2 contro 4 passi) a `1.54e-3` (8 contro 16 passi); l'errore di
andata/ritorno sui link scende da `1.09e-6` a `4.09e-8`. Il lavoro medio
delle due traiettorie passa da -7.266296 a -7.266694. Questo caso richiede
ulteriore raffinamento prima di misure di produzione.

La quadratura della divergenza non è il Jacobiano esatto della mappa numerica
discreta. Prima di usare pesi SNF o accettazioni NEO-REX per misure fisiche,
serve uno studio di convergenza al passo con statistiche sufficienti.

## Verifica del codice

62 test passati nella suite mirata: geometria e covarianza gauge, divergenza
numerica, gradiente del residuo congelato, integrazione, quattro famiglie di
adattatori, campionamento REX e derivate dell'azione sul difetto, caricamento
CNF nel runner SNF e nel CLI NEO-REX, regressioni dei checkpoint e del runner
SNF esistenti. Il test NEO-REX usa Gloo a un rank: verifica costruzione DDP,
caricamento e salvataggio, non scambi fra più processi.

Non verificati GPU/NCCL, più rank o training di produzione. Il percorso CNN
legacy `use_nn` e le reti ritagliate sul difetto non hanno un adattatore CNF;
le architetture supportate usano campi sul reticolo completo.
