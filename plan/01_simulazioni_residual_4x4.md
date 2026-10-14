# Piano locale: residual con loop estesi su 4^4

Preparato il 16 settembre 2026. Destinatario: Luna. Questo documento pianifica
le esecuzioni; alla sua creazione non sono state avviate simulazioni.
Comandi e parametri operativi sono in [02_comandi_luna.md](02_comandi_luna.md).

## 1. Obiettivo e perimetro

Misurare se le convoluzioni dei link congelati migliorano il trasporto SNF
rispetto alla residual a plaquette, distinguendo l'effetto della profondità
`residual_n_pt` da quello della maschera `residual_mask_mod`. Confrontare sia
la qualità statistica sia il costo per campione efficace.

Usare il driver `main/main_nemcmc_snf.py`, dominio beta, SU(3), reticolo
periodico 4x4x4x4. Non usare defect/crop, replica exchange, SLURM o altri volumi.
Non cambiare i layer, l'ottimizzatore o la loss durante il confronto. Non
committare i piani né i risultati. Non sovrascrivere i run già presenti.

Questa campagna riprende beta 1.0 -> 1.5: è un confronto architetturale locale,
non una verifica di prestazioni nel regime beta circa 6 o su volumi grandi.
Il notebook `test/residual_snf_beta_protocol_2to4.ipynb` descrive un test 2^4:
non usarlo come baseline numerica 4^4 e non dedurre il volume dal nome del file.

## 2. Risultati storici effettivamente presenti

Radice del progetto: `/Users/eliacellini/Desktop/scripting/neoqcd`.
I percorsi abbreviati seguenti sono relativi a questa radice:

- **DDP** = `results/nemcmc_snf/beta_1p0_to_1p5_D4_T4_L4_B16_K4_100epochs_ddp`
- **DAEMON** = `results/nemcmc_snf/beta_1p0_to_1p5_D4_T4_L4_B16_K4_100epochs_daemon`

Valori letti dai `summary.json`, campi **pooled**, su 1008 campioni:

| Baseline | Directory di evaluation | Processi | ESS frazione | <W> | DeltaF | Tempo eval, s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| NEMCMC senza NF | DDP/nemcmc_eval_1008 | 2 | 0.034066 | 712.4154 | 709.4405 | 15.35 |
| Residual statica | DDP/residual_static/eval_1008 | 2 | 0.624230 | 709.3953 | 709.1523 | 84.86 |
| Hyper-residual h16 d2 | DDP/hyper_residual/eval_1008 | 2 | 0.711199 | 709.3166 | 709.1429 | 87.18 |
| Hyper-residual h16 d3 | DAEMON/evaluation_correct_1008/hyper-residual_h16_d3 | 1 | 0.682174 | 709.2858 | 709.1082 | 179.31 |
| Hyper-residual h32 d2 | DAEMON/evaluation_correct_1008/hyper-residual_h32_d2 | 1 | 0.757119 | 709.3144 | 709.1748 | 181.54 |
| Smearing statica | DDP/smearing_static/eval_1008 | 2 | 0.617798 | 709.4865 | 709.2329 | 57.78 |
| Hyper-smearing per-link h16 d2 | DDP/hyper_smearing_per_link/eval_1008 | 2 | 0.658774 | — | — | 58.74 |
| Hyper-smearing h16 d3 | DAEMON/evaluation_correct_1008/hyper-smearing_h16_d3 | 1 | 0.713118 | 709.3598 | 709.1847 | 126.48 |
| Hyper-smearing h32 d2 | DAEMON/evaluation_correct_1008/hyper-smearing_h32_d2 | 1 | 0.649811 | 709.4451 | 709.2257 | 128.48 |

Per il confronto principale servono le prime cinque righe; le smearing sono
contesto, non richiedono un nuovo sweep. Per valori non riportati consultare i
JSON originali. I tempi storici non sono confronti di velocità fra nuove reti:
world size, processi concorrenti e condizioni della macchina possono differire.

I checkpoint di training sono presenti sotto `residual_static/train`,
`hyper_residual/train` e `DAEMON/hyper-residual_h32_d2/train`, con nome
`snf_checkpoint.pt`. Usare questi file locali: alcuni `run_meta.json` conservano
percorsi temporanei originari che non sono più il percorso del risultato archiviato.

Escludere i tentativi `hyperparameter_sweep_100epochs`, `*_live`, `*_retry` e
`*_launchd*` senza summary completo. `status=ok` da solo non basta: verificare
conteggio campioni, campi pooled finiti e assenza di aggiornamenti saltati.

## 3. Configurazione fissa

| Voce | Valore |
| --- | --- |
| D, T, L, N | 4, 4, 4, 3 |
| Protocollo | 1.0, 1.125, 1.25, 1.375, 1.5 |
| K | 4 |
| Batch globale | 16 |
| Thermalizzazione | 100 aggiornamenti |
| Prior MCMC per epoca/traiettoria | 1 |
| Updates per layer | 1 |
| Overrelaxation | 0 |
| Training principale | 100 epoche da inizializzazione nuova |
| Ottimizzatore | Adam esistente, LR 0.002, grad clip 10 |
| Feature residual | parti reale e immaginaria; termini quadratici attivi |
| Coefficienti residual | init 0.001, coeff_max 0.0 |
| Hypernetwork primaria | embedding 8, hidden 32, depth 2, SiLU |
| Runtime | CPU/Gloo, 1 processo, 1 thread BLAS/OpenMP; un run alla volta |
| Evaluation principale | 1008 campioni = 63 batch da 16 |
| Logging | ogni epoca/batch; W&B disabilitato |

Per le opzioni legacy dell'hypernetwork mantenere `hyper_normalize_by_nstep=true`
e `hyper_scale_by_delta=true`, come nei checkpoint di riferimento. Non copiare
la configurazione del job cluster `production_snf_residual_beta.sh`: usa volume,
beta e altre opzioni differenti.

100 epoche significano 400 tentativi di aggiornamento Adam, uno per ciascuno dei
quattro blocchi a ogni epoca. L'hyper-residual condivide il layer fra i blocchi;
la residual statica ha layer distinti per blocco. Non attribuire una differenza
fra queste due famiglie soltanto alla presenza dell'hypernetwork.

## 4. Matrice principale: sei ablation controllate

Tutte le righe usano `--nf-architecture hyper-residual`, h32 d2.

| ID | n_pt | mod | Sottopassi per layer 4D | Domanda |
| --- | ---: | ---: | ---: | --- |
| H02 | 0 | 2 | 8 | Baseline contemporanea, stessa rete delle nuove prove |
| H04 | 0 | 4 | 16 | Quanto cambia soltanto la partizione/ordine di aggiornamento? |
| H12 | 1 | 2 | 8 | Beneficio del primo passaggio covariante |
| H14 | 1 | 4 | 16 | Stessa profondità, maschera più sparsa |
| H22 | 2 | 2 | 8 | Beneficio della profondità a maschera fissa |
| H24 | 2 | 4 | 16 | Combinazione di profondità e sparsità |

Non cambiare contemporaneamente hidden size, profondità MLP, K o LR.
`n_pt` conta convoluzioni, non la lunghezza di un unico Wilson loop: i layer
costruiscono combinazioni di percorsi. I nuovi pesi covarianti sono 12*n_pt per
smearing step in 4D, condivisi secondo le regole del layer; le 12 feature finali
restano le stesse. Il supporto periodico può avvolgersi su un reticolo 4^4:
un successo qui non dimostra automaticamente trasferibilità a volumi maggiori.

## 5. Seed, cache e indipendenza dei confronti

- Training iniziale: **1234**, uguale per tutte le sei varianti.
- Validation per scegliere la variante: **2234**, mai usato per il training.
- Test finale: **3234, 3235, 3236**, esclusi dalla scelta della variante.
- Eventuali repliche di training: **1235 e 1236**, fase opzionale separata.

Creare una nuova radice risultati, ad esempio
`results/nemcmc_snf/residual_frozen_4x4_20260916_v1`.
Copiarvi la cache storica `DDP/prior_cache` senza modificarne l'originale;
per il training seed 1234 usare la copia, tag `beta1to15_D4_B16`, modalità `read`.
Il manifest `DONE` schema 4 e il payload indicato devono esistere. La cache
archiviata attuale risulta aggiornata il 15 settembre: non assumere che basti a
riprodurre bit per bit tutte le catene dei run precedenti.

Validation e test usano la nuova cache, tag rispettivamente
`residual4_val` e `residual4_test`, modalità `auto`; il seed distingue le entry.
Non includere l'ID della rete nel tag: a parità di seed tutte le reti devono
partire dallo stesso batch termalizzato. Eseguire serialmente per evitare scritture
concorrenti della cache.

La cache contiene solo il batch iniziale, non un ensemble di 1008 configurazioni.
La catena prior viene avanzata durante ogni run. Stesso seed/cache non garantisce
identiche traiettorie fra architetture: cambia il consumo della casualità, e per
i run storici cambia anche la partizione per rank. Non trattare quindi i risultati
come dati appaiati configurazione per configurazione.

## 6. Sequenza di esecuzione

### A. Preflight

1. Leggere questo piano e i comandi; controllare lo stato Git senza modificarlo.
2. Usare `.sunenv/bin/python`; eseguire i due moduli smoke residual indicati nei comandi.
3. Registrare versione Python/PyTorch, CPU disponibile, numero di processi/thread,
   commit corrente e una copia del diff. Il commit da solo non identifica il codice
   perché i nuovi layer e i test sono ancora non committati.
4. Registrare anche copie o hash dei file nuovi non tracciati rilevanti: il diff
   Git non include gli untracked. Verificare spazio libero e assenza di altre
   simulazioni avviate per questa campagna.
5. Creare directory e copia cache; verificare checkpoint storici e manifest.

### B. Pilot di costo e stabilità

Eseguire **3 epoche** per ciascuna delle sei righe, con B=16 e tutti gli altri
parametri invariati. Salvare i pilot in directory separate. Facoltativamente
una evaluation da 64 campioni per verificare il caricamento del checkpoint;
non usare quell'ESS per scegliere l'architettura.

Per ogni pilot registrare tempo totale del processo, `elapsed_seconds`, secondi
per epoca, memoria di picco se disponibile, `skipped_updates` e `finite_count`.
Stimare il costo totale delle sei prove da 100 epoche dai pilot, aggiungendo
thermalizzazione, evaluation e margine. I precedenti h32 d2 su un processo hanno
impiegato circa 1961 s per il training e 182 s per l'evaluation, ma questa è solo
una scala storica: il costo delle convoluzioni va misurato.

Completare un run prima del successivo. Se un pilot fallisce, non lanciare la
corrispondente prova lunga e non abbassare silenziosamente LR, K, B o precisione.
Registrare il fallimento; gli altri run validi possono procedere. Nessun limite
temporale rigido viene imposto da questo documento: Luna deve riportare la stima
prima delle prove lunghe, senza promettere tempi ricavati dai vecchi run.

### C. Campagna principale

Per H02, H04, H12, H14, H22, H24, nell'ordine:

1. Training **da zero**, 100 epoche, seed 1234. Non caricare i pesi dei pilot.
2. Conservare checkpoint e summary; verificare nessun update saltato.
3. Evaluation del checkpoint fisso su 1008 campioni, seed 2234.
4. Compilare subito la riga della tabella risultati, inclusi i tempi.

Il driver salva i pesi finali, non il migliore checkpoint di validation e non lo
stato Adam. `--checkpoint` in training è un warm start con ottimizzatore nuovo,
non una ripresa esatta. Non usare training 3+97 epoche per simulare le 100 epoche.
Un run interrotto senza checkpoint finale va rifatto in una nuova directory.

### D. Ponte con i run storici

Senza riaddestrarli, rivalutare a un processo, seed 2234 e 1008 campioni:

- hyper-residual h32 d2 dal checkpoint DAEMON;
- hyper-residual h16 d2 dal checkpoint DDP;
- residual statica dal checkpoint DDP;
- NEMCMC puro, senza checkpoint.

Usare la stessa cache validation delle nuove reti. Per ogni checkpoint replicare
esattamente i suoi argomenti architetturali e la semantica salvata. Confrontare
H02 con il vecchio h32 d2 per separare differenze dovute alla replica di training
da quelle introdotte dai nuovi percorsi. Nella tabella mantenere tre etichette:
`storico_originale`, `storico_rivalutato`, `nuovo_training`.

### E. Test finale, con scelta bloccata prima di leggere i risultati

Selezionare la variante estesa con maggiore ESS pooled di validation tra le
prove valide. Se due varianti sono entro il **5% relativo** di ESS, scegliere
quella più veloce in evaluation; la soglia è una regola operativa, non un test
di significatività. Annotare separatamente se il migliore ESS/s è un altro modello.

Confrontare **H02 e la variante scelta** sui tre seed test 3234/3235/3236,
1008 campioni per seed. Rivalutare anche il checkpoint storico h32 d2 e NEMCMC
su questi seed per mantenere il collegamento con il lavoro precedente.
Non cambiare vincitore dopo aver guardato il test. Se nessuna estensione supera
H02, riportare il risultato negativo e testare comunque la migliore estensione
per quantificare il confronto.

Questa fase misura la variabilità dell'evaluation a pesi fissi. Non misura la
variabilità del training. La campagna base termina qui.

### F. Estensioni opzionali, dopo il rapporto della campagna base

1. **Repliche di training:** H02 e vincitore, seed 1235 e 1236, 100 epoche da zero;
   mantenere gli stessi seed di evaluation finali per ogni checkpoint e distinguere
   variazione fra training e fra evaluation. Le nuove cache di training si generano
   con `auto`, conservando il tag condiviso.
2. **Famiglia statica:** S02 (`residual`, n_pt=0, mod2) e Sbest (n_pt/mod del
   vincitore hyper), entrambe da zero, stesso LR/budget, hidden/depth legacy 16/2.
   Non confrontare Sbest soltanto con la statica storica addestrata in DDP.
3. **Budget temporale uguale:** scegliere un budget dal tempo misurato di H02;
   calcolare un numero di epoche per variante dai pilot, poi eseguire prove nuove.
   Il driver non offre uno stop nativo per tempo né resume Adam: non interrompere
   brutalmente a una deadline e chiamarlo confronto a budget uguale.
4. Controllo thermalizzazione 100 vs 500 e prior spacing 1 vs 4, applicato sia alla
   baseline sia al vincitore e chiaramente separato dalla replica storica. Cambiare
   `prior_mcmc_steps` cambia la semantica del checkpoint e richiede una campagna
   coerente, non la modifica del JSON per aggirare il controllo.

Non avviare queste estensioni automaticamente nella campagna base.

## 7. Metriche e criterio di lettura

Usare in evaluation `summary.json`:

- `pooled_ess_fraction` = ESS/N, metrica primaria a 100 epoche;
- `N_eff_weights = pooled_finite_count * pooled_ess_fraction`;
- `N_eff_weights / elapsed_seconds`, indicatore secondario di costo in evaluation;
- `pooled_work_mean`, `pooled_work_sem`, `pooled_deltaF`;
- `s2_work = pooled_finite_count * pooled_work_sem**2`, varianza campionaria ricostruita;
- `<W> - DeltaF` come indicatore empirico di dissipazione, con DeltaF stimata dagli
  stessi pesi e quindi soggetta a errore/bias finito;
- tempi train/eval, wall time esterno, update saltati, campioni finiti.

`elapsed_seconds` esclude preparazione iniziale/cache/thermalizzazione e non è il
tempo totale del processo: riportare entrambi. L'ESS dei pesi non include la
correzione per autocorrelazione delle catene: non chiamarla numero esatto di
configurazioni indipendenti. Lo stesso vale per il SEM attualmente prodotto dal
driver; non usarlo come prova definitiva di significatività fra reti.

Non mediare gli ESS dei 63 batch per ricostruire l'ESS pooled. `ess_fraction`
senza prefisso pooled nel summary descrive soltanto l'ultimo batch. Durante il
training i campi pooled sono null: l'ESS finale di training non è il risultato
di evaluation. Non inferire accettanza Metropolis, tempi di autocorrelazione o
tunneling topologico da questi output: non vengono misurati da questo runner.

Il driver non salva tutti i work individuali: i soli summary/CSV non permettono
bootstrap completo dell'ESS pooled. Per la campagna base riportare i tre risultati
test separati, media e deviazione standard fra seed; non inventare barre di errore
da dati mancanti. Un'analisi di autocorrelazione/bootstrap richiede acquisizione
aggiuntiva dei dati, fuori dal perimetro di questo piano.

## 8. Controlli di validità e problemi

- Training: `status=ok`, `skipped_updates=0`, 100 epoche completate; con log_every=1
  ogni riga deve avere `finite_count=16` e tutti i `block_skipped_json` a zero.
- Evaluation: 63 batch, `evaluation_generated_samples=1008`,
  `pooled_finite_count=1008`, metriche finite, ESS frazione in (0,1].
- Verificare su un piccolo batch anche i checkpoint **addestrati**: SU(3),
  equivarianza locale, Jacobiano locale vs differenze finite e ricostruzione con
  fixed point a parametri allenati. I test unitari esistenti coprono inizializzazioni
  di test, non certificano automaticamente ogni checkpoint allenato.
- Per questa diagnostica Luna può caricare il modello tramite il builder esistente
  e adattare in una diagnostica locale alla campagna gli helper dei test; non
  modificare la libreria per far passare il controllo. Controllare anche il segno
  e l'eventuale quasi-singolarità del Jacobiano prima di `abs`/clamp. Il solo logdet
  finito non garantisce invertibilità globale; una verifica campionaria non è una prova.
- Non assegnare successi ai run con NaN esclusi dal conteggio solo perché il driver
  ha scritto `status=ok`. Non promuovere un checkpoint con diagnostiche fallite.
- Se manca un checkpoint storico, segnalarlo e proseguire con la baseline nuova;
  non addestrare silenziosamente un sostituto etichettandolo come storico.
- Se il controllo semantico rifiuta un checkpoint, correggere gli argomenti rispetto
  al suo metadata; non disabilitare il controllo e non usare `strict=False`.

## 9. Consegna attesa da Luna

Nella radice della nuova campagna conservare:

1. `campaign_notes.md`: ambiente, comando di ogni run, tempi stimati/misurati,
   stato completato/fallito, provenienza cache, snapshot/hash sorgenti e checkpoint.
2. Directory separate per pilot, training, validation, test e baseline storiche rivalutate.
3. `comparison.csv`: ID, provenienza, famiglia, n_pt/mod, hidden/depth, seed train/eval,
   world_size, epoche, N, ESS pooled, N_eff_weights, ESS/s, work, varianza, DeltaF,
   dissipazione, tempi e validità. Nessuna media fra seed senza conservare le righe originali.
4. `comparison.md`: tabella storica, confronto attuale a 100 epoche, confronto di costo,
   tre repliche finali e conclusione separata su qualità e convenienza computazionale.
5. Grafici ESS vs profondità per ciascuna maschera, ESS vs tempo eval e curve di
   training. Non sovrapporre curve training ed evaluation come se fossero lo stesso stimatore.

Una conclusione utile può anche essere: più ESS ma meno ESS/s, oppure nessun
miglioramento rilevabile su 4^4. Non è richiesto trovare un vincitore positivo.
