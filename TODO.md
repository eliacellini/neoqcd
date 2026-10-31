# TODO

1. **CUDA graph per i layer di flow in valutazione** (NEO-REX e SNF, senza gradienti).
   I layer finiti sono limitati dai lanci di kernel: costano quasi uguale a batch 8 e 64
   (residual a classi, commit 8a2af0e: forward/inversa circa 196/337 ms a batch 8 e 199/385 ms
   a batch 64; smearing 22/60 ms; uno sweep HBOR 128/373 ms). Fatto sul branch `cuda-graphs`
   (31/10): `--cuda-graphs` cattura gli sweep HBOR e la mappa/inversa dello smearing con buffer
   statici, serie di exp(iQ) con controllo di c1 dopo il replay (`DeferredChecks`), inversa a 10
   iterazioni fisse con controllo finale e ripiego senza graph; test su GPU in
   `test/test_cuda_graphs.py`. Validato a 2 nodi il 31/10 (K=2, batch 64, patch h=1, update
   simmetrici): O-REX da 1.31 a 0.92 s/step, NEO-REX smearing class da 2.05 a 1.72 s/step, stessa
   fisica; con NCCL serve `NCCL_ASYNC_ERROR_HANDLING=0` (lo esporta `job/debug_neorex_eval.sh`,
   va aggiunto a ogni job script di produzione che passi `--cuda-graphs`). Resta il residual: prima
   vanno tolte le copie dell'intero reticolo per chiamata (192 memcpy nel profilo del 30/10), poi il
   controllo `delta == 0` va fuori dal graph. Lo sweep HBOR non guadagna dai graph (è al pavimento
   della latenza dei kernel GPU): servono meno kernel per stadio (fusione, Triton).

2. **HBOR solo sulla regione del difetto** lungo i protocolli, come nel paper O-REX
   (arXiv:2510.25704): fatto con `--protocol-patch-halfsize h` (scatola attorno al difetto,
   h = 0 è la più piccola che contiene i link delle plaquette che dipendono da bc) e
   `--protocol-symmetric-updates` (sweep HB+OR o il suo inverso con probabilità 1/2, kernel
   autoaggiunto). Gli sweep di equilibrio restano sull'intero reticolo. Resta da fare lo scan di
   h (0, 1, 2, intero reticolo) su 8^4 per O-REX e NEO-REX, con round trip e tau_int come metrica:
   su GPU il costo di uno sweep non scala con il volume della patch (lanci di kernel, vedi punto 1).

3. **Scala delle repliche ottimale** invertendo la lunghezza termodinamica: dal lavoro di scambio
   per coppia di repliche di O-REX/NEO-REX si costruisce L(lambda) e si mettono i gradini a
   lambda_r = L^-1(r L_tot / (R - 1)), così da equalizzare l'accettazione. Oggi
   `ParallelTempering` e `DistributedCNFREXTrainer` accettano solo scale uniformi. Le produzioni
   salvano il lavoro per coppia (`--save-link-work-values`). (Rob pensaci tu)

4. **Carica topologica, suscettibilità e cooling.** Nel codice non c'è ancora nulla:
   - cooling di Wilson (il paper arXiv:2510.25704 usa n_cool = 60, R_s circa sqrt(8 n_cool / 3));
   - carica Q_clov a clover sui link raffreddati;
   - suscettibilità chi = <Q^2>/V all'estremo periodico (bc = 1), con la media pesata di
     Jarzynski per NE-MCMC/SNF;
   - autocorrelazione di Q lungo la catena REX, come misura di merito contro O-REX e il PT.
   (Rob pensaci tu)

5. **Pulizia del codice quando sarà assestato.** Finché le produzioni in coda girano, ogni modifica
   deve restare compatibile con i job già sottomessi (stesse flag, stessi default, stessi file di
   output). Una volta fissato il codice finale: togliere gli script e le flag non più usati
   (`job/*_array.sh` con `--chdir` su `/leonardo_scratch`, le modalità di smearing scartate),
   unificare i raccoglitori `main/collect_*` con `main/summarize_neorex_validation.py`, far scrivere
   il commit anche a `main/main_pt_obc.py` (oggi lo fa solo `main_cnf_rex.py`) e chiudere la nota
   `notes/neorex_class_smearing_issue.txt`.

