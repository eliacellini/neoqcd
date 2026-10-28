# TODO

1. **CUDA graph per i layer di flow in valutazione** (NEO-REX e SNF, senza gradienti).
   I layer finiti sono limitati dai lanci di kernel: costano quasi uguale a batch 8 e 64
   (residual a classi, commit 8a2af0e: forward/inversa circa 196/337 ms a batch 8 e 199/385 ms
   a batch 64; smearing 22/60 ms; uno sweep HBOR 128/373 ms). Da fare:
   - buffer statici per configurazioni, bc e passo;
   - siti delle maschere e tabelle delle convoluzioni calcolati prima della cattura;
   - il controllo `delta == 0` fuori dal graph;
   - `exp_coefficients` sceglie tra serie e formule chiuse di Morningstar-Peardon secondo c1:
     catturare il ramo della serie e controllare c1 dopo il replay, con ripiego senza graph;
   - inversa: graph di un blocco di 4 iterazioni del punto fisso, rilanciato fino a convergenza;
   - test su GPU: replay contro esecuzione normale a 1e-15.

2. **HBOR solo sulla regione del difetto**, come un defect coupling layer, invece dello sweep
   dell'intero reticolo (in CP(N-1) batte il PT standard). Punto di partenza: gli aggiornamenti
   su scatole annidate di `main_nemcmc_snf.py` (`DefectBox`, `--hierarchical-*`).

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

5. **Storia delle etichette delle repliche e tempo di round trip.** `main/main_pt_obc.py` salva
   accettazione e lavoro per coppia di repliche, ma non quale configurazione sta su quale
   replica a ogni step. Salvare le etichette (array di interi step x repliche x batch) e
   ricavarne il tempo di round trip bc = 0 -> bc = 1 -> bc = 0: limita il tau_int di Q sulla
   replica periodica ed è misurabile già su 8^4, prima di avere cooling e carica topologica.
   Da portare su Leonardo solo dopo che le run di produzione in coda sono partite.
