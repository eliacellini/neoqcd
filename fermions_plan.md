# Fermioni dinamici in neoqcd: piano (branch `fermions`)

Stato: piano del 4 novembre 2026; prima implementazione completata nella notte del 6-7 novembre 2026 (vedi "Stato al 7 novembre" in fondo e `doc/fermions.md`).

## Decisioni prese

- **Difetto solo nel settore di gauge** (Bonanno, Clemente, D'Elia, Maio, Parente, arXiv:2404.14151, "Full QCD with milder topological freezing"). Il fattore c(r) sui link che attraversano il difetto entra solo in S_YM; il determinante fermionico non partecipa al tempering. La differenza d'azione dello swap resta puramente di gauge e locale (contributo nullo oltre 2 passi reticolari dal difetto). Gli autori scrivono che il setup "can be applied verbatim" a Wilson. Lo scheletro di scambio in `distributedPT.py` resta quindi invariato.
- **Prima Wilson, poi staggered.** Wilson con N_f = 2: det(D†D) è esattamente due sapori degeneri, niente rooting, niente RHMC. Staggered: N_f = 4 con HMC esatto come primo passo, N_f = 2 con RHMC (approssimazione di Zolotarev per (M†M)^(-1/2), multishift CG) come secondo.
- **Niente clover all'inizio.** Si aggiunge dopo come termine di forza separato; i benchmark di letteratura scelti sono tutti a c_sw = 0.
- **Obiettivi di questo piano: SNF e parallel tempering (REX) con fermioni.** O-REX e NEO-REX con fermioni sono rimandati a un piano successivo, che partirà da questi due. Ordine: Dirac + solver → HMC → NE-MCMC → SNF; REX in parallelo dopo l'HMC; DD-HMC in parallelo come blocco locale, non prerequisito.
- **Protocollo libero.** Il flusso e il NE-MCMC devono accettare come parametro del protocollo β, κ o il difetto (bc), con la stessa interfaccia; la scelta è a runtime, come oggi `protocol_type` in `Flow`. Il costo per passo cambia (vedi fase 4) ma il codice è uno solo.
- **Condizioni al bordo fermioniche:** periodiche nello spazio, antiperiodiche nel tempo (convenzione di tutti i benchmark sotto). Il difetto non le tocca.
- Tutto in PyTorch, stesso layout dei tensori di gauge (B, D, reticolo, N, N). Campi fermionici: (B, reticolo, 4, N) complessi.
- Test unitari sul Mac; ogni run su Leonardo (debug per test e pilota, produzione per benchmark lunghi).

## Principio di validazione

Ogni blocco ha tre livelli di controllo, in quest'ordine:

1. **Esatto su 4^4 e 8^4.** La matrice di Wilson-Dirac su 4^4 ha dimensione 4^4 x 12 = 3072 (staggered: 768) e il determinante denso si calcola in frazioni di secondo; su 8^4 con even-odd è 24576 x 24576 e si calcola su A100 in secondi. Questo rende esatti: il controllo di det e di log det ratio, un Metropolis con determinante esatto come algoritmo indipendente da confrontare con l'HMC, e la verifica dei rapporti di partizione per Jarzynski.
2. **Identità interne.** Reversibilità, <exp(-ΔH)> = 1, forza contro differenze finite, γ5-hermiticità, teoria libera analitica.
3. **Codice indipendente su 4^4 e 8^4** (benchmark principale a volume piccolo, deciso il 4/11): openQCD per Wilson (c_sw = 0, bc_type 3 periodiche, fermioni antiperiodici nel tempo), Grid per staggered naive e come seconda opinione su Wilson. Noi scegliamo i punti: i due MIT su 8^4 (β = 5.6, κ = 0.153 e 0.1545) più un quark più leggero per stressare il solver. Su 8^4 un run da 1000 traiettorie in openQCD è questione di minuti e dà <P> a ~1e-4. Da verificare in un job di debug: build su Leonardo, e se openQCD accetta 4^4 (ha vincoli sui blocchi locali; Grid non ha problemi a 4^4).
4. **Letteratura** (tabella sotto) come verifica finale, un solo punto a volume medio.

## Benchmark di letteratura

### Wilson N_f = 2, azione di gauge di Wilson, c_sw = 0

| Reticolo | β | κ | <P> | am_PCAC | am_π | Fonte |
|---|---|---|---|---|---|---|
| 4^4 | 1.0 | 0.1 | solo istogrammi (HMC chroma: τ = 0.5, 5 passi leapfrog, acc. ~95%, 65k conf.) | | | Abbott et al., PoS LATTICE2022 036, arXiv:2208.03832 |
| 8^4 | 5.6 | 0.1530 e 0.1545 | solo figure normalizzate (9k conf. HMC con pseudofermioni; pesi con determinante esatto su 8^4) | | | Abbott et al., PRD 109 (2024) 094514, arXiv:2401.10874, Model C |
| 2D SU(3), 16^2 | 6.0 | 0.265 | solo istogrammi (3200 conf. HMC) | | | Abbott et al., arXiv:2207.08945, Sez. IV B |
| 12^3 x 24 | 5.6 | 0.156 | 0.56991(2) (24540 traiettorie); run brevi 0.56982(7), 0.56991(6), 0.56995(4) | | | Hasenbusch, arXiv:1805.03560, Tab. I e testo |
| 32 x 16^3 | 5.6 | 0.15750 | 0.57255(6) | 0.0279(4) | 0.298(5) | Lüscher, CPC 165 (2005) 199, hep-lat/0409106, Tab. 3 |
| 32 x 16^3 | 5.6 | 0.15800 | 0.57335(4) | 0.0153(4) | 0.242(4) | idem |
| 32 x 16^3 | 5.6 | 0.15825 | 0.57387(5) | 0.0092(4) | 0.209(7) | idem |
| 32 x 24^3 | 5.6 | 0.15750 | 0.57250(4) | 0.0273(4) | 0.280(3) | idem |
| 32 x 24^3 | 5.6 | 0.15800 | 0.57344(3) | 0.0143(3) | 0.188(5) | idem |
| 32 x 24^3 | 5.6 | 0.15825 | 0.57383(2) | 0.0084(4) | 0.153(4) | idem |
| 24^3 x 32 | 5.6 | 0.15750 / 0.15800 / 0.15825 | 0.57250(3) / 0.57339(3) / 0.57384(4) | | | Urbach, Jansen, Shindler, Wenger, hep-lat/0506011, Tab. 3 |

Note:
- I lavori MIT fissano i punti a volume piccolo con Wilson N_f = 2 non migliorato (4^4 a β = 1, κ = 0.1; 8^4 a β = 5.6, κ = 0.153 e 0.1545; 2D SU(3) su 16^2) ma non pubblicano valori numerici: le figure sono istogrammi normalizzati all'HMC o derivate normalizzate al flusso. Servono come scelta dei parametri, non come numeri. Il 2D SU(3) gira nel codice così com'è (D parametrico).
- Il determinante esatto arriva fino a 8^4: con even-odd la matrice è 24576 x 24576 complessa (9.6 GB in fp64), LU su A100 in secondi; senza even-odd 49152 x 49152 (38 GB), ancora fattibile. Abbott et al. 2024 usano proprio il determinante esatto su 8^4. Quindi il Metropolis a determinante esatto e i rapporti di partizione esatti valgono anche a 8^4, nei punti β = 5.6, κ = 0.153 e 0.1545.
- Il punto 4^4 a β = 1, κ = 0.1 è a coupling forte e quark pesante: è il caso in cui l'HMC in PyTorch gira in minuti e il determinante esatto chiude il cerchio. Non ci sono numeri nel testo, solo figure; il nostro riferimento è il Metropolis a determinante esatto.
- Il punto 12^3 x 24, κ = 0.156 è l'unico benchmark di letteratura fattibile: quark pesante (m_PS ≈ 0.90 GeV, a ≈ 0.098 fm da Orth, Lippert, Schilling, hep-lat/0503016), τ_int(P) ≈ 13 traiettorie. Circa 1000 traiettorie bastano per un errore di ~1e-4 su P: stima ~1 giorno di A100, da fare una sola volta in coda di produzione come verifica finale.
- I punti 32 x 16^3 e 32 x 24^3 sono riportati solo come riferimento: Lüscher usò 8000 traiettorie con τ_int(P) ≈ 50-70; in PyTorch sono settimane di GPU. Non sono benchmark per noi.
- Il motivo per cui la letteratura non ha tabelle a 4^4 o 8^4 per Wilson N_f = 2 puro è che nessuno le pubblica. Il benchmark a volume piccolo lo costruiamo noi: determinante esatto su 4^4 e openQCD su 8^4 (c_sw = 0, bc periodiche, minuti di run).
- Punto aggiuntivo economico, SU(2) N_f = 2 Wilson: 12^4, β = 2.12, κ = 0.15, <P> = 0.5777(3) con HMC (Jansen, Liu, hep-lat/9506020; valore dal riassunto della ricerca bibliografica, da verificare nel paper). Il codice è SU(N) con N parametrico; a due colori (8 componenti per sito, matrici 2 x 2) un 12^4 costa qualche volta un 8^4 a tre colori, ordine di ore. Controllo opzionale: gradino tra il 4^4 esatto e il 12^3 x 24.

Stime d'ordine di grandezza per l'HMC Wilson in PyTorch su A100, fp64 (da misurare in fase 1):

| Reticolo | κ | Siti | Solve | Traiettoria | Traiettorie per δP ~ 1e-4 | Totale |
|---|---|---|---|---|---|---|
| 4^4 | 0.1 | 256 | ms | secondi | centinaia | minuti |
| 8^4 | ~0.15 | 4k | ~0.1 s | ~10 s | ~1000 | ore |
| 12^3 x 24 | 0.156 | 41k | ~0.5-1 s | ~1 min | ~1000 | ~1 giorno |
| 32 x 16^3 | 0.1575 | 131k | ~3-6 s | ~5 min | ~8000 | settimane |

### DD-HMC (Lüscher 2005, stessa tabella)

Parametri di Tab. 1: blocchi 8^4 su 32 x 16^3, blocchi 8 x 6^2 x 12 su 32 x 24^3, accettanza 0.71-0.90. Link attivi: entrambi gli estremi nel blocco e al più uno sul bordo interno (Fig. 2). Traslazione casuale del campo a ogni ciclo (eq. 3.9): il codice ha già la traslazione periodica (commit b30ce34). Fattorizzazione eq. (3.5): det D = det D_Ω det D_Ω* det{1 - D_Ω^-1 D_∂Ω D_Ω*^-1 D_∂Ω*}; il complemento di Schur agisce solo sui campi su ∂Ω*.

### Staggered

Nessuna tabella pulita a volume piccolo con azione staggered non migliorata trovata. Quello che c'è:
- Kennedy, "Algorithms for dynamical fermions", hep-lat/0607038: RHMC N_f = 2+1 staggered naive su 8^3 x 4, m_s = 0.25, azione di Wilson; cumulante di Binder di <ψ̄ψ> in Tab. 1, plaquette solo in figura (Fig. 5).
- Clark, "Algorithm shootout: R versus RHMC", hep-lat/0510004; Clark, Joó, Kennedy, hep-lat/0209035: confronti R vs RHMC con plaquette, da estrarre.
- Test MILC `ks_imp_rhmc/test`: input 6^4, β = 6.5, masse 0.01/0.05, ma azioni migliorate (asqtad/HISQ) e output di run corti: utili come controllo di regressione del codice MILC, non come riferimento fisico.

Strategia: per staggered il riferimento è (a) il determinante esatto su 4^4 e 8^4 e (b) un run nostro di Grid (NaiveStaggeredFermion, HMC per N_f = 4 e RHMC per N_f = 2) su 4^4 e 8^4 a due o tre valori di (β, m). MILC come alternativa se Grid non si costruisce su Leonardo.

Checklist di convenzioni da allineare prima di confrontare con un codice esterno: definizione di κ (1/(2 m0 + 8)) e di m per staggered, fasi staggered, bc fermioniche nel tempo, normalizzazione della plaquette stampata (openQCD e Grid usano normalizzazioni diverse), lunghezza di traiettoria e definizione di τ_int, segno e normalizzazione di Q.

### Modello di Schwinger (opzionale, banco di prova 2D a determinante esatto)

- Albergo et al., arXiv:2202.11712: N_f = 2 Wilson, β = 2.0, κ = 0.276, L = 16. Baseline con HMC aumentato (1.2 x 10^7 configurazioni): <ψ̄ψ> = 1.50918(9), χ_Q = 0.003875(4). HMC a determinante esatto, τ = 1, 10 passi, acc. 94%.
- Finkenrath, arXiv:2201.02216, Tab. I (N_f = 2 Wilson, verificare L nel testo prima dell'uso): β = 3.0, m0 = -0.082626, am_PS = 0.2241(38), P = 0.82705(12); β = 6.0, m0 = -0.034249, am_PS = 0.1649(23), P = 0.91659(10); β = 8.45, m0 = 0, am_PS = 0.1951(17), P = 0.94084(7). Stesso paper: decomposizione a domini del determinante via Schur ricorsivo con accept/reject gerarchico, cioè la versione 2D di ciò che vogliamo in NEO-REX.
- Richiede un modulo U(1) separato (il codice è SU(N)). Da fare solo se NE-MCMC/SNF con fermioni danno problemi che vogliamo isolare a costo zero.

### NE-MCMC e SNF con fermioni

Non trovato nessun lavoro con Jarzynski e pseudofermioni in 4D. Caselle, Nada, Panero (arXiv:1801.03110) discutono l'estensione a QCD solo nel testo. Le referenze per i flussi con pseudofermioni sono Abbott et al., arXiv:2207.08945 (2D U(1) e SU(3), N_f = 2) e arXiv:2208.03832 (4^4). Questa parte è nuova: i benchmark sono le identità esatte su 4^4.

## Fasi

### Fase 0: operatore di Dirac e solver

Modulo `src/neoqcd/fermions/dirac.py`, `solver.py`.
- D = 1 - κ H nella convenzione di Hasenbusch (κ = 1/(2 m0 + 8)), matrici γ euclidee esplicite nel file con test di anticommutazione.
- Precondizionamento even-odd: D_ee - κ^2 H_eo H_oe. Operatore γ5 D hermitiano; CG su D†D e BiCGstab su D.
- Operatore staggered con fasi η_μ(x), stesso modulo.
- Test: (1) D denso su 4^4 e 2^3 x 4 costruito per colonne e confrontato con l'applicazione sparsa; (2) γ5-hermiticità; (3) teoria libera: correlatore pionico analitico su reticolo finito; (4) log det esatto vs tr log via precondizionamento even-odd; (5) solver: residuo e simmetria.

### Fase 1: pseudofermioni e HMC

Modulo `pseudofermion.py`, `hmc.py`.
- Azione S_pf = φ† (D†D)^-1 φ (even-odd), refresh φ = D† η.
- Forza: derivata di S_pf rispetto ai link, con la proiezione su su(3) già in `sun_utils`. Test contro differenze finite dell'azione, per link.
- Integratori leapfrog e Omelyan, due scale temporali (gauge/fermione), mass preconditioning di Hasenbusch (hep-lat/0107019) come opzione.
- Test: reversibilità (|ΔH| andata-ritorno a precisione macchina), <exp(-ΔH)> = 1, scala ΔH ∝ δτ^2, confronto con Metropolis a determinante esatto su 4^4 (plaquette, Polyakov, correlatore pionico).
- Benchmark: 4^4 (β = 1, κ = 0.1) e 8^4 (β = 5.6, κ = 0.153 e 0.1545, i punti MIT) con determinante esatto; 8^4 contro openQCD; SU(2) 12^4 (Jansen, Liu, opzionale); infine 12^3 x 24 β = 5.6, κ = 0.156 una sola volta.
- Misura del costo per solve e per traiettoria su A100 (fp64) a 8^4, 12^3 x 24, 16^3 x 32: decide il resto del piano.

### Fase 2 (traccia E, opzionale e parallela): DD-HMC

Modulo `ddhmc.py`.
- Blocchi, link attivi, fattorizzazione (3.5), pseudofermioni di blocco con bc di Dirichlet e pseudofermione del complemento di Schur.
- Test: la fattorizzazione con matrici dense su 4^4 con blocchi 2^4; stesse medie dell'HMC pieno su 4^4 e 8^4.
- Collegamento con il difetto: il blocco attivo coincide con la regione del difetto più il suo alone, così il lavoro di non equilibrio nelle fasi 3-6 ha la parte di blocco locale e solo il complemento di Schur non locale.

### Fase 3: NE-MCMC (primo passo della fase 4)

- Protocollo in β o in κ (da fissare; con il trucco di Bonanno un protocollo in bc ha lavoro puramente di gauge e i fermioni entrano solo nei passi stocastici).
- Lavoro: ΔS_g + ΔS_pf a φ fisso tra passi consecutivi; refresh di φ dentro il passo di equilibrio. L'identità di Jarzynski vale sulla distribuzione congiunta (U, φ).
- Test esatto: su 4^4 il rapporto Z(κ_2)/Z(κ_1) si confronta con la media di det ratio esatti su un campione di equilibrio; <exp(-W)> deve coincidere entro errori.

### Fase 4: SNF con fermioni (obiettivo 1)

Punto di partenza nel codice: `Flow` in `flow.py` con `flow_type='nemcmc'`/`'snf'` e `protocol_type='beta'` o `'bc'`, strati stout/residual tra blocchi stocastici che oggi sono sweep di heatbath e overrelaxation (`NEMCMC_update`); driver `main/main_nemcmc_snf.py` con `--protocol-file`; training in `cnf_training.py`.

**Cosa cambia con i fermioni**
- *Blocco stocastico:* una o più traiettorie HMC al valore corrente del parametro, con refresh degli pseudofermioni all'inizio del blocco. In alternativa DD-HMC sulla regione attiva, quando la fase 2 è pronta (non è un prerequisito).
- *Lavoro:* a campi fissi tra un passo del protocollo e il successivo, ΔS = ΔS_g + ΔS_pf; per uno strato deterministico U → U'(U), ΔS = S(U') − S(U) − log det J, con S che include S_pf a φ fisso. L'identità di Jarzynski vale sulla distribuzione congiunta (U, φ), perché il refresh di φ è un passo di equilibrio esatto.
- *Gradiente per il training:* dS_pf/dU è la forza fermionica. Lo stesso solve X = (D†D)^-1 φ dà il valore di S_pf e il suo gradiente: funzione autograd custom, un solve per strato deterministico, nessuna retropropagazione attraverso il CG. La loss resta quella attuale (lavoro, KL, path gradient).
- *Niente flusso condizionale sugli pseudofermioni:* nei flussi di Abbott et al. serve un modello condizionale per φ perché φ è campionato una volta sola; nell'SNF φ viene ricampionato in ogni blocco stocastico, quindi basta il modello marginale sui link. È un argomento a favore dell'SNF da misurare, non da assumere.

**Tre protocolli, una interfaccia.** L'azione S(U, φ; λ) espone `action(x, parameter)` e `action_difference(x, p_old, p_new)` per λ ∈ {β, κ, bc}, come oggi per bc in `mcmc.py`; `Flow` sceglie con `protocol_type`. Cosa costa ogni passo a campi fissi:
- *β, κ fisso*: S_pf non dipende da β, il lavoro stocastico è Δβ S_g, zero solve fuori dall'HMC e dagli strati deterministici. Il più economico, primo da far girare.
- *κ, β fisso*: ΔS_pf = φ†[(D†D)^-1(κ_new) − (D†D)^-1(κ_old)]φ, due solve per passo (uno se si tiene la soluzione del passo precedente). Confronto diretto con il Model C di Abbott et al. (κ 0.153 → 0.1545 su 8^4).
- *bc (difetto)*: con il difetto solo di gauge il lavoro è puramente di gauge e locale, come oggi; i fermioni entrano solo nei blocchi HMC. È il protocollo che condivide tutto con il REX della fase 5.
- *β lungo una linea di fisica costante*: β e κ insieme, due solve per passo; estensione dei due casi sopra, non un caso a parte.

**Passi**
1. Blocco stocastico HMC dentro `Flow` al posto di `NEMCMC_update`, con protocollo in β e senza strati deterministici: è NE-MCMC con fermioni (fase 3) ed è il primo deliverable.
2. Strati deterministici con S_pf nel lavoro e autograd della forza. Test: gradiente della loss contro differenze finite su 4^4; lavoro medio e identità di Jarzynski contro i rapporti di partizione esatti (det denso) su 4^4 e 8^4.
3. Training su 4^4 e 8^4 nei punti di benchmark, nei tre protocolli, tutti con il punto finale su un benchmark verificabile:
   - *β a κ fisso:* β 5.45 → 5.6 a κ = 0.153 (arrivo sul punto MIT 8^4) e a κ = 0.156 (arrivo sul 12^3 x 24 di Hasenbusch). L'intervallo è scelto per analogia con i protocolli di pura gauge di arXiv:2412.00200 (a2: 5.896 → 6.037, a3: 6.02 → 6.178, b0: 6.0 → 6.2), che riducono a di circa il 22%; con N_f = 2 la funzione β è simile in grandezza, quindi Δβ ≈ 0.15. Il rapporto a(β0)/a(βt) va misurato con t0 sui nostri ensemble, perché a κ fisso cambia anche la massa del quark: è un protocollo algoritmico, non a fisica costante.
   - *κ a β fisso:* κ 0.153 → 0.1545 a β = 5.6 (partenza dal punto MIT 8^4, Model C di Abbott et al.).
   - *bc:* 1 → 0 sul difetto, nei punti dell'SNF pure gauge della fase 5. Confronto con HMC puro: ESS, accettanza, osservabili. Confronto con i numeri di Abbott et al. (ESS 48% su 8^4 per il flusso puro nel protocollo in κ) a parità di punto.
4. Costo: solve per configurazione prodotta, contro HMC e contro flusso puro, nel metodo già usato per il paper SNF.

**Benchmark esatti:** su 4^4 e 8^4 ogni quantità dell'SNF è verificabile con il determinante denso: la media di exp(−W) contro Z_1/Z_0 esatto, e l'ESS contro i pesi esatti.

### Fase 5: REX con difetto di gauge (obiettivo 2, in parallelo alle fasi 3 e 4)

- Repliche aggiornate con HMC (o DD-HMC), swap con la differenza d'azione locale già implementata.
- Test: osservabili consistenti con HMC singolo; τ_int(Q) e round trip con `replica_history`.

### Fuori da questo piano: O-REX e NEO-REX con fermioni

Rimandati a un piano successivo che parte da SNF (fase 4) e REX (fase 5). Nota per allora: il protocollo locale sull'alone del difetto acquista ΔS_pf (un solve per blocco, o la parte di blocco di DD-HMC se il blocco attivo copre l'alone); lo Jacobiano locale dello stout resta invariato; il test di bilancio dettagliato va esteso con S_pf.

### Fase 6: staggered

- N_f = 4 HMC esatto con tutti i test della fase 1 (det denso 768 x 768 su 4^4).
- N_f = 2 RHMC: Zolotarev, multishift CG, test della forza e di <exp(-ΔH)> = 1.
- Riferimenti: run nostro MILC/Grid su 8^4; Kennedy hep-lat/0607038 per il cumulante di Binder su 8^3 x 4.

## Dipendenze e parallelismo tra fasi

La spina dorsale è operatore di Dirac → HMC fermionico → NE-MCMC → SNF. REX parte dopo l'HMC e corre in parallelo. DD-HMC è un miglioramento laterale dell'aggiornamento, non un prerequisito. O-REX e NEO-REX sono fuori da questo piano.

```
giorno 1 ─┬─ A0 Dirac Wilson + solver ──► A1 forza + HMC ──► A3 NE-MCMC ──► A4 SNF (β | κ | bc)   ◄── obiettivo 1
          │                                   │
          │                                   ├──► D5 REX con difetto di gauge (swap già pronto)   ◄── obiettivo 2
          │                                   ├──► E2 DD-HMC (blocco locale, opzionale) ──► blocco stocastico di A4, update di D5
          │                                   └──► C2 HMC staggered N_f = 4 ──► C3 RHMC N_f = 2 ──► SNF e REX staggered
          ├─ B1 HMC pure gauge (integratori, reversibilità, exp(-ΔH)) ──► entra in A1
          ├─ B2 determinante denso 4^4 / 8^4, Metropolis esatto ─────────► test di A0, A1, A3, A4, D5
          ├─ B3 openQCD + Grid su Leonardo, run di riferimento ──────────► benchmark-compare
          └─ C1 Dirac staggered + fasi + test ─────────────────────────────► C2
```

Tracce:
- **A (spina dorsale, sequenziale):** 0 → 1 → 3 → 4. Un solo agente con tutto il contesto per stadio. L'SNF (A4), con i tre protocolli, è il primo obiettivo.
- **D (parallel tempering, dopo A1):** REX con difetto di gauge, aggiornamento HMC nelle repliche, swap locale già implementato. Secondo obiettivo; corre in parallelo ad A3 e A4 e condivide con A4 il protocollo in bc.
- **B (infrastruttura, dal giorno 1, indipendente):** HMC di pura gauge con integratori e test di reversibilità (il codice oggi ha solo heatbath), matrice densa e Metropolis a determinante esatto, build e run di riferimento di openQCD e Grid, checklist delle convenzioni. Tutto questo non tocca il codice fermionico e può andare in workflow.
- **C (staggered, parallela ad A):** l'operatore staggered con le sue fasi e i test densi (768 x 768 su 4^4) si scrive insieme a quello di Wilson; l'HMC N_f = 4 riusa l'infrastruttura di A1; l'RHMC viene dopo. SNF e REX staggered riusano A4 e D5 cambiando solo l'operatore.
- **E (DD-HMC, dopo A1, opzionale):** blocco locale per l'aggiornamento; entra nell'SNF come blocco stocastico alternativo e nel REX come update delle repliche. Si fa se A4 e D5 mostrano che l'HMC globale è il collo di bottiglia.

Prerequisito per il parallelismo: **fissare il giorno 1 l'interfaccia** dell'operatore (apply, apply_dagger, even-odd, forma del campo, convenzione di κ e delle bc) e dell'azione parametrica (`action`, `action_difference` per β, κ, bc) in un file piccolo e condiviso, altrimenti le tracce A, B, C, D si scontrano al merge. Ogni traccia su un branch figlio di `fermions` (o worktree degli agenti), merge su `fermions` a fine stadio.

Due avvertenze:
- Il parallelismo riduce il tempo di calendario, non i token. Se il collo di bottiglia è il saldo settimanale, due tracce attive alla volta sono il massimo sensato; la terza aspetta il reset.
- B3 (openQCD e Grid) è l'unica traccia che conviene lanciare subito a prescindere: costa poco in token, dipende dal build su Leonardo che può richiedere tentativi, e i suoi numeri servono a tutte le altre.

## Organizzazione del codice

- Pacchetto `src/neoqcd/fermions/` con `dirac.py`, `solver.py`, `pseudofermion.py`, `hmc.py`, `ddhmc.py`, `staggered.py`, `rhmc.py`.
- Interfaccia: l'azione usata da `distributedPT.py` (`action`, `action_difference`, `local_nf_action_change`) acquista un termine fermionico opzionale; con il difetto solo di gauge `action_difference` in bc non cambia.
- Test in `test/test_fermions_*.py`, tutti su CPU e su 4^4 o 2^3 x 4.
- Script in `main/main_hmc.py` per i benchmark, con il salvataggio delle configurazioni in `cnf_io`.

## Workflow multiagente (ultracode)

Regole generali:
- **Fable orchestra, rivede e giudica. Opus 5.5 scrive, lancia e legge.** Gli agenti Opus pesano sulla metà ordinaria del saldo settimanale, quelli Fable sulla quota Fable del 50%: la quota Fable va spesa solo in verifica e giudizio.
- **Sviluppo sequenziale in diretto, non in workflow.** Operatore di Dirac, solver, forza, HMC dipendono l'uno dall'altro: un solo agente con tutto il contesto batte una catena di consegne. Il workflow entra dove i compiti sono indipendenti (test, derivazioni da confrontare, revisioni, benchmark) o dove serve una seconda opinione.
- **Un workflow per stadio, non per fase.** Tra uno stadio e l'altro leggo il risultato e decido il successivo.
- **Sforzo basso per il lavoro meccanico** (sottomettere job, leggere log, raccogliere numeri), alto o massimo solo per i giudici.
- **Run solo su Leonardo**, via sotto-agente Opus; test unitari sul Mac.

### Chi fa cosa, per fase

| Fase | Sviluppo (diretto) | Verifica (workflow) | Run e raccolta | Giudizio |
|---|---|---|---|---|
| 0 Dirac + solver | Opus, con Fable sulle convenzioni (γ, κ, bc, fasi staggered) | `verify-module`: 3 derivazioni indipendenti di D e di γ5 D; test denso 4^4 scritti da agenti separati | Opus basso, coda debug | Fable: confronto tra derivazioni |
| 1 Pseudofermioni + HMC | Opus; Fable sulla forza | `verify-module` sulla forza (differenze finite, 3 confutatori) + `benchmark-compare` sui punti 4^4 e 8^4 | Opus basso; openQCD e nostro HMC | Fable sulle discrepanze |
| E DD-HMC (opzionale, parallela) | `design-panel` prima di scrivere; poi Opus | `verify-module`: fattorizzazione (3.5) con matrici dense; equivalenza con HMC pieno | Opus basso | Fable |
| 3 NE-MCMC | `design-panel` sul parametro del protocollo e sul lavoro con φ; poi Opus | test di Jarzynski contro det ratio esatti su 4^4, 3 confutatori | Opus basso | Fable |
| 4 SNF (β, κ, bc) | Opus; Fable sull'autograd della forza e sul lavoro nei tre protocolli | `verify-module` + `benchmark-compare` (ESS, osservabili vs HMC, exp(-W) vs Z esatta) | Opus basso, produzione per il training | Fable |
| 5 REX | Opus | riuso di `test_orex_detailed_balance` esteso; `benchmark-compare` su τ_int(Q) | Opus basso | Fable |
| 6 Staggered + RHMC | Opus; Fable su Zolotarev e multishift | `verify-module` + `benchmark-compare` contro Grid | Opus basso | Fable |

### I tre workflow riutilizzabili

**`verify-module`** (ogni modulo nuovo). Input: il modulo, la specifica matematica, i test esatti disponibili.
1. *Derive* (Fable, 3 agenti, sforzo alto, prompt da angoli diversi: dalla definizione nel continuo, dal paper di riferimento, dal codice esistente): ciascuno scrive la formula attesa (per esempio la forza) senza vedere il codice. Schema: espressione, convenzioni assunte, casi limite.
2. *Compare* (codice JS, nessun agente): le tre derivazioni coincidono? Se no, stop e si torna in diretto.
3. *Test* (Opus, pipeline su una lista di proprietà: hermiticità γ5, det denso su 4^4, forza vs differenze finite, reversibilità, teoria libera): un agente per proprietà scrive un test in `test/test_fermions_*.py` e lo esegue sul Mac.
4. *Refute* (Fable, 3 confutatori per ogni test fallito o sospetto, prompt "assumi che il codice sia sbagliato e trova dove"): il modulo passa se nessun confutatore trova un errore riproducibile.
5. *Critic* (Fable, 1 agente): cosa manca? Proprietà non testate, convenzioni non fissate. Il suo output è la lista del giro successivo.

**`benchmark-compare`** (ogni punto di benchmark). Input: lista di punti (reticolo, β, κ o m, codice di riferimento).
1. *Submit* (Opus, sforzo basso, pipeline per punto): sottomette il run di riferimento (openQCD o Grid) e il nostro, sulla coda giusta, e aspetta con Monitor.
2. *Collect* (Opus, basso): legge i log, converte le normalizzazioni (checklist delle convenzioni), calcola medie ed errori con il metodo Gamma già in `neoqcd.analysis`.
3. *Judge* (Fable, 1 agente per punto, sforzo alto): discrepanza entro errori? Se no, ipotesi ordinate per probabilità (convenzione, bc, termalizzazione, bug). Schema: verdetto, deviazione in σ, ipotesi.
4. *Report* (codice JS): tabella in `reports/fermions_benchmarks.md`.

**`design-panel`** (prima delle fasi 3 e E). Input: la domanda di disegno.
1. *Propose* (Fable, 3 agenti, sforzo massimo, angoli diversi: minimo costo, massima località, massima riusabilità del codice esistente): ciascuno propone un disegno con formule, costo per passo in solve, cosa cambia nel codice.
2. *Judge* (Fable, 2 giudici indipendenti): punteggio su correttezza, costo, rischio. Schema: punteggi e motivazioni.
3. *Synthesize* (Fable, 1 agente): disegno finale dal vincitore con gli innesti migliori degli altri. Lo leggo io e lo discutiamo prima di scrivere codice.

### Schizzo dello script di `verify-module`

```js
export const meta = {
  name: 'verify-module',
  description: 'Derivazioni indipendenti, test, confutazione e critica di un modulo fermionico',
  phases: [{ title: 'Derive' }, { title: 'Test' }, { title: 'Refute' }, { title: 'Critic' }],
}
const { module, spec, properties } = args
phase('Derive')
const angles = ['dalla definizione nel continuo', 'dal paper di riferimento', 'dal codice esistente in src/neoqcd']
const derivations = (await parallel(angles.map(a => () =>
  agent(`Deriva ${spec} partendo ${a}, senza leggere ${module}.`,
        { phase: 'Derive', effort: 'high', schema: DERIVATION })))).filter(Boolean)
if (!agree(derivations)) return { stop: 'derivazioni discordanti', derivations }
const tests = await pipeline(properties,
  p => agent(`Scrivi ed esegui un test di "${p}" per ${module} in test/, CPU, 4^4 o 2^3x4.`,
             { phase: 'Test', model: 'opus', schema: TEST_RESULT }),
  (t, p) => t && !t.passed
    ? parallel([0,1,2].map(i => () =>
        agent(`Assumi che ${module} sia sbagliato in "${p}". Trova l'errore e riproducilo. Angolo ${i}.`,
              { phase: 'Refute', effort: 'high', schema: VERDICT })))
        .then(v => ({ ...t, refuters: v.filter(Boolean) }))
    : t)
phase('Critic')
const missing = await agent(`Cosa non è stato verificato di ${module}? Proprietà, convenzioni, casi limite.`,
                            { phase: 'Critic', schema: MISSING })
return { derivations, tests: tests.filter(Boolean), missing }
```

Dimensioni tipiche: `verify-module` 10-15 agenti di cui 4-7 Fable; `benchmark-compare` 3 agenti per punto di cui 1 Fable; `design-panel` 6 agenti, tutti Fable. Il saldo Fable del progetto va quasi tutto nei `design-panel` e nei confutatori.

## Questioni aperte

1. Intervallo in β: proposta 5.45 → 5.6 a κ fisso (fase 4, passo 3), da confermare dopo la misura di t0 ai due estremi.
2. Fare o no il modello di Schwinger come banco 2D (meno necessario ora che i benchmark a 4^4 e 8^4 vengono da openQCD/Grid e dal determinante esatto).
3. Clover: quando.
4. Il 12^3 x 24 a κ = 0.156: confermare dopo la misura di costo della fase 1 (stima ~1 giorno di A100).

## Riferimenti

- Bonanno, Clemente, D'Elia, Maio, Parente, arXiv:2404.14151.
- Lüscher, Comput. Phys. Commun. 165 (2005) 199, hep-lat/0409106.
- Urbach, Jansen, Shindler, Wenger, hep-lat/0506011.
- Hasenbusch, arXiv:1805.03560; Hasenbusch, Phys. Lett. B 519 (2001) 177, hep-lat/0107019.
- Orth, Lippert, Schilling, Phys. Rev. D 72 (2005) 014503, hep-lat/0503016.
- Abbott et al., PoS LATTICE2022 036, arXiv:2208.03832; Abbott et al., arXiv:2207.08945; Abbott et al., PRD 109 (2024) 094514, arXiv:2401.10874; arXiv:2502.00263.
- Albergo et al., arXiv:2202.11712; Finkenrath, arXiv:2201.02216.
- Kennedy, hep-lat/0607038; Clark, hep-lat/0510004; Clark, Joó, Kennedy, hep-lat/0209035.
- Caselle, Nada, Panero, arXiv:1801.03110.
- Bulgarelli, Cellini, Nada, Phys. Rev. D 111 (2025) 074517, arXiv:2412.00200 (SNF SU(3), protocolli in β).
- Jansen, Liu, hep-lat/9506020 (SU(2) N_f = 2 Wilson, 12^4).

## Stato al 7 novembre 2026

Branch `fermions`, 20 commit nella notte del 6-7 novembre, tutti con test verdi; nessun modulo SU(3) esistente modificato. Ogni modulo è stato scritto da un agente Opus e confutato da un agente Fable con derivazioni indipendenti e mutazioni deliberate; i rilievi sono entrati in commit di rifinitura separati. Dettagli, convenzioni e limiti in `doc/fermions.md`.

Fatto (verificato solo su 2D 4x4, 2^3 x 4, 2^4 e 4^4, sul Mac):
- Fase 0: operatore di Wilson, even-odd, CG/BiCGstab, matrice densa, spettro libero, correlatore pionico libero, log det. Staggered con fasi, even-odd, spettro libero.
- B1: HMC di pura gauge (momenti, forza dalle staple, leapfrog e Omelyan a più livelli, confronto con l'heatbath).
- Fase 1: pseudofermioni N_f = 2 even-odd, forza analitica (contro differenze finite e autograd su solve densi), HMC a due livelli, Metropolis a determinante esatto come riferimento indipendente, Hasenbusch.
- C2: pseudofermioni staggered (N_f = 2^{D/2}), forza, HMC, Metropolis esatto.
- Fase 3: `QCDAction` con parametro β, κ (o massa) o bc, blocchi HMC con stato (U, φ), NE-MCMC con identità di Jarzynski verificata contro i rapporti esatti dei determinanti nei tre protocolli (Wilson e staggered).
- Fase 4 passo 2: azione pseudofermionica come `autograd.Function` con backward dalla forza (nessuna retropropagazione nel CG), strati deterministici con S_pf nel lavoro, `FermionSNF` allenabile, log-Jacobiano di Haar verificato esattamente per SU(3).
- Fase 5: REX sul difetto di sola gauge con repliche HMC, swap senza solve, storia delle repliche, passo distribuito su `ParallelTempering`.
- Driver `main_hmc.py`, `main_snf_fermions.py`, `main_rex_fermions.py` con ripresa da checkpoint; job di debug per Leonardo (non lanciati).

Da fare, in ordine:
1. Primi run su Leonardo (coda di debug): `job/debug_hmc_fermions.sh` (4^4 β = 1, κ = 0.1 con log det esatto; 8^4 β = 5.6, κ = 0.153), poi `debug_snf_fermions.sh` e `debug_rex_fermions.sh`. Nessun codice fermionico ha ancora girato su GPU.
2. openQCD e Grid su Leonardo per i punti 4^4 e 8^4 (traccia B3).
3. Fase 4 passo 3: training SNF nei tre protocolli ai punti di benchmark; confronto con HMC e con i flussi MIT; costo in solve per configurazione.
4. SNF con strati per staggered (in corso), RHMC N_f = 2 staggered, clover, DD-HMC (opzionale), protocollo a fisica costante.
5. Rilievo ereditato dal codice puro gauge: la regola "2x" di `Flow._full_logdet` per SU(2) non è il Jacobiano di Haar esatto (errore relativo ~1e-3 a ρ = 0.01-0.1); per SU(3) il log det dello strato è esatto a 1e-14.

## Aggiornamento del 13 novembre 2026

Fatto dal 7 novembre: Hasenbusch e parte staggered di azione/SNF confutati e rifiniti; RHMC per staggered con radice (CG multishift, Zolotarev e Remez); termine clover per Wilson; job di debug spostati sul worktree di test di Leonardo (`neoqcd_test`, mai il checkout di produzione).

Primo run su GPU (A100, complex128, job 64423735, commit 4a95cca): il codice gira su CUDA senza modifiche.
- 4^4, β = 1, κ = 0.1: accettanza 0.9925(25), <exp(-dH)> = 1.00021(59), <P> = 0.06091(19), <log det> = 0.9295(50), 0.81 s per traiettoria.
- 8^4, β = 5.6, κ = 0.153: incompleto (30 minuti finiti in termalizzazione); accettanza 0.957, 154 iterazioni CG per solve, 5.8 ms per iterazione, 20.6 s per traiettoria.
- Il costo per iterazione cresce solo di 2 volte per 16 volte il volume: il kernel è dominato dal lancio delle operazioni, non dall'aritmetica. Le stime di costo della tabella dei benchmark erano troppo ottimistiche di circa un ordine di grandezza a 8^4.

### Fase P: prestazioni (solo dopo la validazione)

Decisione del 13 novembre: CUDA graph e kernel Triton entrano nel piano, ma solo quando la correttezza è chiusa (benchmark 4^4 e 8^4 contro determinante esatto e openQCD/Grid). Fino ad allora si misura soltanto.
1. Misura: `main/bench_fermions.py` (operatori, CG con e senza sincronizzazioni con l'host, forze, varianti del termine di hopping con tabelle di indici e proiezione di spin, replay con CUDA graph, conteggio dei kernel).
2. Ottimizzazioni a costo zero di rischio, ammesse anche prima: batch più grandi, CG senza sincronizzazioni a ogni iterazione, operatore even-odd su metà reticolo.
3. Dopo la validazione: CUDA graph per l'applicazione dell'operatore e per l'iterazione del CG (come già fatto per gli sweep di pura gauge), poi kernel Triton per il termine di hopping fuso (shift, moltiplicazione SU(3), proiezione di spin in un solo kernel) e per l'aggiornamento dei link. Ogni variante veloce resta dietro un'opzione e ha un test di uguaglianza con la versione di riferimento.
4. Precisione mista (solve in complex64 con correzione in complex128) e supporto complex64 degli operatori, oggi assente.

### Dopo la fase P (decisione del 13 novembre)

1. **DD-HMC** (Lüscher 2005): da fare, non più opzionale. Blocchi con bc di Dirichlet, link attivi, complemento di Schur; blocco attivo sulla regione del difetto per avere un aggiornamento locale anche con i fermioni. Viene dopo la validazione e le ottimizzazioni.
2. **Bordi aperti con i fermioni: sono quelli di Bonanno, già implementati** (precisazione dell'utente, 13 novembre). Il difetto agisce solo sull'azione di gauge, i fermioni restano (anti)periodici, lo swap e il lavoro in bc non costano solve. Nessun operatore di Dirac con bordo aperto è previsto. Resta da fare solo l'uso su reticoli veri: difetto di taglia L_d come nel paper di pura gauge, scala ottima, e il driver multi-GPU con un gradino per rank.
