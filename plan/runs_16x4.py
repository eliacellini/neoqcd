"""Rows of the 16^4 phase-A run tracker, written as one JSON file per batch of documents."""
import json
from pathlib import Path

OUT = Path(__file__).parent / "runs16_out"
OUT.mkdir(exist_ok=True)
SS = {2: 7, 3: 6, 4: 6}           # space slice: defect centred in the 16^3 volume
NODES = {4: 1, 8: 2, 12: 3, 16: 4}
rows = []

def add(**r):
    r.setdefault("deps", [])
    r.setdefault("cond", "")
    r.setdefault("note", "")
    r.setdefault("cmd", "")
    r.setdefault("Ld", None); r.setdefault("nrep", None); r.setdefault("nodes", None); r.setdefault("nstep", None)
    r.update(status="todo", owner=None, job="", order=len(rows))
    rows.append(r)

def sb(nodes, limit, name, queue="normal"):
    q = f" --qos=normal" if queue == "normal" else ""
    n = f" --nodes={nodes}" if nodes is not None else ""
    t = f" --time={limit}" if limit else ""
    return f'sbatch --chdir="$PWD"{q}{n}{t} --job-name={name}'

def geom(d):
    return f"NT=16 NS=16 DSIZE={d} TSLICE=8 SSLICE={SS[d]} BATCH=64"

def flags(d, n):
    return ("--save-link-work-values --cuda-graphs --protocol-patch-halfsize 1 --protocol-symmetric-updates \\\n"
            f"  --periodic-translation --parameter-ladder ladders/16x4_Ld{d}_n{n}.json")

# ---------------------------------------------------------------- before anything
add(id="CODE", group="pre", title="Codice aggiornato su Leonardo", method="codice", queue="—", limit="", est="",
    note="Merge di cuda-graphs e optimal-ladder in main, push, pull nel checkout su Leonardo. Lo decide Elia, "
         "quando i job a 8^4 in coda lo permettono. Il commit va nei valori (COMMIT).")
add(id="PRE-1", group="pre", title="Test GPU della ladder e dei CUDA graph", method="test", queue="debug",
    nodes=1, limit="00:15:00", est="5 min", deps=["CODE"],
    cmd=f'{sb(None, "", "pre1-gpu-tests", "debug")} job/debug_gpu_tests.sh test.test_ladder test.test_cuda_graphs')
add(id="PRE-2a", group="pre", title="PT standard a 8^4 con S_D registrata", method="PT", queue="debug", Ld=2, nrep=8,
    nodes=2, nstep=0, limit="00:30:00", est="15 min", deps=["CODE"],
    note="Geometria della produzione 8^4. Prova il PT con i CUDA graph e dà Var_c(S_D) su 8 gradini: da qui "
         "la ladder ottimale ladders/8x4_Ld2_n8.json (main/optimal_ladder.py, poi commit).",
    cmd=f'BATCH=64 STEPS=400 K=0 METHODS=pt {sb(None, "", "pre2a-pt-8x4", "debug")} \\\n  job/debug_neorex_eval.sh --cuda-graphs --periodic-translation \\\n  --swap-parity even_odd')
add(id="PRE-2b", group="pre", title="O-REX K=2 a 8^4 sulla ladder ottimale", method="O-REX", queue="debug", Ld=2, nrep=8,
    nodes=2, nstep=2, limit="00:30:00", est="15 min", deps=["PRE-2a"],
    note="Le accettanze per link devono uscire quasi uguali tra loro. Confronto: job 63026332, stesse impostazioni "
         "sulla ladder lineare (media 0.40).",
    cmd=f'BATCH=64 STEPS=300 K=2 METHODS=orex {sb(None, "", "pre2b-orex-8x4-opt", "debug")} \\\n'
        '  job/debug_neorex_eval.sh --save-link-work-values --cuda-graphs --protocol-patch-halfsize 1 \\\n'
        '  --protocol-symmetric-updates --periodic-translation --parameter-ladder ladders/8x4_Ld2_n8.json')
add(id="PRE-3a", group="pre", title="PG smearing class a 16^4, 5 epoche (tempi)", method="PG class", queue="debug", Ld=2,
    nrep=8, nodes=2, limit="00:30:00", est="15 min", deps=["CODE"],
    note="Misura il tempo per epoca a 16^4 (serve per PG_TIME_class) e produce una rete per PRE-3b. Il job id va in PRE3A_JOB.",
    cmd=f'{sb(None, "", "pre3a-pg-class-16x4", "debug")} job/debug_pg_distributed.sh \\\n'
        '  --domain defect --beta 6 --T 16 --L 16 --parameter-min 0 --parameter-max 1 \\\n'
        '  --defect-size 2 --time-slice 8 --space-slice 7 --batch-size 64 --replicas 8 --warmup 20 \\\n'
        '  --epochs 5 --save-every 5 --pg-direction-batch 8 --pg-chunk-size 32 \\\n'
        '  --architecture defect-hyper-smearing --hyper-smearing-mode class --lr 1e-2 --swap-parity even_odd')
add(id="PRE-3b", group="pre", title="NEO-REX smearing class a 16^4 (tempi, v_NF)", method="NEO-REX class", queue="debug",
    Ld=2, nrep=8, nodes=2, nstep=2, limit="00:30:00", est="10 min", deps=["PRE-3a"],
    note="Solo tempi: il profilo per fase dà v_NF. Se il layer costa 16 volte più che a 8^4, va reso locale prima "
         "della fase A. I limiti di tempo dei NEO-REX nei blocchi si aggiornano da qui.",
    cmd=f'NT=16 NS=16 DSIZE=2 TSLICE=8 SSLICE=7 THERMAL=20 BATCH=64 STEPS=30 K=2 SMEARING_MODE=class \\\n'
        '  METHODS="smearing:reports/pg_distributed_{PRE3A_JOB}/cnf_checkpoint.pt" \\\n'
        f'  {sb(None, "", "pre3b-neorex-16x4", "debug")} job/debug_neorex_eval.sh \\\n'
        '  --cuda-graphs --protocol-patch-halfsize 1 --protocol-symmetric-updates --profile-steps 5 --no-cfg-cache')
add(id="PRE-3c", group="pre", title="PG residual class a 16^4, 5 epoche (tempi)", method="PG residual", queue="debug",
    Ld=2, nrep=8, nodes=2, limit="00:30:00", est="20 min", deps=["CODE"], cond="residual",
    note="Tempo per epoca della residual a 16^4 (serve per PG_TIME_res).",
    cmd=f'{sb(None, "", "pre3c-pg-res-16x4", "debug")} job/debug_pg_distributed.sh \\\n'
        '  --domain defect --beta 6 --T 16 --L 16 --parameter-min 0 --parameter-max 1 \\\n'
        '  --defect-size 2 --time-slice 8 --space-slice 7 --batch-size 64 --replicas 8 --warmup 20 \\\n'
        '  --epochs 5 --save-every 5 --pg-direction-batch 8 --pg-chunk-size 16 \\\n'
        '  --architecture defect-hyper-residual --residual-n-pt 1 --residual-mask-mod 4 --residual-mode class --lr 1e-2 \\\n'
        '  --swap-parity even_odd')

# ---------------------------------------------------------------- A0 pilots
for d in (2, 3, 4):
    add(id=f"A0-d{d}", group="a0", title=f"Pilota PT, L_d = {d}, ladder lineare a 16 gradini", method="PT", queue="normal",
        Ld=d, nrep=16, nodes=4, nstep=0, limit="03:00:00", est="1.5 h", deps=["PRE-1", "PRE-2a"],
        note="Senza termalizzazione: la storia di plaquetta e S_D dal primo step dice quanti step servono (THERMAL). "
             "La parte termalizzata dà Var_c(S_D) su 16 gradini, quindi Lambda e la ladder ottimale.",
        cmd=f'{geom(d)} THERMAL=0 STEPS=1000 K=0 METHODS=pt \\\n'
            f'  {sb(4, "03:00:00", f"a0-pt-d{d}")} \\\n  job/debug_neorex_eval.sh --cuda-graphs --periodic-translation \\\n  --swap-parity even_odd --no-cfg-cache')

add(id="A1", group="a1", title="Termalizzazione e ladder dai pilota", method="analisi", queue="locale", est="",
    deps=["A0-d2", "A0-d3", "A0-d4"],
    note="Elia e Claude. Dalla storia: THERMAL e il tempo dei job di termalizzazione (THERM_TIME). Con "
         "main/optimal_ladder.py le 6 ladder dei blocchi, ladders/16x4_Ld{L_d}_n{n_rep}.json, e Lambda per L_d "
         "(cioè k). Commit delle ladder su main.",
    cmd='python main/optimal_ladder.py reports/neorex_eval_<job A0-d3>/pt_K0 --discard <frazione> \\\n'
        '  --n-rep 8 --output ladders/16x4_Ld3_n8.json')

# ---------------------------------------------------------------- blocks
OREX_LIMIT = {1: "01:30:00", 2: "01:30:00", 4: "01:30:00", 8: "02:00:00", 16: "02:00:00", 32: "03:00:00"}
OREX_EST = {1: "50 min", 2: "50 min", 4: "55 min", 8: "1 h", 16: "1.2 h", 32: "1.7 h"}
NEO_LIMIT = {1: "01:30:00", 2: "02:00:00", 4: "02:00:00", 8: "03:00:00", 16: "04:00:00", 32: "06:00:00"}
RES_LIMIT = {1: "02:00:00", 2: "03:00:00", 4: "04:00:00", 8: "06:00:00", 16: "10:00:00", 32: "16:00:00"}

def block(d, n, ks):
    g = f"b-d{d}-n{n}"
    nodes = NODES[n]
    tid = f"T-d{d}-n{n}"
    add(id=tid, group=g, title=f"Termalizzazione, L_d = {d}, n_rep = {n}", method="termalizzazione", queue="normal",
        Ld=d, nrep=n, nodes=nodes, limit="{THERM_TIME}", est="da A1", deps=["A1"],
        note=(f"Scrive la cache in data/cfgs del checkout da cui si lancia: tutto il blocco va lanciato da lì. "
              + (f"Il percorso della cache (stampato nel log) va in CACHE_d{d}_n8." if n == 8 else "")),
        cmd=f'{sb(nodes, "{THERM_TIME}", f"a2-therm-d{d}-n{n}")} job/thermalize_defect.sh \\\n'
            f'  --T 16 --L 16 --defect-size {d} --time-slice 8 --space-slice {SS[d]} --batch-size 64 \\\n'
            f'  --thermal-steps {{THERMAL}} --parameter-ladder ladders/16x4_Ld{d}_n{n}.json')
    if n == 8:
        add(id=f"PG-class-d{d}", group=g, title=f"Training PG smearing class, L_d = {d}", method="PG class",
            queue="normal", Ld=d, nrep=8, nodes=2, limit="{PG_TIME_class}", est="da PRE-3a", deps=[tid, "PRE-3a"],
            note=f"500 epoche (a 8^4 la loss arriva al plateau in 20-50). Il checkpoint "
                 f"reports/pg_distributed_<job>/cnf_checkpoint.pt va in CKPT_class_d{d}.",
            cmd=f'{sb(2, "{PG_TIME_class}", f"a2-pg-class-d{d}")} job/debug_pg_distributed.sh \\\n'
                f'  --domain defect --beta 6 --T 16 --L 16 --parameter-min 0 --parameter-max 1 \\\n'
                f'  --defect-size {d} --time-slice 8 --space-slice {SS[d]} --batch-size 64 --replicas 8 \\\n'
                f'  --parameter-ladder ladders/16x4_Ld{d}_n8.json --init-cfgs {{CACHE_d{d}_n8}} --warmup 0 \\\n'
                '  --save-every 50 --pg-direction-batch 8 --pg-chunk-size 32 \\\n'
                '  --architecture defect-hyper-smearing --hyper-smearing-mode class \\\n'
                '  --epochs 500 --lr 1e-2 --lr-schedule cosine --lr-min 1e-4 --swap-parity even_odd')
        add(id=f"PG-res-d{d}", group=g, title=f"Training PG residual class, L_d = {d}", method="PG residual",
            queue="normal", Ld=d, nrep=8, nodes=2, limit="{PG_TIME_res}", est="da PRE-3c", deps=[tid, "PRE-3c"],
            cond="residual",
            note=f"Il checkpoint va in CKPT_res_d{d}.",
            cmd=f'{sb(2, "{PG_TIME_res}", f"a2-pg-res-d{d}")} job/debug_pg_distributed.sh \\\n'
                f'  --domain defect --beta 6 --T 16 --L 16 --parameter-min 0 --parameter-max 1 \\\n'
                f'  --defect-size {d} --time-slice 8 --space-slice {SS[d]} --batch-size 64 --replicas 8 \\\n'
                f'  --parameter-ladder ladders/16x4_Ld{d}_n8.json --init-cfgs {{CACHE_d{d}_n8}} --warmup 0 \\\n'
                '  --save-every 50 --pg-direction-batch 8 --pg-chunk-size 16 \\\n'
                '  --architecture defect-hyper-residual --residual-n-pt 1 --residual-mask-mod 4 --residual-mode class \\\n'
                '  --epochs 500 --lr 1e-2 --lr-schedule cosine --lr-min 1e-4 --swap-parity even_odd')
    head = f'{geom(d)} THERMAL={{THERMAL}} STEPS=500'
    add(id=f"PT-d{d}-n{n}", group=g, title=f"PT standard", method="PT", queue="normal", Ld=d, nrep=n, nodes=nodes,
        nstep=0, limit="01:30:00", est="50 min", deps=[tid],
        cmd=f'{head} K=0 METHODS=pt \\\n  {sb(nodes, "01:30:00", f"a3-pt-d{d}-n{n}")} \\\n'
            f'  job/debug_neorex_eval.sh {flags(d, n)} --swap-parity even_odd')
    for k in ks:
        add(id=f"OREX-d{d}-n{n}-K{k}", group=g, title=f"O-REX, n_step = {k}", method="O-REX", queue="normal",
            Ld=d, nrep=n, nodes=nodes, nstep=k, limit=OREX_LIMIT[k], est=OREX_EST[k], deps=[tid],
            cmd=f'{head} K={k} METHODS=orex \\\n  {sb(nodes, OREX_LIMIT[k], f"a3-orex-d{d}-n{n}-K{k}")} \\\n'
                f'  job/debug_neorex_eval.sh {flags(d, n)}')
    for k in ks:
        add(id=f"NEO-d{d}-n{n}-K{k}", group=g, title=f"NEO-REX smearing class, n_step = {k}", method="NEO-REX class",
            queue="normal", Ld=d, nrep=n, nodes=nodes, nstep=k, limit=NEO_LIMIT[k], est="da PRE-3b",
            deps=[tid, f"PG-class-d{d}"],
            note="Limite di tempo provvisorio, da aggiornare con PRE-3b.",
            cmd=f'{head} K={k} SMEARING_MODE=class METHODS="smearing:{{CKPT_class_d{d}}}" \\\n'
                f'  {sb(nodes, NEO_LIMIT[k], f"a3-neo-d{d}-n{n}-K{k}")} \\\n'
                f'  job/debug_neorex_eval.sh {flags(d, n)}')
    for k in ks:
        add(id=f"RES-d{d}-n{n}-K{k}", group=g, title=f"NEO-REX residual class, n_step = {k}", method="NEO-REX residual",
            queue="normal", Ld=d, nrep=n, nodes=nodes, nstep=k, limit=RES_LIMIT[k], est="da PRE-3c",
            deps=[tid, f"PG-res-d{d}"], cond="residual",
            note="Limite di tempo molto provvisorio.",
            cmd=f'{head} K={k} RESIDUAL_MODE=class METHODS="residual:{{CKPT_res_d{d}}}" \\\n'
                f'  {sb(nodes, RES_LIMIT[k], f"a3-res-d{d}-n{n}-K{k}")} \\\n'
                f'  job/debug_neorex_eval.sh {flags(d, n)}')

block(3, 8, [1, 2, 4, 8, 16, 32])
for n in (4, 12, 16):
    block(3, n, [8])
for d in (2, 4):
    block(d, 8, [8])

groups = [
    ("pre", "Controlli prima di tutto", "Coda di debug, 30 minuti al massimo. Vanno fatti prima di qualunque run di produzione."),
    ("a0", "A0 · Run pilota", "PT standard sulla ladder lineare a 16 gradini (4 nodi), uno per dimensione del difetto."),
    ("a1", "A1 · Analisi dei pilota", "In locale: termalizzazione e ladder ottimali. Nessun job."),
    ("b-d3-n8", "Blocco L_d = 3, n_rep = 8 · asse n_step", "Punto centrale. Contiene il training della rete per L_d = 3, che serve anche ai blocchi dell'asse n_rep."),
    ("b-d3-n4", "Blocco L_d = 3, n_rep = 4 · asse n_rep", "1 nodo. NEO-REX usa la rete di L_d = 3 del blocco centrale."),
    ("b-d3-n12", "Blocco L_d = 3, n_rep = 12 · asse n_rep", "3 nodi. NEO-REX usa la rete di L_d = 3 del blocco centrale."),
    ("b-d3-n16", "Blocco L_d = 3, n_rep = 16 · asse n_rep", "4 nodi. NEO-REX usa la rete di L_d = 3 del blocco centrale."),
    ("b-d2-n8", "Blocco L_d = 2, n_rep = 8 · asse L_d", "Con il proprio training della rete per L_d = 2."),
    ("b-d4-n8", "Blocco L_d = 4, n_rep = 8 · asse L_d", "Con il proprio training della rete per L_d = 4."),
]
values = {
    "COMMIT": "Commit di main con cui girano i run (dopo il merge).",
    "THERMAL": "Step di termalizzazione a 16^4, da A1.",
    "THERM_TIME": "Limite di tempo dei job di termalizzazione, da A1 (hh:mm:ss).",
    "PG_TIME_class": "Limite di tempo dei training PG class (500 epoche), da PRE-3a.",
    "PG_TIME_res": "Limite di tempo dei training PG residual (500 epoche), da PRE-3c.",
    "PRE3A_JOB": "Job id di PRE-3a.",
    "CACHE_d2_n8": "Cache di termalizzazione di T-d2-n8 (percorso nel log).",
    "CACHE_d3_n8": "Cache di termalizzazione di T-d3-n8 (percorso nel log).",
    "CACHE_d4_n8": "Cache di termalizzazione di T-d4-n8 (percorso nel log).",
    "CKPT_class_d2": "Checkpoint di PG-class-d2.",
    "CKPT_class_d3": "Checkpoint di PG-class-d3.",
    "CKPT_class_d4": "Checkpoint di PG-class-d4.",
    "CKPT_res_d2": "Checkpoint di PG-res-d2.",
    "CKPT_res_d3": "Checkpoint di PG-res-d3.",
    "CKPT_res_d4": "Checkpoint di PG-res-d4.",
}
meta_values = {"items": {k: {"value": "", "help": h, "order": i} for i, (k, h) in enumerate(values.items())}}
meta_groups = {"groups": [{"id": g, "title": t, "text": x} for g, t, x in groups]}

writes = [{"op": "set", "path": f"runs/{r['id']}", "data": r} for r in rows]
writes += [{"op": "set", "path": "meta/values", "data": meta_values},
           {"op": "set", "path": "meta/groups", "data": meta_groups}]
(OUT / "writes.json").write_text(json.dumps(writes, indent=1))
print(len(rows), "rows;", sum(1 for r in rows if r["cond"]), "conditional;", len(writes), "writes")
import collections
print(collections.Counter(r["group"] for r in rows))
