from collections import defaultdict, Counter
import os
import re
import pandas as pd
import numpy as np
from datetime import datetime
from io import BytesIO
from openpyxl import Workbook
from openpyxl.utils import get_column_letter
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

# --- Timezone France (optionnel)
try:
    from zoneinfo import ZoneInfo
    PARIS_TZ = ZoneInfo("Europe/Paris")
except Exception:
    PARIS_TZ = None


# ======================================================
# Utils
# ======================================================
def _norm(s):
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()

def _norm_space(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()

def sanitize_sheet_name(name):
    return re.sub(r'[\\/*?:\[\]]', '', name)[:31]


# ---------- Parsing robuste km / min ----------
def _parse_km_cell(x):
    if isinstance(x, (int, float)) and not pd.isna(x):
        return float(x)
    if isinstance(x, str):
        s = x.strip().lower().replace(",", ".")
        m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*km", s)
        if m:
            return float(m.group(1))
        try:
            return float(re.sub(r"[^\d\.]", "", s))
        except Exception:
            return 0.0
    return 0.0

def _parse_min_cell(x):
    if isinstance(x, (int, float)) and not pd.isna(x):
        return float(x)
    if isinstance(x, str):
        s = x.strip().lower().replace(",", ".")
        m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*min", s)
        if m:
            return float(m.group(1))
        parts = s.split("/")
        if len(parts) >= 2:
            try:
                return float(re.sub(r"[^\d\.]", "", parts[1]))
            except Exception:
                pass
        try:
            return float(re.sub(r"[^\d\.]", "", s))
        except Exception:
            return 0.0
    return 0.0


# ======================================================
# Capacités véhicules
# ======================================================
def adjust_vehicle_capacities(vehicles):
    def extract_additional_capacity(info):
        if pd.isna(info):
            return 0
        m = re.search(r"(\d+)\s*kg", str(info))
        return int(m.group(1)) if m else 0

    vehicles = vehicles.copy()
    base = pd.to_numeric(vehicles.get("Poids (kg)"), errors="coerce").fillna(0)
    # IMPORTANT : Series par défaut pour éviter .apply sur int
    extra_series = vehicles.get(
        "Informations supplémentaires",
        pd.Series(0, index=vehicles.index)
    ).apply(extract_additional_capacity)

    vehicles["Capacité ajustée (kg)"] = base + extra_series
    return vehicles


# ======================================================
# MAJ feuilles Excel (compactes)
# ======================================================
def update_kilometrage_file(workbook, delivery_data):
    """Rebuild compact 'Kilométrage' replacing dates present in delivery_data."""
    HEADER = ("Date de livraison", "Chauffeur", "Tournée", "Distance (km)")

    def to_ymd(v):
        from datetime import date, datetime as _dt
        if v is None: return None
        if isinstance(v, _dt):  return v.date().isoformat()
        if isinstance(v, date): return v.isoformat()
        s = str(v).strip()
        if " " in s: s = s.split(" ", 1)[0]
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%m/%d/%Y"):
            try: return _dt.strptime(s, fmt).date().isoformat()
            except: pass
        return s

    new_entries = []
    for e in delivery_data:
        if not e or len(e) < 4: continue
        new_entries.append((to_ymd(e[0]), e[1], e[2], e[3]))
    target_dates = {e[0] for e in new_entries if e[0]}

    kept_rows = []
    if "Kilométrage" in workbook.sheetnames:
        ws_old = workbook["Kilométrage"]
        for r in range(2, (ws_old.max_row or 1) + 1):
            vals = [ws_old.cell(r, c).value for c in range(1, 5)]
            if all(v in (None, "") for v in vals): continue
            d = to_ymd(vals[0])
            if d in target_dates:
                continue
            kept_rows.append((d, vals[1], vals[2], vals[3]))

    tmp_name = "Kilométrage_tmp"
    if tmp_name in workbook.sheetnames:
        workbook.remove(workbook[tmp_name])
    ws_new = workbook.create_sheet(tmp_name)
    for c, v in enumerate(HEADER, start=1):
        ws_new.cell(1, c, v)

    row = 2
    for d, ch, tour, dist in kept_rows + new_entries:
        ws_new.cell(row, 1, d)
        ws_new.cell(row, 2, ch)
        ws_new.cell(row, 3, tour)
        ws_new.cell(row, 4, dist)
        row += 1

    if "Kilométrage" in workbook.sheetnames:
        workbook.remove(workbook["Kilométrage"])
    ws_new.title = "Kilométrage"


def update_distance_parcourue(workbook):
    if not {"Liste", "Kilométrage"}.issubset(workbook.sheetnames):
        return
    ws_list = workbook["Liste"]
    ws_km   = workbook["Kilométrage"]

    totals = defaultdict(float)
    for _, chauffeur, _, dist in ws_km.iter_rows(min_row=2, values_only=True):
        if dist is None: continue
        try: d = float(dist)
        except:
            s = re.sub(r"[^\d\.,]", "", str(dist)).replace(",", ".")
            try: d = float(s)
            except: d = 0.0
        totals[chauffeur] += d

    headers = [cell.value for cell in ws_list[1]]
    if "Distance parcourue (km)" in headers:
        col = headers.index("Distance parcourue (km)") + 1
    else:
        col = len(headers) + 1
        ws_list.cell(1, col, "Distance parcourue (km)")

    for r in range(2, ws_list.max_row + 1):
        nom    = ws_list.cell(r, 1).value
        prenom = ws_list.cell(r, 2).value
        if nom and prenom:
            key = f"{nom} {prenom}"
            ws_list.cell(r, col, totals.get(key, 0.0))
        else:
            ws_list.cell(r, col, None)


# ======================================================
# Post-traitements de routes (km réels + redistribution)
# ======================================================
def _compute_route_distance_km(seq_nodes, nodes, km_matrix):
    """km Depot -> seq -> Depot, km_matrix alignée à 'nodes'."""
    if not seq_nodes: return 0.0
    idx = {n: i for i, n in enumerate(nodes)}
    d = 0.0
    d += km_matrix[idx["FRESH DISTRIB"]][idx[seq_nodes[0]]]
    for a, b in zip(seq_nodes, seq_nodes[1:]):
        d += km_matrix[idx[a]][idx[b]]
    d += km_matrix[idx[seq_nodes[-1]]][idx["FRESH DISTRIB"]]
    return float(d)


def _best_insertion(seq, node, nodes, km_matrix):
    """Insère 'node' dans 'seq' à la meilleure position (min km)."""
    if not seq:
        return [node]
    best_seq = None
    best_cost = float("inf")
    for pos in range(len(seq) + 1):
        cand = seq[:pos] + [node] + seq[pos:]
        cost = _compute_route_distance_km(cand, nodes, km_matrix)
        if cost < best_cost:
            best_cost, best_seq = cost, cand
    return best_seq


def _redistribute_net_gain(routes, demands_map, cartons_map, nodes, km_matrix, cap_w, cap_c, force_one_tour=False):
    """
    Remplit les tournées vides uniquement si le déplacement diminue la distance totale.
    Si force_one_tour=True et aucun mouvement n'améliore, prend le mouvement de plus faible pénalité.
    """
    empties  = [r for r in routes if len(r["seq"]) == 0]
    nonempty = [r for r in routes if len(r["seq"]) > 0]
    if not empties:
        return routes

    nonempty.sort(key=lambda x: x["dist"], reverse=True)

    for empty in empties:
        best_move = None  # (delta_total, donor, client, new_seq_empty, donor_seq_new, w, c, donor_new_dist, empty_new_dist)
        for donor in nonempty:
            if not donor["seq"]:
                continue
            # candidats proches du dépôt (proxy simple)
            candidates = sorted(
                donor["seq"],
                key=lambda cli: km_matrix[nodes.index("FRESH DISTRIB")][nodes.index(cli)]
            )
            for cli in candidates:
                w = float(demands_map.get(cli, 0.0))
                c = float(cartons_map.get(cli, 0.0))
                rem_w = cap_w(empty["vid"]) - float(empty["w"])
                rem_c = cap_c(empty["vid"]) - float(empty["c"])
                if w > rem_w or c > rem_c:
                    continue
                before = donor["dist"] + empty["dist"]
                donor_seq_new = donor["seq"][:]; donor_seq_new.remove(cli)
                donor_new_dist = _compute_route_distance_km(donor_seq_new, nodes, km_matrix)
                empty_new_seq  = _best_insertion(empty["seq"], cli, nodes, km_matrix)
                empty_new_dist = _compute_route_distance_km(empty_new_seq, nodes, km_matrix)
                delta = (donor_new_dist + empty_new_dist) - before
                if (best_move is None) or (delta < best_move[0]):
                    best_move = (delta, donor, cli, empty_new_seq, donor_seq_new, w, c, donor_new_dist, empty_new_dist)
                    if delta < -1e-6:
                        break
            if best_move and best_move[0] < -1e-6:
                break

        if best_move and (best_move[0] < 0 or force_one_tour):
            delta, donor, cli, empty_seq_new, donor_seq_new, w, c, donor_new_dist, empty_new_dist = best_move
            donor["seq"] = donor_seq_new
            empty["seq"] = empty_seq_new
            donor["w"]  -= w; donor["c"]  -= c; donor["dist"] = donor_new_dist
            empty["w"]  += w; empty["c"]  += c; empty["dist"] = empty_new_dist

    return routes


# ======================================================
# Optimisation
# ======================================================
def run_optimization(
    distance_file,
    orders_file,
    vehicles_file,
    chauffeurs_file,
    critere,
    unavailable_vehicles=None,
    unavailable_chauffeurs=None,

    # contraintes & qualité
    require_one_tour: bool = True,  # impose ≥1 client par chauffeur
    balance_span: bool = True,      # équilibre les distances (span du max)
    span_coeff: int = 40,           # 20..80 conseillé
    time_limit_s: int = 180         # qualité correcte et stable
):
    # ---------- Lecture sources ----------
    dist_mat_raw = pd.read_excel(distance_file, index_col=0)
    dist_mat_raw.index = dist_mat_raw.index.map(_norm_space)
    dist_mat_raw.columns = dist_mat_raw.columns.map(_norm_space)

    orders       = pd.read_excel(orders_file)
    vehs         = pd.read_excel(vehicles_file, skiprows=1)
    ch_df_full   = pd.read_excel(chauffeurs_file, sheet_name="Liste")

    vehs["Véhicule"] = vehs["Véhicule"].apply(_norm_space)
    for col in ["Nom", "Prénom", "Véhicule affecté", "Statut"]:
        if col not in ch_df_full.columns:
            ch_df_full[col] = ""

    ch_df_full["Nom Complet"] = (ch_df_full["Nom"].astype(str) + " " + ch_df_full["Prénom"].astype(str)).str.strip()
    ch_df_full["norm_name"]   = ch_df_full["Nom Complet"].map(_norm)
    ch_df_full["veh_norm"]    = ch_df_full["Véhicule affecté"].astype(str).map(_norm_space)
    ch_df_full["statut_norm"] = ch_df_full["Statut"].astype(str).str.strip().str.lower().replace({"": "permanent"})

    unv_ch_norm = set(_norm(n) for n in (unavailable_chauffeurs or []))
    unv_vh_set  = set(_norm_space(v) for v in (unavailable_vehicles or []))

    # ---------- Choix des chauffeurs ----------
    perms = ch_df_full[
        (ch_df_full["statut_norm"] == "permanent")
        & (~ch_df_full["norm_name"].isin(unv_ch_norm))
        & (~ch_df_full["veh_norm"].isin(unv_vh_set))
    ][["Nom Complet", "veh_norm"]].rename(columns={"veh_norm": "Véhicule affecté"}).reset_index(drop=True)

    pairs = perms.to_dict(orient="records")

    # temporaires de remplacement (même véhicule) uniquement s'il y a des indisponibles
    if unv_ch_norm:
        need = Counter(
            ch_df_full[ch_df_full["norm_name"].isin(unv_ch_norm) & (~ch_df_full["veh_norm"].isin(unv_vh_set))]["veh_norm"]
            .tolist()
        )
        temps_pool = ch_df_full[
            (ch_df_full["statut_norm"] == "temporaire")
            & (~ch_df_full["norm_name"].isin(unv_ch_norm))
            & (~ch_df_full["veh_norm"].isin(unv_vh_set))
        ][["Nom Complet", "veh_norm"]].rename(columns={"veh_norm": "Véhicule affecté"})

        already = set(p["Nom Complet"] for p in pairs)
        for v, k in need.items():
            cand = temps_pool[temps_pool["Véhicule affecté"] == v]["Nom Complet"].tolist()
            for nm in cand[:k]:
                if nm not in already:
                    pairs.append({"Nom Complet": nm, "Véhicule affecté": v})
                    already.add(nm)

    # filtre véhicules indispo
    pairs = [p for p in pairs if _norm_space(p["Véhicule affecté"]) not in unv_vh_set]
    if not pairs:
        return "Aucun chauffeur disponible après filtrage.", None

    # capacités véhicules
    vehs = adjust_vehicle_capacities(vehs)
    veh_dict = vehs.set_index("Véhicule").to_dict("index")
    cartons_col = next((c for c in vehs.columns if re.search(r"\bcartons?\b", str(c), flags=re.I)), None)

    # ne garder que les paires avec véhicule connu
    pairs = [p for p in pairs if p["Véhicule affecté"] in veh_dict]
    if not pairs:
        return "Aucun véhicule trouvable pour les chauffeurs retenus.", None

    # **Ordre fixe par véhicule** (stabilité quand on remplace le chauffeur)
    pairs = sorted(pairs, key=lambda p: _norm_space(p["Véhicule affecté"]))

    # ---------- Demandes ----------
    orders["Code postal"]      = orders["Code postal"].astype(str).str.split(".").str[0]
    orders["Adresse complète"] = orders["Adresse"] + ", " + orders["Code postal"] + ", " + orders["Ville"]
    orders_f = orders[~orders["Libellé"].str.contains("ECHANTILLON RUSTIK", na=False)].copy()

    orders_f["Quantité"] = pd.to_numeric(orders_f["Quantité"], errors="coerce").fillna(0.0)
    orders_f["Cartons"]  = np.where(orders_f["Unité"] == "U",  orders_f["Quantité"]/30.0, 0.0)
    orders_f["Poids"]    = np.where(orders_f["Unité"] == "KG", orders_f["Quantité"],      0.0)
    orders_f["Code client"] = orders_f["Code client"].astype(str).map(_norm_space)

    agg = (
        orders_f.groupby(["Code client","Adresse complète"], dropna=True)
                .agg({"Poids":"sum","Cartons":"sum"})
                .reset_index()
    )
    addr_by_code    = agg.drop_duplicates("Code client").set_index("Code client")["Adresse complète"].to_dict()
    poids_by_code   = dict(zip(agg["Code client"], agg["Poids"]))
    cartons_by_code = dict(zip(agg["Code client"], agg["Cartons"]))

    client_codes = orders_f["Code client"].dropna().astype(str).unique().tolist()
    valid = set(dist_mat_raw.index) & set(dist_mat_raw.columns)
    cust  = [c for c in client_codes if c in valid]
    nodes = ["FRESH DISTRIB"] + cust
    if len(nodes) <= 1:
        return "Aucune commande exploitable.", None

    # Si on exige ≥1 client par chauffeur, on ne peut pas avoir plus de véhicules que de clients
    if require_one_tour and len(cust) < len(pairs):
        note_short = f"ℹ️ {len(pairs)} chauffeurs mais {len(cust)} clients : véhicules actifs réduits à {len(cust)}."
        pairs = pairs[:len(cust)]
    else:
        note_short = ""

    submat = dist_mat_raw.loc[nodes, nodes]

    km_matrix = submat.applymap(_parse_km_cell).values
    np.fill_diagonal(km_matrix, 0.0)

    if critere == "distance":
        cost_float = km_matrix
    else:
        cost_float = submat.applymap(_parse_min_cell).values
        np.fill_diagonal(cost_float, 0.0)

    if float(np.max(cost_float)) <= 0.0:
        return "Matrice de coûts invalide (toutes valeurs nulles). Vérifie le format km/min.", None

    SCALE = 100
    cost_int = np.rint(cost_float * SCALE).astype(np.int64)
    if cost_int.max() <= 0:
        return "Alerte: coûts nuls après parsing. Vérifie le fichier distances/temps.", None

    # ---- Capacités & demandes en décimaux (×10) pour réduire l'erreur d'arrondi
    W_SCALE = 10  # 0.1 kg
    C_SCALE = 10  # 0.1 carton
    w_dem = [0] + [int(round(float(poids_by_code.get(c, 0.0)) * W_SCALE))   for c in nodes[1:]]
    c_dem = [0] + [int(round(float(cartons_by_code.get(c, 0.0)) * C_SCALE)) for c in nodes[1:]]

    def cap_w_scaled(vid):
        try:    return int(round(float(veh_dict[pairs[vid]["Véhicule affecté"]]["Capacité ajustée (kg)"]) * W_SCALE))
        except: return int(1e12)
    def cap_c_scaled(vid):
        if cartons_col is None: return int(1e12)
        try:    return int(round(float(veh_dict[pairs[vid]["Véhicule affecté"]][cartons_col]) * C_SCALE))
        except: return int(1e12)

    vehicle_weights = [cap_w_scaled(v) for v in range(len(pairs))]
    vehicle_cartons = [cap_c_scaled(v) for v in range(len(pairs))]

    # ---------- Builder + Solve (enveloppe pour tenter, avec ou sans contrainte "1 visite") ----------
    def _build_and_solve(enforce_one_tour: bool):
        data = {
            "cost_int": cost_int,
            "km_matrix": km_matrix,
            "w_dem": w_dem,
            "c_dem": c_dem,
            "vehicle_weights": vehicle_weights,
            "vehicle_cartons": vehicle_cartons,
            "num_vehicles": len(pairs),
            "depot": 0
        }

        mgr = pywrapcp.RoutingIndexManager(
            len(cost_int),
            data["num_vehicles"],
            [data["depot"]]*data["num_vehicles"],
            [data["depot"]]*data["num_vehicles"],
        )
        routing = pywrapcp.RoutingModel(mgr)

        def dist_cb(i, j):
            return int(data["cost_int"][mgr.IndexToNode(i)][mgr.IndexToNode(j)])
        cb_idx = routing.RegisterTransitCallback(dist_cb)
        routing.SetArcCostEvaluatorOfAllVehicles(cb_idx)

        # Distance balance
        if balance_span and int(span_coeff) > 0:
            ub = int(max(1, data["cost_int"].max()) * max(3, len(nodes)))
            routing.AddDimension(cb_idx, 0, ub, True, "Distance")
            dist_dim = routing.GetDimensionOrDie("Distance")
            dist_dim.SetGlobalSpanCostCoefficient(int(span_coeff))

        # Weight / Cartons
        w_cb_idx = routing.RegisterUnaryTransitCallback(lambda i: data["w_dem"][mgr.IndexToNode(i)])
        c_cb_idx = routing.RegisterUnaryTransitCallback(lambda i: data["c_dem"][mgr.IndexToNode(i)])
        routing.AddDimensionWithVehicleCapacity(w_cb_idx, 0, data["vehicle_weights"], True, "Weight")
        routing.AddDimensionWithVehicleCapacity(c_cb_idx, 0, data["vehicle_cartons"], True, "Cartons")

        # Visits dimension pour forcer ≥1 client/vehicule
        if enforce_one_tour:
            vis_cb_idx = routing.RegisterUnaryTransitCallback(lambda i: 0 if mgr.IndexToNode(i) == 0 else 1)
            routing.AddDimensionWithVehicleCapacity(vis_cb_idx, 0, [10**9]*data["num_vehicles"], True, "Visits")
            vis_dim = routing.GetDimensionOrDie("Visits")
            # impose au moins 1 visite par route
            for v in range(data["num_vehicles"]):
                vis_dim.CumulVar(routing.End(v)).SetMin(1)

        # Recherche — multithread + opérateurs
        params = pywrapcp.DefaultRoutingSearchParameters()
        params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
        params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        params.time_limit.seconds = int(time_limit_s)
        params.log_search = False
        params.random_seed = 42
        try:
            params.num_search_workers = max(1, os.cpu_count() or 1)
        except Exception:
            pass
        try:
            params.first_solution_neighbors_ratio = 0.2
        except Exception:
            pass
        try:
            ops = params.local_search_operators
            ops.use_relocate = routing_enums_pb2.BOOL_TRUE
            ops.use_exchange = routing_enums_pb2.BOOL_TRUE
            ops.use_or_opt = routing_enums_pb2.BOOL_TRUE
            ops.use_2opt = routing_enums_pb2.BOOL_TRUE
            if hasattr(ops, "use_lin_kernighan"):
                ops.use_lin_kernighan = routing_enums_pb2.BOOL_TRUE
        except Exception:
            pass

        solution = routing.SolveWithParameters(params)
        return solution, routing, mgr, data

    # 1) tenter avec contrainte dure "1 visite"
    solution, routing, mgr, data = _build_and_solve(enforce_one_tour=require_one_tour)

    # 2) si échec et qu'on avait imposé "1 visite", relâcher proprement et on remplira via redistribution
    if not solution and require_one_tour:
        solution, routing, mgr, data = _build_and_solve(enforce_one_tour=False)

    if not solution:
        return "Aucune solution trouvée.", None

    # ---------- Extraction ----------
    routes = []
    for vid in range(len(pairs)):
        idx = routing.Start(vid)
        seq, w_sum, c_sum = [], 0, 0
        while not routing.IsEnd(idx):
            n = mgr.IndexToNode(idx)
            if n != 0:
                code = nodes[n]
                seq.append(code)
                w_sum += data["w_dem"][n]
                c_sum += data["c_dem"][n]
            idx = solution.Value(routing.NextVar(idx))
        dist_km = _compute_route_distance_km(seq, nodes, data["km_matrix"])
        routes.append({
            "vid": vid,
            "seq": seq,
            "w": float(w_sum) / 10.0,   # retour en kg
            "c": float(c_sum) / 10.0,   # retour en cartons
            "dist": dist_km
        })

    # ---------- Redistribution (si besoin) ----------
    if require_one_tour:
        # assure au moins 1 client par chauffeur même si la contrainte a été relâchée
        if any(len(r["seq"]) == 0 for r in routes):
            routes = _redistribute_net_gain(
                routes,
                demands_map={k: float(v) for k, v in dict(zip(nodes[1:], [float(x)/10.0 for x in data["w_dem"][1:]])).items()},
                cartons_map={k: float(v) for k, v in dict(zip(nodes[1:], [float(x)/10.0 for x in data["c_dem"][1:]])).items()},
                nodes=nodes,
                km_matrix=data["km_matrix"],
                cap_w=lambda vid: float(veh_dict[pairs[vid]["Véhicule affecté"]]["Capacité ajustée (kg)"]),
                cap_c=lambda vid: (float(veh_dict[pairs[vid]["Véhicule affecté"]].get(cartons_col, 1e12)) if cartons_col else 1e12),
                force_one_tour=True
            )

    # ---------- Export Excel ----------
    wb = Workbook()
    summary = wb.active
    summary.title = "Résumé"
    summary.append(["Tournée", "Lien vers la feuille"])

    result_str = ""
    if note_short:
        result_str += note_short + "\n"

    delivery_data = []

    routes_sorted = sorted(routes, key=lambda r: r["dist"], reverse=True)
    for i, r in enumerate(routes_sorted, start=1):
        chauffeur = pairs[r["vid"]]["Nom Complet"]
        vehicle   = pairs[r["vid"]]["Véhicule affecté"]
        name = sanitize_sheet_name(f"Tournée {i}")
        ws   = wb.create_sheet(name)

        ws.append([f"Chauffeur : {chauffeur}"])
        ws.append([f"Véhicule : {vehicle}"])
        ws.append(["Code client", "Adresse", "Ordre de visite"])

        full_seq = ["FRESH DISTRIB"] + r["seq"] + ["FRESH DISTRIB"]
        for j, code in enumerate(full_seq, start=1):
            adresse = "DEPOT - FRESH DISTRIB" if code == "FRESH DISTRIB" else addr_by_code.get(code, "")
            ws.append([code, adresse, j])

        for col in ws.columns:
            width = max((len(str(c.value)) for c in col if c.value), default=0) + 2
            ws.column_dimensions[get_column_letter(col[0].column)].width = width

        summary.append([f"Tournée {i}", f"=HYPERLINK(\"#'{name}'!A1\",\"Aller à {name}\")"])

        result_str += (
            f"Tournée {i} : {chauffeur} via {vehicle}\n"
            f"  Clients  : {' -> '.join(full_seq)}\n"
            f"  Distance : {r['dist']:.0f} km\n"
            f"  Poids    : {r['w']:.1f} kg, Cartons : {r['c']:.1f}\n\n"
        )

        now_str = datetime.now(PARIS_TZ).strftime("%Y-%m-%d") if PARIS_TZ else datetime.now().strftime("%Y-%m-%d")
        delivery_data.append((now_str, chauffeur, f"Tournée {i}", round(r["dist"], 0)))

    if "Sheet" in wb.sheetnames:
        wb.remove(wb["Sheet"])

    update_kilometrage_file(wb, delivery_data)
    update_distance_parcourue(wb)

    out = BytesIO()
    wb.save(out); out.seek(0)

    total_d = sum(r["dist"] for r in routes)
    total_w = sum(r["w"]    for r in routes)
    total_c = sum(r["c"]    for r in routes)

    empty_left = sum(1 for r in routes if not r["seq"])
    if empty_left:
        result_str += f"⚠️ {empty_left} véhicule(s) sans client (contraintes et demandes insuffisantes).\n"

    result_str += f"\nTotal : {int(round(total_d))} km | {total_w:.1f} kg | {total_c:.1f} cartons"
    return result_str, out
