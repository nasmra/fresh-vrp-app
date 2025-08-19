from collections import defaultdict, Counter
import re
import pandas as pd
import numpy as np
from datetime import datetime
from io import BytesIO
from openpyxl import Workbook
from openpyxl.utils import get_column_letter
from ortools.constraint_solver import pywrapcp, routing_enums_pb2


# ======================================================
# Utils
# ======================================================
def _norm(s): return re.sub(r"\s+", " ", str(s or "")).strip().lower()
def sanitize_sheet_name(name): return re.sub(r'[\\/*?:\[\]]', '', name)[:31]


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
    vehicles["Capacité ajustée (kg)"] = (
        pd.to_numeric(vehicles.get("Poids (kg)"), errors="coerce").fillna(0)
        + vehicles.get("Informations supplémentaires", 0).apply(extract_additional_capacity)
    )
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


def _redistribute_to_fill_empty_routes(routes, demands_map, cartons_map, nodes, km_matrix, cap_w, cap_c):
    """Garantie: au moins 1 tournée par chauffeur (si possible en capacité)."""
    empties  = [r for r in routes if len(r["seq"]) == 0]
    nonempty = [r for r in routes if len(r["seq"]) > 0]
    if not empties:
        return routes

    # plus longues d'abord comme donneurs
    nonempty.sort(key=lambda x: x["dist"], reverse=True)

    for empty in empties:
        for donor in nonempty:
            if not donor["seq"]:
                continue
            # candidats (lourds d'abord)
            cand_sorted = sorted(
                donor["seq"],
                key=lambda c: (demands_map.get(c, 0.0), cartons_map.get(c, 0.0)),
                reverse=True
            )
            moved = False
            for cli in cand_sorted:
                w = float(demands_map.get(cli, 0.0))
                c = float(cartons_map.get(cli, 0.0))
                if w <= cap_w(empty["vid"]) and c <= cap_c(empty["vid"]):
                    # insertion optimale chez empty
                    new_seq = _best_insertion(empty["seq"], cli, nodes, km_matrix)
                    new_dist = _compute_route_distance_km(new_seq, nodes, km_matrix)

                    # déplacer
                    donor["seq"].remove(cli)
                    empty["seq"] = new_seq
                    donor["w"]  -= w
                    donor["c"]  -= c
                    empty["w"]  += w
                    empty["c"]  += c
                    donor["dist"] = _compute_route_distance_km(donor["seq"], nodes, km_matrix)
                    empty["dist"] = new_dist
                    moved = True
                    break
            if moved:
                break
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
    
    balance_span: bool = True,   # activer l’équilibrage des distances (jour J)
    span_coeff: int = 50,       # doser l’équilibrage (0=off, ~50-200 recommandé)
    time_limit_s: int = 120      # temps limite de recherche
):
    # ---------- Lecture sources ----------
    dist_mat_raw = pd.read_excel(distance_file, index_col=0)
    orders       = pd.read_excel(orders_file)
    vehs         = pd.read_excel(vehicles_file, skiprows=1)
    ch_df_full   = pd.read_excel(chauffeurs_file, sheet_name="Liste")

    vehs["Véhicule"] = vehs["Véhicule"].apply(lambda s: re.sub(r"\s+", " ", str(s or "")).strip())
    for col in ["Nom", "Prénom", "Véhicule affecté", "Statut"]:
        if col not in ch_df_full.columns:
            ch_df_full[col] = ""

    ch_df_full["Nom Complet"] = (ch_df_full["Nom"].astype(str) + " " + ch_df_full["Prénom"].astype(str)).str.strip()
    ch_df_full["norm_name"]   = ch_df_full["Nom Complet"].map(_norm)
    ch_df_full["veh_norm"]    = ch_df_full["Véhicule affecté"].astype(str).map(lambda s: re.sub(r"\s+", " ", s).strip())
    ch_df_full["statut_norm"] = ch_df_full["Statut"].astype(str).str.strip().str.lower().replace({"": "permanent"})

    unv_ch_norm = set(_norm(n) for n in (unavailable_chauffeurs or []))
    unv_vh_set  = set(re.sub(r"\s+"," ",str(v or "")).strip() for v in (unavailable_vehicles or []))

    # ---------- Choix des chauffeurs ----------
    # permanents dispo
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

    # véhicules indispo → filtre défensif
    pairs = [p for p in pairs if p["Véhicule affecté"] not in unv_vh_set]
    if not pairs:
        return "Aucun chauffeur disponible après filtrage.", None

    # ---------- Capacités véhicules ----------
    vehs = adjust_vehicle_capacities(vehs)
    veh_dict = vehs.set_index("Véhicule").to_dict("index")

    # colonne cartons (souvent nommée “Cartons (30 unités de pain)”)
    cartons_col = None
    for c in vehs.columns:
        if "Cartons" in c:
            cartons_col = c
            break

    def cap_w(vid):
        try:    return int(veh_dict[pairs[vid]["Véhicule affecté"]]["Capacité ajustée (kg)"])
        except: return 10**9

    def cap_c(vid):
        if cartons_col is None:
            return 10**9
        try:    return int(veh_dict[pairs[vid]["Véhicule affecté"]][cartons_col])
        except: return 10**9

    # garder seulement les pairs ayant des capacités connues
    pairs = [p for p in pairs if p["Véhicule affecté"] in veh_dict]
    if not pairs:
        return "Aucun véhicule trouvable pour les chauffeurs retenus.", None
    
    # 🔒 Ordre déterministe pour stabiliser l'heuristique
    pairs.sort(key=lambda p: (str(p["Véhicule affecté"]), str(p["Nom Complet"])))



    
    # ---------- Demandes ----------
    orders["Code postal"]      = orders["Code postal"].astype(str).str.split(".").str[0]
    orders["Adresse complète"] = orders["Adresse"] + ", " + orders["Code postal"] + ", " + orders["Ville"]
    orders_f = orders[~orders["Libellé"].str.contains("ECHANTILLON RUSTIK", na=False)].copy()

    orders_f["Quantité"] = pd.to_numeric(orders_f["Quantité"], errors="coerce").fillna(0.0)
    orders_f["Cartons"]  = np.where(orders_f["Unité"] == "U",  orders_f["Quantité"]/30.0, 0.0)
    orders_f["Poids"]    = np.where(orders_f["Unité"] == "KG", orders_f["Quantité"],      0.0)

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

    submat = dist_mat_raw.loc[nodes, nodes]

    # km réels (pour reporting/redistribution)
    km_matrix = submat.applymap(
        lambda x: float(str(x).split(" km")[0].replace(",", ".")) if isinstance(x, str) and "km" in x else 0.0
    ).values
    np.fill_diagonal(km_matrix, 0.0)

    # matrice de coût (distance ou temps) → ENTIER avec SCALE
    if critere == "distance":
        cost_float = km_matrix
    else:
        cost_float = submat.applymap(
            lambda x: float(str(x).split("/")[1].split(" min")[0].strip().replace(",", ".")) if isinstance(x, str) and "min" in x else 0.0
        ).values
        np.fill_diagonal(cost_float, 0.0)

    SCALE = 100  # centi-km ou centi-min
    cost_int = np.rint(cost_float * SCALE).astype(np.int64)

    # demandes entières (arrondi)
    w_dem = [0] + [int(round(float(poids_by_code.get(c, 0.0))))   for c in nodes[1:]]
    c_dem = [0] + [int(round(float(cartons_by_code.get(c, 0.0)))) for c in nodes[1:]]

    vehicle_weights = [cap_w(v) for v in range(len(pairs))]
    vehicle_cartons = [cap_c(v) for v in range(len(pairs))]

    # ---------- OR-Tools ----------
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

    # dimension Distance avec un plafond sûr (>0)
    ub = int(max(1, data["cost_int"].max()) * max(3, len(nodes)))  # marge large
    routing.AddDimension(cb_idx, 0, ub, True, "Distance")
    dist_dim = routing.GetDimensionOrDie("Distance")

    # >>> ÉQUILIBRAGE JOUR J (dosage)
    dist_dim.SetGlobalSpanCostCoefficient(int(span_coeff) if balance_span and span_coeff > 0 else 0)

    # capacités
    w_cb_idx = routing.RegisterUnaryTransitCallback(lambda i: data["w_dem"][mgr.IndexToNode(i)])
    c_cb_idx = routing.RegisterUnaryTransitCallback(lambda i: data["c_dem"][mgr.IndexToNode(i)])
    routing.AddDimensionWithVehicleCapacity(w_cb_idx, 0, data["vehicle_weights"], True, "Weight")
    routing.AddDimensionWithVehicleCapacity(c_cb_idx, 0, data["vehicle_cartons"], True, "Cartons")

    params = pywrapcp.DefaultRoutingSearchParameters()
    
    # Meilleure solution initiale pour VRP multi-véhicules
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    
    # Métaheuristique pour sortir des optima locaux
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    
    # Temps de recherche
    params.time_limit.seconds = int(time_limit_s)
    
    # Reproductibilité (si dispo dans ta version)
    try:
        params.random_seed = 42
    except AttributeError:
        # Anciennes versions : ce champ n’existe pas -> on ignore
        pass
    
    # (optionnel) logs pour debug
    # params.log_search = True



    solution = routing.SolveWithParameters(params)
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
        routes.append({"vid": vid, "seq": seq, "w": float(w_sum), "c": float(c_sum), "dist": dist_km})

    # ---------- Forcer 1 tournée par chauffeur (redistribution optimale simple) ----------
    routes = _redistribute_to_fill_empty_routes(
        routes,
        demands_map={k: float(v) for k, v in dict(zip(nodes[1:], w_dem[1:])).items()},
        cartons_map={k: float(v) for k, v in dict(zip(nodes[1:], c_dem[1:])).items()},
        nodes=nodes,
        km_matrix=data["km_matrix"],
        cap_w=lambda vid: vehicle_weights[vid],
        cap_c=lambda vid: vehicle_cartons[vid],
    )

    # ---------- Export Excel ----------
    wb = Workbook()
    summary = wb.active
    summary.title = "Résumé"
    summary.append(["Tournée", "Lien vers la feuille"])

    result_str = ""
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
        delivery_data.append((datetime.now().strftime("%Y-%m-%d"), chauffeur, f"Tournée {i}", round(r["dist"], 0)))

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
        result_str += f"⚠️ {empty_left} véhicule(s) sans client (demande insuffisante).\n"

    result_str += f"\nTotal : {int(round(total_d))} km | {total_w:.1f} kg | {total_c:.1f} cartons"
    return result_str, out


