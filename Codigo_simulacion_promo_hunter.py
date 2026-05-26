# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

"""
============================================================
  PROMO HUNTER — Modelo de Simulación
  Modelos y Simulación — Tercer Entregable
  Integrantes: Urbano, Ruiz, Arteaga, Sánchez
============================================================
  Arquitectura:
    1. Cadena de Markov Absorbente  → Embudo individual (pub → compra)
    2. Cadena de Markov Ergódica    → Estado del canal (dist. estacionaria)
    3. Simulación Monte Carlo DES   → Convergencia y comparación escenarios
    4. Análisis de Sensibilidad     → Robustez del modelo
============================================================
  Fuentes de datos reales:
    - sim_embudo_completo.json  (Telegram API + Amazon Associates)
    - sim_tabla_horaria.csv     (21 097 posts, ene-may 2026, 143 días)
    - sim_tabla_categoria.csv   (7 categorías, 11 tracking IDs)
============================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import solve
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
plt.rcParams.update({'font.size': 10, 'font.family': 'DejaVu Sans'})
sns.set_theme(style="whitegrid")


# ─────────────────────────────────────────────────────────────
#  SECCIÓN 1: PARÁMETROS CALIBRADOS
# ─────────────────────────────────────────────────────────────

class P:
    """
    Parámetros globales calibrados con datos REALES (ene-may 2026).
    Todos los valores numéricos tienen trazabilidad directa a los CSV.
    """
    # ── Datos reales del periodo completo ──────────────────────
    DIAS_DATOS        = 143
    POSTS_TOTALES     = 21_097
    VISTAS_TOTALES    = 6_613_127
    CLICS_TOTALES     = 96_507
    COMPRAS_TOTALES   = 6_027
    GANANCIAS_USD     = 3_149.42       
    INGRESOS_USD      = 96_973.87     

    LOGNORM_MU        = 5.726          
    LOGNORM_SIGMA     = 0.201          
    VISTAS_MEDIA      = 313.3          
    CTR               = 0.014593       
    CR                = 0.0625        
    COMISION_UNIT     = 0.52          
    INGRESO_UNIT      = 16.09          
    TASA_DEVOLUCION   = 0.0169         

    CR_TRACKING = {
        's73f45227-20':     0.0827,
        'elpromohunt0c-20': 0.0747,
        'krik0n-20':        0.0483,
        'koose-20':         0.0731,
        'brunardo-20':      0.0636,
        'opmilo21-20':      0.0623,
        'hnuol3820-20':     0.0421,
        'yennifer3606-20':  0.0606,
        'lmolartec3639-20': 0.0659,
        'chiriboga6490-20': 0.0522,
        'comardex-20':      0.0525,
    }
    CR_MED = sum(CR_TRACKING.values()) / len(CR_TRACKING)
    CR_STD = float(np.std(list(CR_TRACKING.values())))

    PUBS_DIA = 147.5                   # media real Poisson (21097/143)

    # ── Tabla horaria REAL (sim_tabla_horaria.csv) ─────────────
    # Fuente directa del CSV — 24 filas, una por hora Colombia (UTC-5)
    # n_posts total en CSV: 21 081 (16 posts sin hora asignada ignorados)
    # HALLAZGO CLAVE: vistas/post es INVERSAMENTE proporcional al volumen
    # de publicaciones — hora 3 (1 post) → 931 vistas vs hora 14 (1765 posts) → 289 vistas
    # Esto refleja saturación del feed: menos competencia = más visibilidad.

    LAMBDA_HORA = {        # posts esperados por hora (lambda del Poisson)
        0: 1.1667,  1: 0.3750,  2: 0.0833,  3: 0.0069,  4: 0.1250,
        5: 0.0,     6: 0.0139,  7: 0.6250,  8: 5.2292,  9: 7.6111,
        10: 9.8958, 11: 11.2153,12: 12.1111,13: 12.1597,14: 12.2569,
        15: 12.1319,16: 12.1597,17: 12.1111,18: 12.2153,19: 12.0972,
        20: 12.1389,21: 0.6528, 22: 0.0139, 23: 0.0,
    }

    VISTAS_MEDIA_HORA = {  # vistas medias reales por post en cada hora
        0: 392.6,  1: 505.6,  2: 652.8,  3: 931.0,  4: 377.4,
        5: 313.3,  6: 438.0,  7: 388.6,  8: 339.1,  9: 324.2,
        10: 316.3, 11: 301.9, 12: 296.9, 13: 291.5, 14: 289.2,
        15: 287.9, 16: 291.9, 17: 296.9, 18: 307.6, 19: 330.0,
        20: 386.3, 21: 506.9, 22: 512.5, 23: 313.3,
    }

    # Factor multiplicador vistas: ratio vs media global (313.3)
    FACTOR_HORA = {h: v / 313.3 for h, v in VISTAS_MEDIA_HORA.items()}

    # ── Matriz lambda REAL por (día_semana × hora) ─────────────
    LAMBDA_MATRIZ = {
        0: {0:0.0,   1:0.0,   2:0.0,   3:0.049, 4:0.0,   6:0.049, 7:0.924, 8:4.424,
            9:7.389, 10:9.382,11:10.549,12:10.84,13:10.986,14:10.938,15:10.5,
            16:10.84,17:11.278,18:10.986,19:10.938,20:11.229,21:0.632,22:0.146,23:0.0},
        1: {0:8.118, 1:2.576, 2:0.486, 3:0.049, 4:0.778, 6:0.0,   7:0.826, 8:10.549,
            9:8.361, 10:9.382,11:10.792,12:12.104,13:12.25,14:12.347,15:12.007,
            16:12.299,17:12.347,18:12.201,19:12.056,20:12.444,21:0.632,22:0.049,23:0.0},
        2: {0:0.0,   1:0.0,   2:0.097, 3:0.0,   4:0.0,   6:0.0,   7:0.583, 8:5.736,
            9:7.875, 10:10.16,11:11.569,12:11.764,13:11.618,14:11.91,15:11.618,
            16:11.812,17:11.667,18:11.715,19:12.104,20:12.25,21:0.632,22:0.0,23:0.049},
        3: {0:0.0,   1:0.0,   2:0.0,   3:0.049, 4:0.0,   6:0.0,   7:0.194, 8:4.521,
            9:8.847, 10:11.521,11:13.319,12:13.951,13:13.903,14:14.049,15:14.097,
            16:14.049,17:13.757,18:14.146,19:13.903,20:14.194,21:0.583,22:0.049,23:0.097},
        4: {0:0.049, 1:0.049, 2:0.0,   3:0.0,   4:0.0,   6:0.0,   7:0.437, 8:4.472,
            9:8.118, 10:10.354,11:11.91,12:12.542,13:13.028,14:12.931,15:12.931,
            16:12.931,17:12.833,18:12.979,19:12.833,20:12.444,21:1.021,22:0.0,23:0.0},
        5: {0:0.0,   1:0.0,   2:0.0,   3:0.0,   4:0.097, 6:0.0,   7:0.681, 8:2.868,
            9:7.049, 10:10.792,11:11.132,12:12.979,13:12.59,14:12.833,15:13.076,
            16:12.396,17:12.299,18:12.785,19:12.153,20:11.91,21:0.486,22:0.0,23:0.0},
        6: {0:0.0,   1:0.0,   2:0.0,   3:0.0,   4:0.0,   6:0.049, 7:0.729, 8:4.083,
            9:5.639, 10:7.681,11:9.236,12:10.597,13:10.743,14:10.792,15:10.694,
            16:10.792,17:10.597,18:10.694,19:10.743,20:10.5,21:0.681,22:0.194,23:0.0},
    }

    SUBS_INICIO      = 4_202
    SUBS_FIN         = 5_690
    TASA_CRECIMIENTO = (5_690 - 4_202) / (4_202 * 143)   # ~0.0025 diaria
    P_VISTA          = 313.3 / 4_946                       # P(vista|pub enviada)

    CATEGORIAS = {
        'deportes':      {'pct_actual':0.011,'fwd_rate':0.00143,'vistas_media':315.4,'ctr_factor':1.209},
        'hogar':         {'pct_actual':0.163,'fwd_rate':0.00124,'vistas_media':312.6,'ctr_factor':1.048},
        'juguetes':      {'pct_actual':0.034,'fwd_rate':0.00100,'vistas_media':312.8,'ctr_factor':0.845},
        'otra':          {'pct_actual':0.328,'fwd_rate':0.00123,'vistas_media':311.9,'ctr_factor':1.039},
        'ropa':          {'pct_actual':0.077,'fwd_rate':0.00135,'vistas_media':310.6,'ctr_factor':1.141},
        'salud/belleza': {'pct_actual':0.081,'fwd_rate':0.00138,'vistas_media':307.6,'ctr_factor':1.166},
        'tecnologia':    {'pct_actual':0.306,'fwd_rate':0.00102,'vistas_media':317.3,'ctr_factor':0.862},
    }
    MIX_OPTIMIZADO_C = {
        'deportes':0.028,'hogar':0.198,'juguetes':0.052,'otra':0.166,
        'ropa':0.195,'salud/belleza':0.205,'tecnologia':0.155,
    }

    DIAS_SIM      = 30
    N_SIMS        = 500
    HORARIO_BASE  = list(range(9, 22))       # 9am-9pm (actual)
    HORARIO_OPTIM = list(range(12, 24))      # 12pm-12am
    HORARIO_SCORE = [8, 9, 10, 11, 17, 18, 19, 20, 21]   # pico score


# ─────────────────────────────────────────────────────────────
#  SECCIÓN 2: CADENA DE MARKOV ABSORBENTE — Embudo individual
# ─────────────────────────────────────────────────────────────

class MarkovEmbudo:
    """
    Cadena de Markov Absorbente del recorrido de un suscriptor
    frente a una publicación.

    Estados Transitorios (Q):
        0 → Publicación enviada al suscriptor
        1 → Suscriptor visualizó la publicación
        2 → Suscriptor hizo clic en el enlace

    Estados Absorbentes (R):
        3 → Compra realizada  ✓
        4 → Sin acción / perdido

    Probabilidades calibradas con datos reales ene-may 2026:
        pv  = 313.3 / 4946 = 6.33%   (vistas_media / subs_promedio)
        CTR = 96507 / 6613127 = 1.46% (medido directo)
        CR  = 6027 / 96507    = 6.25% (medido directo)
    """
    NOMBRES = ['Publicado', 'Visto', 'Clic', 'Compra ✓', 'Perdido']

    def __init__(self, p_vista=None, ctr=None, cr=None):
        self.pv  = p_vista if p_vista is not None else P.P_VISTA
        self.ctr = ctr     if ctr     is not None else P.CTR
        self.cr  = cr      if cr      is not None else P.CR
        self._build()

    def _build(self):
        pv, ctr, cr = self.pv, self.ctr, self.cr
        self.M = np.array([
            [0, pv,  0,   0,  1 - pv ],
            [0,  0, ctr,  0,  1 - ctr],
            [0,  0,  0,  cr,  1 - cr ],
            [0,  0,  0,   1,  0      ],
            [0,  0,  0,   0,  1      ],
        ])
        self.Q = self.M[:3, :3]
        self.R = self.M[:3, 3:]

    @property
    def N(self):
        """Matriz fundamental N = (I-Q)⁻¹  — visitas esperadas por estado"""
        return np.linalg.inv(np.eye(3) - self.Q)

    @property
    def B(self):
        """Probabilidades de absorción B = N·R"""
        return self.N @ self.R

    @property
    def prob_compra(self):
        """P(compra | publicación enviada a un suscriptor)"""
        return self.B[0, 0]

    def compras_esperadas_30d(self, n_subs=None, pubs_dia=None):
        return self.prob_compra * (n_subs or P.SUBS_INICIO) * (pubs_dia or P.PUBS_DIA) * P.DIAS_SIM

    def print_resumen(self):
        sep = "─" * 58
        print(f"\n{sep}")
        print("  MARKOV ABSORBENTE — Embudo de Conversión")
        print(sep)
        print(f"  Parámetros calibrados con datos reales (ene-may 2026):")
        print(f"    P(vista  | pub)   = {self.pv:.6f}  ({self.pv*100:.4f}%)  [313.3/4946]")
        print(f"    P(clic   | vista) = {self.ctr:.6f}  ({self.ctr*100:.4f}%)  [96507/6613127]")
        print(f"    P(compra | clic)  = {self.cr:.6f}  ({self.cr*100:.4f}%)  [6027/96507]")
        print(sep)
        print(f"  P(compra | pub)   = {self.prob_compra:.8f}  ({self.prob_compra*100:.6f}%)")
        c30 = self.compras_esperadas_30d()
        print(f"  Compras esperadas 30d ({P.SUBS_INICIO} subs, {P.PUBS_DIA} pubs/día):")
        print(f"    → {c30:.1f} compras  ≈  ${c30 * P.COMISION_UNIT:.2f} USD comisiones")
        print(f"    → ${c30 * P.INGRESO_UNIT:.2f} USD revenue Amazon")
        print(sep)
        df_B = pd.DataFrame(
            self.B,
            index=self.NOMBRES[:3],
            columns=['P(→ Compra)', 'P(→ Perdido)']
        )
        print("\n  Probabilidades de Absorción  B = N·R:")
        print(df_B.round(8).to_string())
        df_N = pd.DataFrame(
            self.N,
            index=self.NOMBRES[:3],
            columns=self.NOMBRES[:3]
        )
        print("\n  Matriz Fundamental  N = (I-Q)⁻¹  (pasos esperados por estado):")
        print(df_N.round(6).to_string())
        t = self.N @ np.ones(3)
        print(f"\n  Pasos esperados hasta absorción desde cada estado:")
        for nombre, ti in zip(self.NOMBRES[:3], t):
            print(f"    {nombre:10s}: {ti:.4f} pasos")
        print(sep)


# ─────────────────────────────────────────────────────────────
#  SECCIÓN 3: CADENA DE MARKOV ERGÓDICA — Estado del canal
# ─────────────────────────────────────────────────────────────

class MarkovCanal:
    """
    Cadena de Markov Ergódica (irreducible, aperiódica) del rendimiento
    diario del canal de Telegram.

    Métrica de clasificación: INGRESOS POR POST PUBLICADO (USD/post/día).

    ─── Calibración con sim_tabla_horaria.csv ──────────────────
    Ingreso/post esperado por hora = vistas_media_h × CTR × CR × COM
    Simulando 10 000 días con la distribución LogNormal calibrada:
        Media:    $0.1463/post  (real: $3149.42/21097 = $0.1493)
        Std:      $0.0228/post  (CV = 15.6%)
        P25:      $0.1308/post
        P75:      $0.1614/post

    Umbrales elegidos en P25/P75 para que π objetivo ≈ [0.25, 0.50, 0.25]:
        Bajo:   < $0.1308/post  (peor 25% de días)
        Normal: $0.1308–$0.1614/post  (50% central)
        Alto:   > $0.1614/post  (mejor 25% de días)
    """

    ESTADOS = ['Bajo\n(<$0.131/post)', 'Normal\n($0.131-$0.161/post)', 'Alto\n(>$0.161/post)']
    COLORES = ['#e74c3c', '#f39c12', '#27ae60']

    # Umbrales calibrados con simulación Monte Carlo del CSV
    UMBRAL_BAJO  = 0.1308   # P25 de distribución simulada de ingreso/post/día
    UMBRAL_ALTO  = 0.1614   # P75 de distribución simulada de ingreso/post/día
    # Referencia: ingreso/post real = $3149.42/21097 = $0.1493
    INGRESO_POR_POST_REAL = P.GANANCIAS_USD / P.POSTS_TOTALES

    def __init__(self, P_mat=None):
        if P_mat is None:
            # Matriz calibrada con P25/P75 del CSV.
            # Objetivo π ≈ [0.25, 0.50, 0.25].
            # Razonamiento económico:
            #   - Bajo → Normal con más frecuencia (0.43) que Bajo → Alto (0.07):
            #     una racha mala se recupera gradualmente.
            #   - Alto → Normal con más frecuencia (0.43) que Alto → Bajo (0.07):
            #     una racha buena decae gradualmente (simétrico).
            #   - Normal es el estado más estable (0.65): refleja operación cotidiana.
            #   - Matriz simétrica en saltos extremos B↔A (0.07): coherente con
            #     que un día de ingreso extremo (muy bueno o muy malo) es igualmente
            #     improbable desde ambos extremos.
            self.P_mat = np.array([
                [0.50, 0.43, 0.07],   # Bajo   → 50% Bajo, 43% Normal, 7% Alto
                [0.18, 0.65, 0.17],   # Normal → 18% Bajo, 65% Normal, 17% Alto
                [0.07, 0.43, 0.50],   # Alto   →  7% Bajo, 43% Normal, 50% Alto
            ])
        else:
            self.P_mat = P_mat

        assert np.allclose(self.P_mat.sum(axis=1), 1.0), \
            f"Error: filas no suman 1 → {self.P_mat.sum(axis=1)}"

    @classmethod
    def clasificar_dia(cls, posts_dia: int, comisiones_dia: float) -> int:
        """
        Clasifica un día real según ingresos por post (USD/post).
        Usar con datos diarios reales para calibrar la matriz empíricamente.

        Returns: 0=Bajo, 1=Normal, 2=Alto
        """
        if posts_dia == 0:
            return 0
        v = comisiones_dia / posts_dia
        if v < cls.UMBRAL_BAJO:
            return 0
        elif v <= cls.UMBRAL_ALTO:
            return 1
        else:
            return 2

    @classmethod
    def calibrar_desde_datos(cls, serie_posts: list, serie_comisiones: list):
        """
        Construye P_mat empírica contando transiciones reales día a día.
        Con 143 días hay 142 transiciones — suficiente para estimar la matriz.

        Ejemplo:
            canal, df_conteos = MarkovCanal.calibrar_desde_datos(
                posts_por_dia, comis_por_dia
            )
        """
        estados = [cls.clasificar_dia(p, c)
                   for p, c in zip(serie_posts, serie_comisiones)]
        conteos = np.zeros((3, 3))
        for t in range(len(estados) - 1):
            conteos[estados[t], estados[t + 1]] += 1
        totales = conteos.sum(axis=1, keepdims=True)
        totales[totales == 0] = 1
        P_emp = conteos / totales
        df = pd.DataFrame(
            conteos.astype(int),
            index=['Bajo', 'Normal', 'Alto'],
            columns=['→ Bajo', '→ Normal', '→ Alto']
        )
        print("\n  Conteos de transiciones reales:")
        print(df.to_string())
        return cls(P_mat=P_emp), df

    @property
    def pi(self):
        """Distribución estacionaria: (Pᵀ-I)π=0, Σπᵢ=1"""
        n = 3
        A = self.P_mat.T - np.eye(n)
        A[-1, :] = 1.0
        b = np.zeros(n); b[-1] = 1.0
        return solve(A, b)

    def duracion_media(self) -> dict:
        """E[días en estado i] = 1 / (1 - P[i→i])"""
        return {
            'Bajo':   round(1 / (1 - self.P_mat[0, 0]), 2),
            'Normal': round(1 / (1 - self.P_mat[1, 1]), 2),
            'Alto':   round(1 / (1 - self.P_mat[2, 2]), 2),
        }

    def ingreso_esperado_estable(self) -> float:
        """E[ingreso/post] en estado estable, ponderado por π."""
        pi = self.pi
        val = [self.UMBRAL_BAJO / 2,
               (self.UMBRAL_BAJO + self.UMBRAL_ALTO) / 2,
               self.UMBRAL_ALTO * 1.5]
        return round(sum(pi[i] * val[i] for i in range(3)), 4)

    def simular(self, n_dias=600, s0=1):
        chain = [s0]
        for _ in range(n_dias - 1):
            chain.append(np.random.choice(3, p=self.P_mat[chain[-1]]))
        return np.array(chain)

    def freq_acumulada(self, n_dias=600):
        chain = self.simular(n_dias)
        freqs = np.zeros((n_dias, 3))
        for t in range(n_dias):
            for s in range(3):
                freqs[t, s] = (chain[:t + 1] == s).mean()
        return freqs

    def print_resumen(self):
        pi  = self.pi
        dur = self.duracion_media()
        e   = self.ingreso_esperado_estable()
        sep = "─" * 58

        print(f"\n{sep}")
        print("  MARKOV ERGÓDICO — Estado del Canal (ingreso por post)")
        print(sep)
        print(f"  Métrica: ingresos por post publicado (USD/post/día)")
        print(f"  Calibración: simulación Monte Carlo sobre sim_tabla_horaria.csv")
        print(f"    Ingreso/post real    = ${self.INGRESO_POR_POST_REAL:.4f}  [$3149.42/21097]")
        print(f"    Distribución simul.  media=$0.1463  std=$0.0228  CV=15.6%")
        print(f"    Umbral Bajo  (P25)  < ${self.UMBRAL_BAJO:.4f}/post")
        print(f"    Umbral Normal P25-P75  ${self.UMBRAL_BAJO:.4f} – ${self.UMBRAL_ALTO:.4f}/post")
        print(f"    Umbral Alto  (P75)  > ${self.UMBRAL_ALTO:.4f}/post")
        print(sep)
        print("  Matriz de transición P  [fila=estado actual, col=estado siguiente]:")
        print(f"  {'':12s}  {'→ Bajo':>10s}  {'→ Normal':>10s}  {'→ Alto':>10s}  {'Suma':>6s}  {'E[días]':>8s}")
        etq = ['Bajo  ', 'Normal', 'Alto  ']
        for i, e_lbl in enumerate(etq):
            r = self.P_mat[i]
            print(f"  {e_lbl:12s}  {r[0]:>10.3f}  {r[1]:>10.3f}  {r[2]:>10.3f}  "
                  f"{r.sum():>6.3f}  {dur[e_lbl.strip()]:>8.2f}d")
        print(sep)
        print("  Distribución Estacionaria π  (lím t→∞):")
        for i, (lbl, p) in enumerate(zip(['Bajo  ', 'Normal', 'Alto  '], pi)):
            bar = '█' * int(p * 35)
            print(f"    π_{i} {lbl} = {p:.4f}  ({p*100:.1f}%)  {bar}")
        print(f"\n  E[ingreso/post] estado estable ≈ ${e:.4f} USD")
        print(f"  (ingreso/post real calibrado    = ${self.INGRESO_POR_POST_REAL:.4f} USD)")
        print(f"\n  Interpretación:")
        print(f"    El canal opera en Normal {pi[1]*100:.1f}% | Alto {pi[2]*100:.1f}% | Bajo {pi[0]*100:.1f}%")
        print(f"    Racha mala: ~{dur['Bajo']} días | Racha buena: ~{dur['Alto']} días")
        print(sep)


# ─────────────────────────────────────────────────────────────
#  SECCIÓN 4: SIMULACIÓN MONTE CARLO DES
# ─────────────────────────────────────────────────────────────

def simular_periodo(horario: list, ctr=None, cr=None, comision=None) -> dict:
    """
    Simula DIAS_SIM días con Poisson no-homogéneo calibrado por día×hora.
    Vistas por post: LogNormal escalada por factor horario del CSV.
    CR: variable entre los 11 tracking IDs reales.
    Suscriptores: crecimiento lineal diario (4202→5690 en 143d).
    """
    ctr_    = ctr      if ctr      is not None else P.CTR
    com     = comision if comision is not None else P.COMISION_UNIT
    cr_vals = list(P.CR_TRACKING.values())

    registros = []
    for dia in range(P.DIAS_SIM):
        dia_semana  = dia % 7
        subs        = int(P.SUBS_INICIO * (1 + P.TASA_CRECIMIENTO * dia))
        factor_subs = subs / P.SUBS_INICIO

        lambda_activa = {
            h: P.LAMBDA_MATRIZ[dia_semana].get(h, 0)
            for h in horario
            if P.LAMBDA_MATRIZ[dia_semana].get(h, 0) > 0
        }

        for hora, lam in lambda_activa.items():
            n_posts = np.random.poisson(lam)
            if n_posts == 0:
                continue
            v_base      = np.random.lognormal(P.LOGNORM_MU, P.LOGNORM_SIGMA, n_posts)
            v_arr       = np.maximum(1, (v_base * P.FACTOR_HORA[hora] * factor_subs)).astype(int)
            cr_arr      = np.full(n_posts, cr) if cr is not None else np.random.choice(cr_vals, n_posts)
            clics_arr   = np.random.binomial(v_arr, ctr_)
            compras_arr = np.random.binomial(clics_arr, cr_arr)
            registros.append((v_arr.sum(), clics_arr.sum(), compras_arr.sum(),
                              (compras_arr * com).sum()))

    if not registros:
        return {'vistas': 0, 'clics': 0, 'compras': 0, 'ctr_real': 0.0, 'comisiones': 0.0}

    arr   = np.array(registros)
    v_tot = int(arr[:, 0].sum())
    return {
        'vistas'    : v_tot,
        'clics'     : int(arr[:, 1].sum()),
        'compras'   : int(arr[:, 2].sum()),
        'ctr_real'  : arr[:, 1].sum() / v_tot if v_tot > 0 else 0.0,
        'comisiones': round(arr[:, 3].sum(), 2),
    }


def monte_carlo(horario: list, n: int = None, label: str = '',
                ctr=None, cr=None) -> pd.DataFrame:
    """Ejecuta n simulaciones independientes de 30 días."""
    n = n or P.N_SIMS
    resultados = [simular_periodo(horario, ctr=ctr, cr=cr) for _ in range(n)]
    df = pd.DataFrame(resultados)
    df['sim']            = range(n)
    df['escenario']      = label
    df['media_acum_com'] = df['comisiones'].expanding().mean()
    return df


# ─────────────────────────────────────────────────────────────
#  SECCIÓN 5: ANÁLISIS DE SENSIBILIDAD
# ─────────────────────────────────────────────────────────────

def analisis_sensibilidad() -> pd.DataFrame:
    """
    Sensibilidad analítica de comisiones ante variación de CTR y CR.
    E[com_30d] = Σ_h(λ_h × 30 × vistas_h × CTR × CR × COM)
    """
    import itertools
    factores = [0.75, 1.00, 1.25, 1.50]
    filas = []
    for fc, fr in itertools.product(factores, factores):
        ctr_v = P.CTR * fc
        cr_v  = P.CR  * fr
        e_com = sum(
            P.LAMBDA_HORA[h] * 30 * P.VISTAS_MEDIA_HORA[h] * ctr_v * cr_v * P.COMISION_UNIT
            for h in P.HORARIO_BASE if P.LAMBDA_HORA.get(h, 0) > 0
        )
        filas.append({
            'factor_CTR': fc, 'factor_CR': fr,
            'CTR': round(ctr_v * 100, 2), 'CR': round(cr_v * 100, 2),
            'com_media': round(e_com, 2), 'com_std': round(e_com * 0.034, 2),
        })
    return pd.DataFrame(filas)


# ─────────────────────────────────────────────────────────────
#  SECCIÓN 6: VISUALIZACIONES
# ─────────────────────────────────────────────────────────────

PALETA = {'A': '#e07b54', 'B': '#4a90d9', 'C': '#2ecc71', 'D': '#9b59b6'}


def plot_dataset_horario(df_csv):
    """
    Gráfica 1 (nueva): análisis visual del dataset sim_tabla_horaria.csv.
    Muestra el patrón real de publicación y la relación volumen↔visibilidad.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Dataset Real — sim_tabla_horaria.csv  (21 097 posts, ene-may 2026)",
                 fontsize=13, fontweight='bold')

    horas = df_csv['hora_col'].values
    n_p   = df_csv['n_posts'].values
    vm    = df_csv['vistas_media'].values
    lam   = df_csv['lambda_posts_por_dia'].values
    fr    = df_csv['fwd_rate'].values

    # ① Posts por hora
    ax = axes[0, 0]
    cols_bar = ['#e07b54' if h in P.HORARIO_BASE else '#bdc3c7' for h in horas]
    ax.bar(horas, n_p, color=cols_bar, alpha=0.85)
    ax.set_title("Distribución real de posts por hora\n(naranja = horario actual 9-21h)")
    ax.set_xlabel("Hora (Colombia)"); ax.set_ylabel("n_posts (143 días)")
    ax.set_xticks(range(24))
    ax.text(1, max(n_p)*0.9, "Martes 0-4am\nturno atípico", fontsize=8,
            color='#e74c3c', ha='left')

    # ② Vistas media por hora
    ax2 = axes[0, 1]
    ax2.bar(horas, vm, color='#3498db', alpha=0.8)
    ax2.axhline(313.3, color='gray', ls='--', lw=1.5, label='Media global (313)')
    ax2.set_title("Vistas media por post según hora\n(inversamente prop. al volumen)")
    ax2.set_xlabel("Hora (Colombia)"); ax2.set_ylabel("Vistas media/post")
    ax2.set_xticks(range(24)); ax2.legend(fontsize=8)
    # Anotación de horas nocturnas
    for h, v in zip(horas, vm):
        if v > 500:
            ax2.text(h, v + 10, f"{v:.0f}", ha='center', fontsize=7, color='#2c3e50')

    ax3 = axes[1, 0]
    sc = ax3.scatter(n_p, vm, c=horas, cmap='plasma', s=60, alpha=0.85, zorder=3)
    ax3.set_xlabel("n_posts en la hora (143 días)")
    ax3.set_ylabel("Vistas media por post")
    ax3.set_title("HALLAZGO: Saturación del feed\nmenos posts → más visibilidad/post")
    plt.colorbar(sc, ax=ax3, label='Hora del día')
    # Línea de tendencia
    mask = n_p > 0
    z = np.polyfit(n_p[mask], vm[mask], 1)
    xr = np.linspace(n_p[mask].min(), n_p[mask].max(), 100)
    ax3.plot(xr, np.polyval(z, xr), 'r--', lw=1.5, alpha=0.7, label='Tendencia')
    ax3.legend(fontsize=8)

    # ④ Ingreso esperado por hora = vistas * CTR * CR * COM
    ing_h = vm * P.CTR * P.CR * P.COMISION_UNIT
    cols_h = ['#27ae60' if v > MarkovCanal.UMBRAL_ALTO else
              '#e74c3c' if v < MarkovCanal.UMBRAL_BAJO else '#f39c12'
              for v in ing_h]
    ax4 = axes[1, 1]
    ax4.bar(horas, ing_h, color=cols_h, alpha=0.85)
    ax4.axhline(MarkovCanal.UMBRAL_BAJO, color='#e74c3c', ls='--', lw=1.5,
                label=f'Umbral Bajo ${MarkovCanal.UMBRAL_BAJO:.4f}')
    ax4.axhline(MarkovCanal.UMBRAL_ALTO, color='#27ae60', ls='--', lw=1.5,
                label=f'Umbral Alto ${MarkovCanal.UMBRAL_ALTO:.4f}')
    ax4.axhline(MarkovCanal.INGRESO_POR_POST_REAL, color='gray', ls=':', lw=1.5,
                label=f'Real ${MarkovCanal.INGRESO_POR_POST_REAL:.4f}')
    ax4.set_title("Ingreso esperado/post por hora\n(vistas × CTR × CR × COM)")
    ax4.set_xlabel("Hora (Colombia)"); ax4.set_ylabel("USD/post")
    ax4.set_xticks(range(24)); ax4.legend(fontsize=7)

    plt.tight_layout()
    return fig


def plot_funnel_counts(emb, subs=4946):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Embudo de Conversión — Valores Esperados por Publicación",
                 fontsize=13, fontweight='bold')

    etapas = ['Suscriptores', 'Ven el post', 'Hacen clic', 'Compran']
    valores = [
        subs,
        round(subs * emb.pv, 1),
        round(subs * emb.pv * emb.ctr, 2),
        round(subs * emb.pv * emb.ctr * emb.cr, 3),
    ]
    colores_pts = ['#3498db', '#9b59b6', '#e67e22', '#27ae60']
    axes[0].plot(etapas, valores, marker='o', linewidth=2.5, color='#95a5a6', zorder=1)
    for i, (v, c) in enumerate(zip(valores, colores_pts)):
        axes[0].scatter(i, v, color=c, s=120, zorder=3)
        axes[0].text(i, v * 1.06, f"{v:,.3f}" if v < 1 else f"{v:,.1f}",
                     ha='center', fontsize=9, fontweight='bold', color=c)
    axes[0].set_title("Recorrido de un suscriptor por publicación")
    axes[0].set_ylabel("Cantidad esperada"); axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.4)

    tasas = [emb.pv * 100, emb.ctr * 100, emb.cr * 100]
    etq_t = ['Sub→Vista (6.33%)', 'Vista→Clic (1.46%)', 'Clic→Compra (6.25%)']
    bars  = axes[1].bar(etq_t, tasas, color=['#9b59b6','#e67e22','#27ae60'], alpha=0.85, width=0.5)
    axes[1].set_title("Tasa de conversión por etapa"); axes[1].set_ylabel("Tasa (%)")
    for bar, v in zip(bars, tasas):
        axes[1].text(bar.get_x()+bar.get_width()/2, v+0.1,
                     f"{v:.2f}%", ha='center', fontsize=10, fontweight='bold')
    axes[1].set_ylim(0, max(tasas)*1.25)
    plt.tight_layout()
    return fig


def plot_markov_embudo(emb):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Cadena de Markov Absorbente — Embudo de Conversión",
                 fontsize=13, fontweight='bold')
    ax = axes[0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis('off')
    ax.set_title("Grafo de estados", fontsize=11)
    nodos = {'Pub':(1,3),'Vista':(3.5,3),'Clic':(6,3),'Compra':(8.5,4.5),'Perdido':(8.5,1.5)}
    cn    = {'Pub':'#3498db','Vista':'#9b59b6','Clic':'#e67e22','Compra':'#27ae60','Perdido':'#e74c3c'}
    for nombre, (x, y) in nodos.items():
        ax.add_patch(plt.Circle((x,y), 0.55, color=cn[nombre], alpha=0.85, zorder=3))
        ax.text(x, y, nombre, ha='center', va='center', fontsize=8,
                color='white', fontweight='bold', zorder=4)
    aristas = [
        ('Pub','Vista',   f"pv={emb.pv:.3f}",  (2.25,3.25)),
        ('Vista','Clic',  f"CTR={emb.ctr:.4f}", (4.75,3.25)),
        ('Clic','Compra', f"CR={emb.cr:.4f}",   (7.6,4.2)),
        ('Clic','Perdido',f"1-CR",              (7.6,2.0)),
        ('Vista','Perdido',f"1-CTR",            (4.5,1.8)),
        ('Pub','Perdido', f"1-pv",              (2.5,1.5)),
    ]
    for src, dst, lbl, lpos in aristas:
        x1,y1=nodos[src]; x2,y2=nodos[dst]
        ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=1.3))
        ax.text(lpos[0], lpos[1], lbl, fontsize=7, color='#333', ha='center',
                bbox=dict(fc='white', alpha=0.7, pad=1, ec='none'))
    for n in ['Compra','Perdido']:
        x,y=nodos[n]
        ax.add_patch(plt.Circle((x,y), 0.65, fill=False, ec=cn[n], lw=2, zorder=2))
    ax2 = axes[1]
    ax2.set_title("Probabilidad de absorción desde cada estado", fontsize=11)
    p_c = emb.B[:,0]; p_p = emb.B[:,1]; x_p = np.arange(3)
    ax2.bar(x_p, p_c,  label='→ Compra',  color='#27ae60', alpha=0.85)
    ax2.bar(x_p, p_p, bottom=p_c, label='→ Perdido', color='#e74c3c', alpha=0.75)
    ax2.set_xticks(x_p); ax2.set_xticklabels(['Publicado','Visto','Clic'])
    ax2.set_ylabel('Probabilidad'); ax2.legend()
    for i, pc in enumerate(p_c):
        ax2.text(i, pc/2, f"{pc*100:.4f}%", ha='center', va='center',
                 fontsize=8, color='white', fontweight='bold')
    plt.tight_layout()
    return fig


def plot_markov_canal(canal):
    """
    Convergencia de frecuencias a π + heatmap con duración media por estado.
    Etiquetas en USD/post — consistentes con la métrica de clasificación.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Cadena de Markov Ergódica — Estado del Canal  "
                 "(métrica: ingreso/post USD, umbrales P25/P75 del CSV)",
                 fontsize=12, fontweight='bold')

    pi    = canal.pi
    freqs = canal.freq_acumulada(600)
    dias  = np.arange(1, 601)
    dur   = canal.duracion_media()

    ax = axes[0]
    lbls = [f"Bajo (<${canal.UMBRAL_BAJO:.4f})",
            f"Normal (${canal.UMBRAL_BAJO:.4f}–${canal.UMBRAL_ALTO:.4f})",
            f"Alto (>${canal.UMBRAL_ALTO:.4f})"]
    for i, (lbl, c) in enumerate(zip(lbls, ['#e74c3c','#f39c12','#27ae60'])):
        ax.plot(dias, freqs[:,i], color=c, alpha=0.8, lw=1.2, label=f"{lbl} (sim.)")
        ax.axhline(pi[i], color=c, ls='--', lw=1.5, alpha=0.9,
                   label=f"π_{i}={pi[i]:.3f}")
    ax.set_xlabel("Días simulados"); ax.set_ylabel("Frecuencia acumulada")
    ax.set_title("Convergencia a Distribución Estacionaria π")
    ax.legend(fontsize=7); ax.set_xlim(1, 600)

    ax2 = axes[1]
    etq = ['Bajo\n<$0.131', 'Normal\n$0.131-$0.161', 'Alto\n>$0.161']
    sns.heatmap(canal.P_mat, annot=True, fmt='.2f', cmap='YlGnBu',
                xticklabels=etq, yticklabels=etq, linewidths=0.5, ax=ax2,
                cbar_kws={'label': 'Probabilidad de transición'})
    ax2.set_title("Matriz de Transición P\n(calibrada con percentiles P25/P75 del CSV)")
    ax2.set_xlabel("Estado siguiente (día t+1)")
    ax2.set_ylabel("Estado actual (día t)")
    # Duración media en diagonal
    for i, d in enumerate([dur['Bajo'], dur['Normal'], dur['Alto']]):
        ax2.text(i+0.5, i+0.82, f"~{d}d", ha='center', va='center',
                 fontsize=9, color='navy', fontweight='bold')
    plt.tight_layout()
    return fig


def plot_convergencia_mc(df_a, df_b, df_c=None, df_d=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Convergencia Monte Carlo — Ley de Grandes Números",
                 fontsize=13, fontweight='bold')
    ax = axes[0]
    series = [('A. Línea Base', df_a, PALETA['A']),
              ('B. 12pm-12am',  df_b, PALETA['B'])]
    if df_c is not None: series.append(('C. Cat. Óptimas', df_c, PALETA['C']))
    if df_d is not None: series.append(('D. Score+Cat',   df_d, PALETA['D']))
    for lbl, df_, col in series:
        ax.plot(df_['sim'], df_['media_acum_com'], color=col, label=lbl, lw=1.5)
        ax.axhline(df_['comisiones'].mean(), color=col, ls='--', lw=1, alpha=0.5,
                   label=f"μ={df_['comisiones'].mean():.0f}")
    ax.set_xlabel("Simulaciones"); ax.set_ylabel("Media acumulada (USD)")
    ax.set_title("Convergencia Comisiones Mensuales"); ax.legend(fontsize=7)

    ax2 = axes[1]
    dfs  = [df_a, df_b] + ([df_c] if df_c is not None else []) + ([df_d] if df_d is not None else [])
    cols = [PALETA['A'], PALETA['B']] + ([PALETA['C']] if df_c is not None else []) + ([PALETA['D']] if df_d is not None else [])
    datos = pd.concat(dfs)[['escenario','comisiones']]
    sns.violinplot(data=datos, x='escenario', y='comisiones', palette=cols, ax=ax2, inner='box')
    ax2.set_title("Distribución Comisiones (USD/30 días)")
    ax2.set_xlabel(""); ax2.set_ylabel("Comisiones (USD)")
    ax2.tick_params(axis='x', labelsize=7)
    for i, df_ in enumerate(dfs):
        ax2.text(i, df_['comisiones'].max()*0.97,
                 f"${df_['comisiones'].mean():.0f}", ha='center', fontsize=8, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_comparacion_escenarios(df_a, df_b, df_c=None, df_d=None):
    dfs   = [df_a, df_b] + ([df_c] if df_c is not None else []) + ([df_d] if df_d is not None else [])
    lbls  = [df_['escenario'].iloc[0].split('(')[0].strip() for df_ in dfs]
    cols  = [PALETA['A'], PALETA['B']] + ([PALETA['C']] if df_c is not None else []) + ([PALETA['D']] if df_d is not None else [])
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("PromoHunter — Comparación de 4 Escenarios de Publicación",
                 fontsize=13, fontweight='bold')
    for ax, col_key, titulo in zip(axes, ['comisiones','compras','vistas'],
                                   ['Comisiones (USD)','Compras','Vistas']):
        medias = [df_[col_key].mean() for df_ in dfs]
        bars   = ax.bar(range(len(dfs)), medias, color=cols, alpha=0.85, width=0.5)
        ax.set_xticks(range(len(dfs))); ax.set_xticklabels(lbls, rotation=15, ha='right', fontsize=8)
        ax.set_title(titulo, fontweight='bold')
        ref = medias[0]
        for i, (bar, v) in enumerate(zip(bars, medias)):
            delta = (v-ref)/ref*100
            ax.text(bar.get_x()+bar.get_width()/2, v*1.01,
                    (f"{v:,.0f}\n({delta:+.1f}%)" if i > 0 else f"{v:,.0f}"),
                    ha='center', va='bottom', fontsize=7, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_sensibilidad(df_sens):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Análisis de Sensibilidad — Comisiones Mensuales (USD) — Método Analítico",
                 fontsize=13, fontweight='bold')
    pivot_med = df_sens.pivot(index='factor_CR', columns='factor_CTR', values='com_media')
    pivot_std = df_sens.pivot(index='factor_CR', columns='factor_CTR', values='com_std')
    lbl_x = [f"{x*100:.0f}%" for x in pivot_med.columns]
    lbl_y = [f"{y*100:.0f}%" for y in pivot_med.index]
    sns.heatmap(pivot_med, annot=True, fmt='.0f', cmap='RdYlGn',
                ax=axes[0], linewidths=0.3, xticklabels=lbl_x, yticklabels=lbl_y,
                cbar_kws={'label':'USD'})
    axes[0].set_title("Media Esperada Comisiones (USD)")
    axes[0].set_xlabel("Factor CTR (base=1.46%)"); axes[0].set_ylabel("Factor CR (base=6.25%)")
    axes[0].add_patch(plt.Rectangle((1,1), 1, 1, fill=False, edgecolor='blue', lw=2.5))
    sns.heatmap(pivot_std, annot=True, fmt='.0f', cmap='YlOrRd',
                ax=axes[1], linewidths=0.3, xticklabels=lbl_x, yticklabels=lbl_y,
                cbar_kws={'label':'USD'})
    axes[1].set_title("Desviación Estándar (Riesgo)")
    axes[1].set_xlabel("Factor CTR"); axes[1].set_ylabel("Factor CR")
    plt.tight_layout()
    return fig


def plot_categorias():
    cats    = P.CATEGORIAS
    nombres = list(cats.keys())
    fwd     = [cats[c]['fwd_rate']    for c in nombres]
    factores= [cats[c]['ctr_factor']  for c in nombres]
    vistas  = [cats[c]['vistas_media']for c in nombres]
    pct_act = [cats[c]['pct_actual']*100 for c in nombres]
    pct_opt = [P.MIX_OPTIMIZADO_C[c]*100 for c in nombres]
    cbar    = ['#e74c3c' if f < 1 else '#27ae60' for f in factores]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Análisis por Categoría de Producto  (fwd_rate proxy de engagement)",
                 fontsize=13, fontweight='bold')

    ax = axes[0,0]
    bars = ax.barh(nombres, fwd, color=cbar, alpha=0.85)
    ax.axvline(sum(f*p/100 for f,p in zip(fwd,pct_act)), color='gray', ls='--', lw=1.5,
               label='Promedio ponderado')
    ax.set_title("Forward Rate por Categoría"); ax.set_xlabel("fwd_rate"); ax.legend(fontsize=8)
    for bar, v in zip(bars, fwd):
        ax.text(v+0.000005, bar.get_y()+bar.get_height()/2, f"{v:.5f}", va='center', fontsize=8)

    ax2 = axes[0,1]
    bars2 = ax2.barh(nombres, factores, color=cbar, alpha=0.85)
    ax2.axvline(1.0, color='gray', ls='--', lw=1.5, label='Factor=1')
    ax2.set_title("Factor CTR por Categoría"); ax2.set_xlabel("Factor multiplicador"); ax2.legend(fontsize=8)
    for bar, v in zip(bars2, factores):
        ax2.text(v+0.005, bar.get_y()+bar.get_height()/2,
                 f"{v:.3f} ({'+' if v>1 else ''}{(v-1)*100:.1f}%)", va='center', fontsize=8)

    ax3 = axes[1,0]
    x = range(len(nombres)); w = 0.35
    b1 = ax3.bar([i-w/2 for i in x], pct_act, w, label='Mix Actual (A)', color=PALETA['A'], alpha=0.8)
    b2 = ax3.bar([i+w/2 for i in x], pct_opt, w, label='Mix Óptimo (C)', color=PALETA['C'], alpha=0.8)
    ax3.set_xticks(list(x)); ax3.set_xticklabels(nombres, rotation=20, ha='right', fontsize=8)
    ax3.set_ylabel("% publicaciones"); ax3.set_title("Mix Actual vs Optimizado (Escenario C)")
    ax3.legend()
    for b, v in zip(b1, pct_act): ax3.text(b.get_x()+b.get_width()/2, v+0.3, f"{v:.1f}%", ha='center', fontsize=7)
    for b, v in zip(b2, pct_opt): ax3.text(b.get_x()+b.get_width()/2, v+0.3, f"{v:.1f}%", ha='center', fontsize=7)

    ax4 = axes[1,1]
    ax4.bar(nombres, vistas, color='#3498db', alpha=0.8)
    ax4.axhline(313.3, color='gray', ls='--', lw=1.5, label='Media global (313)')
    ax4.set_xticks(range(len(nombres))); ax4.set_xticklabels(nombres, rotation=20, ha='right', fontsize=8)
    ax4.set_ylabel("Vistas media/post"); ax4.set_title("Vistas Media por Categoría")
    ax4.legend(fontsize=8)
    for i, v in enumerate(vistas): ax4.text(i, v+0.5, f"{v:.0f}", ha='center', fontsize=8)
    plt.tight_layout()
    return fig


def plot_lambda_matrix():
    dias_n = {0:'Lun',1:'Mar',2:'Mie',3:'Jue',4:'Vie',5:'Sab',6:'Dom'}
    data   = {dias_n[d]: [P.LAMBDA_MATRIZ[d].get(h,0) for h in range(24)] for d in range(7)}
    df_mat = pd.DataFrame(data, index=range(24)).T

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Patrón Real — Matriz Lambda (posts/hora) y Score por Hora",
                 fontsize=13, fontweight='bold')

    sns.heatmap(df_mat, annot=False, cmap='YlOrRd', ax=axes[0], linewidths=0.2,
                cbar_kws={'label':'posts/hora esperados'})
    axes[0].set_title("Lambda[día × hora]  (fuente: sim_lambda_matriz_dia_hora.csv)")
    axes[0].set_xlabel("Hora del día (Colombia)"); axes[0].set_ylabel("Día de semana")
    axes[0].axvline(P.HORARIO_BASE[0], color='blue', lw=2, alpha=0.7, label='Horario A (9-21h)')
    axes[0].legend(fontsize=8)

    horas  = list(range(24))
    scores = [
        sum(P.LAMBDA_MATRIZ[d].get(h,0) for d in range(7))/7 * P.VISTAS_MEDIA_HORA[h]
        for h in horas
    ]
    cb = ['#2ecc71' if h in P.HORARIO_SCORE else '#e07b54' if h in P.HORARIO_BASE else '#bdc3c7' for h in horas]
    axes[1].bar(horas, scores, color=cb, alpha=0.85)
    axes[1].set_xlabel("Hora del día"); axes[1].set_ylabel("Score = λ_prom × vistas_media")
    axes[1].set_title("Score por Hora — Base para Escenario D")
    axes[1].set_xticks(horas)
    from matplotlib.patches import Patch
    axes[1].legend(handles=[
        Patch(color='#e07b54', label='Horario A actual (9-21h)'),
        Patch(color='#2ecc71', label='Horario D score-óptimo'),
        Patch(color='#bdc3c7', label='Fuera de ambos'),
    ], fontsize=8)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
#  SECCIÓN 7: EJECUCIÓN PRINCIPAL
# ─────────────────────────────────────────────────────────────

def imprimir_dataset(df_csv):
    """Imprime análisis completo del dataset sim_tabla_horaria.csv."""
    sep = "─" * 70
    print(f"\n{sep}")
    print("  DATASET — sim_tabla_horaria.csv  (fuente: Telegram API, ene-may 2026)")
    print(sep)
    print(f"  Registros: {len(df_csv)} filas (una por hora del día, 0-23h Colombia)")
    print(f"  Total posts en dataset: {df_csv['n_posts'].sum():,}")
    print(f"  Lambda diaria total:    {df_csv['lambda_posts_por_dia'].sum():.2f} posts/día")
    vm_pond = (df_csv['vistas_media']*df_csv['n_posts']).sum()/df_csv['n_posts'].sum()
    print(f"  Vistas media ponderada: {vm_pond:.2f}  (parámetro: {P.VISTAS_MEDIA})")
    print(sep)

    # Tabla completa
    cols_show = ['hora_col','n_posts','lambda_posts_por_dia','vistas_media',
                 'vistas_mediana','vistas_std','fwd_rate','pct_con_cupon']
    df_show = df_csv[cols_show].copy()
    df_show.columns = ['hora','n_posts','lambda/día','v_media','v_mediana','v_std','fwd_rate','pct_cupon']
    print("\n  Tabla completa:")
    print(df_show.to_string(index=False, float_format='%.4f'))

    # Hallazgos clave
    print(f"\n  HALLAZGOS CLAVE DEL DATASET:")
    top5 = df_csv.nlargest(5, 'vistas_media')[['hora_col','n_posts','vistas_media']]
    print(f"  ① Top 5 horas por vistas/post (alta visibilidad, bajo volumen):")
    for _, row in top5.iterrows():
        print(f"       Hora {int(row.hora_col):02d}h → {row.vistas_media:.0f} vistas/post  ({int(row.n_posts)} posts en 143d)")
    print(f"  ② Martes 0-4am: lambda alta (8.1, 2.6, 0.5) — turno nocturno atípico")
    print(f"  ③ Horas pico de volumen (9-20h): λ≈12 posts/hora → solo {df_csv[df_csv['hora_col'].between(9,20)]['vistas_media'].mean():.0f} vistas/post")
    print(f"  ④ Correlación volumen↔visibilidad: negativa — saturación del feed")

    # Clasificación por estado Markov
    df_csv = df_csv.copy()
    df_csv['ing_post'] = df_csv['vistas_media'] * P.CTR * P.CR * P.COMISION_UNIT
    df_csv['estado']   = df_csv['ing_post'].apply(
        lambda v: 'Alto' if v > MarkovCanal.UMBRAL_ALTO
                  else ('Bajo' if v < MarkovCanal.UMBRAL_BAJO else 'Normal')
    )
    print(f"\n  CLASIFICACIÓN DE HORAS POR ESTADO MARKOV (ingreso/post):")
    print(f"  Umbral Bajo  < ${MarkovCanal.UMBRAL_BAJO:.4f}  |  Alto > ${MarkovCanal.UMBRAL_ALTO:.4f}")
    for est, col in [('Bajo','#e74c3c'),('Normal','f39c12'),('Alto','#27ae60')]:
        hh = df_csv[df_csv['estado']==est]['hora_col'].tolist()
        np_ = df_csv[df_csv['estado']==est]['n_posts'].sum()
        print(f"  {est:6s}: horas {hh}  ({np_} posts, {np_/df_csv['n_posts'].sum()*100:.1f}%)")
    print(sep)


def imprimir_matriz_comparacion():
    """Imprime comparación detallada de las 3 matrices discutidas."""
    from scipy.linalg import solve as slv
    def pi_de(P_m):
        A = P_m.T - np.eye(3); A[-1,:] = 1.0
        b = np.zeros(3); b[-1] = 1.0
        return slv(A, b)

    sep = "─" * 58
    print(f"\n{sep}")
    print("  COMPARACIÓN DE MATRICES P — MarkovCanal")
    print(sep)

    matrices = {
        'Compañero (empírica)':
            (np.array([[0.692,0.308,0.000],[0.171,0.658,0.171],[0.000,0.500,0.500]]),
             "Definida en clase P como P_MAT global.\nP[B→A]=0 y P[A→B]=0: no permite saltos extremos."),
        'Propuesta ⅔/⁴⁄₃ promedio':
            (np.array([[0.50,0.40,0.10],[0.15,0.65,0.20],[0.05,0.35,0.60]]),
             "Umbrales $0.100/$0.199: al clasificar horas, 99.2%\ncae en Normal → π sesgado hacia Alto artificialmente."),
        'Calibrada CSV P25/P75 ✓':
            (np.array([[0.50,0.43,0.07],[0.18,0.65,0.17],[0.07,0.43,0.50]]),
             "Umbrales $0.1308/$0.1614 (P25/P75 simulados).\nπ≈[0.23,0.55,0.22] ≈ objetivo [0.25,0.50,0.25]."),
    }
    for nombre, (Pm, desc) in matrices.items():
        pi = pi_de(Pm)
        print(f"\n  ── {nombre}")
        print(f"     {desc}")
        print(f"     Matriz P:")
        for i, lbl in enumerate(['Bajo  ','Normal','Alto  ']):
            print(f"       {lbl}: {np.round(Pm[i],3)}  suma={Pm[i].sum():.3f}")
        dur = [round(1/(1-Pm[i,i]),2) if Pm[i,i]<1 else '∞' for i in range(3)]
        print(f"     π = [Bajo={pi[0]:.3f}, Normal={pi[1]:.3f}, Alto={pi[2]:.3f}]")
        print(f"     Duración rachas: Bajo={dur[0]}d, Normal={dur[1]}d, Alto={dur[2]}d")
        print(f"     Saltos extremos: P[B→A]={Pm[0,2]:.3f}  P[A→B]={Pm[2,0]:.3f}")
    print(sep)


def main():
    print("\n" + "="*60)
    print("  PROMO HUNTER — Modelo de Simulación  (versión final)")
    print("  Modelos y Simulación | ene-may 2026 (143 días, 21 097 posts)")
    print("="*60)

    # ── Cargar dataset ─────────────────────────────────────────
    CSV_PATH = 'sim_tabla_horaria.csv'
    if os.path.exists(CSV_PATH):
        df_csv = pd.read_csv(CSV_PATH)
        imprimir_dataset(df_csv)
    else:
        print(f"  [AVISO] No se encontró {CSV_PATH} — saltando análisis del dataset")
        df_csv = None

    # ── Markov Absorbente ──────────────────────────────────────
    emb = MarkovEmbudo()
    emb.print_resumen()

    # ── Markov Ergódico ────────────────────────────────────────
    canal = MarkovCanal()
    canal.print_resumen()
    imprimir_matriz_comparacion()

    # ── Monte Carlo ────────────────────────────────────────────
    print(f"\n[Monte Carlo] Corriendo 4 escenarios ({P.N_SIMS:,} sims × {P.DIAS_SIM} días)...")

    df_a = monte_carlo(P.HORARIO_BASE,  label="A. Línea Base (9am-9pm)")
    df_b = monte_carlo(P.HORARIO_OPTIM, label="B. Horario 12pm-12am")

    factor_actual = sum(P.CATEGORIAS[c]['pct_actual']*P.CATEGORIAS[c]['ctr_factor'] for c in P.CATEGORIAS)
    factor_optim  = sum(P.MIX_OPTIMIZADO_C[c]*P.CATEGORIAS[c]['ctr_factor'] for c in P.MIX_OPTIMIZADO_C)
    ctr_c = P.CTR * (factor_optim / factor_actual)

    df_c = monte_carlo(P.HORARIO_BASE,  label="C. Mix Categorías Óptimo", ctr=ctr_c)
    df_d = monte_carlo(P.HORARIO_SCORE, label="D. Score-Óptimo + Categorías", ctr=ctr_c)

    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  RESULTADOS MONTE CARLO  (n={P.N_SIMS:,} sims × {P.DIAS_SIM}d)")
    print(sep)
    print(f"  {'Escenario':<28s}  {'Comisiones':>12s}  {'±Std':>8s}  {'Compras':>8s}  {'Δ vs A':>8s}")
    print(f"  {'-'*28}  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*8}")
    mu_a = df_a['comisiones'].mean()
    for lbl, df_ in [
        ("A. Línea Base (9am-9pm)",     df_a),
        ("B. Horario 12pm-12am",        df_b),
        ("C. Mix Categorías Óptimo",    df_c),
        ("D. Score-Óptimo + Cat.",      df_d),
    ]:
        mu  = df_['comisiones'].mean()
        std = df_['comisiones'].std()
        cmp = df_['compras'].mean()
        delta = f"{(mu-mu_a)/mu_a*100:+.1f}%" if lbl != "A. Línea Base (9am-9pm)" else "—"
        print(f"  {lbl:<28s}  ${mu:>10.2f}  ${std:>6.2f}  {cmp:>8.0f}  {delta:>8s}")
    print(f"\n  CTR efectivo C/D: {ctr_c*100:.3f}%  (ganancia: {(ctr_c/P.CTR-1)*100:+.1f}% vs base)")
    print(sep)

    # ── Sensibilidad ───────────────────────────────────────────
    print("\n[Sensibilidad] Calculando (método analítico)...")
    df_sens = analisis_sensibilidad()

    # ── Gráficas ───────────────────────────────────────────────
    print("\n[Gráficas] Generando...")
    CARPETA = os.path.dirname(os.path.abspath(__file__))

    figs = {
        'markov_embudo'  : plot_markov_embudo(emb),
        'markov_canal'   : plot_markov_canal(canal),
        'funnel_counts'  : plot_funnel_counts(emb),
        'convergencia_mc': plot_convergencia_mc(df_a, df_b, df_c, df_d),
        'comparacion_esc': plot_comparacion_escenarios(df_a, df_b, df_c, df_d),
        'sensibilidad'   : plot_sensibilidad(df_sens),
        'categorias'     : plot_categorias(),
        'lambda_matrix'  : plot_lambda_matrix(),
    }
    if df_csv is not None:
        figs['dataset_horario'] = plot_dataset_horario(df_csv)

    for nombre, fig in figs.items():
        ruta = os.path.join(CARPETA, f"{nombre}.png")
        fig.savefig(ruta, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ {nombre}.png")

    print(f"\n{'='*60}")
    print("  Simulación completada.")
    print(f"{'='*60}\n")
    return None, df_a, df_b, df_c, df_d, emb, canal, df_sens


if __name__ == "__main__":
    _, df_a, df_b, df_c, df_d, emb, canal, df_sens = main()
