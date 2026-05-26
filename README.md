# 🏹 El Promo Hunter - Simulador del Embudo de Conversión Estocástico

Este repositorio contiene el **Gemelo Digital** y el modelo de simulación matemática desarrollado para el canal de Telegram de **"El Promo Hunter"** (marketing de afiliados bajo el programa de Amazon Associates). 

El proyecto implementa un enfoque avanzado de simulación de eventos discretos combinado con la teoría estocástica de **Cadenas de Markov** para analizar, predecir y optimizar las comisiones mensuales generadas por la plataforma.

---

## 📄 Descripción del Proyecto

El objetivo principal de este desarrollo es modelar formalmente el embudo de conversión y evaluar cuantitativamente el impacto de diferentes decisiones operativas (horarios de publicación y mezcla de categorías de productos) antes de su implementación real. El núcleo matemático busca maximizar la función objetivo de ganancias esperadas:

$$\mathbb{E}[G] = \sum \text{Comisiones Netas}$$

Este desarrollo se fundamenta de manera rigurosa en el documento técnico institucional **"Desarrollo del Modelo y la Simulación del Embudo de Conversión en el Canal de Telegram de El Promo Hunter"** (Mayo, 2026).

---

## 🏗️ Arquitectura del Modelo (Multi-Capa)

El sistema de software traduce computacionalmente un modelo estructurado en tres niveles jerárquicos e interconectados:

### 1. Capa Micro: El Embudo de Conversión (`MarkovEmbudo`)
Implementa una **Cadena de Markov Absorbente** con 5 estados lógicos:
* `Estado 0: Publicado` (Estado transitorio inicial)
* `Estado 1: Visto` (Estado transitorio)
* `Estado 2: Clic` (Estado transitorio)
* `Estado 3: Compra` (Estado absorbente de éxito)
* `Estado 4: Perdido` (Estado absorbente de salida)

El código calcula analíticamente la matriz fundamental $N = (I - Q)^{-1}$ y la probabilidad de absorción de éxito, validando de forma exacta que la probabilidad teórica de alcanzar una compra por publicación es de $5.77 \times 10^{-5}$.

### 2. Capa Media: Salud del Canal (`MarkovCanal`)
Modela el engagement global de largo plazo usando una **Cadena de Markov Ergódica**. Resuelve algebraicamente el sistema lineal para hallar el vector de estado estacionario $\pi$:
* **Distribución obtenida:** $\pi = [0.293, 0.527, 0.180]$
* **Conclusión:** Demuestra matemáticamente que el canal se mantendrá operando en un estado de rendimiento **Normal** el **52.7%** del tiempo a largo plazo.

### 3. Capa Macro: Simulación Monte Carlo (`simular_periodo`)
El motor de ejecución principal del simulador. Genera trayectorias estocásticas a lo largo de un horizonte temporal de **30 días**, ejecutando **500 réplicas independientes** mediante procesamiento paralelo (`ProcessPoolExecutor`) para garantizar la convergencia estadística.
* **Llegadas de Publicaciones:** Modeladas como un Proceso de Poisson No Homogéneo mediante una matriz de intensidad horaria diaria ($7 \times 24$).
* **Vistas por Publicación:** Variable aleatoria continua que sigue una distribución **LogNormal** ($\mu = 5.726, \sigma = 0.201$), afectada por factores estacionales de saturación según el bloque horario.
* **Interacciones (Clics y Compras):** Ensayos de Bernoulli simulados a través de distribuciones binomiales basadas en tasas CTR y CR reales indexadas por Tracking IDs.

---

## 🧪 Experimentos y Escenarios de Decisión

El código evalúa automáticamente cuatro alternativas estratégicas de negocio:

| Escenario | Nombre | Configuración Horaria | Estrategia de Contenido |
| :--- | :--- | :--- | :--- |
| **Escenario A** | Base Histórico | 09:00 a 21:00 | Distribución uniforme histórica de categorías. |
| **Escenario B** | Cambio de Horario | 12:00 a 23:00 | Desplazamiento hacia horas de mayor tráfico nocturno. |
| **Escenario C** | Mezcla Optimizada | 09:00 a 21:00 | Priorización de categorías con alto CTR (Deportes, Salud, Ropa). |
| **Escenario D** | Enfoque Combinado | Bloques Pico (8-11h, 17-21h) | Mezcla optimizada del Escenario C concentrada en horas pico. |

> 💡 **Hallazgo Contraintuitivo del Modelo:** Los experimentos demuestran que reducir o contraer el bloque operativo (Escenarios B y D) perjudica el rendimiento financiero neto. Aunque las vistas por post aumentan en horas pico, la pérdida en el volumen total acumulado de publicaciones diarias reduce significativamente las comisiones globales. La estrategia óptima resulta ser el **Escenario C**, incrementando las utilidades en un **5.3%** de manera eficiente.

---

## 🛠️ Requisitos e Instalación

El entorno requiere Python 3.8 o superior y las siguientes dependencias de análisis de datos y computación científica:

```bash
pip install numpy pandas scipy matplotlib
