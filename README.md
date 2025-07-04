# 🏠 Trabajo Final - Inteligencia de Negocios 2025

**Maestría en Economía Aplicada - Facultad de Ciencias Económicas - UBA**

## 📋 Descripción del Proyecto

Este proyecto implementa un sistema completo de **predicción de precios de inmuebles** utilizando técnicas de Machine Learning y Procesamiento de Lenguaje Natural (NLP). El objetivo es predecir el precio de propiedades basándose en características numéricas, categóricas y descripciones textuales.

## 🗂️ Estructura del Proyecto

### 📁 Archivos de Datos
- `train_bi_2025.csv` - Dataset de entrenamiento original (~400k registros)
- `test_bi_2025.csv` - Dataset de prueba para evaluación final
- `train_bi_2025_filtered.csv` - Dataset filtrado (generado automáticamente)

### 📓 Notebooks del Análisis

#### 1. **01_analisis_exploratorio.ipynb** 🔍
**Duración estimada:** 10-15 minutos
- ✅ Carga y exploración inicial del dataset
- ✅ Análisis de tipos de datos y valores faltantes
- ✅ Matriz de correlaciones y estadísticas descriptivas
- ✅ Análisis por tipo de propiedad con visualizaciones
- ✅ Detección y eliminación de outliers (método IQR)
- ✅ Optimización de variables temporales para ML
- ✅ Generación de dataset filtrado y optimizado

#### 2. **02_modelos_tradicionales.ipynb** 📈
**Duración estimada:** 5-10 minutos
- ✅ Preparación de datos con encoding automático
- ✅ **Regresión Lineal:** Análisis de coeficientes y significancia
- ✅ **LASSO:** Optimización con validación cruzada, selección de variables
- ✅ Interpretación económica de coeficientes
- ✅ Análisis de significatividad estadística vs práctica
- ✅ Comparación final con visualizaciones

#### 3. **03_modelos_ml.ipynb** 🤖
**Duración estimada:** 15-25 minutos
- ✅ **Random Forest:** Optimización con RandomizedSearchCV
- ✅ **XGBoost:** Configuración optimizada con verbose para seguimiento
- ✅ **Redes Neuronales (MLP):** Arquitecturas optimizadas
- ✅ **Análisis de overfitting:** Clasificación de robustez
- ✅ Importancia de variables y visualizaciones
- ✅ Comparación con métricas de generalización

#### 4. **04_modelos_nlp.ipynb** 📝
**Duración estimada:** 10-20 minutos
- ✅ **Procesamiento de texto:** Limpieza con stopwords en español
- ✅ **TF-IDF + SVD:** Reducción dimensional (500→100 componentes)
- ✅ **Modelos híbridos:** Combinación de features tradicionales + texto
- ✅ **Ridge, LASSO, RF, XGB, MLP:** Versiones híbridas
- ✅ **Análisis de términos:** Palabras más predictivas
- ✅ **Comparación integral:** Todos los modelos con análisis de robustez

## 🚀 Cómo Ejecutar el Proyecto

### Opción 1: Ejecución Secuencial (Recomendada)
```bash
1. Ejecutar: 01_analisis_exploratorio.ipynb
2. Ejecutar: 02_modelos_tradicionales.ipynb
3. Ejecutar: 03_modelos_ml.ipynb
4. Ejecutar: 04_modelos_nlp.ipynb
```

### Opción 2: Ejecución Independiente
Cada notebook es **auto-contenida** y puede ejecutarse independientemente. Cada una carga y prepara sus propios datos.

## 📦 Dependencias

### Librerías Requeridas:
```python
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
xgboost>=1.5.0
tqdm>=4.62.0
```

### Instalación:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost tqdm
```

## 📊 Resultados Obtenidos

### 🎯 Mejores Modelos por Categoría:

| Categoría | Mejor Modelo | RMSE | R² | Features | Overfitting |
|-----------|-------------|------|----|---------|-----------| 
| **Tradicional** | LASSO | $42,966 | 0.7284 | 77 | 0.991 ✅ |
| **Machine Learning** | XGBoost | $36,452 | 0.8045 | 93 | 1.012 ✅ |
| **NLP Híbrido** | **XGBoost Híbrido** | **$29,649** | **0.8707** | **193** | **1.114 ✅** |

### 🏆 **Modelo Ganador: XGBoost Híbrido**
- **RMSE:** $29,649 (mejor rendimiento absoluto)
- **R²:** 0.8707 (explica 87% de la variabilidad)
- **Overfitting Ratio:** 1.114 (buena generalización)
- **Features:** 193 (93 tradicionales + 100 texto SVD)
- **Mejora vs XGBoost sin texto:** 18.7%

### 📈 **Ranking Completo de Modelos (Top 5)**:
1. **XGBoost Híbrido** - $29,649 RMSE, R² 0.8707
2. **Random Forest Híbrido** - $31,330 RMSE, R² 0.8556 (⚠️ overfitting severo)
3. **Random Forest** - $33,518 RMSE, R² 0.8347 (⚠️ overfitting severo)
4. **MLP Híbrido** - $35,465 RMSE, R² 0.8150
5. **XGBoost** - $36,452 RMSE, R² 0.8045

## 🔍 Características del Dataset

### Variables Principales:
- **Numéricas:** rooms, bathrooms, surface_total, surface_covered
- **Temporales:** created_year, created_month, created_quarter, created_weekday
- **Categóricas:** l2, l3, prop_type  
- **Texto:** description (procesada con TF-IDF + SVD)
- **Target:** price

### Procesamiento:
- **Dataset filtrado:** 311,660 registros (eliminación de outliers)
- **Features tradicionales:** 93 (post one-hot encoding)
- **Features de texto:** 100 (post SVD de 500 términos TF-IDF)
- **Features híbridas:** 193 (tradicionales + texto)

## ⚡ Optimizaciones Implementadas

### Rendimiento:
- ✅ **Estrategia 2-fases:** Optimización en muestra + entrenamiento completo
- ✅ **RandomizedSearchCV** para búsqueda eficiente de hiperparámetros
- ✅ **Verbose=2** para seguimiento en tiempo real
- ✅ **SVD** para reducción dimensional del texto
- ✅ **Early stopping** en redes neuronales

### Robustez:
- ✅ **Análisis de overfitting:** Clasificación automática
- ✅ **Validación cruzada** en todos los modelos
- ✅ **Métricas múltiples:** RMSE, R², MAE, ratios de generalización
- ✅ **Comparación integral:** 11 modelos evaluados

## 📈 Insights Clave

### Variables Más Importantes (Modelos Tradicionales):
1. **surface_total** - Superficie total (+$49,878 por m²)
2. **bathrooms** - Número de baños (+$13,806 por baño)
3. **rooms** - Número de habitaciones (+$5,618 por habitación)
4. **Puerto Madero** - Ubicación premium (+$178,390)
5. **Villa Soldati** - Ubicación desfavorable (-$86,924)

### Impacto del NLP:
- ✅ **Mejora promedio:** 10.0% en RMSE
- ✅ **XGBoost:** 18.7% mejor con texto
- ✅ **MLP:** 4.8% mejor con texto
- ✅ **Ridge solo texto:** R² 0.4913 (aporte sustancial)

### Hallazgos de Robustez:
- ✅ **Modelos robustos:** XGBoost, MLP (ratio ≤ 1.15)
- ⚠️ **Overfitting severo:** Random Forest (ratio > 1.30)
- ✅ **Texto mejora robustez:** Modelos híbridos más estables

## 🎯 Aplicaciones Prácticas

### Para el Negocio:
- 🏠 **Sistema de valuación automática** (error típico $29,649)
- 📊 **Detección de precios anómalos** (R² 0.87)
- 📈 **Dashboard de monitoreo** por barrio/tipo
- 🤖 **Recomendación de precios** basada en descripción

### Para Inversores:
- 💰 **Identificación de oportunidades** (modelo vs mercado)
- 📊 **Análisis de mercado** por zona geográfica
- 🔍 **Evaluación de propiedades** con texto descriptivo

## 🔬 Metodología Técnica

### Análisis de Overfitting:
- **Excelente (≤1.05):** Generalización perfecta
- **Bueno (≤1.15):** Generalización aceptable
- **Moderado (≤1.30):** Overfitting controlado
- **Severo (>1.30):** Overfitting problemático

### Procesamiento NLP:
- **Limpieza:** Texto en español, stopwords personalizadas
- **Vectorización:** TF-IDF con bigramas
- **Reducción:** SVD 500→100 dimensiones (80% varianza)
- **Combinación:** Features tradicionales + texto

## 🔜 Trabajo Futuro

1. **Embeddings avanzados:** Word2Vec, FastText, BERT en español
2. **Análisis de sentimientos:** Polaridad de descripciones
3. **Geolocalización:** Features basadas en coordenadas
4. **Ensemble methods:** Combinación de mejores modelos
5. **Optimización de memoria:** Técnicas para datasets más grandes

## 🏆 Recomendación Final

### **Modelo Recomendado para Producción:**
**XGBoost Híbrido** ($29,649 RMSE, R² 0.8707)

**Justificación:**
- ✅ **Mejor rendimiento:** 31% mejor que LASSO
- ✅ **Robustez:** Overfitting controlado (1.114)
- ✅ **Escalabilidad:** Maneja bien features híbridas
- ✅ **Interpretabilidad:** Importancia de variables clara
- ✅ **Aplicabilidad:** Funciona con y sin descripción

## 👥 Información del Curso

- **Materia:** Inteligencia de Negocios
- **Institución:** Maestría en Economía Aplicada - UBA
- **Año:** 2025
- **Formato entrega:** Jupyter Notebooks ejecutados

## 📧 Entrega

- **Formato:** `tp_final_bi_2025_{apellido}.ipynb`
- **Email:** fmastelli@gmail.com
- **Asunto:** "TP final bi 2025 {apellido}"
- **Fecha límite:** 20 de julio, 23:59:59

---

## ✅ **Proyecto Completado Exitosamente**

**Todas las consignas del trabajo práctico han sido implementadas y documentadas con resultados reales.**

🎉 **¡Listo para entregar!**

### 📊 **Resumen de Logros:**
- ✅ **4 notebooks completos** con análisis exhaustivo
- ✅ **11 modelos evaluados** con métricas de robustez
- ✅ **Metodología híbrida** que combina ML + NLP
- ✅ **Análisis de overfitting** para recomendaciones de producción
- ✅ **Interpretabilidad económica** de todos los resultados