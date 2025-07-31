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
**Duración estimada:** 15-25 minutos
- ✅ **Procesamiento de texto:** Limpieza con stopwords en español
- ✅ **TF-IDF + SVD:** Reducción dimensional (5000→300 componentes)
- ✅ **Modelos híbridos:** Combinación de features tradicionales + texto
- ✅ **Random Forest NLP:** Optimización con HalvingRandomSearchCV
- ✅ **XGBoost NLP:** Optimización avanzada con GPU
- ✅ **Redes Neuronales NLP:** 3 arquitecturas diferentes (PyTorch GPU)
- ✅ **Análisis de términos:** Palabras más predictivas del precio
- ✅ **Comparación integral:** Modelos con y sin texto

#### 5. **06_evaluacion_final_performance.ipynb** 📊
**Duración estimada:** 5-10 minutos
- ✅ **Consigna 6:** Evaluación de 6 modelos optimizados sobre test
- ✅ **Análisis comparativo:** Modelos con vs sin descripciones
- ✅ **Métricas RMSE y MAE:** Reportes detallados
- ✅ **Conclusiones:** Información relevante en descripciones de texto
- ✅ **Visualizaciones:** Gráficos comparativos de performance

## 🚀 Cómo Ejecutar el Proyecto

### Opción 1: Ejecución Secuencial (Recomendada)
```bash
1. Ejecutar: 01_analisis_exploratorio.ipynb
2. Ejecutar: 02_modelos_tradicionales.ipynb
3. Ejecutar: 03_modelos_ml.ipynb
4. Ejecutar: 04_modelos_nlp.ipynb
5. Ejecutar: 06_evaluacion_final_performance.ipynb
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

| Categoría | Mejor Modelo | RMSE | R² | MAE | Features | Overfitting |
|-----------|-------------|------|----|----|---------|-------------|
| **Tradicional** | Regresión Lineal | $42,960 | 0.7285 | $30,922 | 93 | 0.991 ✅ |
| **Machine Learning** | XGBoost Optimizado | $34,623 | 0.8236 | $23,967 | 93 | 1.076 ✅ |
| **NLP Híbrido** | **XGBoost NLP** | **$26,471** | **0.8982** | **$17,284** | **393** | **1.487 ⚠️** |

### 🏆 **Modelo Ganador: XGBoost NLP**
- **RMSE:** $26,471 (mejor rendimiento absoluto)
- **R²:** 0.8982 (explica 89.8% de la variabilidad)
- **MAE:** $17,284 (error absoluto medio más bajo)
- **Overfitting Ratio:** 1.487 (overfitting moderado pero aceptable)
- **Features:** 393 (93 tradicionales + 300 texto TF-IDF/SVD)
- **Mejora vs XGBoost tradicional:** 23.5% en RMSE

### 📈 **Ranking Completo de Modelos (Top 8)**:
1. **XGBoost NLP** - $26,471 RMSE, R² 0.8982, MAE $17,284 ⚠️
2. **Red Neuronal NLP Simple** - $28,193 RMSE, R² 0.8845, MAE $19,013 ✅
3. **Random Forest NLP** - $29,900 RMSE, R² 0.8701, MAE $19,455 ❌
4. **XGBoost Optimizado** - $34,623 RMSE, R² 0.8236, MAE $23,967 ✅
5. **Red Neuronal NLP Estándar** - $35,097 RMSE, R² 0.8210, MAE $24,640 ✅
6. **Red Neuronal Optimizada** - $40,321 RMSE, R² 0.7608, MAE $28,231 ✅
7. **Random Forest Optimizado** - $40,837 RMSE, R² 0.7547, MAE $28,963 ✅
8. **Regresión Lineal** - $42,960 RMSE, R² 0.7285, MAE $30,922 ✅

## 🔍 Características del Dataset

### Variables Principales:
- **Numéricas:** rooms, bathrooms, surface_total, surface_covered
- **Temporales:** created_year, created_month, created_quarter, created_weekday
- **Categóricas:** l2, l3, prop_type  
- **Texto:** description (procesada con TF-IDF + SVD)
- **Target:** price

### Procesamiento:
- **Dataset original:** 400k+ registros
- **Dataset filtrado:** 311,660 registros (eliminación de outliers IQR)
- **Features tradicionales:** 93 (post one-hot encoding)
- **Features de texto:** 300 (post SVD de 5,000 términos TF-IDF)
- **Features híbridas:** 393 (tradicionales + texto)

## ⚡ Optimizaciones Implementadas

### Rendimiento:
- ✅ **HalvingRandomSearchCV** para búsqueda eficiente de hiperparámetros
- ✅ **PyTorch GPU** para redes neuronales aceleradas
- ✅ **SVD** para reducción dimensional del texto (5000→300)
- ✅ **Early stopping** en redes neuronales
- ✅ **Memory management** para datasets grandes

### Robustez:
- ✅ **Análisis de overfitting:** Clasificación automática por ratios
- ✅ **Validación cruzada** en optimización de hiperparámetros
- ✅ **Métricas múltiples:** RMSE, R², MAE, ratios de generalización
- ✅ **Comparación integral:** 8+ modelos evaluados
- ✅ **Evaluación separada:** Train, validación y test independientes

## 📈 Insights Clave

### Variables Más Importantes (Modelos Tradicionales):
1. **surface_total** - Superficie total (coef: +$49,878 por m²)
2. **bathrooms** - Número de baños (coef: +$13,806 por baño)
3. **rooms** - Número de habitaciones (coef: +$5,618 por habitación)
4. **l3_Puerto Madero** - Ubicación premium (coef: +$178,390)
5. **l3_Villa Soldati** - Ubicación desfavorable (coef: -$86,924)

### Variables Más Importantes (Modelos NLP):
1. **trad_surface_covered** - Superficie cubierta (34.1% importancia RF)
2. **trad_surface_total** - Superficie total (20.3% importancia RF)
3. **trad_l3_Puerto Madero** - Ubicación premium (6.1% importancia XGB)
4. **texto_dim_12** - Dimensión textual 12 (4.8% importancia RF)
5. **trad_bathrooms** - Número de baños (6.3% importancia RF)

### Impacto del NLP:
- ✅ **XGBoost:** 23.5% mejora en RMSE con texto
- ✅ **Red Neuronal Simple:** 30.1% mejora en RMSE con texto  
- ✅ **Random Forest:** 20.4% mejora en RMSE con texto
- ✅ **Contribución promedio:** ~25% mejora en performance
- ✅ **Información valiosa:** Las descripciones SÍ contienen información relevante

### Análisis de Overfitting (Clasificación por Ratios):
- ✅ **Excelente (≤1.05):** Red Neuronal NLP Estándar (1.007)
- ✅ **Bueno (≤1.15):** XGBoost Optimizado (1.076), Red Neuronal Optimizada (0.992)
- ⚠️ **Moderado (≤1.50):** XGBoost NLP (1.487)
- ❌ **Severo (>1.50):** Random Forest NLP (1.686)

## 🎯 Aplicaciones Prácticas

### Para el Negocio:
- 🏠 **Sistema de valuación automática** (error típico $26,471)
- 📊 **Detección de precios anómalos** (R² 0.898)
- 📈 **Dashboard de monitoreo** por barrio/tipo de propiedad
- 🤖 **Recomendación de precios** basada en descripción textual
- 💬 **Análisis de texto:** Identificación de términos que incrementan valor

### Para Inversores:
- 💰 **Identificación de oportunidades** (modelo vs mercado)
- 📊 **Análisis de mercado** por zona geográfica (l2, l3)
- 🔍 **Evaluación de propiedades** con descripciones optimizadas
- 📈 **Predicción de ROI** basada en características textuales

### Para Desarrolladores:
- 🏗️ **Optimización de descripciones** para maximizar valor percibido
- 📝 **Guidelines de marketing** basadas en términos más influyentes
- 🎯 **Segmentación de mercado** por preferencias textuales

## 🔬 Metodología Técnica

### Análisis de Overfitting (Ratios train/test RMSE):
- **Excelente (≤1.05):** Generalización perfecta
- **Bueno (≤1.15):** Generalización aceptable  
- **Moderado (≤1.50):** Overfitting controlado
- **Severo (>1.50):** Overfitting problemático

### Procesamiento NLP:
- **Limpieza:** Texto español, stopwords personalizadas + inmobiliarias
- **Vectorización:** TF-IDF con unigramas y bigramas (max_features=5000)
- **Reducción:** SVD 5000→300 dimensiones (retiene >80% varianza)
- **Combinación:** Features tradicionales (93) + texto (300) = 393 totales
- **Memoria:** Optimización para datasets grandes con batch processing

### Optimización de Hiperparámetros:
- **Random Forest:** HalvingRandomSearchCV con resource=n_estimators
- **XGBoost:** HalvingRandomSearchCV + early stopping
- **Redes Neuronales:** PyTorch con GPU, 3 arquitecturas diferentes
- **Validación:** 2-fold CV para eficiencia en datasets grandes

## 🔜 Trabajo Futuro

1. **Embeddings avanzados:** Word2Vec, FastText, BERT en español
2. **Análisis de sentimientos:** Polaridad y emociones en descripciones
3. **Geolocalización:** Features basadas en coordenadas lat/lon
4. **Ensemble methods:** Stacking de mejores modelos (XGB NLP + NN Simple)
5. **Optimización de memoria:** Streaming para datasets >1M registros
6. **Transfer learning:** Modelos pre-entrenados en español inmobiliario
7. **Feature engineering:** Interacciones texto-numéricas automáticas
8. **Deployment:** API REST para predicciones en tiempo real

## 🏆 Recomendación Final

### **Modelo Recomendado para Producción:**
**XGBoost NLP** ($26,471 RMSE, R² 0.8982, MAE $17,284)

**Justificación:**
- ✅ **Mejor rendimiento:** 38% mejor que modelos tradicionales
- ⚠️ **Overfitting moderado:** 1.487 ratio (aceptable para el rendimiento obtenido)
- ✅ **Escalabilidad:** Maneja bien 393 features híbridas
- ✅ **Interpretabilidad:** Feature importance clara y explicable
- ✅ **Aplicabilidad:** Funciona con y sin descripción de texto
- ✅ **ROI:** Reduce error de predicción significativamente vs alternativas

### **Modelo Alternativo Robusto:**
**Red Neuronal NLP Simple** ($28,193 RMSE, R² 0.8845)
- ✅ **Excelente generalización:** Sin overfitting
- ✅ **Performance sólida:** 2° mejor modelo
- ✅ **Estabilidad:** Ideal para ambientes conservadores

### **Estrategia Híbrida Recomendada:**
1. **Producción principal:** XGBoost NLP (máximo rendimiento)
2. **Validación cruzada:** Red Neuronal Simple (robustez)
3. **Fallback:** XGBoost tradicional (cuando falta descripción)

## 👥 Información del Curso

- **Materia:** Inteligencia de Negocios
- **Institución:** Maestría en Economía Aplicada - UBA
- **Año:** 2025
- **Formato entrega:** Jupyter Notebooks ejecutados

---